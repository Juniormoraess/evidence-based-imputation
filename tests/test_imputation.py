"""Suíte de testes unitários para o Módulo 3 (Imputação Baseada em Evidências)."""

import json
from types import SimpleNamespace

from psycopg2 import sql
from pydantic import ValidationError
import pytest
from pytest_mock import MockerFixture

from src.imputation import (
    COLUMN_LINEAGE_MAP,
    ExtractionResult,
    build_extraction_prompt,
    extract_value_with_llm,
    run_imputation_pipeline,
    update_dw_record,
)
from src.retrieval import ImputationTask, RetrievedEvidence


# ===========================================================================
# 1. Testes de Sanitização e Validação Pydantic (ExtractionResult)
# ===========================================================================


@pytest.mark.parametrize(
    ("raw_input", "expected_float"),
    [
        ("£15,500.50", 15500.50),
        ("£ 12,345.67", 12345.67),
        ("15,000", 15000.0),
        ("GBP 980,100.25", 980100.25),
        (45000, 45000.0),
        (123.45, 123.45),
        (None, None),
        ("", None),
        ("null", None),
        ("N/A", None),
    ],
)
def test_extraction_result_sanitize_currency(
    raw_input: str | int | float | None,
    expected_float: float | None,
) -> None:
    """Deve converter strings monetárias variadas para float ou None automaticamente."""
    result = ExtractionResult(
        value_found=expected_float is not None,
        extracted_value=raw_input,  # type: ignore[arg-type]
        source_page=2,
        exact_quote="Total income £15,500.50",
        reasoning="Valor literal localizado na demonstração financeira.",
        confidence_score=0.95,
    )
    assert result.extracted_value == expected_float


def test_extraction_result_invalid_currency_and_confidence_bounds() -> None:
    """Deve rejeitar strings numéricas inválidas e scores de confiança fora de [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        ExtractionResult(
            value_found=True,
            extracted_value="quinze mil libras",  # type: ignore[arg-type]
            reasoning="Inválido",
            confidence_score=0.8,
        )

    with pytest.raises(ValidationError):
        ExtractionResult(
            value_found=True,
            extracted_value=1000.0,
            reasoning="Confiança acima do limite",
            confidence_score=1.5,
        )


# ===========================================================================
# 2. Testes de Construção do Prompt (build_extraction_prompt)
# ===========================================================================


def test_build_extraction_prompt_includes_metadata_auditing_rules_and_page_references() -> None:
    """Deve injetar charity_name, report_date, target_column, as 3 regras contábeis e evidências com [Page X]."""
    task = ImputationTask(
        row_id=42,
        document_id="charity_42.pdf",
        charity_name="Royal Voluntary Service",
        report_date="2024-03-31",
        target_column="income_annually_in_british_pounds",
    )
    evidences = [
        RetrievedEvidence(
            document_id="charity_42.pdf",
            page_number=3,
            chunk_index=5,
            text_content="Statement of Financial Activities: Total income £2,450,000.",
            similarity_score=0.92,
        ),
        RetrievedEvidence(
            document_id="charity_42.pdf",
            page_number=7,
            chunk_index=14,
            text_content="Notes to the accounts: Donations and legacies £1,100,000.",
            similarity_score=0.85,
        ),
    ]

    prompt = build_extraction_prompt(task=task, evidences=evidences)

    assert "Royal Voluntary Service" in prompt
    assert "2024-03-31" in prompt
    assert "income_annually_in_british_pounds" in prompt
    assert "[Page 3]" in prompt
    assert "Total income £2,450,000." in prompt
    assert "[Page 7]" in prompt
    assert "REGRA 1" in prompt
    assert "143,120.55" in prompt
    assert "Charitable activities" in prompt
    assert "Total expenditure" in prompt
    assert "REGRA 2" in prompt
    assert "F000's" in prompt
    assert "510,348,000" in prompt
    assert "REGRA 3" in prompt
    assert "Unrestricted" in prompt
    assert "value_found" in prompt
    assert "extracted_value" in prompt


# ===========================================================================
# 3. Testes de Extração com LLM (extract_value_with_llm) - Mockado
# ===========================================================================


def test_extract_value_with_llm_valid_json(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deve chamar o cliente OpenAI/DeepSeek com temperature=0.0, json_object e system prompt contábil."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    mocker.patch("src.imputation.llm_client.load_dotenv")

    llm_json_payload = json.dumps(
        {
            "value_found": True,
            "extracted_value": "£1,250,500.75",
            "source_page": 4,
            "exact_quote": "Total incoming resources for the year were £1,250,500.75",
            "reasoning": "Extraído diretamente da linha Total Incoming Resources na página 4.",
            "confidence_score": 0.98,
        }
    )

    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=llm_json_payload))]
    )
    mock_openai_cls = mocker.patch("src.imputation.llm_client.OpenAI")
    mock_client = mock_openai_cls.return_value
    mock_client.chat.completions.create.return_value = mock_response

    result = extract_value_with_llm("Prompt de teste")

    mock_openai_cls.assert_called_once_with(
        api_key="sk-deepseek-test-key",
        base_url="https://api.deepseek.com",
    )
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-chat"
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["response_format"] == {"type": "json_object"}
    system_msg = call_kwargs["messages"][0]["content"]
    assert "REGRA 1" in system_msg
    assert "REGRA 2" in system_msg
    assert "REGRA 3" in system_msg

    assert result.value_found is True
    assert result.extracted_value == 1250500.75
    assert result.source_page == 4
    assert result.exact_quote == "Total incoming resources for the year were £1,250,500.75"
    assert result.confidence_score == 0.98


def test_extract_value_with_llm_corrupted_json_returns_defensive_fallback(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deve capturar ValidationError em JSON malformado/alucinado e retornar fallback seguro."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test-key")
    mocker.patch("src.imputation.llm_client.load_dotenv")

    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{json_invalido: true"))]
    )
    mock_openai_cls = mocker.patch("src.imputation.llm_client.OpenAI")
    mock_openai_cls.return_value.chat.completions.create.return_value = mock_response

    result = extract_value_with_llm("Prompt de teste")

    assert result.value_found is False
    assert result.extracted_value is None
    assert result.source_page is None
    assert result.exact_quote is None
    assert result.reasoning == "LLM parsing or API error"
    assert result.confidence_score == 0.0


def test_extract_value_with_llm_api_exception_returns_defensive_fallback(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deve retornar fallback seguro quando a chamada à API do DeepSeek lançar exceção."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test-key")
    mocker.patch("src.imputation.llm_client.load_dotenv")

    mock_openai_cls = mocker.patch("src.imputation.llm_client.OpenAI")
    mock_openai_cls.return_value.chat.completions.create.side_effect = RuntimeError(
        "API Connection Timeout"
    )

    result = extract_value_with_llm("Prompt de teste")

    assert result.value_found is False
    assert result.extracted_value is None
    assert result.reasoning == "LLM parsing or API error"
    assert result.confidence_score == 0.0


# ===========================================================================
# 4. Testes de Escrita no Data Warehouse (update_dw_record) - Mockado
# ===========================================================================


def test_update_dw_record_when_value_found_true(mocker: MockerFixture) -> None:
    """Deve executar UPDATE com sql.Identifier e preencher todas as colunas de linhagem."""
    mock_connect = mocker.patch("src.imputation.writer.psycopg2.connect")
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    task = ImputationTask(
        row_id=15,
        document_id="doc_15.pdf",
        charity_name="Test Charity",
        report_date="2024-03-31",
        target_column="income_annually_in_british_pounds",
    )
    result = ExtractionResult(
        value_found=True,
        extracted_value=750000.00,
        source_page=None,  # Deve usar evidences[0].page_number (página 6) como fallback
        exact_quote="Total income for the year ended 31 March 2024 was £750,000.",
        reasoning="Valor exato localizado no demonstrativo financeiro.",
        confidence_score=0.94,
    )
    evidences = [
        RetrievedEvidence(
            document_id="doc_15.pdf",
            page_number=6,
            chunk_index=10,
            text_content="Total income for the year ended 31 March 2024 was £750,000.",
            similarity_score=0.91,
        )
    ]

    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"
    updated = update_dw_record(
        task=task,
        result=result,
        evidences=evidences,
        db_connection_string=dsn,
    )

    assert updated is True
    mock_connect.assert_called_once_with(dsn)
    mock_cursor.execute.assert_called_once()

    query_obj, params = mock_cursor.execute.call_args.args
    assert isinstance(query_obj, sql.Composed)
    assert sql.Identifier("income_annually_in_british_pounds") in query_obj._wrapped
    assert sql.Identifier("income_pagina") in query_obj._wrapped
    assert sql.Identifier("income_trecho_evidenciado") in query_obj._wrapped
    assert sql.Identifier("income_confianca") in query_obj._wrapped
    assert sql.Identifier("spending_pagina") not in query_obj._wrapped
    assert params == (
        750000.00,
        6,  # income_pagina (fallback para evidences[0].page_number)
        "Total income for the year ended 31 March 2024 was £750,000.",  # income_trecho_evidenciado
        0.94,  # income_confianca
        "doc_15.pdf",  # fonte_documento
        6,  # pagina (retrocompatibilidade)
        "Total income for the year ended 31 March 2024 was £750,000.",  # trecho_evidenciado
        0.94,  # confianca
        15,  # row_id
    )
    mock_conn.commit.assert_called_once()


def test_update_dw_record_separate_lineage_for_income_and_spending(
    mocker: MockerFixture,
) -> None:
    """Deve atualizar income_* e spending_* separadamente sem que spending sobrescreva income_*."""
    mock_connect = mocker.patch("src.imputation.writer.psycopg2.connect")
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    assert COLUMN_LINEAGE_MAP["income_annually_in_british_pounds"] == (
        "income_pagina",
        "income_trecho_evidenciado",
        "income_confianca",
    )
    assert COLUMN_LINEAGE_MAP["spending_annually_in_british_pounds"] == (
        "spending_pagina",
        "spending_trecho_evidenciado",
        "spending_confianca",
    )

    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"

    # 1. Atualização de Income (página 14)
    income_task = ImputationTask(
        row_id=20,
        document_id="doc_20.pdf",
        charity_name="Charity Twenty",
        report_date="2024-03-31",
        target_column="income_annually_in_british_pounds",
    )
    income_result = ExtractionResult(
        value_found=True,
        extracted_value=143120.55,
        source_page=14,
        exact_quote="Total Income 143,120.55",
        reasoning="Tabela Profit & Loss na página 14.",
        confidence_score=0.99,
    )
    update_dw_record(income_task, income_result, [], dsn)

    # 2. Atualização de Spending (página 15) na mesma linha row_id=20
    spending_task = ImputationTask(
        row_id=20,
        document_id="doc_20.pdf",
        charity_name="Charity Twenty",
        report_date="2024-03-31",
        target_column="spending_annually_in_british_pounds",
    )
    spending_result = ExtractionResult(
        value_found=True,
        extracted_value=108294.79,
        source_page=15,
        exact_quote="Total Expense 108,294.79",
        reasoning="Tabela Profit & Loss na página 15.",
        confidence_score=0.97,
    )
    update_dw_record(spending_task, spending_result, [], dsn)

    assert mock_cursor.execute.call_count == 2

    income_query, income_params = mock_cursor.execute.call_args_list[0].args
    assert sql.Identifier("income_pagina") in income_query._wrapped
    assert sql.Identifier("income_trecho_evidenciado") in income_query._wrapped
    assert sql.Identifier("income_confianca") in income_query._wrapped
    assert sql.Identifier("spending_pagina") not in income_query._wrapped
    assert income_params[:4] == (143120.55, 14, "Total Income 143,120.55", 0.99)

    spending_query, spending_params = mock_cursor.execute.call_args_list[1].args
    assert sql.Identifier("spending_pagina") in spending_query._wrapped
    assert sql.Identifier("spending_trecho_evidenciado") in spending_query._wrapped
    assert sql.Identifier("spending_confianca") in spending_query._wrapped
    assert sql.Identifier("income_pagina") not in spending_query._wrapped
    assert spending_params[:4] == (108294.79, 15, "Total Expense 108,294.79", 0.97)


def test_update_dw_record_when_value_found_false_does_not_write(
    mocker: MockerFixture,
) -> None:
    """Não deve abrir conexão nem executar UPDATE quando value_found=False."""
    mock_connect = mocker.patch("src.imputation.writer.psycopg2.connect")

    task = ImputationTask(
        row_id=15,
        document_id="doc_15.pdf",
        charity_name="Test Charity",
        report_date="2024-03-31",
        target_column="spending_annually_in_british_pounds",
    )
    result = ExtractionResult(
        value_found=False,
        extracted_value=None,
        source_page=None,
        exact_quote=None,
        reasoning="Métrica ausente nos fragmentos recuperados.",
        confidence_score=0.0,
    )

    updated = update_dw_record(
        task=task,
        result=result,
        evidences=[],
        db_connection_string="postgresql://postgres:secret@localhost:5432/evidence_db",
    )

    assert updated is False
    mock_connect.assert_not_called()


def test_update_dw_record_rejects_sql_injection_column() -> None:
    """Deve bloquear qualquer target_column fora da allowlist estrita."""
    malicious_task = ImputationTask(
        row_id=1,
        document_id="doc.pdf",
        target_column="income_annually_in_british_pounds; DROP TABLE dw_charity_data;--",
    )
    result = ExtractionResult(
        value_found=True,
        extracted_value=100.0,
        source_page=1,
        exact_quote="£100",
        reasoning="Teste",
        confidence_score=1.0,
    )

    with pytest.raises(ValueError, match="não permitida"):
        update_dw_record(
            task=malicious_task,
            result=result,
            evidences=[],
            db_connection_string="postgresql://postgres:secret@localhost:5432/evidence_db",
        )


# ===========================================================================
# 5. Teste de Orquestração Completa do Pipeline (run_imputation_pipeline)
# ===========================================================================


def test_run_imputation_pipeline_orchestration(mocker: MockerFixture) -> None:
    """Deve orquestrar busca de tasks, retrieval de evidências (default top_k=12), extração LLM e escrita no DW."""
    task_1 = ImputationTask(
        row_id=1,
        document_id="doc_1.pdf",
        charity_name="Charity One",
        report_date="2024-03-31",
        target_column="income_annually_in_british_pounds",
    )
    task_2 = ImputationTask(
        row_id=2,
        document_id="doc_2_no_chunks.pdf",
        charity_name="Charity Two",
        report_date="2024-03-31",
        target_column="spending_annually_in_british_pounds",
    )
    task_3 = ImputationTask(
        row_id=3,
        document_id="doc_3.pdf",
        charity_name="Charity Three",
        report_date="2024-03-31",
        target_column="spending_annually_in_british_pounds",
    )

    mocker.patch(
        "src.imputation.pipeline.get_pending_tasks",
        return_value=[task_1, task_2, task_3],
    )
    mocker.patch(
        "src.imputation.pipeline.embed_search_query",
        return_value=[0.1] * 1536,
    )

    evidence_sample = [
        RetrievedEvidence(
            document_id="doc_1.pdf",
            page_number=2,
            chunk_index=0,
            text_content="Total income: £500,000",
            similarity_score=0.95,
        )
    ]

    # task_1 tem evidências e encontra valor; task_2 não tem evidências; task_3 tem evidências mas LLM retorna False
    mock_fetch = mocker.patch(
        "src.imputation.pipeline.fetch_evidence",
        side_effect=[evidence_sample, [], evidence_sample],
    )

    result_found = ExtractionResult(
        value_found=True,
        extracted_value=500000.0,
        source_page=2,
        exact_quote="Total income: £500,000",
        reasoning="Encontrado.",
        confidence_score=0.95,
    )
    result_not_found = ExtractionResult(
        value_found=False,
        extracted_value=None,
        source_page=None,
        exact_quote=None,
        reasoning="Não encontrado.",
        confidence_score=0.0,
    )

    mocker.patch(
        "src.imputation.pipeline.extract_value_with_llm",
        side_effect=[result_found, result_not_found],
    )
    mock_update = mocker.patch(
        "src.imputation.pipeline.update_dw_record",
        side_effect=[True, False],
    )

    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"
    summary = run_imputation_pipeline(db_connection_string=dsn)

    assert summary == {
        "total_tasks": 3,
        "imputed_count": 1,
        "skipped_count": 2,
    }
    assert mock_fetch.call_args.kwargs["top_k"] == 12
    assert mock_update.call_count == 2
