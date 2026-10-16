"""Suíte de testes unitários para o Módulo 2 (Recuperação / Retrieval)."""

from datetime import date
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from src.retrieval import (
    ImputationTask,
    RetrievedEvidence,
    embed_search_query,
    fetch_evidence,
    get_pending_tasks,
)


# ===========================================================================
# 1. Testes de Mapeamento de Lacunas (get_pending_tasks)
# ===========================================================================


def test_get_pending_tasks_generates_one_or_two_tasks_per_row(mocker: MockerFixture) -> None:
    """Deve gerar 1 task quando apenas uma coluna é NULL e 2 tasks quando ambas são NULL."""
    mock_connect = mocker.patch("src.retrieval.search.psycopg2.connect")
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    # Linha 1: apenas income é NULL -> 1 task
    # Linha 2: apenas spending é NULL -> 1 task
    # Linha 3: ambas as colunas são NULL -> 2 tasks
    mock_cursor.fetchall.return_value = [
        (10, "doc_1.pdf", "British Red Cross", date(2024, 3, 31), None, 450000.00),
        (20, "doc_2.pdf", "Save The Children", date(2024, 12, 31), 800000.00, None),
        (30, "doc_3.pdf", "Oxfam GB", date(2025, 6, 30), None, None),
    ]

    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"
    tasks = get_pending_tasks(dsn)

    mock_connect.assert_called_once_with(dsn)
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args.args[0]
    assert "FROM dw_charity_data" in executed_sql
    assert "income_annually_in_british_pounds IS NULL" in executed_sql
    assert "spending_annually_in_british_pounds IS NULL" in executed_sql

    assert len(tasks) == 4

    # Task 0 (Linha 10 - income)
    assert tasks[0] == ImputationTask(
        row_id=10,
        document_id="doc_1.pdf",
        charity_name="British Red Cross",
        report_date="2024-03-31",
        target_column="income_annually_in_british_pounds",
    )

    # Task 1 (Linha 20 - spending)
    assert tasks[1] == ImputationTask(
        row_id=20,
        document_id="doc_2.pdf",
        charity_name="Save The Children",
        report_date="2024-12-31",
        target_column="spending_annually_in_british_pounds",
    )

    # Tasks 2 e 3 (Linha 30 - income e spending)
    assert tasks[2] == ImputationTask(
        row_id=30,
        document_id="doc_3.pdf",
        charity_name="Oxfam GB",
        report_date="2025-06-30",
        target_column="income_annually_in_british_pounds",
    )
    assert tasks[3] == ImputationTask(
        row_id=30,
        document_id="doc_3.pdf",
        charity_name="Oxfam GB",
        report_date="2025-06-30",
        target_column="spending_annually_in_british_pounds",
    )


def test_get_pending_tasks_invalid_dsn() -> None:
    """Deve lançar ValueError se a string de conexão for vazia."""
    with pytest.raises(ValueError, match="db_connection_string"):
        get_pending_tasks("   ")


# ===========================================================================
# 2. Testes de Geração da Query Semântica (embed_search_query) - Mockado
# ===========================================================================


def test_embed_search_query_formats_and_embeds_income_and_spending(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deve formatar a query semântica combinando coluna, charity_name e report_date e retornar 1536 dims."""
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-retrieval-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    mocker.patch("src.retrieval.search.load_dotenv")

    fake_embedding = [0.05] * 1536
    mock_response = SimpleNamespace(data=[SimpleNamespace(embedding=fake_embedding)])

    mock_openai_cls = mocker.patch("src.retrieval.search.OpenAI")
    mock_client = mock_openai_cls.return_value
    mock_client.embeddings.create.return_value = mock_response

    income_task = ImputationTask(
        row_id=1,
        document_id="charity_report.pdf",
        charity_name="Age UK",
        report_date="2024-03-31",
        target_column="income_annually_in_british_pounds",
    )

    vector = embed_search_query(income_task)

    assert len(vector) == 1536
    assert vector == fake_embedding
    mock_openai_cls.assert_called_once_with(api_key="sk-test-retrieval-key")

    call_kwargs = mock_client.embeddings.create.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-3-small"
    query_sent = call_kwargs["input"][0]
    assert "income" in query_sent.lower()
    assert "Age UK" in query_sent
    assert "2024-03-31" in query_sent

    # Testa também para a coluna de despesas (spending)
    spending_task = ImputationTask(
        row_id=1,
        document_id="charity_report.pdf",
        charity_name="Age UK",
        report_date="2024-03-31",
        target_column="spending_annually_in_british_pounds",
    )
    embed_search_query(spending_task)
    spending_query_sent = mock_client.embeddings.create.call_args.kwargs["input"][0]
    assert "spending" in spending_query_sent.lower()
    assert "Age UK" in spending_query_sent
    assert "2024-03-31" in spending_query_sent


def test_embed_search_query_missing_api_key(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deve lançar ValueError quando EMBEDDING_API_KEY não estiver definida."""
    mocker.patch("src.retrieval.search.load_dotenv")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    task = ImputationTask(
        row_id=1,
        document_id="doc.pdf",
        target_column="income_annually_in_british_pounds",
    )

    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        embed_search_query(task)


# ===========================================================================
# 3. Testes de Busca Híbrida e Prevenção de Alucinação (fetch_evidence) - Mockado
# ===========================================================================


def test_fetch_evidence_executes_hybrid_search_with_hard_filtering(
    mocker: MockerFixture,
) -> None:
    """Deve filtrar obrigatoriamente por document_id e ordenar por distância de cosseno (<=>)."""
    mock_connect = mocker.patch("src.retrieval.search.psycopg2.connect")
    mock_register_vector = mocker.patch("src.retrieval.search.register_vector")

    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    doc_id = "1ada336f29e8247f9f55a8d7e1b1c0da.pdf"
    mock_cursor.fetchall.return_value = [
        (doc_id, 4, 12, "Total incoming resources: £1,250,000", 0.9123),
        (doc_id, 5, 15, "Total resources expended: £1,100,000", 0.8450),
    ]

    task = ImputationTask(
        row_id=7,
        document_id=doc_id,
        charity_name="Sample Charity",
        report_date="2024-03-31",
        target_column="income_annually_in_british_pounds",
    )
    query_vec = [0.1] * 1536
    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"

    evidences = fetch_evidence(
        task=task,
        query_embedding=query_vec,
        db_connection_string=dsn,
        top_k=5,
    )

    mock_connect.assert_called_once_with(dsn)
    mock_register_vector.assert_called_once_with(mock_conn)
    mock_cursor.execute.assert_called_once()

    executed_sql, executed_params = mock_cursor.execute.call_args.args
    assert "WHERE document_id = %s" in executed_sql
    assert "1 - (embedding <=> %s::vector) AS similarity_score" in executed_sql
    assert "ORDER BY embedding <=> %s::vector" in executed_sql
    assert "LIMIT %s" in executed_sql
    assert executed_params == (query_vec, doc_id, query_vec, 5)

    assert len(evidences) == 2
    assert evidences[0] == RetrievedEvidence(
        document_id=doc_id,
        page_number=4,
        chunk_index=12,
        text_content="Total incoming resources: £1,250,000",
        similarity_score=0.9123,
    )
    assert evidences[1] == RetrievedEvidence(
        document_id=doc_id,
        page_number=5,
        chunk_index=15,
        text_content="Total resources expended: £1,100,000",
        similarity_score=0.8450,
    )


def test_fetch_evidence_returns_empty_list_when_document_has_no_chunks(
    mocker: MockerFixture,
) -> None:
    """Se o document_id não possuir chunks, deve retornar [] sem realizar fallback em outros documentos."""
    mock_connect = mocker.patch("src.retrieval.search.psycopg2.connect")
    mocker.patch("src.retrieval.search.register_vector")

    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
    mock_cursor.fetchall.return_value = []

    task = ImputationTask(
        row_id=99,
        document_id="missing_doc.pdf",
        charity_name="Ghost Charity",
        report_date="2024-01-01",
        target_column="spending_annually_in_british_pounds",
    )
    query_vec = [0.2] * 1536
    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"

    evidences = fetch_evidence(
        task=task,
        query_embedding=query_vec,
        db_connection_string=dsn,
        top_k=3,
    )

    assert evidences == []
    # Garante que apenas 1 query foi executada (nenhum fallback sem WHERE document_id = %s)
    assert mock_cursor.execute.call_count == 1
    executed_sql, executed_params = mock_cursor.execute.call_args.args
    assert "WHERE document_id = %s" in executed_sql
    assert executed_params[1] == "missing_doc.pdf"
