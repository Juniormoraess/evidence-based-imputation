"""Suíte de testes unitários para o Módulo 1 (Ingestão Raw-RAG)."""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List

import fitz
import pytest
from pytest_mock import MockerFixture

from src.ingestion import (
    DocumentChunk,
    create_chunks,
    extract_text_from_pdf,
    generate_embeddings,
    ingest_directory,
    save_chunks_to_db,
)


# ===========================================================================
# 1. Testes de Extração de Texto (extract_text_from_pdf)
# ===========================================================================


def test_extract_text_from_pdf_success(tmp_path: Path) -> None:
    """Deve extrair texto de múltiplas páginas mantendo page_number 1-indexed."""
    pdf_file = tmp_path / "sample_report.pdf"
    with fitz.open() as doc:
        page1 = doc.new_page()
        page1.insert_text((72, 72), "Annual Charity Report - Page 1: Income £500,000")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Financial Breakdown - Page 2: Spending £450,000")
        doc.save(str(pdf_file))

    pages_data = extract_text_from_pdf(str(pdf_file))

    assert len(pages_data) == 2
    assert pages_data[0]["page_number"] == 1
    assert "Income £500,000" in str(pages_data[0]["text"])
    assert pages_data[1]["page_number"] == 2
    assert "Spending £450,000" in str(pages_data[1]["text"])


def test_extract_text_from_pdf_file_not_found() -> None:
    """Deve lançar FileNotFoundError quando o arquivo PDF não existe."""
    with pytest.raises(FileNotFoundError, match="Arquivo PDF não encontrado"):
        extract_text_from_pdf("data/raw/nao_existe.pdf")


def test_extract_text_from_pdf_corrupted_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Deve logar erro e lançar RuntimeError ao encontrar um arquivo PDF corrompido."""
    corrupted_pdf = tmp_path / "corrupted.pdf"
    corrupted_pdf.write_bytes(b"conteudo invalido que nao e um PDF")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Erro ao processar PDF"):
            extract_text_from_pdf(str(corrupted_pdf))

    assert "Falha ao ler o arquivo PDF" in caplog.text


# ===========================================================================
# 2. Testes de Fragmentação / Chunking (create_chunks)
# ===========================================================================


def test_create_chunks_sliding_window_and_overlap() -> None:
    """Deve fragmentar texto usando janela deslizante com sobreposição exata e ignorar páginas em branco."""
    doc_id = "1ada336f29e8247f9f55a8d7e1b1c0da.pdf"
    # Cria texto de 1500 caracteres na página 1, página 2 em branco e 400 caracteres na página 3
    page_1_text = "A" * 800 + "B" * 200 + "C" * 500
    page_3_text = "D" * 400

    pages_data: List[dict[str, int | str]] = [
        {"page_number": 1, "text": page_1_text},
        {"page_number": 2, "text": "   \n\t  "},  # Página em branco deve ser ignorada
        {"page_number": 3, "text": page_3_text},
    ]

    chunks = create_chunks(
        document_id=doc_id,
        pages_data=pages_data,
        chunk_size=1000,
        overlap=200,
    )

    assert len(chunks) == 3

    # Chunk 0: primeiros 1000 chars da página 1 (800 'A' + 200 'B')
    assert chunks[0].document_id == doc_id
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert len(chunks[0].text_content) == 1000
    assert chunks[0].text_content.endswith("B" * 200)
    assert chunks[0].embedding is None

    # Chunk 1: sobreposição de 200 'B' + 500 'C' = 700 chars na página 1
    assert chunks[1].document_id == doc_id
    assert chunks[1].page_number == 1
    assert chunks[1].chunk_index == 1
    assert len(chunks[1].text_content) == 700
    assert chunks[1].text_content.startswith("B" * 200)
    assert chunks[1].text_content.endswith("C" * 500)

    # Chunk 2: página 3 (página 2 foi ignorada)
    assert chunks[2].document_id == doc_id
    assert chunks[2].page_number == 3
    assert chunks[2].chunk_index == 2
    assert chunks[2].text_content == page_3_text


def test_create_chunks_invalid_parameters() -> None:
    """Deve validar parâmetros inválidos de chunk_size, overlap e document_id."""
    pages_data: List[dict[str, int | str]] = [{"page_number": 1, "text": "Texto de teste"}]

    with pytest.raises(ValueError, match="document_id"):
        create_chunks(document_id="", pages_data=pages_data)

    with pytest.raises(ValueError, match="chunk_size"):
        create_chunks(document_id="doc.pdf", pages_data=pages_data, chunk_size=0)

    with pytest.raises(ValueError, match="overlap"):
        create_chunks(
            document_id="doc.pdf", pages_data=pages_data, chunk_size=500, overlap=500
        )


# ===========================================================================
# 3. Testes de Geração de Embeddings (generate_embeddings) - Mockado
# ===========================================================================


def test_generate_embeddings_success(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deve gerar embeddings em lote usando o cliente OpenAI mockado sem chamadas reais à rede."""
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-fake-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    mocker.patch("src.ingestion.embedder.load_dotenv")

    fake_vector_1 = [0.01] * 1536
    fake_vector_2 = [0.02] * 1536
    mock_response = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=fake_vector_1),
            SimpleNamespace(embedding=fake_vector_2),
        ]
    )

    mock_openai_cls = mocker.patch("src.ingestion.embedder.OpenAI")
    mock_client = mock_openai_cls.return_value
    mock_client.embeddings.create.return_value = mock_response

    chunks = [
        DocumentChunk(
            document_id="charity_01.pdf",
            page_number=1,
            chunk_index=0,
            text_content="Total income: £1,200,000",
        ),
        DocumentChunk(
            document_id="charity_01.pdf",
            page_number=1,
            chunk_index=1,
            text_content="Total spending: £980,000",
        ),
    ]

    result = generate_embeddings(chunks)

    mock_openai_cls.assert_called_once_with(api_key="sk-test-fake-key")
    mock_client.embeddings.create.assert_called_once_with(
        input=["Total income: £1,200,000", "Total spending: £980,000"],
        model="text-embedding-3-small",
    )

    assert len(result) == 2
    assert result[0].embedding == fake_vector_1
    assert result[1].embedding == fake_vector_2
    assert len(result[0].embedding) == 1536


def test_generate_embeddings_missing_api_key(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deve lançar ValueError se EMBEDDING_API_KEY não estiver definida."""
    mocker.patch("src.ingestion.embedder.load_dotenv")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    chunks = [
        DocumentChunk(
            document_id="charity_01.pdf",
            page_number=1,
            chunk_index=0,
            text_content="Conteúdo qualquer",
        )
    ]

    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        generate_embeddings(chunks)


# ===========================================================================
# 4. Testes de Persistência e Idempotência no Banco (save_chunks_to_db) - Mockado
# ===========================================================================


def test_save_chunks_to_db_idempotent_transaction(mocker: MockerFixture) -> None:
    """Deve executar DELETE por document_id seguido de INSERTs dentro da mesma transação."""
    mock_connect = mocker.patch("src.ingestion.repository.psycopg2.connect")
    mock_register_vector = mocker.patch("src.ingestion.repository.register_vector")

    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    vec_1 = [0.1] * 1536
    vec_2 = [0.2] * 1536
    doc_id = "1ada336f29e8247f9f55a8d7e1b1c0da.pdf"

    chunks = [
        DocumentChunk(
            document_id=doc_id,
            page_number=1,
            chunk_index=0,
            text_content="Fragmento 1",
            embedding=vec_1,
        ),
        DocumentChunk(
            document_id=doc_id,
            page_number=2,
            chunk_index=1,
            text_content="Fragmento 2",
            embedding=vec_2,
        ),
    ]

    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"
    save_chunks_to_db(chunks, dsn)

    mock_connect.assert_called_once_with(dsn)
    mock_register_vector.assert_called_once_with(mock_conn)

    # 1 chamada de DELETE + 2 chamadas de INSERT = 3 execuções no cursor
    assert mock_cursor.execute.call_count == 3

    first_call_sql, first_call_params = mock_cursor.execute.call_args_list[0].args
    assert "DELETE FROM knowledge_base WHERE document_id = %s" in first_call_sql
    assert first_call_params == (doc_id,)

    second_call_sql, second_call_params = mock_cursor.execute.call_args_list[1].args
    assert "INSERT INTO knowledge_base" in second_call_sql
    assert second_call_params == (doc_id, 1, 0, "Fragmento 1", vec_1)

    third_call_sql, third_call_params = mock_cursor.execute.call_args_list[2].args
    assert "INSERT INTO knowledge_base" in third_call_sql
    assert third_call_params == (doc_id, 2, 1, "Fragmento 2", vec_2)

    mock_conn.commit.assert_called_once()


def test_save_chunks_to_db_raises_without_embedding() -> None:
    """Deve rejeitar a persistência caso algum chunk não possua embedding calculado."""
    chunk_without_embedding = DocumentChunk(
        document_id="doc.pdf",
        page_number=1,
        chunk_index=0,
        text_content="Texto sem embedding",
        embedding=None,
    )

    with pytest.raises(ValueError, match="não possui vetor de embedding"):
        save_chunks_to_db(
            [chunk_without_embedding],
            "postgresql://postgres:secret@localhost:5432/evidence_db",
        )


# ===========================================================================
# 5. Teste de Resiliência do Pipeline em Lote (ingest_directory)
# ===========================================================================


def test_ingest_directory_skips_corrupted_and_blank_pdfs(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Arquivos corrompidos ou em branco devem ser logados e pulados sem derrubar o lote."""
    # 1. PDF corrompido
    bad_pdf = tmp_path / "01_corrupted.pdf"
    bad_pdf.write_bytes(b"dados invalidos")

    # 2. PDF em branco (sem texto)
    blank_pdf = tmp_path / "02_blank.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(str(blank_pdf))

    # 3. PDF válido
    valid_pdf = tmp_path / "03_valid.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Valid financial report content for charity.")
        doc.save(str(valid_pdf))

    mock_generate = mocker.patch(
        "src.ingestion.pipeline.generate_embeddings",
        side_effect=lambda chunks: [
            chunk.model_copy(update={"embedding": [0.5] * 1536}) for chunk in chunks
        ],
    )
    mock_save = mocker.patch("src.ingestion.pipeline.save_chunks_to_db")

    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"
    stats = ingest_directory(raw_dir=str(tmp_path), db_connection_string=dsn)

    assert stats == {"processed": 1, "skipped": 1, "failed": 1}
    mock_generate.assert_called_once()
    mock_save.assert_called_once()
