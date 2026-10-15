"""Pipeline de orquestração do Módulo de Ingestão (Raw-RAG)."""

import logging
from pathlib import Path
from typing import List

from src.ingestion.chunker import create_chunks
from src.ingestion.embedder import generate_embeddings
from src.ingestion.models import DocumentChunk
from src.ingestion.parser import extract_text_from_pdf
from src.ingestion.repository import save_chunks_to_db

logger = logging.getLogger(__name__)


def process_document(
    file_path: str,
    db_connection_string: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[DocumentChunk]:
    """Executa o fluxo completo de ingestão para um único arquivo PDF.

    Args:
        file_path: Caminho para o arquivo PDF.
        db_connection_string: String de conexão com o banco PostgreSQL.
        chunk_size: Tamanho máximo de caracteres por chunk (default: 1000).
        overlap: Sobreposição de caracteres entre chunks (default: 200).

    Returns:
        Lista de DocumentChunk processados e persistidos.
    """
    pdf_path = Path(file_path)
    document_id = pdf_path.name

    pages_data = extract_text_from_pdf(str(pdf_path))
    if not pages_data:
        logger.warning("Documento '%s' não retornou páginas extraíveis. Pulando.", document_id)
        return []

    chunks = create_chunks(
        document_id=document_id,
        pages_data=pages_data,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        logger.warning("Documento '%s' não contém texto válido para chunking. Pulando.", document_id)
        return []

    embedded_chunks = generate_embeddings(chunks)
    save_chunks_to_db(embedded_chunks, db_connection_string)
    return embedded_chunks


def ingest_directory(
    raw_dir: str,
    db_connection_string: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> dict[str, int]:
    """Processa todos os arquivos PDF de um diretório de forma resiliente.

    Arquivos corrompidos ou inválidos são registrados no log e pulados sem interromper o lote.

    Args:
        raw_dir: Caminho do diretório contendo os arquivos PDF (ex: "data/raw").
        db_connection_string: String de conexão com o PostgreSQL.
        chunk_size: Tamanho de cada chunk de texto (default: 1000).
        overlap: Sobreposição entre chunks (default: 200).

    Returns:
        Dicionário com estatísticas do processamento: {"processed": int, "skipped": int, "failed": int}.
    """
    directory = Path(raw_dir)
    stats: dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0}

    if not directory.exists() or not directory.is_dir():
        logger.warning("Diretório de entrada '%s' não encontrado.", raw_dir)
        return stats

    pdf_files = sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        logger.warning("Nenhum arquivo PDF encontrado em '%s'.", raw_dir)
        return stats

    total_files = len(pdf_files)
    logger.info("Iniciando ingestão de %d arquivo(s) PDF em '%s'.", total_files, raw_dir)

    for index, pdf_file in enumerate(pdf_files, start=1):
        logger.info("[%d/%d] Processando documento '%s'...", index, total_files, pdf_file.name)
        try:
            chunks = process_document(
                file_path=str(pdf_file),
                db_connection_string=db_connection_string,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if chunks:
                stats["processed"] += 1
                logger.info(
                    "[%d/%d] Documento '%s' ingerido com sucesso (%d chunks).",
                    index,
                    total_files,
                    pdf_file.name,
                    len(chunks),
                )
            else:
                stats["skipped"] += 1
                logger.warning(
                    "[%d/%d] Documento '%s' pulado por não conter conteúdo textual.",
                    index,
                    total_files,
                    pdf_file.name,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            stats["failed"] += 1
            logger.error(
                "[%d/%d] Erro ao ingerir documento '%s': %s. Pulando para o próximo arquivo.",
                index,
                total_files,
                pdf_file.name,
                exc,
            )

    logger.info(
        "Ingestão concluída. Processados: %d | Pulados: %d | Falhas: %d",
        stats["processed"],
        stats["skipped"],
        stats["failed"],
    )
    return stats
