"""Módulo de Ingestão (Raw-RAG) de documentos PDF para o PostgreSQL com pgvector."""

from src.ingestion.chunker import create_chunks
from src.ingestion.embedder import generate_embeddings
from src.ingestion.models import DocumentChunk
from src.ingestion.parser import extract_text_from_pdf
from src.ingestion.pipeline import ingest_directory, process_document
from src.ingestion.repository import save_chunks_to_db

__all__ = [
    "DocumentChunk",
    "extract_text_from_pdf",
    "create_chunks",
    "generate_embeddings",
    "save_chunks_to_db",
    "process_document",
    "ingest_directory",
]
