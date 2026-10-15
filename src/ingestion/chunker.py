"""Fragmentação (chunking) de texto por janela deslizante de caracteres."""

import logging
from typing import List

from src.ingestion.models import DocumentChunk

logger = logging.getLogger(__name__)


def create_chunks(
    document_id: str,
    pages_data: List[dict[str, int | str]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[DocumentChunk]:
    """Divide o texto das páginas de um documento em fragmentos com sobreposição (sliding window).

    Args:
        document_id: Nome exato do arquivo PDF (ex: "1ada336f29e8247f9f55a8d7e1b1c0da.pdf").
        pages_data: Lista de dicionários no formato {"page_number": int, "text": str}.
        chunk_size: Tamanho máximo de caracteres por fragmento (default: 1000).
        overlap: Quantidade de caracteres sobrepostos entre fragmentos consecutivos (default: 200).

    Returns:
        Lista de objetos DocumentChunk mantendo rastreabilidade de document_id e page_number.

    Raises:
        ValueError: Se document_id for vazio ou se chunk_size/overlap forem inválidos.
    """
    if not document_id or not document_id.strip():
        raise ValueError("O 'document_id' não pode ser vazio.")
    if chunk_size <= 0:
        raise ValueError("O 'chunk_size' deve ser maior que zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("O 'overlap' deve ser maior ou igual a zero e menor que 'chunk_size'.")

    chunks: List[DocumentChunk] = []
    chunk_index: int = 0
    step: int = chunk_size - overlap

    for page in pages_data:
        page_number = int(page["page_number"])
        raw_text = str(page.get("text", ""))

        # Ignorar páginas em branco
        cleaned_text = raw_text.strip()
        if not cleaned_text:
            continue

        text_length = len(cleaned_text)
        start = 0

        while start < text_length:
            end = start + chunk_size
            fragment = cleaned_text[start:end].strip()

            if fragment:
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text_content=fragment,
                    )
                )
                chunk_index += 1

            if end >= text_length:
                break

            start += step

    return chunks
