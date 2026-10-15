"""Geração de embeddings semânticos utilizando o SDK oficial da OpenAI."""

import logging
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src.ingestion.models import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100


def generate_embeddings(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Gera embeddings em lote (batch) para uma lista de DocumentChunk usando a API da OpenAI.

    Lê as variáveis de ambiente EMBEDDING_API_KEY, EMBEDDING_BASE_URL (opcional)
    e EMBEDDING_MODEL (default: text-embedding-3-small).

    Args:
        chunks: Lista de objetos DocumentChunk a serem vetorizados.

    Returns:
        Lista de objetos DocumentChunk com o atributo 'embedding' preenchido.

    Raises:
        ValueError: Se a variável de ambiente EMBEDDING_API_KEY não estiver configurada.
        RuntimeError: Se ocorrer falha na comunicação com a API de embeddings.
    """
    if not chunks:
        return []

    load_dotenv()

    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key or not api_key.strip():
        logger.error("Variável de ambiente EMBEDDING_API_KEY ausente ou vazia.")
        raise ValueError("A variável de ambiente 'EMBEDDING_API_KEY' é obrigatória.")

    base_url = os.getenv("EMBEDDING_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url and base_url.strip():
        client_kwargs["base_url"] = base_url.strip()

    client = OpenAI(**client_kwargs)

    try:
        for batch_start in range(0, len(chunks), DEFAULT_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + DEFAULT_BATCH_SIZE]
            texts: List[str] = [chunk.text_content for chunk in batch]

            response = client.embeddings.create(
                input=texts,
                model=model,
            )

            if len(response.data) != len(batch):
                raise RuntimeError(
                    f"Quantidade de embeddings retornados ({len(response.data)}) "
                    f"difere do tamanho do lote enviado ({len(batch)})."
                )

            for chunk, embedding_item in zip(batch, response.data):
                chunk.embedding = list(embedding_item.embedding)

        return chunks
    except Exception as exc:
        logger.error("Erro ao gerar embeddings com o modelo '%s': %s", model, exc)
        raise RuntimeError(f"Falha na geração de embeddings: {exc}") from exc
