"""Persistência vetorial na tabela knowledge_base do PostgreSQL com pgvector."""

import logging
from typing import List

import psycopg2
from pgvector.psycopg2 import register_vector

from src.ingestion.models import DocumentChunk

logger = logging.getLogger(__name__)


def save_chunks_to_db(chunks: List[DocumentChunk], db_connection_string: str) -> None:
    """Persiste uma lista de DocumentChunk na tabela knowledge_base de maneira idempotente.

    Remove registros prévios do mesmo document_id antes de inserir os novos chunks
    dentro da mesma transação.

    Args:
        chunks: Lista de objetos DocumentChunk contendo embeddings calculados.
        db_connection_string: String de conexão (DSN/URI) para o PostgreSQL.

    Raises:
        ValueError: Se db_connection_string for vazia ou se algum chunk não possuir embedding.
        psycopg2.Error: Em caso de erro de banco de dados durante a transação.
    """
    if not db_connection_string or not db_connection_string.strip():
        raise ValueError("A string de conexão 'db_connection_string' não pode ser vazia.")

    if not chunks:
        logger.warning("Lista de chunks vazia recebida; nenhuma operação realizada no banco.")
        return

    for chunk in chunks:
        if chunk.embedding is None:
            raise ValueError(
                f"O chunk {chunk.chunk_index} do documento '{chunk.document_id}' "
                "não possui vetor de embedding calculado."
            )

    document_ids: List[str] = list(dict.fromkeys(chunk.document_id for chunk in chunks))

    delete_sql = "DELETE FROM knowledge_base WHERE document_id = %s"
    insert_sql = """
        INSERT INTO knowledge_base (document_id, page_number, chunk_index, text_content, embedding)
        VALUES (%s, %s, %s, %s, %s)
    """

    try:
        with psycopg2.connect(db_connection_string) as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                for doc_id in document_ids:
                    cursor.execute(delete_sql, (doc_id,))

                for chunk in chunks:
                    cursor.execute(
                        insert_sql,
                        (
                            chunk.document_id,
                            chunk.page_number,
                            chunk.chunk_index,
                            chunk.text_content,
                            chunk.embedding,
                        ),
                    )
            conn.commit()
            logger.info(
                "Persistidos com sucesso %d chunks para %d documento(s).",
                len(chunks),
                len(document_ids),
            )
    except Exception as exc:
        logger.error("Falha ao persistir chunks no PostgreSQL: %s", exc)
        raise
