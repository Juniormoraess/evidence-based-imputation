"""Mapeamento de lacunas no Data Warehouse e busca híbrida vetorial no PostgreSQL (pgvector)."""

from datetime import date, datetime
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

from src.retrieval.models import ImputationTask, RetrievedEvidence

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
TARGET_INCOME_COLUMN = "income_annually_in_british_pounds"
TARGET_SPENDING_COLUMN = "spending_annually_in_british_pounds"


def _format_report_date(raw_date: date | datetime | str | None) -> str | None:
    """Converte o valor de report_date do banco para string ISO (YYYY-MM-DD) quando presente."""
    if raw_date is None:
        return None
    if isinstance(raw_date, datetime):
        return raw_date.date().isoformat()
    if isinstance(raw_date, date):
        return raw_date.isoformat()
    cleaned = str(raw_date).strip()
    return cleaned if cleaned else None


def _build_query_text(task: ImputationTask) -> str:
    """Constrói uma string de busca semântica rica combinando target_column, charity_name e report_date."""
    if task.target_column == TARGET_INCOME_COLUMN:
        metric_phrase = "Total annual income in British pounds (£) financial statement"
    elif task.target_column == TARGET_SPENDING_COLUMN:
        metric_phrase = (
            "Total annual spending and expenditure in British pounds (£) financial statement"
        )
    else:
        readable_col = task.target_column.replace("_", " ")
        metric_phrase = f"Financial report {readable_col} in British pounds (£) financial statement"

    parts: list[str] = [metric_phrase]

    if task.charity_name and task.charity_name.strip():
        parts.append(f"for charity {task.charity_name.strip()}")

    if task.report_date and task.report_date.strip():
        parts.append(f"for report period ending {task.report_date.strip()}")

    return " ".join(parts)


def get_pending_tasks(db_connection_string: str) -> list[ImputationTask]:
    """Varre a tabela dw_charity_data em busca de registros com lacunas (NULL) nas colunas financeiras.

    Se uma mesma linha tiver ambas as colunas nulas, gera duas instâncias de ImputationTask.

    Args:
        db_connection_string: String de conexão (DSN/URI) para o PostgreSQL.

    Returns:
        Lista de objetos ImputationTask pendentes de imputação.

    Raises:
        ValueError: Se db_connection_string for vazia.
        psycopg2.Error: Em caso de falha na consulta ao banco de dados.
    """
    if not db_connection_string or not db_connection_string.strip():
        raise ValueError("A string de conexão 'db_connection_string' não pode ser vazia.")

    select_sql = """
        SELECT
            id,
            document_id,
            charity_name,
            report_date,
            income_annually_in_british_pounds,
            spending_annually_in_british_pounds
        FROM dw_charity_data
        WHERE income_annually_in_british_pounds IS NULL
           OR spending_annually_in_british_pounds IS NULL
        ORDER BY id ASC
    """

    tasks: list[ImputationTask] = []

    try:
        with psycopg2.connect(db_connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_sql)
                rows = cursor.fetchall()

        for row in rows:
            row_id, document_id, charity_name, report_date, income_val, spending_val = row
            iso_report_date = _format_report_date(report_date)

            if income_val is None:
                tasks.append(
                    ImputationTask(
                        row_id=int(row_id),
                        document_id=str(document_id),
                        charity_name=str(charity_name) if charity_name is not None else None,
                        report_date=iso_report_date,
                        target_column=TARGET_INCOME_COLUMN,
                    )
                )

            if spending_val is None:
                tasks.append(
                    ImputationTask(
                        row_id=int(row_id),
                        document_id=str(document_id),
                        charity_name=str(charity_name) if charity_name is not None else None,
                        report_date=iso_report_date,
                        target_column=TARGET_SPENDING_COLUMN,
                    )
                )

        logger.info("Foram mapeadas %d tarefa(s) de imputação pendente(s).", len(tasks))
        return tasks
    except Exception as exc:
        logger.error("Erro ao consultar tarefas pendentes em dw_charity_data: %s", exc)
        raise


def embed_search_query(task: ImputationTask) -> list[float]:
    """Gera o vetor de embedding semântico (1536 dimensões) para uma tarefa de imputação.

    Lê as variáveis de ambiente EMBEDDING_API_KEY, EMBEDDING_BASE_URL (opcional)
    e EMBEDDING_MODEL (default: text-embedding-3-small).

    Args:
        task: Objeto ImputationTask contendo os metadados da lacuna.

    Returns:
        Vetor de floats representando o embedding da query semântica.

    Raises:
        ValueError: Se a variável de ambiente EMBEDDING_API_KEY não estiver configurada.
        RuntimeError: Se ocorrer falha na comunicação com a API de embeddings.
    """
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

    query_text = _build_query_text(task)
    client = OpenAI(**client_kwargs)

    try:
        response = client.embeddings.create(
            input=[query_text],
            model=model,
        )

        if not response.data or not response.data[0].embedding:
            raise RuntimeError("A API de embeddings retornou uma resposta vazia.")

        return list(response.data[0].embedding)
    except Exception as exc:
        logger.error(
            "Erro ao gerar embedding da query para document_id='%s', coluna='%s': %s",
            task.document_id,
            task.target_column,
            exc,
        )
        raise RuntimeError(f"Falha na geração de embedding da query: {exc}") from exc


def fetch_evidence(
    task: ImputationTask,
    query_embedding: list[float],
    db_connection_string: str,
    top_k: int = 5,
) -> list[RetrievedEvidence]:
    """Executa busca híbrida (Hard Filtering por document_id + similaridade de cosseno) no PostgreSQL.

    Args:
        task: Tarefa de imputação contendo o document_id alvo.
        query_embedding: Vetor de embedding da query semântica.
        db_connection_string: String de conexão (DSN/URI) para o PostgreSQL.
        top_k: Quantidade máxima de fragmentos mais relevantes a retornar (default: 5).

    Returns:
        Lista de objetos RetrievedEvidence ordenados por maior similaridade.
        Retorna lista vazia se o document_id não possuir chunks na base.

    Raises:
        ValueError: Se parâmetros obrigatórios (db_connection_string, query_embedding ou top_k) forem inválidos.
        psycopg2.Error: Em caso de falha na execução da consulta SQL.
    """
    if not db_connection_string or not db_connection_string.strip():
        raise ValueError("A string de conexão 'db_connection_string' não pode ser vazia.")
    if not query_embedding:
        raise ValueError("O vetor 'query_embedding' não pode ser vazio.")
    if top_k <= 0:
        raise ValueError("O parâmetro 'top_k' deve ser maior que zero.")

    search_sql = """
        SELECT
            document_id,
            page_number,
            chunk_index,
            text_content,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM knowledge_base
        WHERE document_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    try:
        with psycopg2.connect(db_connection_string) as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                cursor.execute(
                    search_sql,
                    (query_embedding, task.document_id, query_embedding, top_k),
                )
                rows = cursor.fetchall()

        if not rows:
            logger.warning(
                "Nenhum chunk encontrado na knowledge_base para document_id='%s'.",
                task.document_id,
            )
            return []

        evidences: list[RetrievedEvidence] = [
            RetrievedEvidence(
                document_id=str(doc_id),
                page_number=int(page_number),
                chunk_index=int(chunk_index),
                text_content=str(text_content),
                similarity_score=float(similarity_score),
            )
            for doc_id, page_number, chunk_index, text_content, similarity_score in rows
        ]

        return evidences
    except Exception as exc:
        logger.error(
            "Erro na busca híbrida de evidências para document_id='%s': %s",
            task.document_id,
            exc,
        )
        raise
