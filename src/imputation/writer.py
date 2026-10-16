"""Persistência de valores imputados e metadados de linhagem no Data Warehouse (dw_charity_data)."""

import logging

import psycopg2
from psycopg2 import sql

from src.imputation.models import ExtractionResult
from src.retrieval.models import ImputationTask, RetrievedEvidence

logger = logging.getLogger(__name__)

ALLOWED_TARGET_COLUMNS: frozenset[str] = frozenset(
    {
        "income_annually_in_british_pounds",
        "spending_annually_in_british_pounds",
    }
)


def update_dw_record(
    task: ImputationTask,
    result: ExtractionResult,
    evidences: list[RetrievedEvidence],
    db_connection_string: str,
) -> bool:
    """Atualiza a coluna alvo e as colunas de auditoria/linhagem na tabela dw_charity_data.

    Args:
        task: Tarefa de imputação contendo row_id, document_id e target_column.
        result: Resultado validado da extração via LLM.
        evidences: Lista de evidências utilizadas na extração (usada para fallback de página).
        db_connection_string: String de conexão (DSN/URI) para o PostgreSQL.

    Returns:
        True se o registro foi atualizado no banco, False se value_found=False ou extracted_value=None.

    Raises:
        ValueError: Se db_connection_string for vazia ou se task.target_column não estiver na allowlist.
        psycopg2.Error: Em caso de erro de banco de dados durante o UPDATE.
    """
    if not db_connection_string or not db_connection_string.strip():
        raise ValueError("A string de conexão 'db_connection_string' não pode ser vazia.")

    if task.target_column not in ALLOWED_TARGET_COLUMNS:
        raise ValueError(
            f"Coluna alvo '{task.target_column}' não permitida. "
            f"Valores aceitos: {sorted(ALLOWED_TARGET_COLUMNS)}"
        )

    if not result.value_found or result.extracted_value is None:
        logger.info(
            "Valor não encontrado para row_id=%d, document_id='%s', coluna='%s' (motivo: %s). "
            "Mantendo NULL no Data Warehouse.",
            task.row_id,
            task.document_id,
            task.target_column,
            result.reasoning,
        )
        return False

    determined_page: int | None = result.source_page
    if determined_page is None and evidences:
        determined_page = evidences[0].page_number

    update_query = sql.SQL(
        """
        UPDATE dw_charity_data
        SET {target_col} = %s,
            imputado_por_ia = TRUE,
            fonte_documento = %s,
            pagina = %s,
            trecho_evidenciado = %s,
            confianca = %s,
            data_imputacao = CURRENT_TIMESTAMP
        WHERE id = %s
        """
    ).format(target_col=sql.Identifier(task.target_column))

    try:
        with psycopg2.connect(db_connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    update_query,
                    (
                        result.extracted_value,
                        task.document_id,
                        determined_page,
                        result.exact_quote,
                        result.confidence_score,
                        task.row_id,
                    ),
                )
            conn.commit()

        logger.info(
            "Registro id=%d atualizado com sucesso: %s = %.2f (página=%s, confiança=%.2f).",
            task.row_id,
            task.target_column,
            result.extracted_value,
            determined_page,
            result.confidence_score,
        )
        return True
    except Exception as exc:
        logger.error(
            "Falha ao atualizar dw_charity_data para row_id=%d, coluna='%s': %s",
            task.row_id,
            task.target_column,
            exc,
        )
        raise
