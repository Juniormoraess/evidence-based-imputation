"""Pipeline de orquestração do Módulo de Imputação Baseada em Evidências."""

import logging

from src.imputation.llm_client import build_extraction_prompt, extract_value_with_llm
from src.imputation.writer import update_dw_record
from src.retrieval.search import embed_search_query, fetch_evidence, get_pending_tasks

logger = logging.getLogger(__name__)


def run_imputation_pipeline(
    db_connection_string: str,
    top_k: int = 12,
) -> dict[str, int]:
    """Orquestra o fluxo completo de busca de lacunas, recuperação de evidências e imputação via LLM.

    Args:
        db_connection_string: String de conexão (DSN/URI) para o PostgreSQL.
        top_k: Quantidade máxima de chunks base de evidência por tarefa (default: 12).

    Returns:
        Dicionário resumo com {"total_tasks": int, "imputed_count": int, "skipped_count": int}.
    """
    tasks = get_pending_tasks(db_connection_string)
    total_tasks = len(tasks)
    imputed_count = 0
    skipped_count = 0

    logger.info("Iniciando pipeline de imputação para %d tarefa(s) pendente(s).", total_tasks)

    for index, task in enumerate(tasks, start=1):
        logger.info(
            "[%d/%d] Processando row_id=%d | document_id='%s' | coluna='%s'...",
            index,
            total_tasks,
            task.row_id,
            task.document_id,
            task.target_column,
        )
        try:
            query_embedding = embed_search_query(task)
            evidences = fetch_evidence(
                task=task,
                query_embedding=query_embedding,
                db_connection_string=db_connection_string,
                top_k=top_k,
            )

            if not evidences:
                logger.warning(
                    "[%d/%d] Nenhuma evidência encontrada para document_id='%s'. Pulando tarefa.",
                    index,
                    total_tasks,
                    task.document_id,
                )
                skipped_count += 1
                continue

            prompt = build_extraction_prompt(task=task, evidences=evidences)
            extraction_result = extract_value_with_llm(prompt)

            was_updated = update_dw_record(
                task=task,
                result=extraction_result,
                evidences=evidences,
                db_connection_string=db_connection_string,
            )

            if was_updated:
                imputed_count += 1
            else:
                skipped_count += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "[%d/%d] Erro ao processar tarefa row_id=%d (%s): %s. Pulando tarefa.",
                index,
                total_tasks,
                task.row_id,
                task.target_column,
                exc,
            )
            skipped_count += 1

    summary: dict[str, int] = {
        "total_tasks": total_tasks,
        "imputed_count": imputed_count,
        "skipped_count": skipped_count,
    }
    logger.info(
        "Pipeline de imputação finalizado. Total: %d | Imputados: %d | Pulados: %d",
        total_tasks,
        imputed_count,
        skipped_count,
    )
    return summary
