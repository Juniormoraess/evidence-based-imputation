"""Script executável para o Módulo 3: Imputação Baseada em Evidências (Raw-RAG + DeepSeek)."""

import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imputation.pipeline import run_imputation_pipeline  # noqa: E402

logger = logging.getLogger(__name__)


def build_db_connection_string() -> str:
    """Constrói a string de conexão PostgreSQL a partir das variáveis de ambiente do .env.

    Returns:
        String de conexão PostgreSQL no formato URI (postgresql://user:pass@host:port/dbname).

    Raises:
        ValueError: Se alguma das variáveis obrigatórias de banco de dados estiver ausente.
    """
    required_vars = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    env_values: dict[str, str] = {}
    missing_vars: list[str] = []

    for var_name in required_vars:
        value = os.getenv(var_name)
        if value is None or not value.strip():
            missing_vars.append(var_name)
        else:
            env_values[var_name] = value.strip()

    if missing_vars:
        missing_str = ", ".join(missing_vars)
        raise ValueError(
            f"Variáveis de ambiente de banco de dados obrigatórias ausentes: {missing_str}"
        )

    user = quote_plus(env_values["DB_USER"])
    password = quote_plus(env_values["DB_PASSWORD"])
    host = env_values["DB_HOST"]
    port = env_values["DB_PORT"]
    dbname = env_values["DB_NAME"]

    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def validate_required_api_keys() -> tuple[str, str]:
    """Valida a presença de DEEPSEEK_API_KEY e EMBEDDING_API_KEY no ambiente.

    Returns:
        Tupla (deepseek_model, embedding_model).

    Raises:
        ValueError: Se DEEPSEEK_API_KEY ou EMBEDDING_API_KEY estiverem ausentes.
    """
    missing_keys: list[str] = []

    for key_name in ("DEEPSEEK_API_KEY", "EMBEDDING_API_KEY"):
        val = os.getenv(key_name)
        if not val or not val.strip():
            missing_keys.append(key_name)

    if missing_keys:
        raise ValueError(
            f"Chaves de API obrigatórias não configuradas no .env: {', '.join(missing_keys)}"
        )

    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat"
    embedding_model = (
        os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
    )
    return deepseek_model.strip(), embedding_model.strip()


def main(top_k: int = 12) -> int:
    """Ponto de entrada principal para execução do pipeline de imputação.

    Args:
        top_k: Quantidade de evidências recuperadas por tarefa de imputação (default: 12).

    Returns:
        Código de saída do processo (0 para sucesso, 1 para erro de configuração ou execução).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    load_dotenv()

    try:
        db_connection_string = build_db_connection_string()
        deepseek_model, embedding_model = validate_required_api_keys()
    except ValueError as exc:
        logger.error("Erro de configuração de ambiente: %s", exc)
        return 1

    logger.info(
        "Iniciando pipeline de Imputação Baseada em Evidências "
        "(LLM: %s | Embedding: %s | top_k: %d).",
        deepseek_model,
        embedding_model,
        top_k,
    )

    summary = run_imputation_pipeline(
        db_connection_string=db_connection_string,
        top_k=top_k,
    )

    logger.info(
        "Resumo final da imputação -> Total de Tarefas: %d | Imputados: %d | Pulados: %d",
        summary["total_tasks"],
        summary["imputed_count"],
        summary["skipped_count"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
