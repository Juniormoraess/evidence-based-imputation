"""Script executável para o Módulo 1: Ingestão de Documentos PDF (Raw-RAG)."""

import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import ingest_directory  # noqa: E402

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


def validate_embedding_env() -> tuple[str, str]:
    """Valida e retorna as configurações de ambiente para geração de embeddings.

    Returns:
        Tupla (embedding_api_key, embedding_model).

    Raises:
        ValueError: Se EMBEDDING_API_KEY não estiver definida.
    """
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "Variável de ambiente obrigatória 'EMBEDDING_API_KEY' não configurada no .env."
        )

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
    os.environ["EMBEDDING_MODEL"] = model.strip()
    return api_key.strip(), model.strip()


def main(raw_dir: str = "data/raw") -> int:
    """Ponto de entrada principal para execução do pipeline de ingestão.

    Args:
        raw_dir: Diretório contendo os documentos PDF brutos.

    Returns:
        Código de saída do processo (0 para execução concluída, 1 para erro de configuração).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    load_dotenv()

    try:
        db_connection_string = build_db_connection_string()
        _, embedding_model = validate_embedding_env()
    except ValueError as exc:
        logger.error("Erro de configuração de ambiente: %s", exc)
        return 1

    logger.info(
        "Iniciando pipeline de ingestão Raw-RAG (Modelo de Embedding: %s).",
        embedding_model,
    )

    raw_path = Path(raw_dir)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    stats = ingest_directory(
        raw_dir=str(raw_path),
        db_connection_string=db_connection_string,
    )

    logger.info(
        "Resumo final da ingestão -> Processados: %d | Pulados: %d | Falhas: %d",
        stats["processed"],
        stats["skipped"],
        stats["failed"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
