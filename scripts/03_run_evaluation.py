"""Script executável para o Módulo 4: Avaliação Determinística de Imputação e Linhagem."""

import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    calculate_metrics,
    load_ground_truth,
    load_imputed_data,
)
from src.evaluation.report import export_reports  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_PATH = "data/ground_truth/expected_with_doc_id.xlsx"
DEFAULT_REPORTS_DIR = "reports"


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


def main(
    ground_truth_path: str | None = None,
    output_dir: str = DEFAULT_REPORTS_DIR,
) -> int:
    """Ponto de entrada principal para execução do pipeline de avaliação determinística.

    Args:
        ground_truth_path: Caminho opcional para o arquivo de Ground Truth.
        output_dir: Diretório de saída para os relatórios gerados (default: "reports").

    Returns:
        Código de saída do processo (0 para sucesso, 1 para erro).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    load_dotenv()

    try:
        db_connection_string = build_db_connection_string()
    except ValueError as exc:
        logger.error("Erro de configuração de ambiente: %s", exc)
        return 1

    configured_gt_path = (
        ground_truth_path
        or os.getenv("GROUND_TRUTH_PATH", DEFAULT_GROUND_TRUTH_PATH)
        or DEFAULT_GROUND_TRUTH_PATH
    ).strip()

    gt_path_obj = Path(configured_gt_path)
    if not gt_path_obj.is_absolute():
        gt_path_obj = PROJECT_ROOT / gt_path_obj

    out_dir_obj = Path(output_dir)
    if not out_dir_obj.is_absolute():
        out_dir_obj = PROJECT_ROOT / out_dir_obj

    try:
        logger.info("Carregando Ground Truth de '%s'...", gt_path_obj)
        df_truth = load_ground_truth(str(gt_path_obj))

        logger.info("Carregando dados imputados de dw_charity_data...")
        df_imputed = load_imputed_data(db_connection_string)

        logger.info("Calculando métricas de qualidade de imputação e completude de linhagem...")
        report, df_merged = calculate_metrics(df_truth=df_truth, df_imputed=df_imputed)

        json_path, csv_path = export_reports(
            report=report,
            df_merged=df_merged,
            output_dir=str(out_dir_obj),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Falha durante a execução da avaliação: %s", exc)
        return 1

    logger.info("Relatórios salvos em: JSON='%s' | CSV='%s'", json_path, csv_path)
    logger.info("Resumo Consolidado da Avaliação:\n%s", report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
