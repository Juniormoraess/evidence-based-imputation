"""Exportação dos artefatos de avaliação (summary_metrics.json e detailed_evaluation.csv)."""

import logging
import os
from pathlib import Path

import pandas as pd

from src.evaluation.models import EvaluationReport

logger = logging.getLogger(__name__)


def export_reports(
    report: EvaluationReport,
    df_merged: pd.DataFrame,
    output_dir: str = "reports",
) -> tuple[str, str]:
    """Salva o resumo das métricas em JSON e o detalhamento linha a linha em CSV.

    Args:
        report: Objeto EvaluationReport contendo as métricas consolidadas.
        df_merged: DataFrame resultante do cruzamento entre Ground Truth e dados imputados.
        output_dir: Diretório de destino para os relatórios (default: "reports").

    Returns:
        Tupla (caminho_summary_json, caminho_detailed_csv).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir)

    json_file = out_path / "summary_metrics.json"
    csv_file = out_path / "detailed_evaluation.csv"

    json_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    df_merged.to_csv(csv_file, index=False, encoding="utf-8")

    logger.info(
        "Relatórios de avaliação exportados com sucesso em '%s' e '%s'.",
        json_file,
        csv_file,
    )
    return str(json_file), str(csv_file)
