"""Carregamento de dados e cálculo determinístico de métricas de imputação e linhagem."""

import logging
from pathlib import Path

import pandas as pd
import psycopg2

from src.evaluation.models import ColumnMetrics, EvaluationReport

logger = logging.getLogger(__name__)

DEFAULT_TARGET_COLUMNS: list[str] = [
    "income_annually_in_british_pounds",
    "spending_annually_in_british_pounds",
]


def _normalize_document_id(value: object) -> str:
    """Normaliza o identificador do documento removendo espaços e garantindo o sufixo .pdf."""
    if pd.isna(value):
        return ""
    cleaned = str(value).strip()
    if not cleaned:
        return ""
    if not cleaned.lower().endswith(".pdf"):
        return f"{cleaned}.pdf"
    return cleaned


def load_ground_truth(file_path: str) -> pd.DataFrame:
    """Carrega a planilha gabarito (Ground Truth) a partir de arquivo .xlsx ou .csv.

    Normaliza a coluna 'document_id' aplicando strip de espaços e garantindo o sufixo '.pdf'.

    Args:
        file_path: Caminho para o arquivo de Ground Truth (.xlsx ou .csv).

    Returns:
        DataFrame contendo os dados esperados com 'document_id' normalizado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o formato do arquivo não for suportado ou se a coluna 'document_id' não existir.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        logger.error("Arquivo de Ground Truth não encontrado: %s", file_path)
        raise FileNotFoundError(f"Arquivo de Ground Truth não encontrado: {file_path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Formato de arquivo não suportado ('{suffix}'). Utilize '.xlsx' ou '.csv'."
        )

    if "document_id" not in df.columns:
        raise ValueError("A coluna obrigatória 'document_id' não foi encontrada no Ground Truth.")

    df = df.copy()
    df["document_id"] = df["document_id"].map(_normalize_document_id)
    return df


def load_imputed_data(db_connection_string: str) -> pd.DataFrame:
    """Carrega todos os registros da tabela dw_charity_data do PostgreSQL em um DataFrame.

    Args:
        db_connection_string: String de conexão (DSN/URI) para o PostgreSQL.

    Returns:
        DataFrame com os registros da tabela dw_charity_data.

    Raises:
        ValueError: Se db_connection_string for vazia.
        psycopg2.Error: Em caso de erro de consulta no banco de dados.
    """
    if not db_connection_string or not db_connection_string.strip():
        raise ValueError("A string de conexão 'db_connection_string' não pode ser vazia.")

    select_sql = "SELECT * FROM dw_charity_data"

    try:
        with psycopg2.connect(db_connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_sql)
                rows = cursor.fetchall()
                columns: list[str] = (
                    [ str(col.name if hasattr(col, "name") else col[0]) for col in cursor.description ]
                    if cursor.description
                    else []
                )

        df = pd.DataFrame(rows, columns=columns)
        if "document_id" in df.columns and not df.empty:
            df["document_id"] = df["document_id"].map(_normalize_document_id)
        return df
    except Exception as exc:
        logger.error("Erro ao carregar dados imputados de dw_charity_data: %s", exc)
        raise


def is_match_financial(
    imputed_val: float | None,
    truth_val: float | None,
    tolerance: float = 0.05,
) -> bool:
    """Compara dois valores financeiros dentro de uma margem de tolerância absoluta.

    Args:
        imputed_val: Valor imputado pela IA.
        truth_val: Valor esperado no Ground Truth.
        tolerance: Tolerância numérica absoluta permitida (default: 0.05).

    Returns:
        True se ambos os valores forem válidos e a diferença absoluta for <= tolerance,
        False caso qualquer um seja None/NaN ou fora da margem.
    """
    if imputed_val is None or truth_val is None:
        return False
    if pd.isna(imputed_val) or pd.isna(truth_val):
        return False

    try:
        imp_float = float(imputed_val)
        truth_float = float(truth_val)
    except (ValueError, TypeError):
        return False

    return bool(abs(imp_float - truth_float) <= tolerance)


def _is_non_empty_field(value: object) -> bool:
    """Verifica se um campo de linhagem não é nulo/NaN e não é uma string vazia."""
    if value is None or pd.isna(value):
        return False
    return bool(str(value).strip())


def _calculate_lineage_completeness(df_merged: pd.DataFrame) -> float:
    """Calcula a proporção de registros com imputado_por_ia == True que possuem linhagem completa."""
    ia_col = (
        "imputado_por_ia_imputed"
        if "imputado_por_ia_imputed" in df_merged.columns
        else "imputado_por_ia"
    )
    if ia_col not in df_merged.columns or df_merged.empty:
        return 0.0

    ia_mask = df_merged[ia_col].fillna(False).astype(bool)
    df_ia = df_merged[ia_mask]
    if df_ia.empty:
        return 0.0

    source_col = (
        "fonte_documento_imputed"
        if "fonte_documento_imputed" in df_ia.columns
        else "fonte_documento"
    )
    page_col = "pagina_imputed" if "pagina_imputed" in df_ia.columns else "pagina"
    quote_col = (
        "trecho_evidenciado_imputed"
        if "trecho_evidenciado_imputed" in df_ia.columns
        else "trecho_evidenciado"
    )

    for required_col in (source_col, page_col, quote_col):
        if required_col not in df_ia.columns:
            return 0.0

    complete_count = 0
    for _, row in df_ia.iterrows():
        has_source = _is_non_empty_field(row[source_col])
        has_page = _is_non_empty_field(row[page_col])
        has_quote = _is_non_empty_field(row[quote_col])
        if has_source and has_page and has_quote:
            complete_count += 1

    return round(complete_count / len(df_ia), 4)


def calculate_metrics(
    df_truth: pd.DataFrame,
    df_imputed: pd.DataFrame,
    target_columns: list[str] | None = None,
) -> tuple[EvaluationReport, pd.DataFrame]:
    """Cruza os dados do Ground Truth com o Data Warehouse e calcula as métricas de avaliação.

    Args:
        df_truth: DataFrame contendo o gabarito (Ground Truth).
        df_imputed: DataFrame contendo os registros de dw_charity_data.
        target_columns: Lista de colunas financeiras a avaliar.

    Returns:
        Tupla contendo (EvaluationReport, df_merged).
    """
    columns_to_evaluate = (
        list(target_columns) if target_columns is not None else list(DEFAULT_TARGET_COLUMNS)
    )

    truth_copy = df_truth.copy()
    imputed_copy = df_imputed.copy()

    if "document_id" in truth_copy.columns:
        truth_copy["document_id"] = truth_copy["document_id"].map(_normalize_document_id)
    if "document_id" in imputed_copy.columns:
        imputed_copy["document_id"] = imputed_copy["document_id"].map(_normalize_document_id)

    df_merged = pd.merge(
        truth_copy,
        imputed_copy,
        on="document_id",
        suffixes=("_truth", "_imputed"),
    )

    evaluated_metrics: list[ColumnMetrics] = []

    for col in columns_to_evaluate:
        truth_col = f"{col}_truth" if f"{col}_truth" in df_merged.columns else col
        imputed_col = f"{col}_imputed" if f"{col}_imputed" in df_merged.columns else col
        match_col = f"{col}_is_match"

        if df_merged.empty or truth_col not in df_merged.columns or imputed_col not in df_merged.columns:
            df_merged[match_col] = pd.Series(dtype=bool)
            evaluated_metrics.append(
                ColumnMetrics(
                    column_name=col,
                    total_expected=0,
                    total_imputed=0,
                    correct_matches=0,
                    accuracy_rate=0.0,
                    precision_rate=0.0,
                    fill_rate=0.0,
                )
            )
            continue

        df_merged[match_col] = [
            is_match_financial(imp_val, tr_val)
            for imp_val, tr_val in zip(df_merged[imputed_col], df_merged[truth_col])
        ]

        expected_mask = ~df_merged[truth_col].isna()
        imputed_among_expected_mask = expected_mask & (~df_merged[imputed_col].isna())

        total_expected = int(expected_mask.sum())
        total_imputed = int(imputed_among_expected_mask.sum())
        correct_matches = int((expected_mask & df_merged[match_col]).sum())

        accuracy_rate = (
            round(correct_matches / total_expected, 4) if total_expected > 0 else 0.0
        )
        precision_rate = (
            round(correct_matches / total_imputed, 4) if total_imputed > 0 else 0.0
        )
        fill_rate = (
            round(total_imputed / total_expected, 4) if total_expected > 0 else 0.0
        )

        evaluated_metrics.append(
            ColumnMetrics(
                column_name=col,
                total_expected=total_expected,
                total_imputed=total_imputed,
                correct_matches=correct_matches,
                accuracy_rate=accuracy_rate,
                precision_rate=precision_rate,
                fill_rate=fill_rate,
            )
        )

    lineage_completeness = _calculate_lineage_completeness(df_merged)

    report = EvaluationReport(
        total_documents_evaluated=len(df_merged),
        columns_evaluated=evaluated_metrics,
        lineage_completeness=lineage_completeness,
    )

    return report, df_merged
