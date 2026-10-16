"""Suíte de testes unitários para o Módulo 4 (Avaliação / Evaluation)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.evaluation import (
    ColumnMetrics,
    EvaluationReport,
    calculate_metrics,
    export_reports,
    is_match_financial,
    load_ground_truth,
    load_imputed_data,
)


# ===========================================================================
# 1. Testes do Comparador Financeiro (is_match_financial)
# ===========================================================================


@pytest.mark.parametrize(
    ("imputed_val", "truth_val", "tolerance", "expected"),
    [
        (150000.00, 150000.00, 0.05, True),    # Exato
        (150000.03, 150000.00, 0.05, True),    # Dentro da tolerância (diff = 0.03 <= 0.05)
        (149999.95, 150000.00, 0.05, True),    # No limite inferior da tolerância
        (150000.06, 150000.00, 0.05, False),   # Fora da tolerância (diff = 0.06 > 0.05)
        (None, 150000.00, 0.05, False),        # Imputado é None
        (150000.00, None, 0.05, False),        # Ground truth é None
        (None, None, 0.05, False),             # Ambos None
        (np.nan, 150000.00, 0.05, False),      # Imputado é NaN
        (150000.00, np.nan, 0.05, False),      # Ground truth é NaN
        (float("nan"), float("nan"), 0.05, False),
    ],
)
def test_is_match_financial(
    imputed_val: float | None,
    truth_val: float | None,
    tolerance: float,
    expected: bool,
) -> None:
    """Deve validar igualdade financeira considerando margem de tolerância e tratar None/NaN."""
    assert is_match_financial(imputed_val, truth_val, tolerance=tolerance) is expected


# ===========================================================================
# 2. Testes de Carregamento de Ground Truth e DW (load_ground_truth / load_imputed_data)
# ===========================================================================


def test_load_ground_truth_xlsx_and_csv_normalizes_document_id(tmp_path: Path) -> None:
    """Deve carregar .xlsx e .csv aplicando strip e garantindo o sufixo .pdf em document_id."""
    raw_df = pd.DataFrame(
        {
            "document_id": ["  doc_sem_extensao  ", "doc_com_extensao.pdf", "DOC_MAIUSCULO.PDF"],
            "income_annually_in_british_pounds": [1000.0, 2000.0, 3000.0],
            "spending_annually_in_british_pounds": [800.0, 1500.0, 2500.0],
        }
    )

    xlsx_path = tmp_path / "ground_truth.xlsx"
    csv_path = tmp_path / "ground_truth.csv"
    raw_df.to_excel(xlsx_path, index=False)
    raw_df.to_csv(csv_path, index=False)

    df_xlsx = load_ground_truth(str(xlsx_path))
    df_csv = load_ground_truth(str(csv_path))

    expected_ids = ["doc_sem_extensao.pdf", "doc_com_extensao.pdf", "DOC_MAIUSCULO.PDF"]
    assert list(df_xlsx["document_id"]) == expected_ids
    assert list(df_csv["document_id"]) == expected_ids


def test_load_ground_truth_missing_file_raises() -> None:
    """Deve lançar FileNotFoundError se a planilha de Ground Truth não existir."""
    with pytest.raises(FileNotFoundError, match="Ground Truth não encontrado"):
        load_ground_truth("data/ground_truth/inexistente.xlsx")


def test_load_imputed_data_mocked_postgres(mocker: MockerFixture) -> None:
    """Deve consultar dw_charity_data via psycopg2 mockado e retornar DataFrame estruturado."""
    mock_connect = mocker.patch("src.evaluation.metrics.psycopg2.connect")
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    mock_cursor.description = [
        SimpleNamespace(name="id"),
        SimpleNamespace(name="document_id"),
        SimpleNamespace(name="income_annually_in_british_pounds"),
        SimpleNamespace(name="spending_annually_in_british_pounds"),
        SimpleNamespace(name="imputado_por_ia"),
        SimpleNamespace(name="fonte_documento"),
        SimpleNamespace(name="pagina"),
        SimpleNamespace(name="trecho_evidenciado"),
        SimpleNamespace(name="confianca"),
    ]
    mock_cursor.fetchall.return_value = [
        (1, "doc_1.pdf", 50000.0, 40000.0, True, "doc_1.pdf", 3, "Income £50,000", 0.95),
        (2, "doc_2", None, 20000.0, False, None, None, None, None),
    ]

    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"
    df_imputed = load_imputed_data(dsn)

    mock_connect.assert_called_once_with(dsn)
    mock_cursor.execute.assert_called_once_with("SELECT * FROM dw_charity_data")
    assert len(df_imputed) == 2
    assert list(df_imputed["document_id"]) == ["doc_1.pdf", "doc_2.pdf"]
    assert df_imputed.loc[0, "income_annually_in_british_pounds"] == 50000.0


# ===========================================================================
# 3. Testes de Cálculo de Métricas e Completude de Linhagem (calculate_metrics)
# ===========================================================================


def test_calculate_metrics_and_lineage_completeness() -> None:
    """Deve calcular accuracy, precision, fill_rate e lineage_completeness corretamente."""
    df_truth = pd.DataFrame(
        {
            "document_id": ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf", "doc_4.pdf"],
            # income: 4 esperados (todos não-nulos)
            "income_annually_in_british_pounds": [1000.00, 2000.00, 3000.00, 4000.00],
            # spending: 3 esperados (doc_4 é NaN no Ground Truth)
            "spending_annually_in_british_pounds": [500.00, 1500.00, 2500.00, np.nan],
        }
    )

    df_imputed = pd.DataFrame(
        {
            "document_id": ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf", "doc_4.pdf"],
            # income: doc_1 acerta (1000.02), doc_2 acerta (2000.00), doc_3 erra (9999.00), doc_4 é NaN (não imputado)
            # -> total_expected=4, total_imputed=3, correct_matches=2
            # -> accuracy = 2/4 = 0.5, precision = 2/3 = 0.6667, fill_rate = 3/4 = 0.75
            "income_annually_in_british_pounds": [1000.02, 2000.00, 9999.00, np.nan],
            # spending: doc_1 acerta (500.00), doc_2 não imputado (NaN), doc_3 acerta (2500.00), doc_4 NaN
            # -> total_expected=3, total_imputed=2, correct_matches=2
            # -> accuracy = 2/3 = 0.6667, precision = 2/2 = 1.0, fill_rate = 2/3 = 0.6667
            "spending_annually_in_british_pounds": [500.00, np.nan, 2500.00, np.nan],
            # Linhagem: 3 linhas com imputado_por_ia=True (doc_1, doc_2, doc_3)
            # doc_1 e doc_2 têm linhagem completa; doc_3 está sem trecho_evidenciado -> 2/3 = 0.6667
            "imputado_por_ia": [True, True, True, False],
            "fonte_documento": ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf", None],
            "pagina": [2, 5, 4, None],
            "trecho_evidenciado": ["Quote 1", "Quote 2", "", None],
        }
    )

    report, df_merged = calculate_metrics(df_truth=df_truth, df_imputed=df_imputed)

    assert isinstance(report, EvaluationReport)
    assert report.total_documents_evaluated == 4
    assert report.lineage_completeness == 0.6667

    assert "income_annually_in_british_pounds_is_match" in df_merged.columns
    assert "spending_annually_in_british_pounds_is_match" in df_merged.columns
    assert list(df_merged["income_annually_in_british_pounds_is_match"]) == [
        True,
        True,
        False,
        False,
    ]

    income_metrics: ColumnMetrics = report.columns_evaluated[0]
    assert income_metrics.column_name == "income_annually_in_british_pounds"
    assert income_metrics.total_expected == 4
    assert income_metrics.total_imputed == 3
    assert income_metrics.correct_matches == 2
    assert income_metrics.accuracy_rate == 0.5
    assert income_metrics.precision_rate == 0.6667
    assert income_metrics.fill_rate == 0.75

    spending_metrics: ColumnMetrics = report.columns_evaluated[1]
    assert spending_metrics.column_name == "spending_annually_in_british_pounds"
    assert spending_metrics.total_expected == 3
    assert spending_metrics.total_imputed == 2
    assert spending_metrics.correct_matches == 2
    assert spending_metrics.accuracy_rate == 0.6667
    assert spending_metrics.precision_rate == 1.0
    assert spending_metrics.fill_rate == 0.6667


def test_calculate_metrics_zero_division_protection() -> None:
    """Deve retornar 0.0 para todas as taxas quando total_expected == 0 ou total_imputed == 0."""
    df_truth = pd.DataFrame(
        {
            "document_id": ["doc_1.pdf"],
            "income_annually_in_british_pounds": [np.nan],
            "spending_annually_in_british_pounds": [1000.0],
        }
    )
    df_imputed = pd.DataFrame(
        {
            "document_id": ["doc_1.pdf"],
            "income_annually_in_british_pounds": [np.nan],
            "spending_annually_in_british_pounds": [np.nan],
            "imputado_por_ia": [False],
            "fonte_documento": [None],
            "pagina": [None],
            "trecho_evidenciado": [None],
        }
    )

    report, _ = calculate_metrics(df_truth=df_truth, df_imputed=df_imputed)

    assert report.lineage_completeness == 0.0
    for col_metric in report.columns_evaluated:
        assert col_metric.accuracy_rate == 0.0
        assert col_metric.precision_rate == 0.0
        assert col_metric.fill_rate == 0.0


# ===========================================================================
# 4. Testes de Exportação de Relatórios (export_reports)
# ===========================================================================


def test_export_reports_creates_json_and_csv_in_tmp_path(tmp_path: Path) -> None:
    """Deve criar o diretório de saída e gravar summary_metrics.json e detailed_evaluation.csv."""
    output_dir = tmp_path / "reports_output"
    report = EvaluationReport(
        total_documents_evaluated=2,
        columns_evaluated=[
            ColumnMetrics(
                column_name="income_annually_in_british_pounds",
                total_expected=2,
                total_imputed=2,
                correct_matches=2,
                accuracy_rate=1.0,
                precision_rate=1.0,
                fill_rate=1.0,
            )
        ],
        lineage_completeness=1.0,
    )
    df_merged = pd.DataFrame(
        {
            "document_id": ["doc_1.pdf", "doc_2.pdf"],
            "income_annually_in_british_pounds_truth": [100.0, 200.0],
            "income_annually_in_british_pounds_imputed": [100.0, 200.0],
            "income_annually_in_british_pounds_is_match": [True, True],
        }
    )

    json_path_str, csv_path_str = export_reports(
        report=report,
        df_merged=df_merged,
        output_dir=str(output_dir),
    )

    json_path = Path(json_path_str)
    csv_path = Path(csv_path_str)

    assert json_path.exists()
    assert json_path.name == "summary_metrics.json"
    assert csv_path.exists()
    assert csv_path.name == "detailed_evaluation.csv"

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["total_documents_evaluated"] == 2
    assert loaded_json["lineage_completeness"] == 1.0
    assert loaded_json["columns_evaluated"][0]["accuracy_rate"] == 1.0

    loaded_csv = pd.read_csv(csv_path)
    assert len(loaded_csv) == 2
    assert list(loaded_csv["document_id"]) == ["doc_1.pdf", "doc_2.pdf"]
