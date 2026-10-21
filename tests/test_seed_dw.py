"""Suíte de testes unitários para o script utilitário scripts/00_seed_dw.py."""

from datetime import date
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

seed_dw = importlib.import_module("scripts.00_seed_dw")


# ===========================================================================
# 1. Testes de Parsing de Argumentos (parse_args)
# ===========================================================================


def test_parse_args_defaults_and_flags() -> None:
    """Deve interpretar corretamente os argumentos --only-raw-pdfs, --limit e --truncate."""
    default_args = seed_dw.parse_args([])
    assert default_args.only_raw_pdfs is False
    assert default_args.limit is None
    assert default_args.truncate is False

    custom_args = seed_dw.parse_args(["--only-raw-pdfs", "--limit", "5", "--truncate"])
    assert custom_args.only_raw_pdfs is True
    assert custom_args.limit == 5
    assert custom_args.truncate is True


# ===========================================================================
# 2. Testes de Sanitização de Valores Pandas/NumPy (sanitize_pandas_value)
# ===========================================================================


def test_sanitize_pandas_value_converts_nan_and_types() -> None:
    """Deve converter NaN/NaT para None e tipos escalares do Pandas/NumPy para tipos nativos."""
    assert seed_dw.sanitize_pandas_value(None) is None
    assert seed_dw.sanitize_pandas_value(np.nan) is None
    assert seed_dw.sanitize_pandas_value(float("nan")) is None
    assert seed_dw.sanitize_pandas_value(pd.NaT) is None

    ts = pd.Timestamp("2024-03-31")
    assert seed_dw.sanitize_pandas_value(ts) == date(2024, 3, 31)

    np_int = np.int64(123456)
    sanitized_int = seed_dw.sanitize_pandas_value(np_int)
    assert sanitized_int == 123456
    assert type(sanitized_int) is int

    np_float = np.float64(99.5)
    sanitized_float = seed_dw.sanitize_pandas_value(np_float)
    assert sanitized_float == 99.5
    assert type(sanitized_float) is float


# ===========================================================================
# 3. Testes de Filtro de PDFs e Limite (prepare_seed_dataframe)
# ===========================================================================


def test_prepare_seed_dataframe_normalizes_filters_pdfs_and_limits(tmp_path: Path) -> None:
    """Deve normalizar document_id, filtrar apenas PDFs existentes em data/raw e aplicar --limit."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Cria apenas doc_1.pdf e doc_3.pdf fisicamente
    (raw_dir / "doc_1.pdf").write_bytes(b"%PDF-1.4 dummy")
    (raw_dir / "doc_3.pdf").write_bytes(b"%PDF-1.4 dummy")

    df = pd.DataFrame(
        {
            "document_id": [" doc_1 ", "doc_2_ausente", "doc_3.pdf"],
            "charity_name": ["Charity 1", "Charity 2", "Charity 3"],
        }
    )

    # Sem filtro de PDF físico, deve manter os 3 com sufixo .pdf normalizado
    df_all = seed_dw.prepare_seed_dataframe(df, raw_dir=raw_dir, only_raw_pdfs=False)
    assert list(df_all["document_id"]) == ["doc_1.pdf", "doc_2_ausente.pdf", "doc_3.pdf"]

    # Com --only-raw-pdfs, deve manter apenas doc_1.pdf e doc_3.pdf
    df_filtered = seed_dw.prepare_seed_dataframe(df, raw_dir=raw_dir, only_raw_pdfs=True)
    assert list(df_filtered["document_id"]) == ["doc_1.pdf", "doc_3.pdf"]

    # Com --only-raw-pdfs e --limit 1, deve manter apenas doc_1.pdf
    df_limited = seed_dw.prepare_seed_dataframe(
        df, raw_dir=raw_dir, only_raw_pdfs=True, limit=1
    )
    assert list(df_limited["document_id"]) == ["doc_1.pdf"]


def test_prepare_seed_dataframe_invalid_limit_raises() -> None:
    """Deve lançar ValueError se --limit for menor ou igual a zero."""
    df = pd.DataFrame({"document_id": ["doc_1.pdf"]})
    with pytest.raises(ValueError, match="--limit"):
        seed_dw.prepare_seed_dataframe(df, limit=0)


# ===========================================================================
# 4. Testes de Carga no Banco de Dados (seed_dw_table) - Mockado
# ===========================================================================


def test_seed_dw_table_with_truncate_and_nan_conversion(mocker: MockerFixture) -> None:
    """Deve executar TRUNCATE quando solicitado, converter NaN para None e fazer UPSERT com colunas alvo NULL."""
    mock_connect = mocker.patch.object(seed_dw.psycopg2, "connect")
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    df = pd.DataFrame(
        {
            "document_id": ["doc_1.pdf", "doc_2.pdf"],
            "charity_name": ["Oxfam GB", np.nan],
            "charity_number": [202918, np.nan],
            "address__post_town": ["Oxford", np.nan],
            "address__postcode": ["OX4 2JY", None],
            "address__street_line": ["Oxfam House", np.nan],
            "report_date": [pd.Timestamp("2024-03-31"), pd.NaT],
            # Mesmo que o Ground Truth tenha valores nas colunas alvo, o seed deve forçar NULL no SQL
            "income_annually_in_british_pounds": [1000000.0, 500000.0],
            "spending_annually_in_british_pounds": [900000.0, 450000.0],
        }
    )

    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"
    inserted = seed_dw.seed_dw_table(df=df, db_connection_string=dsn, truncate=True)

    assert inserted == 2
    mock_connect.assert_called_once_with(dsn)

    # 1 chamada de TRUNCATE + 2 chamadas de INSERT ON CONFLICT = 3 execuções
    assert mock_cursor.execute.call_count == 3

    first_sql = mock_cursor.execute.call_args_list[0].args[0]
    assert first_sql == "TRUNCATE TABLE dw_charity_data RESTART IDENTITY;"

    second_sql, second_params = mock_cursor.execute.call_args_list[1].args
    assert "INSERT INTO dw_charity_data" in second_sql
    assert "ON CONFLICT (document_id) DO UPDATE" in second_sql
    assert "income_annually_in_british_pounds = NULL" in second_sql
    assert "spending_annually_in_british_pounds = NULL" in second_sql
    assert "income_pagina = NULL" in second_sql
    assert "income_trecho_evidenciado = NULL" in second_sql
    assert "income_confianca = NULL" in second_sql
    assert "spending_pagina = NULL" in second_sql
    assert "spending_trecho_evidenciado = NULL" in second_sql
    assert "spending_confianca = NULL" in second_sql
    assert "imputado_por_ia = FALSE" in second_sql
    assert second_params == (
        "doc_1.pdf",
        "Oxfam GB",
        202918,
        "Oxford",
        "OX4 2JY",
        "Oxfam House",
        date(2024, 3, 31),
    )

    # Verifica que todos os NaN/NaT da segunda linha foram convertidos para None
    _, third_params = mock_cursor.execute.call_args_list[2].args
    assert third_params == (
        "doc_2.pdf",
        None,
        None,
        None,
        None,
        None,
        None,
    )

    mock_conn.commit.assert_called_once()


def test_seed_dw_table_without_truncate(mocker: MockerFixture) -> None:
    """Não deve executar TRUNCATE quando a flag truncate for False."""
    mock_connect = mocker.patch.object(seed_dw.psycopg2, "connect")
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    df = pd.DataFrame({"document_id": ["doc_1.pdf"], "charity_name": ["Charity One"]})
    dsn = "postgresql://postgres:secret@localhost:5432/evidence_db"

    inserted = seed_dw.seed_dw_table(df=df, db_connection_string=dsn, truncate=False)

    assert inserted == 1
    assert mock_cursor.execute.call_count == 1
    executed_sql = mock_cursor.execute.call_args.args[0]
    assert "TRUNCATE" not in executed_sql
    assert "INSERT INTO dw_charity_data" in executed_sql
