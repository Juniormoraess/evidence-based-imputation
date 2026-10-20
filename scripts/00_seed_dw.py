"""Script utilitário para popular a tabela dw_charity_data a partir do Ground Truth com lacunas (NULL) nas colunas alvo."""

import argparse
from datetime import date, datetime
import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    _normalize_document_id,
    load_ground_truth,
)

logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_PATH = "data/ground_truth/expected_with_doc_id.xlsx"
DEFAULT_RAW_DIR = "data/raw"

METADATA_COLUMNS: tuple[str, ...] = (
    "document_id",
    "charity_name",
    "charity_number",
    "address__post_town",
    "address__postcode",
    "address__street_line",
    "report_date",
)

TRUNCATE_DW_SQL = "TRUNCATE TABLE dw_charity_data RESTART IDENTITY;"

UPSERT_DW_SQL = """
    INSERT INTO dw_charity_data (
        document_id,
        charity_name,
        charity_number,
        address__post_town,
        address__postcode,
        address__street_line,
        report_date,
        income_annually_in_british_pounds,
        spending_annually_in_british_pounds,
        imputado_por_ia,
        fonte_documento,
        pagina,
        trecho_evidenciado,
        confianca,
        data_imputacao
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, NULL, FALSE, NULL, NULL, NULL, NULL, NULL)
    ON CONFLICT (document_id) DO UPDATE SET
        charity_name = EXCLUDED.charity_name,
        charity_number = EXCLUDED.charity_number,
        address__post_town = EXCLUDED.address__post_town,
        address__postcode = EXCLUDED.address__postcode,
        address__street_line = EXCLUDED.address__street_line,
        report_date = EXCLUDED.report_date,
        income_annually_in_british_pounds = NULL,
        spending_annually_in_british_pounds = NULL,
        imputado_por_ia = FALSE,
        fonte_documento = NULL,
        pagina = NULL,
        trecho_evidenciado = NULL,
        confianca = NULL,
        data_imputacao = NULL
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Processa os argumentos de linha de comando para preparação do Data Warehouse.

    Args:
        argv: Lista opcional de argumentos (default: sys.argv[1:]).

    Returns:
        Namespace com as opções parseadas (only_raw_pdfs, limit, truncate).
    """
    parser = argparse.ArgumentParser(
        description="Popula a tabela dw_charity_data a partir do Ground Truth com colunas financeiras NULL."
    )
    parser.add_argument(
        "--only-raw-pdfs",
        action="store_true",
        help="Filtra o Ground Truth mantendo apenas registros cujo PDF exista em data/raw/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a quantidade máxima de documentos inseridos no Data Warehouse.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Executa TRUNCATE TABLE dw_charity_data RESTART IDENTITY antes da carga.",
    )
    return parser.parse_args(argv)


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


def sanitize_pandas_value(value: object) -> str | int | float | date | None:
    """Converte valores NaN/NaT do Pandas e escalares do NumPy para tipos nativos Python aceitos pelo psycopg2.

    Args:
        value: Valor bruto extraído de uma célula do DataFrame.

    Returns:
        Valor nativo Python (str, int, float, date) ou None se for NaN/nulo.
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def prepare_seed_dataframe(
    df: pd.DataFrame,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    only_raw_pdfs: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Normaliza, filtra por PDFs existentes em data/raw/ e aplica limite de linhas ao DataFrame.

    Args:
        df: DataFrame original carregado do Ground Truth.
        raw_dir: Caminho da pasta contendo os PDFs brutos (default: "data/raw").
        only_raw_pdfs: Se True, mantém apenas registros com arquivo .pdf presente em raw_dir.
        limit: Quantidade máxima de registros a manter (opcional).

    Returns:
        DataFrame filtrado e pronto para inserção no banco.

    Raises:
        ValueError: Se a coluna 'document_id' não existir ou se limit <= 0.
    """
    if "document_id" not in df.columns:
        raise ValueError("A coluna obrigatória 'document_id' não foi encontrada no DataFrame.")

    if limit is not None and limit <= 0:
        raise ValueError("O argumento '--limit' deve ser um inteiro maior que zero.")

    result = df.copy()
    result["document_id"] = result["document_id"].map(_normalize_document_id)
    result = result[result["document_id"] != ""].reset_index(drop=True)

    if only_raw_pdfs:
        raw_path = Path(raw_dir)
        existing_mask = result["document_id"].map(
            lambda doc_id: (raw_path / str(doc_id)).is_file()
        )
        result = result[existing_mask].reset_index(drop=True)
        logger.info(
            "Filtro --only-raw-pdfs aplicado: %d registro(s) com PDF presente em '%s'.",
            len(result),
            raw_path,
        )

    if limit is not None:
        result = result.head(limit).reset_index(drop=True)
        logger.info("Limite --limit=%d aplicado: %d registro(s) selecionado(s).", limit, len(result))

    return result


def seed_dw_table(
    df: pd.DataFrame,
    db_connection_string: str,
    truncate: bool = False,
) -> int:
    """Insere ou atualiza os registros de experimento na tabela dw_charity_data forçando NULL nas colunas alvo.

    Args:
        df: DataFrame preparado com os metadados das instituições.
        db_connection_string: String de conexão (DSN/URI) para o PostgreSQL.
        truncate: Se True, executa TRUNCATE TABLE dw_charity_data RESTART IDENTITY antes da carga.

    Returns:
        Total de registros inseridos/atualizados.

    Raises:
        ValueError: Se db_connection_string for vazia.
        psycopg2.Error: Em caso de erro de banco de dados durante a transação.
    """
    if not db_connection_string or not db_connection_string.strip():
        raise ValueError("A string de conexão 'db_connection_string' não pode ser vazia.")

    records_to_insert: list[tuple[str | int | float | date | None, ...]] = []
    for _, row in df.iterrows():
        row_tuple = tuple(
            sanitize_pandas_value(row[col]) if col in df.columns else None
            for col in METADATA_COLUMNS
        )
        records_to_insert.append(row_tuple)

    try:
        with psycopg2.connect(db_connection_string) as conn:
            with conn.cursor() as cursor:
                if truncate:
                    logger.info("Executando TRUNCATE TABLE dw_charity_data RESTART IDENTITY...")
                    cursor.execute(TRUNCATE_DW_SQL)

                for params in records_to_insert:
                    cursor.execute(UPSERT_DW_SQL, params)

            conn.commit()

        logger.info(
            "Carga concluída com sucesso: %d registro(s) preparado(s) em dw_charity_data.",
            len(records_to_insert),
        )
        return len(records_to_insert)
    except Exception as exc:
        logger.error("Falha ao popular a tabela dw_charity_data: %s", exc)
        raise


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada principal do script de preparação do Data Warehouse.

    Args:
        argv: Lista opcional de argumentos de linha de comando.

    Returns:
        Código de saída do processo (0 para sucesso, 1 para falha).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    load_dotenv()

    try:
        args = parse_args(argv)
        db_connection_string = build_db_connection_string()
    except (ValueError, SystemExit) as exc:
        logger.error("Erro nos parâmetros ou variáveis de ambiente: %s", exc)
        return 1

    gt_env_path = (
        os.getenv("GROUND_TRUTH_PATH", DEFAULT_GROUND_TRUTH_PATH) or DEFAULT_GROUND_TRUTH_PATH
    ).strip()

    gt_path = Path(gt_env_path)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path

    raw_dir_path = PROJECT_ROOT / DEFAULT_RAW_DIR

    try:
        logger.info("Lendo Ground Truth de '%s'...", gt_path)
        df_ground_truth = load_ground_truth(str(gt_path))

        df_seed = prepare_seed_dataframe(
            df=df_ground_truth,
            raw_dir=raw_dir_path,
            only_raw_pdfs=bool(args.only_raw_pdfs),
            limit=args.limit,
        )

        inserted_count = seed_dw_table(
            df=df_seed,
            db_connection_string=db_connection_string,
            truncate=bool(args.truncate),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Erro durante a preparação do Data Warehouse: %s", exc)
        return 1

    logger.info(
        "Experimento pronto no Data Warehouse: %d documento(s) com lacunas aguardando imputação.",
        inserted_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
