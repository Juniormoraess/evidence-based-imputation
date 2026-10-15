"""Extração de texto de arquivos PDF utilizando PyMuPDF (fitz)."""

import logging
from pathlib import Path
from typing import List

import fitz

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> List[dict[str, int | str]]:
    """Extrai o texto de todas as páginas de um arquivo PDF.

    Args:
        file_path: Caminho para o arquivo PDF.

    Returns:
        Lista de dicionários contendo {"page_number": int, "text": str} (1-indexed).

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho informado.
        RuntimeError: Se o arquivo PDF estiver corrompido, vazio ou ilegível.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        logger.error("Arquivo PDF não encontrado: %s", file_path)
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {file_path}")

    try:
        pages_data: List[dict[str, int | str]] = []
        with fitz.open(file_path) as pdf_document:
            if len(pdf_document) == 0:
                logger.warning("O arquivo PDF '%s' não possui páginas.", file_path)
                return []

            for page_idx, page in enumerate(pdf_document):
                page_text: str = str(page.get_text("text"))
                pages_data.append(
                    {
                        "page_number": page_idx + 1,
                        "text": page_text,
                    }
                )

        return pages_data
    except Exception as exc:
        logger.error("Falha ao ler o arquivo PDF '%s': %s", file_path, exc)
        raise RuntimeError(f"Erro ao processar PDF '{file_path}': {exc}") from exc
