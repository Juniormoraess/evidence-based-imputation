"""Modelos de dados (Data Contracts) do módulo de ingestão."""

from typing import List, Optional
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """Representa um fragmento de texto extraído de um documento PDF com seu respectivo embedding."""

    document_id: str  # Nome exato do arquivo PDF (ex: "1ada336f29e8247f9f55a8d7e1b1c0da.pdf")
    page_number: int
    chunk_index: int
    text_content: str
    embedding: Optional[List[float]] = None
