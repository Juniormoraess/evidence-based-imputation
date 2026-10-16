"""Modelos de dados (Data Contracts) do módulo de recuperação (Retrieval)."""

from typing import Optional
from pydantic import BaseModel


class ImputationTask(BaseModel):
    """Representa uma lacuna (NULL) identificada no Data Warehouse a ser preenchida."""

    row_id: int
    document_id: str
    charity_name: Optional[str] = None
    report_date: Optional[str] = None
    target_column: str  # 'income_annually_in_british_pounds' ou 'spending_annually_in_british_pounds'


class RetrievedEvidence(BaseModel):
    """Representa um fragmento de documento recuperado via busca híbrida no pgvector."""

    document_id: str
    page_number: int
    chunk_index: int
    text_content: str
    similarity_score: float
