"""Módulo de Recuperação (Retrieval) para busca híbrida de evidências no PostgreSQL (pgvector)."""

from src.retrieval.models import ImputationTask, RetrievedEvidence
from src.retrieval.search import embed_search_query, fetch_evidence, get_pending_tasks

__all__ = [
    "ImputationTask",
    "RetrievedEvidence",
    "get_pending_tasks",
    "embed_search_query",
    "fetch_evidence",
]
