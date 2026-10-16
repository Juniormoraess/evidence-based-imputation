"""Modelos de dados (Data Contracts) do módulo de avaliação (Evaluation)."""

from pydantic import BaseModel


class ColumnMetrics(BaseModel):
    """Métricas de avaliação de imputação para uma coluna financeira específica."""

    column_name: str
    total_expected: int        # Total de registros com valor não-nulo no Ground Truth
    total_imputed: int         # Total de registros (dentre os esperados) preenchidos pela IA
    correct_matches: int       # Total de acertos dentro da margem de tolerância financeira
    accuracy_rate: float       # correct_matches / total_expected (0.0 se total_expected == 0)
    precision_rate: float      # correct_matches / total_imputed (0.0 se total_imputed == 0)
    fill_rate: float           # total_imputed / total_expected (0.0 se total_expected == 0)


class EvaluationReport(BaseModel):
    """Relatório consolidado de avaliação científica da imputação e completude de linhagem."""

    total_documents_evaluated: int
    columns_evaluated: list[ColumnMetrics]
    lineage_completeness: float  # Proporção de registros imputados (imputado_por_ia=True) com linhagem completa
