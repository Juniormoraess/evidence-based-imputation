"""Modelos de dados (Data Contracts) do módulo de imputação."""

import re
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class ExtractionResult(BaseModel):
    """Contrato de saída estruturada retornado pela LLM após análise das evidências."""

    value_found: bool = Field(
        description="True somente se o valor financeiro exato foi localizado nas evidências."
    )
    extracted_value: Optional[float] = Field(
        default=None,
        description="Valor numérico absoluto em libras esterlinas (ex: 15500.50). None se não encontrado.",
    )
    source_page: Optional[int] = Field(
        default=None,
        description="Número da página do chunk de onde o valor foi extraído.",
    )
    exact_quote: Optional[str] = Field(
        default=None,
        description="Citação literal exata do texto que comprova o valor extraído.",
    )
    reasoning: str = Field(
        description="Explicação objetiva relacionando o trecho citado à métrica solicitada."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Grau de confiança na extração, entre 0.0 e 1.0.",
    )

    @field_validator("extracted_value", mode="before")
    @classmethod
    def sanitize_currency(cls, v: Any) -> Optional[float]:
        """Converte strings monetárias como '£15,500.50' ou '15,000' em float, ou retorna None."""
        if v is None:
            return None

        if isinstance(v, bool):
            raise ValueError("Valor booleano inválido para 'extracted_value'.")

        if isinstance(v, (int, float)):
            return float(v)

        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned or cleaned.lower() in {"null", "none", "n/a", "na", "-"}:
                return None

            # Remove símbolos monetários (ex: £, GBP, $, €), espaços e separadores de milhar (,)
            cleaned = re.sub(r"(?i)\bgbp\b", "", cleaned)
            cleaned = cleaned.replace("£", "").replace("$", "").replace("€", "")
            cleaned = cleaned.replace(",", "").replace(" ", "").strip()

            # Trata parênteses contábeis para valores negativos, ex: "(15500.50)" -> "-15500.50"
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1].strip()

            try:
                return float(cleaned)
            except ValueError as exc:
                raise ValueError(
                    f"Não foi possível converter '{v}' em um valor numérico float válido."
                ) from exc

        raise ValueError(f"Tipo inválido para 'extracted_value': {type(v).__name__}")
