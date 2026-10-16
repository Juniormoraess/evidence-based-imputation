"""Cliente LLM (DeepSeek via SDK OpenAI) e engenharia de prompt para extração baseada em evidências."""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from src.imputation.models import ExtractionResult
from src.retrieval.models import ImputationTask, RetrievedEvidence

logger = logging.getLogger(__name__)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "income_annually_in_british_pounds": (
        "Total annual income / total incoming resources in British pounds (£ / GBP)"
    ),
    "spending_annually_in_british_pounds": (
        "Total annual spending / total expenditure / resources expended in British pounds (£ / GBP)"
    ),
}


def build_extraction_prompt(
    task: ImputationTask,
    evidences: list[RetrievedEvidence],
) -> str:
    """Constrói o prompt determinístico para extração de fatos contábeis a partir das evidências.

    Args:
        task: Tarefa de imputação contendo metadados da instituição e coluna alvo.
        evidences: Lista de fragmentos de evidência recuperados do documento.

    Returns:
        Prompt formatado exigindo resposta estrita em JSON compatível com ExtractionResult.
    """
    charity_display = task.charity_name if task.charity_name else "Unknown Charity"
    report_date_display = task.report_date if task.report_date else "Unspecified Date"
    metric_description = COLUMN_DESCRIPTIONS.get(
        task.target_column,
        task.target_column.replace("_", " "),
    )

    evidence_blocks: list[str] = []
    for idx, ev in enumerate(evidences, start=1):
        evidence_blocks.append(
            f"--- Evidence #{idx} [Page {ev.page_number}] (Chunk {ev.chunk_index}) ---\n"
            f"{ev.text_content}"
        )

    evidences_text = (
        "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence blocks provided."
    )

    prompt = (
        "You are a deterministic financial auditing data extractor.\n"
        "Your sole objective is to extract an exact accounting fact from the provided evidence chunks.\n\n"
        "STRICT RULES:\n"
        "1. Rely EXCLUSIVELY on the provided evidence chunks below.\n"
        "2. Do NOT guess, extrapolate, interpolate, or calculate/sum values that are not explicitly stated in the text.\n"
        "3. If the requested value is stated in thousands (e.g., '£150k' or '£ in thousands') or only partial figures exist without an explicit total, only extract if the exact absolute figure is unambiguous; otherwise set value_found to false.\n"
        "4. If the metric is not explicitly present in the evidences, set value_found to false and extracted_value to null.\n\n"
        "TARGET CONTEXT:\n"
        f"- Document ID: {task.document_id}\n"
        f"- Charity Name: {charity_display}\n"
        f"- Report Date: {report_date_display}\n"
        f"- Target Column: {task.target_column} ({metric_description})\n\n"
        "RETRIEVED EVIDENCES:\n"
        f"{evidences_text}\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "value_found": boolean,\n'
        '  "extracted_value": number or null,\n'
        '  "source_page": integer or null,\n'
        '  "exact_quote": string or null,\n'
        '  "reasoning": string,\n'
        '  "confidence_score": number between 0.0 and 1.0\n'
        "}"
    )
    return prompt


def _build_fallback_result() -> ExtractionResult:
    """Retorna uma instância segura de ExtractionResult em caso de falha da API ou de validação."""
    return ExtractionResult(
        value_found=False,
        extracted_value=None,
        source_page=None,
        exact_quote=None,
        reasoning="LLM parsing or API error",
        confidence_score=0.0,
    )


def extract_value_with_llm(prompt: str) -> ExtractionResult:
    """Invoca a LLM (DeepSeek via SDK OpenAI) para extrair o dado estruturado em JSON.

    Lê DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL (default: https://api.deepseek.com)
    e DEEPSEEK_MODEL (default: deepseek-chat) do arquivo .env.

    Args:
        prompt: Prompt de extração formatado com as evidências.

    Returns:
        Instância validada de ExtractionResult. Em caso de erro de API ou ValidationError,
        retorna um ExtractionResult defensivo com value_found=False e confidence_score=0.0.
    """
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        logger.error("Variável de ambiente DEEPSEEK_API_KEY ausente ou vazia.")
        return _build_fallback_result()

    base_url = (
        os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL) or DEFAULT_DEEPSEEK_BASE_URL
    )
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL) or DEFAULT_DEEPSEEK_MODEL

    try:
        client = OpenAI(
            api_key=api_key.strip(),
            base_url=base_url.strip(),
        )

        response = client.chat.completions.create(
            model=model.strip(),
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise financial data extraction assistant that outputs "
                        "strictly valid JSON matching the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw_content = response.choices[0].message.content if response.choices else None
        if not raw_content:
            logger.error("A LLM retornou conteúdo vazio.")
            return _build_fallback_result()

        return ExtractionResult.model_validate_json(raw_content)
    except ValidationError as val_exc:
        logger.error("Falha de validação Pydantic na resposta JSON da LLM: %s", val_exc)
        return _build_fallback_result()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Falha na chamada à API da LLM ou parse do JSON: %s", exc)
        return _build_fallback_result()
