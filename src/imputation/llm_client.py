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
        "Total annual income / total incoming resources / total gross income in British pounds (£ / GBP)"
    ),
    "spending_annually_in_british_pounds": (
        "Total annual spending / total expenditure / total expense / total resources expended in British pounds (£ / GBP)"
    ),
}

SYSTEM_AUDITING_PROMPT = (
    "You are a senior UK charity financial auditor and deterministic data extraction assistant.\n"
    "You output strictly valid JSON matching the requested schema and adhere to these 3 mandatory accounting rules:\n"
    "1. REGRA 1 (Hierarquia Contábil / Accounting Hierarchy): Always prioritize figures extracted from formal "
    "financial statement tables (Statement of Financial Activities / SOFA, Profit & Loss Statement, "
    "Income & Expenditure Account, Receipts and Payments) over rounded figures cited in narrative paragraphs "
    "of the Trustees' Report. If a formal Profit & Loss or SOFA table provides exact figures with pence "
    "(e.g., Total Income 143,120.55 / Total Expense 108,294.79) and narrative text cites rounded figures "
    "(e.g., £140,265 / £108,111), you MUST extract the figure from the formal financial table. "
    "Furthermore, NEVER confuse intermediate category subtotals of expenditure (such as the subtotal for "
    "'Charitable activities' alone) with the final consolidated 'Total expenditure' / 'Total resources expended' "
    "line (which sums Raising funds + Charitable activities + Trading/Other activities).\n"
    "2. REGRA 2 (Escala de Milhares e Ruído de OCR / Thousands Scale & OCR Noise): Scanned reports frequently "
    "contain OCR errors where the pound symbol '£' appears as 'F', 'f', or '5'. Table headers such as "
    "'£000', '£'000', 'F000's', 'f000's', or '5000's' mean the table numbers are stated in thousands of pounds "
    "and MUST be multiplied by 1000 (e.g., '10,348' under column 'F000's' = 10348000.00; '9,415' under "
    "'F000's' = 9415000.00). Likewise, figures in text such as '510,348,000' (when compared to '510,204,000') "
    "have the leading '5' as an OCR misread of the '£' symbol (£10,348,000 = 10348000.00).\n"
    "3. REGRA 3 (Fundos Totais e Período / Total Funds & Period): In tables with 'Restricted', 'Unrestricted', "
    "and 'Total' columns, always extract the 'Total' column corresponding to the financial year of report_date."
)


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
        "You are a deterministic UK charity financial auditing data extractor.\n"
        "Your sole objective is to extract the exact accounting figure from the provided evidence chunks.\n\n"
        "MANDATORY ACCOUNTING AUDITING RULES:\n"
        "1. REGRA 1 (Hierarquia Contábil / Accounting Hierarchy): Always prioritize values extracted from formal "
        "financial statement tables (Statement of Financial Activities / SOFA, Profit & Loss Statement, "
        "Income & Expenditure Account, Receipts and Payments) over rounded numbers cited in narrative paragraphs "
        "of the Trustees' Report. If there is a Profit & Loss / SOFA table with exact pence (e.g., Total Income "
        "143,120.55 / Total Expense 108,294.79) and narrative text with rounded figures (e.g., £140,265 / £108,111), "
        "you MUST extract the value from the formal table. NEVER confuse intermediate expenditure category subtotals "
        "(such as the subtotal for 'Charitable activities' alone) with the final consolidated 'Total expenditure' / "
        "'Total resources expended' line (which sums Raising funds + Charitable activities + Trading/Other activities).\n"
        "2. REGRA 2 (Escala de Milhares e Ruído de OCR / Thousands Scale & OCR Noise): Scanned reports frequently "
        "exhibit OCR errors where the symbol '£' appears as 'F', 'f', or '5'. Table headers such as '£000', "
        "'£'000', 'F000's', 'f000's', or '5000's' mean the numbers in the table are in thousands of pounds "
        "and MUST be multiplied by 1000 (e.g., '10,348' under column 'F000's' = 10348000.00; '9,415' under "
        "'F000's' = 9415000.00). Similarly, figures in text like '510,348,000' (compared to '510,204,000') "
        "have the leading '5' as an OCR error for the '£' symbol (£10,348,000 = 10348000.00).\n"
        "3. REGRA 3 (Fundos Totais e Período / Total Funds & Period): In tables with 'Restricted', 'Unrestricted', "
        "and 'Total' columns, always extract the 'Total' column corresponding to the financial year of report_date.\n"
        "4. Rely EXCLUSIVELY on the provided evidence chunks. Do NOT invent or sum partial line items if no total is reported. "
        "If the target metric is not present in the evidences, set value_found to false and extracted_value to null.\n\n"
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
                    "content": SYSTEM_AUDITING_PROMPT,
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
