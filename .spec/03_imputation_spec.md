# Especificação de Desenvolvimento: Módulo de Imputação

## 1. Objetivo do Módulo
O módulo `src/imputation/` é o núcleo de decisão do framework. Ele recebe uma tarefa pendente (`ImputationTask`) e as evidências recuperadas (` list[RetrievedEvidence]`) do módulo de *Retrieval*, constrói o prompt de contexto, invoca a API do DeepSeek V4 com saída estruturada em JSON, valida os dados financeiros via Pydantic e executa o `UPDATE` no Data Warehouse (`dw_charity_data`) preenchendo a coluna alvo e os metadados de linhagem/auditoria.

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `openai` (cliente compatível com a API do DeepSeek), `pydantic` (validação rigorosa de saída), `psycopg2` (escrita segura no banco com `psycopg2.sql`), `python-dotenv`.
* **Configuração da LLM:** Ler `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (default: `https://api.deepseek.com`) e `DEEPSEEK_MODEL` (default: `deepseek-chat`) do arquivo `.env`. A chamada deve usar `temperature=0.0` e `response_format={"type": "json_object"}`.
* **Engenharia de Prompt Determinística:** O prompt deve instruir o modelo a atuar exclusivamente como extrator de fatos contábeis baseados nas evidências fornecidas. É proibido adivinhar, extrapolar ou somar valores que não estejam explícitos no texto.
* **Segurança SQL:** O nome da coluna alvo (`income_annually_in_british_pounds` ou `spending_annually_in_british_pounds`) deve ser validado contra uma *allowlist* estrita e injetado na query usando `psycopg2.sql.Identifier` para evitar SQL Injection.

## 3. Modelos de Dados (Data Contracts)
A IA deve criar o seguinte contrato Pydantic em `src/imputation/models.py`:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any

class ExtractionResult(BaseModel):
    value_found: bool = Field(description="True somente se o valor financeiro exato foi localizado nas evidências.")
    extracted_value: Optional[float] = Field(default=None, description="Valor numérico absoluto em libras esterlinas (ex: 15500.50). None se não encontrado.")
    source_page: Optional[int] = Field(default=None, description="Número da página do chunk de onde o valor foi extraído.")
    exact_quote: Optional[str] = Field(default=None, description="Citação literal exata do texto que comprova o valor extraído.")
    reasoning: str = Field(description="Explicação objetiva relacionando o trecho citado à métrica solicitada.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Grau de confiança na extração, entre 0.0 e 1.0.")

    @field_validator("extracted_value", mode="before")
    @classmethod
    def sanitize_currency(cls, v: Any) -> Optional[float]:
        # Deve converter strings como "£15,500.50" ou "15,000" em float(15500.50), ou retornar None
        ...
```

## 4. Assinaturas de Funções Exigidas
A IA deve implementar as funções em `src/imputation/` (ex: `models.py`, `llm_client.py`, `writer.py`, `pipeline.py`):

1. **Construção do Prompt:**
   `def build_extraction_prompt(task: ImputationTask, evidences: list[RetrievedEvidence]) -> str:`
   * *Regra:* Formatar uma instrução clara especificando a instituição (`task.charity_name`), a data do relatório (`task.report_date`), a métrica desejada (`task.target_column`) e a lista de evidências numeradas com seus respectivos `[Page X]` e texto. Exigir retorno estrito em JSON compatível com o schema de `ExtractionResult`.

2. **Chamada da LLM:**
   `def extract_value_with_llm(prompt: str) -> ExtractionResult:`
   * *Regra:* Invocar o cliente OpenAI configurado para o DeepSeek e validar a resposta com `ExtractionResult.model_validate_json()`.
   * *Programação Defensiva:* Se a API falhar, retornar JSON malformado ou ocorrer `ValidationError`, registrar o erro via `logging` e retornar um `ExtractionResult` seguro com `value_found=False, extracted_value=None, source_page=None, exact_quote=None, reasoning="LLM parsing or API error", confidence_score=0.0`.

3. **Escrita no Data Warehouse (Auditabilidade):**
   `def update_dw_record(task: ImputationTask, result: ExtractionResult, evidences: list[RetrievedEvidence], db_connection_string: str) -> bool:`
   * *Regra:*
     * Se `not result.value_found` ou `result.extracted_value is None`, **não** executar `UPDATE` na coluna financeira (mantendo `NULL` no banco), apenas registrar log e retornar `False`.
     * Se `result.value_found` for `True` e houver `extracted_value`, determinar a página final (`result.source_page` se presente, caso contrário `evidences[0].page_number` como fallback) e executar `UPDATE dw_charity_data` na linha `WHERE id = task.row_id`, atualizando:
       * A coluna dinâmica `task.target_column = result.extracted_value`
       * `imputado_por_ia = TRUE`
       * `fonte_documento = task.document_id`
       * `pagina = pagina_determinada`
       * `trecho_evidenciado = result.exact_quote`
       * `confianca = result.confidence_score`
       * `data_imputacao = CURRENT_TIMESTAMP`
     * Retornar `True` quando o registro for atualizado com sucesso.

4. **Orquestração do Pipeline de Imputação:**
   `def run_imputation_pipeline(db_connection_string: str, top_k: int = 5) -> dict[str, int]:`
   * *Regra:* Integrar com `src/retrieval` (`get_pending_tasks`, `embed_search_query`, `fetch_evidence`). Para cada task pendente, buscar evidências; se houver evidências, construir o prompt, extrair via LLM e atualizar o DW. Retornar dicionário resumo com `{"total_tasks": int, "imputed_count": int, "skipped_count": int}`.

## 5. Script Executável (`scripts/02_run_imputation.py`)
Deve carregar variáveis do `.env`, validar a presença de `DEEPSEEK_API_KEY` e `EMBEDDING_API_KEY`, montar a `db_connection_string` e executar `run_imputation_pipeline`, exibindo o resumo final no log.

## 6. Critérios de Aceite (Definition of Done)
1. **Tipagem e Pacotes:** 100% das funções tipadas (PEP 484) e `src/imputation/__init__.py` exportando contratos e funções principais.
2. **Sanitização Numérica:** Strings monetárias como `"£ 12,345.67"` ou `"12,000"` são convertidas automaticamente para `float` pelo modelo Pydantic.
3. **Proteção contra SQL Injection:** Uso obrigatório de `psycopg2.sql.Identifier` e validação de `target_column`.
4. **Testes Unitários:** Arquivo `tests/test_imputation.py` usando `pytest` e `pytest-mock` cobrindo: sanitização de valores no Pydantic, construção do prompt, extração via LLM com JSON válido e com JSON corrompido (fallback defensivo), escrita no banco quando `value_found=True` e não-escrita quando `value_found=False`, e orquestração completa mockada.