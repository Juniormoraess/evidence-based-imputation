# Especificação de Desenvolvimento: Módulo de Imputação

## 1. Objetivo do Módulo
O módulo `src/imputation/` é o núcleo de decisão do framework. Ele recebe uma tarefa pendente (`ImputationTask`) e as evidências recuperadas (`list[RetrievedEvidence]`) do módulo de *Retrieval*, constrói o prompt de contexto, invoca a API do DeepSeek V4 com saída estruturada em JSON, valida os dados financeiros via Pydantic e executa o `UPDATE` no Data Warehouse (`dw_charity_data`) preenchendo a coluna alvo e os metadados de linhagem/auditoria.

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `openai` (cliente compatível com a API do DeepSeek), `pydantic` (validação rigorosa de saída), `psycopg2` (escrita segura no banco com `psycopg2.sql`), `python-dotenv`.
* **Configuração da LLM:** Ler `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (default: `https://api.deepseek.com`) e `DEEPSEEK_MODEL` (default: `deepseek-chat`) do arquivo `.env`. A chamada deve usar `temperature=0.0` e `response_format={"type": "json_object"}`.
* **Engenharia de Prompt Determinística e Regras de Auditoria Contábil:** O prompt e o system prompt devem instruir o modelo a atuar como auditor contábil determinístico seguindo 3 regras fundamentais:
  1. **REGRA 1 (Hierarquia Contábil):** Sempre priorizar valores extraídos de tabelas de demonstrações financeiras formais (*Statement of Financial Activities / SOFA*, *Profit & Loss Statement*, *Income & Expenditure Account*, *Receipts and Payments*) sobre números arredondados citados em parágrafos narrativos do *Trustees' Report*. Se houver uma tabela *Profit & Loss* com centavos exatos (ex: Total Income `143,120.55` / Total Expense `108,294.79`) e um texto narrativo com valores arredondados (ex: `£140,265` / `£108,111`), extrair obrigatoriamente o valor da tabela formal. Além disso, nunca confundir subtotais intermediários de categorias de despesa (como o subtotal apenas de *"Charitable activities"*) com a linha final consolidada *"Total expenditure"* / *"Total resources expended"* (que soma *Raising funds* + *Charitable activities* + *Trading/Other activities*).
  2. **REGRA 2 (Escala de Milhares e Ruído de OCR):** Relatórios escaneados frequentemente apresentam erros de OCR onde o símbolo `£` aparece como `F`, `f` ou `5`. Cabeçalhos de tabela como `£000`, `£'000`, `F000's`, `f000's` ou `5000's` significam que os números da tabela estão em milhares de libras e DEVEM ser multiplicados por 1000 (ex: `10,348` sob coluna `F000's` = `10348000.00`; `9,415` sob `F000's` = `9415000.00`). Da mesma forma, cifras no texto como `510,348,000` (comparadas a `510,204,000`) têm o `5` inicial como erro de OCR do símbolo `£` (`£10,348,000`).
  3. **REGRA 3 (Fundos Totais e Período):** Em tabelas com colunas *"Restricted"*, *"Unrestricted"* e *"Total"*, extrair sempre a coluna *"Total"* correspondente ao exercício de `report_date`.
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
   * *Regra:* Formatar uma instrução clara especificando a instituição (`task.charity_name`), a data do relatório (`task.report_date`), a métrica desejada (`task.target_column`), as 3 Regras de Auditoria Contábil (Hierarquia Contábil, Escala de Milhares e Ruído de OCR, Fundos Totais e Período) e a lista de evidências numeradas com seus respectivos `[Page X]` e texto. Exigir retorno estrito em JSON compatível com o schema de `ExtractionResult`.

2. **Chamada da LLM:**
   `def extract_value_with_llm(prompt: str) -> ExtractionResult:`
   * *Regra:* Invocar o cliente OpenAI configurado para o DeepSeek reforçando no `system` prompt as regras de auditoria contábil e validar a resposta com `ExtractionResult.model_validate_json()`.
   * *Programação Defensiva:* Se a API falhar, retornar JSON malformado ou ocorrer `ValidationError`, registrar o erro via `logging` e retornar um `ExtractionResult` seguro com `value_found=False, extracted_value=None, source_page=None, exact_quote=None, reasoning="LLM parsing or API error", confidence_score=0.0`.

3. **Escrita no Data Warehouse (Auditabilidade e Linhagem por Métrica):**
   `def update_dw_record(task: ImputationTask, result: ExtractionResult, evidences: list[RetrievedEvidence], db_connection_string: str) -> bool:`
   * *Mapeamento de Linhagem (`COLUMN_LINEAGE_MAP`):*
     * `income_annually_in_british_pounds` -> `("income_pagina", "income_trecho_evidenciado", "income_confianca")`
     * `spending_annually_in_british_pounds` -> `("spending_pagina", "spending_trecho_evidenciado", "spending_confianca")`
   * *Regra:*
     * Se `not result.value_found` ou `result.extracted_value is None`, **não** executar `UPDATE` na coluna financeira (mantendo `NULL` no banco), apenas registrar log e retornar `False`.
     * Se `result.value_found` for `True` e houver `extracted_value`, determinar a página final (`result.source_page` se presente, caso contrário `evidences[0].page_number` como fallback) e executar `UPDATE dw_charity_data` na linha `WHERE id = task.row_id` usando `psycopg2.sql.Identifier`, atualizando:
       * A coluna dinâmica `task.target_column = result.extracted_value`
       * As colunas de linhagem específicas da métrica (`income_pagina`/`spending_pagina`, `income_trecho_evidenciado`/`spending_trecho_evidenciado`, `income_confianca`/`spending_confianca`)
       * As colunas gerais de auditoria para retrocompatibilidade:
         * `imputado_por_ia = TRUE`
         * `fonte_documento = task.document_id`
         * `pagina = pagina_determinada`
         * `trecho_evidenciado = result.exact_quote`
         * `confianca = result.confidence_score`
         * `data_imputacao = CURRENT_TIMESTAMP`
     * Retornar `True` quando o registro for atualizado com sucesso.

4. **Orquestração do Pipeline de Imputação:**
   `def run_imputation_pipeline(db_connection_string: str, top_k: int = 12) -> dict[str, int]:`
   * *Regra:* Integrar com `src/retrieval` (`get_pending_tasks`, `embed_search_query`, `fetch_evidence` com `top_k=12`). Para cada task pendente, buscar evidências; se houver evidências, construir o prompt, extrair via LLM e atualizar o DW. Retornar dicionário resumo com `{"total_tasks": int, "imputed_count": int, "skipped_count": int}`.

## 5. Script Executável (`scripts/02_run_imputation.py`)
Deve carregar variáveis do `.env`, validar a presença de `DEEPSEEK_API_KEY` e `EMBEDDING_API_KEY`, montar a `db_connection_string` e executar `run_imputation_pipeline` (com `top_k=12` por padrão), exibindo o resumo final no log.

## 6. Critérios de Aceite (Definition of Done)
1. **Tipagem e Pacotes:** 100% das funções tipadas (PEP 484) e `src/imputation/__init__.py` exportando contratos e funções principais.
2. **Sanitização Numérica:** Strings monetárias como `"£ 12,345.67"` ou `"12,000"` são convertidas automaticamente para `float` pelo modelo Pydantic.
3. **Proteção contra SQL Injection:** Uso obrigatório de `psycopg2.sql.Identifier` e validação de `target_column`.
4. **Testes Unitários:** Arquivo `tests/test_imputation.py` usando `pytest` e `pytest-mock` cobrindo: sanitização de valores no Pydantic, construção do prompt (incluindo verificação das 3 regras contábeis), extração via LLM com JSON válido e com JSON corrompido (fallback defensivo), escrita no banco quando `value_found=True` e não-escrita quando `value_found=False`, e orquestração completa mockada.
