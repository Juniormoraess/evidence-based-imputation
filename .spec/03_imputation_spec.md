# Especificação de Desenvolvimento: Módulo de Imputação

## 1. Objetivo do Módulo
O módulo `src/imputation/` é o núcleo de decisão do framework. Ele recebe a tarefa pendente e as evidências recuperadas (do módulo de *Retrieval*), constrói o prompt de contexto, invoca a API da LLM (DeepSeek V4), estrutura a resposta financeira e executa a instrução `UPDATE` no Data Warehouse (`dw_charity_data`), preenchendo o valor nulo e todas as colunas de auditoria/linhagem.

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `openai` (cliente para conectar na API do DeepSeek), `pydantic` (validação rigorosa de saída), `psycopg2` (escrita no banco).
* **Engenharia de Prompt:** O prompt DEVE ser instruído a atuar como um extrator de dados determinístico. É estritamente proibido "adivinhar" ou "calcular" valores que não estejam explicitamente no texto.
* **Segurança SQL:** A atualização dinâmica da coluna alvo deve ser feita usando o módulo `psycopg2.sql` para evitar SQL Injection.

## 3. Modelos de Dados (Data Contracts)
A IA deve criar o seguinte contrato Pydantic em `src/imputation/models.py`. Este modelo define a estrutura exata do JSON que a LLM deve retornar:

```python
from pydantic import BaseModel, Field
from typing import Optional

class ExtractionResult(BaseModel):
    value_found: bool = Field(description="Booleano indicando se o valor financeiro foi localizado nas evidências.")
    extracted_value: Optional[float] = Field(description="O valor numérico absoluto extraído (ex: 15500.50). Nulo se não encontrado.")
    exact_quote: Optional[str] = Field(description="A citação literal e exata do texto que comprova o valor extraído.")
    reasoning: str = Field(description="Breve explicação de por que este valor corresponde à métrica solicitada.")
    confidence_score: float = Field(description="Grau de confiança na extração, de 0.0 a 1.0.")
```

## 4. Assinaturas de Funções Exigidas
A IA deve implementar as funções lógicas em `src/imputation/llm_client.py` e `src/imputation/writer.py`:

1. **Construção do Prompt:**
   `def build_extraction_prompt(target_column: str, charity_name: str, evidences: List[RetrievedEvidence]) -> str:`
   * *Regra:* Deve formatar uma string injetando a definição da coluna desejada e concatenando o texto dos chunks recuperados, referenciando o número da página de cada chunk.

2. **Chamada da LLM:**
   `def extract_value_with_llm(prompt: str) -> ExtractionResult:`
   * *Regra:* Usar o cliente OpenAI (apontando para a `DEEPSEEK_API_KEY`). Configurar o parâmetro para forçar saída em JSON. Passar o retorno textual da LLM pelo modelo `ExtractionResult.model_validate_json()` do Pydantic.
   * *Tratamento de Erro:* Se a LLM alucinar a estrutura JSON, capturar a exceção `ValidationError` e retornar um `ExtractionResult` com `value_found=False`.

3. **Escrita no Data Warehouse (Auditabilidade):**
   `def update_dw_record(task: ImputationTask, result: ExtractionResult, best_evidence: RetrievedEvidence, db_connection_string: str) -> None:`
   * *Regra:* Deve executar um `UPDATE` na tabela `dw_charity_data` na linha correspondente (`WHERE id = task.row_id`).
   * *Linhagem:* Se `result.value_found` for True, atualizar a coluna alvo (ex: `income_annually_in_british_pounds`) com `result.extracted_value` e as colunas:
     * `imputado_por_ia = TRUE`
     * `fonte_documento = task.document_id`
     * `pagina = best_evidence.page_number`
     * `trecho_evidenciado = result.exact_quote`
     * `confianca = result.confidence_score`

## 5. Critérios de Aceite (Definition of Done)
1. **Sem Suposições:** Se a LLM retornar `value_found=False`, o sistema não deve atualizar o valor numérico, devendo apenas fazer um log ou marcar a confiança como 0.0, mantendo o valor como NULL no banco para análise humana posterior.
2. **Robustez de Tipagem:** Valores como "15,000" ou "£15000" devem ser convertidos (pela LLM ou pelo Pydantic) para um float numérico válido (`15000.00`) para não quebrar o banco de dados.
3. **Mock em Testes:** Os testes unitários (`tests/test_imputation.py`) não devem gastar créditos da API. A chamada do cliente OpenAI deve ser interceptada com `pytest-mock`, retornando um JSON simulado.