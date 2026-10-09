# Especificação de Desenvolvimento: Módulo de Avaliação (Evaluation)

## 1. Objetivo do Módulo
O módulo `src/evaluation/` atua como o "juiz" do framework. Ele é responsável por extrair os dados finais imputados pela IA no banco de dados (`dw_charity_data`), cruzar essas informações com a planilha original gabarito (*Ground Truth*) usando o `document_id` como chave, e calcular as métricas de performance científica da extração (Acurácia, Taxa de Preenchimento e Validade da Linhagem).

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `pandas` (para cruzamento e vetorização de cálculos), `psycopg2` (para leitura do DW), `pydantic`.
* **Proibição de LLM:** Este módulo DEVE ser estritamente determinístico. Nenhuma chamada à API da LLM deve ser feita aqui.
* **Saída (Output):** Os resultados devem gerar artefatos físicos (CSV com o detalhamento linha a linha e JSON com o resumo das métricas) salvos obrigatoriamente na pasta `reports/`.

## 3. Modelos de Dados (Data Contracts)
A IA deve implementar os modelos de métricas em `src/evaluation/models.py`:

```python
from pydantic import BaseModel

class ColumnMetrics(BaseModel):
    column_name: str
    total_expected: int        # Quantos valores não-nulos existiam no gabarito
    total_imputed: int         # Quantos valores a IA tentou preencher
    correct_matches: int       # Quantos acertos (dentro da margem de tolerância)
    accuracy_rate: float       # (correct_matches / total_expected)
    fill_rate: float           # (total_imputed / total_expected)

class EvaluationReport(BaseModel):
    total_documents_processed: int
    columns_evaluated: list[ColumnMetrics]
    lineage_completeness: float # % de registros imputados que possuem trecho_evidenciado e pagina preenchidos
```

## 4. Assinaturas de Funções Exigidas
A IA deve implementar o pipeline de avaliação em `src/evaluation/metrics.py` e `src/evaluation/report.py`:

1. **Carregamento de Dados:**
   `def load_ground_truth(file_path: str) -> pd.DataFrame:`
   * *Regra:* Ler o arquivo Excel/CSV do gabarito original, normalizando a coluna `document_id`.
   `def load_imputed_data(db_connection_string: str) -> pd.DataFrame:`
   * *Regra:* Executar um `SELECT * FROM dw_charity_data` e retornar o DataFrame.

2. **Comparador Financeiro (Tolerância Numérica):**
   `def is_match_financial(imputed_val: float, truth_val: float, tolerance: float = 0.05) -> bool:`
   * *Regra:* Como lidamos com libras esterlinas extraídas de texto, a comparação deve aceitar um delta mínimo (ex: tolerância de 5 centavos) para evitar que arredondamentos simples sejam marcados como erro de extração pela LLM. Se algum valor for nulo, retorna False.

3. **Cálculo de Métricas:**
   `def calculate_metrics(df_truth: pd.DataFrame, df_imputed: pd.DataFrame, target_columns: list[str]) -> EvaluationReport:`
   * *Regra:* Fazer um `pd.merge` usando `document_id`. Iterar pelas colunas alvo (ex: `income_annually_in_british_pounds`), aplicar a função `is_match_financial` e popular o modelo `EvaluationReport`. Validar se as colunas de linhagem (ex: `trecho_evidenciado`) não estão nulas quando houve imputação.

4. **Geração de Artefatos:**
   `def export_reports(report: EvaluationReport, df_merged: pd.DataFrame, output_dir: str = "reports/") -> None:`
   * *Regra:* Salvar o `report` como `summary_metrics.json` e o DataFrame cruzado (com a flag de acerto/erro linha a linha) como `detailed_evaluation.csv`.

## 5. Critérios de Aceite (Definition of Done)
1. **Resiliência a Nulos:** O código Pandas deve tratar corretamente os casos onde a IA retornou NaN/NULL e o gabarito esperava um valor, ou vice-versa.
2. **Reprodutibilidade:** Rodar o script de avaliação múltiplas vezes sobre o mesmo banco de dados deve produzir o arquivo JSON com os mesmos números exatos.
3. **Testes Unitários:** O arquivo `tests/test_evaluation.py` deve testar a lógica do comparador financeiro (`is_match_financial`) passando cenários como `(15000.0, 15000.0)`, `(15000.0, 15000.01)`, e `(NULL, 15000.0)`.