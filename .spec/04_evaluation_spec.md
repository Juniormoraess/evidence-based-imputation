# Especificação de Desenvolvimento: Módulo de Avaliação (Evaluation)

## 1. Objetivo do Módulo
O módulo `src/evaluation/` atua como o juiz determinístico do framework. Ele extrai os dados imputados pela IA no banco de dados (`dw_charity_data`), cruza essas informações com a planilha gabarito (*Ground Truth* em `data/ground_truth/expected_with_doc_id.xlsx`) usando `document_id` como chave, calcula métricas científicas de qualidade de imputação e completude de linhagem, e exporta os relatórios finais na pasta `reports/`.

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `pandas` e `openpyxl` (leitura de Excel/CSV e cruzamento tabular), `psycopg2` (leitura do DW), `pydantic`, `python-dotenv`.
* **Proibição de LLM:** Este módulo DEVE ser 100% determinístico. Nenhuma chamada de rede para APIs de IA deve ser realizada.
* **Saída (Output):** Os resultados devem gerar dois artefatos físicos salvos em `reports/`: `summary_metrics.json` (métricas consolidadas) e `detailed_evaluation.csv` (comparação linha a linha).

## 3. Modelos de Dados (Data Contracts)
A IA deve implementar os seguintes modelos em `src/evaluation/models.py`:

```python
from pydantic import BaseModel

class ColumnMetrics(BaseModel):
    column_name: str
    total_expected: int        # Total de registros com valor não-nulo no Ground Truth
    total_imputed: int         # Total de registros(dentre os esperados) preenchidos pela IA
    correct_matches: int       # Total de acertos dentro da margem de tolerância financeira
    accuracy_rate: float       # correct_matches / total_expected (0.0 se total_expected == 0)
    precision_rate: float      # correct_matches / total_imputed (0.0 se total_imputed == 0)
    fill_rate: float           # total_imputed / total_expected (0.0 se total_expected == 0)

class EvaluationReport(BaseModel):
    total_documents_evaluated: int
    columns_evaluated: list[ColumnMetrics]
    lineage_completeness: float # % de registros imputados (imputado_por_ia=True) que possuem fonte_documento, pagina e trecho_evidenciado não-nulos e não-vazios
```

## 4. Assinaturas de Funções Exigidas
A IA deve implementar o pipeline em `src/evaluation/` (ex: `models.py`, `metrics.py`, `report.py`):

1. **Carregamento de Dados:**
   `def load_ground_truth(file_path: str) -> pd.DataFrame:`
   * *Regra:* Suportar leitura de `.xlsx` (`pd.read_excel`) e `.csv` (`pd.read_csv`). Normalizar a coluna `document_id` (strip de espaços e garantir sufixo `.pdf` caso ausente).
   `def load_imputed_data(db_connection_string: str) -> pd.DataFrame:`
   * *Regra:* Executar `SELECT * FROM dw_charity_data` via `psycopg2` e retornar como `pd.DataFrame`.

2. **Comparador Financeiro (Tolerância Numérica):**
   `def is_match_financial(imputed_val: float | None, truth_val: float | None, tolerance: float = 0.05) -> bool:`
   * *Regra:* Verificar se qualquer um dos valores é `None` ou `NaN` (`pd.isna`); se for, retornar `False`. Caso contrário, converter para `float` e retornar `abs(float(imputed_val) - float(truth_val)) <= tolerance`.

3. **Cálculo de Métricas e Cruzamento:**
   `def calculate_metrics(df_truth: pd.DataFrame, df_imputed: pd.DataFrame, target_columns: list[str] | None = None) -> tuple[EvaluationReport, pd.DataFrame]:`
   * *Regra:* Por padrão, `target_columns` deve ser `["income_annually_in_british_pounds", "spending_annually_in_british_pounds"]`.
   * Realizar `pd.merge` entre `df_truth` (sufixo `_truth`) e `df_imputed` (sufixo `_imputed`) pela chave `document_id`.
   * Para cada coluna alvo, criar no `df_merged` uma coluna booleana `<col>_is_match` aplicando `is_match_financial` e calcular `ColumnMetrics` arredondando taxas para 4 casas decimais.
   * Calcular `lineage_completeness` verificando, dentre as linhas onde `imputado_por_ia == True`:
     * Se as colunas de linhagem por métrica (`income_pagina`, `income_trecho_evidenciado`, `spending_pagina`, `spending_trecho_evidenciado`) estiverem presentes no DataFrame, avaliar individualmente cada métrica financeira não-nula imputada na linha (exigindo `fonte_documento`, `<metrica>_pagina` e `<metrica>_trecho_evidenciado` válidos).
     * Caso contrário (fallback de retrocompatibilidade para DataFrames legados), verificar qual proporção das linhas com `imputado_por_ia == True` possui `fonte_documento`, `pagina` e `trecho_evidenciado` válidos.
     * Se nenhuma linha tiver `imputado_por_ia == True` (ou nenhum valor imputado), retornar `0.0`.
   * Retornar a tupla `(EvaluationReport, df_merged)`.

4. **Geração de Artefatos:**
   `def export_reports(report: EvaluationReport, df_merged: pd.DataFrame, output_dir: str = "reports") -> tuple[str, str]:`
   * *Regra:* Garantir que o diretório `output_dir` exista (`os.makedirs(..., exist_ok=True)`). Salvar `summary_metrics.json` (usando `model_dump_json(indent=2)`) e `detailed_evaluation.csv` (`index=False`). Retornar os caminhos dos dois arquivos gerados.

## 5. Script Executável (`scripts/03_run_evaluation.py`)
Deve carregar variáveis do `.env`, ler o caminho do gabarito via variável `GROUND_TRUTH_PATH` (default: `data/ground_truth/expected_with_doc_id.xlsx`), carregar os dados do PostgreSQL, executar `calculate_metrics` e `export_reports`, e exibir o JSON consolidado no log.

## 6. Critérios de Aceite (Definition of Done)
1. **Tipagem e Pacotes:** 100% das funções com type hints (PEP 484) e `src/evaluation/__init__.py` exportando contratos e funções principais.
2. **Resiliência a Nulos e Divisão por Zero:** Tratamento seguro de `None`/`np.nan` e proteção contra `ZeroDivisionError` quando `total_expected == 0` ou `total_imputed == 0`.
3. **Testes Unitários:** Arquivo `tests/test_evaluation.py` usando `pytest` e `pytest-mock` cobrindo: `is_match_financial` (exato, dentro da tolerância, fora da tolerância, `None` e `NaN`), `calculate_metrics` com DataFrames em memória (incluindo validação de `lineage_completeness`), carregamento mockado e exportação em diretório temporário (`tmp_path`).