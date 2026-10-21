# Especificação de Desenvolvimento: Módulo de Recuperação (Retrieval)

## 1. Objetivo do Módulo
O módulo `src/retrieval/` é responsável por orquestrar a busca das evidências. Ele deve varrer o Data Warehouse (tabela `dw_charity_data`) em busca de registros com lacunas (`NULL`) nas colunas financeiras alvo e, para cada lacuna, executar uma busca híbrida no PostgreSQL (`knowledge_base`) para resgatar os fragmentos de texto mais relevantes restritos ao documento correspondente.

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `psycopg2` e `pgvector` (conexão e consultas SQL brutas), `openai` (geração de embedding da query), `pydantic`, `python-dotenv`. NENHUM ORM (como SQLAlchemy) ou framework de RAG deve ser utilizado.
* **Configuração de Embedding:** Utilizar o cliente `openai` lendo `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL` (opcional) e `EMBEDDING_MODEL` (default: `text-embedding-3-small`, 1536 dimensões) do arquivo `.env`.
* **Mecanismo de Busca:** Obrigatório o uso do operador de distância por cosseno (`<=>`) do `pgvector`, retornando o `similarity_score` calculado como `1 - (embedding <=> query_vector)`.
* **Segurança de Dados (Hard Filtering):** A busca vetorial DEVE ser obrigatoriamente restrita ao `document_id` da linha alvo (`WHERE document_id = %s`). A IA nunca deve cruzar informações de relatórios de instituições diferentes.

## 3. Modelos de Dados (Data Contracts)
A IA deve criar os seguintes modelos em `src/retrieval/models.py`:

```python
from pydantic import BaseModel
from typing import Optional

class ImputationTask(BaseModel):
    row_id: int
    document_id: str
    charity_name: Optional[str] = None
    report_date: Optional[str] = None
    target_column: str  # 'income_annually_in_british_pounds' ou 'spending_annually_in_british_pounds'

class RetrievedEvidence(BaseModel):
    document_id: str
    page_number: int
    chunk_index: int
    text_content: str
    similarity_score: float
```

## 4. Assinaturas de Funções Exigidas
A IA deve implementar o fluxo principal em `src/retrieval/` (ex: `models.py`, `search.py`), respeitando estritamente as seguintes assinaturas:

1. **Mapeamento de Lacunas:**
   `def get_pending_tasks(db_connection_string: str) -> list[ImputationTask]:`
   * *Regra:* Fazer um `SELECT id, document_id, charity_name, report_date, income_annually_in_british_pounds, spending_annually_in_british_pounds FROM dw_charity_data WHERE income_annually_in_british_pounds IS NULL OR spending_annually_in_british_pounds IS NULL`.
   * *Desdobramento:* Se uma mesma linha tiver ambas as colunas nulas, ela deve gerar **duas** instâncias de `ImputationTask` (uma para `income_annually_in_british_pounds` e outra para `spending_annually_in_british_pounds`). Converter `report_date` para string ISO (`YYYY-MM-DD`) quando presente.

2. **Geração da Query Semântica:**
   `def embed_search_query(task: ImputationTask) -> list[float]:`
   * *Regra:* Construir uma string de busca semântica rica combinando `task.target_column`, `task.charity_name`, `task.report_date` e vocabulário contábil britânico explícito:
     * Para `income_annually_in_british_pounds`: incluir `"Statement of Financial Activities SOFA Profit and Loss Receipts and Payments Total Income Total incoming resources Total gross income unrestricted restricted funds"`.
     * Para `spending_annually_in_british_pounds`: incluir `"Statement of Financial Activities SOFA Profit and Loss Receipts and Payments Total Expenditure Total Expense Total resources expended charitable activities"`.
   * Gerar o vetor de 1536 dimensões via API da OpenAI.

3. **Busca Híbrida com Expansão de Página Completa / Full-Page Context Expansion (SQL Bruto):**
   `def fetch_evidence(task: ImputationTask, query_embedding: list[float], db_connection_string: str, top_k: int = 12) -> list[RetrievedEvidence]:`
   * *Regra:* Executar a consulta no PostgreSQL filtrando por `document_id` e selecionando os `top_k` (default: `12`) chunks mais similares pela distância de cosseno (`embedding <=> %s::vector`).
   * *Expansão de Página Completa (Full-Page Context Expansion):* Para todas as páginas (`page_number`) presentes nos `top_k` fragmentos recuperados pela busca vetorial, buscar **todos** os fragmentos pertencentes àquelas mesmas páginas dentro do mesmo documento (tanto `chunk_index` anteriores quanto posteriores), garantindo que cabeçalhos de escala (ex: `£000` no início da página) e rodapés de totais consolidados (ex: `Total expenditure` no final da página) nunca sejam truncados.
   * *Ordenação:* Retornar a lista final ordenada por `(page_number, chunk_index)`.

## 5. Critérios de Aceite (Definition of Done)
1. **Tipagem e Pacotes:** 100% das funções tipadas (PEP 484), validadas via Pydantic e com `src/retrieval/__init__.py` exportando os contratos e funções principais.
2. **Eficiência de Conexão:** Gerenciamento seguro de contexto (`with psycopg2.connect(...)`) garantindo fechamento de conexão e cursor.
3. **Prevenção de Alucinação:** Se a query retornar zero chunks para um `document_id`, retornar lista vazia `[]` sem remover o filtro `WHERE document_id = %s`.
4. **Testes Unitários:** Arquivo `tests/test_retrieval.py` usando `pytest` e `pytest-mock` cobrindo: (a) geração de 1 ou 2 tasks por linha com NULLs, (b) formatação com vocabulário contábil britânico e geração de embedding da query mockada, (c) execução da busca híbrida verificando `document_id`, `top_k=12` e ordenação por `(page_number, chunk_index)`, e (d) retorno vazio quando o documento não possui chunks.
