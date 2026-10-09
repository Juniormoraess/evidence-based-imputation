# Especificação de Desenvolvimento: Módulo de Recuperação (Retrieval)

## 1. Objetivo do Módulo
O módulo `src/retrieval/` é responsável por orquestrar a busca das evidências. Ele deve varrer o Data Warehouse (tabela `dw_charity_data`) em busca de registros com lacunas (NULL) nas colunas financeiras e, para cada lacuna, executar uma busca híbrida no PostgreSQL (`knowledge_base`) para resgatar os fragmentos de texto mais relevantes do documento correspondente.

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `psycopg2` (conexão e consultas SQL brutas), `openai` (geração de embedding da query), `pydantic`. NENHUM ORM complexo (como SQLAlchemy) ou framework de RAG deve ser usado para a busca vetorial.
* **Mecanismo de Busca:** Obrigatório o uso do operador de distância por cosseno (`<=>`) do `pgvector`.
* **Segurança de Dados (Hard Filtering):** A busca vetorial DEVE ser obrigatoriamente restrita ao `document_id` da linha alvo. A IA nunca deve cruzar informações de relatórios de instituições diferentes.

## 3. Modelos de Dados (Data Contracts)
A IA deve criar os seguintes modelos em `src/retrieval/models.py`:

```python
from pydantic import BaseModel
from typing import List, Optional

class ImputationTask(BaseModel):
    row_id: int
    document_id: str
    charity_name: Optional[str]
    target_column: str  # Qual coluna financeira precisa ser preenchida
    
class RetrievedEvidence(BaseModel):
    page_number: int
    chunk_index: int
    text_content: str
    similarity_score: float
```

## 4. Assinaturas de Funções Exigidas
A IA deve implementar o fluxo principal em `src/retrieval/search.py`, respeitando as seguintes assinaturas:

1. **Mapeamento de Lacunas:**
   `def get_pending_tasks(db_connection_string: str) -> List[ImputationTask]:`
   * *Regra:* Fazer um `SELECT` em `dw_charity_data` buscando linhas onde `income_annually_in_british_pounds IS NULL` ou `spending_annually_in_british_pounds IS NULL`. Cada coluna nula de uma mesma linha gera uma `ImputationTask` separada.

2. **Geração da Query Semântica:**
   `def embed_search_query(task: ImputationTask) -> List[float]:`
   * *Regra:* Construir uma string de busca baseada na tarefa (ex: *"Financial report income annually in british pounds for charity {task.charity_name}"*) e gerar o embedding usando a API da OpenAI.

3. **Busca Híbrida (SQL Bruto):**
   `def fetch_evidence(task: ImputationTask, query_embedding: List[float], db_connection_string: str, top_k: int = 3) -> List[RetrievedEvidence]:`
   * *Regra:* Executar a consulta no PostgreSQL filtrando por `document_id` e ordenando pela similaridade do vetor. Exemplo de cláusula: `WHERE document_id = %s ORDER BY embedding <=> %s LIMIT %s`. O score de similaridade deve ser retornado (`1 - (embedding <=> %s)`).

## 5. Critérios de Aceite (Definition of Done)
1. **Tipagem:** 100% das funções tipadas e validadas via Pydantic.
2. **Eficiência de Conexão:** As funções de banco de dados devem usar cursores corretamente (`with psycopg2.connect(...)`) garantindo o fechamento da conexão.
3. **Prevenção de Alucinação:** Se a query retornar zero chunks para um `document_id`, o sistema deve retornar uma lista vazia, e não buscar em outros documentos.
4. **Testes Unitários:** Testes devem estar em `tests/test_retrieval.py`, fazendo o mock do banco de dados (sem exigir um banco real rodando durante o pytest) e o mock da API de embeddings.