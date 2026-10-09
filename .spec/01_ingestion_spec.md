# Especificação de Desenvolvimento: Módulo de Ingestão (Raw-RAG)

## 1. Objetivo do Módulo
O módulo `src/ingestion/` é responsável por processar documentos não estruturados (PDFs), extrair seu conteúdo textual, aplicar a estratégia de *chunking* (fragmentação) com sobreposição, calcular os vetores semânticos (embeddings) e persistir o conhecimento no banco de dados PostgreSQL usando a extensão `pgvector`.

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `PyMuPDF` (leitura de PDF), `openai` (cliente para geração de embeddings), `psycopg2` (conexão com BD), `pydantic` (modelagem de dados). NENHUM framework de abstração de RAG (como LangChain ou LlamaIndex) deve ser utilizado.
* **Paradigma:** Código modular, funcional e tipado (PEP 484).
* **Tratamento de Erros:** O script não deve falhar silenciosamente. Arquivos corrompidos devem ser logados e "pulados" (skipped), permitindo que o pipeline continue para os demais documentos.

## 3. Modelos de Dados (Data Contracts)
A IA deve implementar os seguintes modelos Pydantic no arquivo `src/ingestion/models.py` para garantir a integridade dos dados trafegados:

```python
from pydantic import BaseModel
from typing import List

class DocumentChunk(BaseModel):
    document_name: str
    page_number: int
    chunk_index: int
    text_content: str
    embedding: List[float] | None = None
```

## 4. Assinaturas de Funções Exigidas
A IA deve implementar o fluxo principal em `src/ingestion/pipeline.py` (ou separando em arquivos lógicos como `parser.py`, `chunker.py`, `embedder.py`), respeitando estritamente as seguintes assinaturas:

1. **Extração de Texto:**
   `def extract_text_from_pdf(file_path: str) -> List[dict]:`
   * *Entrada:* Caminho do arquivo PDF (`data/raw/`).
   * *Saída:* Lista de dicionários no formato `{"page_number": int, "text": str}`.

2. **Chunking (Fragmentação):**
   `def create_chunks(pages_data: List[dict], chunk_size: int = 1000, overlap: int = 200) -> List[DocumentChunk]:`
   * *Regra:* Implementar janela deslizante (sliding window) por caracteres para preservar contexto entre fragmentos. Manter a rastreabilidade da página de origem.

3. **Geração de Embeddings:**
   `def generate_embeddings(chunks: List[DocumentChunk]) -> List[DocumentChunk]:`
   * *Regra:* Utilizar a API client da OpenAI (apontando para o endpoint configurado no `.env`). Enviar requisições em lotes (batch) se possível, para economizar tempo e chamadas de rede.

4. **Persistência (Escrita no Banco):**
   `def save_chunks_to_db(chunks: List[DocumentChunk], db_connection_string: str) -> None:`
   * *Regra:* Usar instrução `INSERT` no PostgreSQL (tabela `knowledge_base`). O vetor `embedding` deve ser inserido no formato compatível com o tipo `vector` do `pgvector`.

## 5. Esquema de Banco de Dados Alvo
O código Python deve assumir que a seguinte estrutura já existe no banco de dados (que será criada via `sql/init_db.sql`):
* Tabela: `knowledge_base`
* Colunas: `id` (UUID/Serial), `document_name` (VARCHAR), `page_number` (INT), `chunk_index` (INT), `text_content` (TEXT), `embedding` (VECTOR).

## 6. Critérios de Aceite (Definition of Done)
1. **Tipagem:** 100% das funções possuem type hints e validação Pydantic.
2. **Resiliência:** O processamento não é interrompido se um arquivo na pasta `data/raw/` não for um PDF válido ou estiver corrompido; o erro é apenas registrado (logging).
3. **Idempotência:** A ingestão deve evitar duplicação de dados. Se o mesmo `document_name` for processado novamente, os chunks antigos desse documento devem ser deletados e substituídos, ou a ingestão deve ser ignorada.
4. **Testabilidade:** Devem ser gerados testes automatizados em `tests/test_ingestion.py` usando `pytest`. As chamadas à API de Embeddings e ao banco de dados DEVEM ser simuladas (mockadas) com `pytest-mock`.