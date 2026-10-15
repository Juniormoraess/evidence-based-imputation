# Especificação de Desenvolvimento: Módulo de Ingestão (Raw-RAG)

## 1. Objetivo do Módulo
O módulo `src/ingestion/` é responsável por processar documentos não estruturados (PDFs) da pasta `data/raw/`, extrair seu conteúdo textual, aplicar a estratégia de *chunking* (fragmentação) com sobreposição, calcular os vetores semânticos (embeddings) e persistir o conhecimento no banco de dados PostgreSQL (`knowledge_base`) usando a extensão `pgvector`.

## 2. Escopo e Restrições (Inegociáveis)
* **Bibliotecas Permitidas:** `PyMuPDF` (`fitz` para leitura de PDF), `openai` (cliente para geração de embeddings), `psycopg2` e `pgvector` (conexão e escrita vetorial no BD), `pydantic` (modelagem de dados), `python-dotenv`. NENHUM framework de abstração de RAG (como LangChain ou LlamaIndex) deve ser utilizado.
* **Paradigma:** Código modular, funcional e estritamente tipado (PEP 484).
* **Configuração de Embedding:** Utilizar o cliente `openai` lendo `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL` (opcional) e `EMBEDDING_MODEL` (default: `text-embedding-3-small`, 1536 dimensões) do arquivo `.env`.
* **Tratamento de Erros:** O script não deve falhar silenciosamente. Arquivos corrompidos devem ser logados (módulo `logging`) e "pulados" (skipped), permitindo que o pipeline continue para os demais documentos.

## 3. Modelos de Dados (Data Contracts)
A IA deve implementar o seguinte modelo Pydantic no arquivo `src/ingestion/models.py`:

```python
from pydantic import BaseModel
from typing import List, Optional

class DocumentChunk(BaseModel):
    document_id: str  # Nome exato do arquivo PDF (ex: "1ada336f29e8247f9f55a8d7e1b1c0da.pdf")
    page_number: int
    chunk_index: int
    text_content: str
    embedding: Optional[List[float]] = None
```

## 4. Assinaturas de Funções Exigidas
A IA deve implementar o fluxo principal em `src/ingestion/` (podendo organizar em `models.py`, `parser.py`, `chunker.py`, `embedder.py`, `repository.py` ou `pipeline.py`), respeitando estritamente as seguintes assinaturas:

1. **Extração de Texto:**
   `def extract_text_from_pdf(file_path: str) -> List[dict[str, int | str]]:`
   * *Entrada:* Caminho do arquivo PDF (`data/raw/`).
   * *Saída:* Lista de dicionários no formato `{"page_number": int, "text": str}` (páginas 1-indexed).

2. **Chunking (Fragmentação):**
   `def create_chunks(document_id: str, pages_data: List[dict[str, int | str]], chunk_size: int = 1000, overlap: int = 200) -> List[DocumentChunk]:`
   * *Regra:* Implementar janela deslizante (sliding window) por caracteres para preservar contexto entre fragmentos. Manter a rastreabilidade de `document_id` e `page_number`. Ignorar páginas em branco.

3. **Geração de Embeddings:**
   `def generate_embeddings(chunks: List[DocumentChunk]) -> List[DocumentChunk]:`
   * *Regra:* Utilizar o client da OpenAI em lotes (batch) para preencher o atributo `embedding` de cada `DocumentChunk`.

4. **Persistência (Escrita no Banco):**
   `def save_chunks_to_db(chunks: List[DocumentChunk], db_connection_string: str) -> None:`
   * *Regra:* Garantir idempotência executando `DELETE FROM knowledge_base WHERE document_id = %s` antes de inserir os novos chunks do mesmo documento dentro da mesma transação. Inserir na tabela `knowledge_base` (`document_id`, `page_number`, `chunk_index`, `text_content`, `embedding`).

## 5. Script Executável (`scripts/01_run_ingestion.py`)
Deve carregar variáveis do `.env`, listar todos os arquivos `.pdf` em `data/raw/`, executar o pipeline de ingestão documento a documento com logs de progresso e tratar falhas individualmente por arquivo.

## 6. Critérios de Aceite (Definition of Done)
1. **Tipagem e Pacotes:** 100% das funções possuem type hints, validação Pydantic e arquivos `__init__.py` criados para resolução limpa de imports.
2. **Resiliência:** Arquivos PDF inválidos ou vazios na pasta `data/raw/` são logados e não derrubam o lote.
3. **Idempotência:** Reprocessar o mesmo `document_id` substitui os registros anteriores sem duplicar chunks no banco.
4. **Testabilidade:** Testes automatizados em `tests/test_ingestion.py` usando `pytest` e `pytest-mock`, cobrindo extração, chunking, geração de embeddings (mockada) e persistência (mockada), passando 100% ao rodar `pytest`.