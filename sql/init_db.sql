-- ===========================================================================
-- Script de Inicialização do Banco de Dados (PostgreSQL + pgvector)
-- Projeto: Evidence-Based Imputation (Raw-RAG)
-- Dataset: UK Charities Financial Reports
-- ===========================================================================

-- 1. Habilitar a extensão pgvector (necessita de privilégio de superusuário)
CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- TABELA 1: BASE DE CONHECIMENTO (KNOWLEDGE BASE)
-- Armazena os fragmentos textuais dos relatórios em PDF e seus vetores.
-- ===========================================================================
CREATE TABLE IF NOT EXISTS knowledge_base (
    id SERIAL PRIMARY KEY,
    -- O 'document_id' atua como Foreign Key lógica para a tabela do DW
    document_id VARCHAR(255) NOT NULL, 
    page_number INT NOT NULL,
    chunk_index INT NOT NULL,
    text_content TEXT NOT NULL,
    
    -- Nota: 1536 é o tamanho gerado pela OpenAI (text-embedding-ada-002 ou v3).
    -- Se for utilizar um modelo local mais leve no futuro (ex: all-MiniLM-L6-v2), altere para 384.
    embedding vector(1536), 
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índice de Filtro Exato (Rota A - Hard Filtering):
-- Acelera a consulta 'WHERE document_id = XYZ' antes de calcular a similaridade vetorial.
CREATE INDEX IF NOT EXISTS idx_kb_document_id ON knowledge_base (document_id);

-- Índice HNSW para Busca Vetorial Semântica (Rota B):
-- Garante performance na busca por similaridade semântica (cosseno).
CREATE INDEX IF NOT EXISTS idx_kb_embedding ON knowledge_base USING hnsw (embedding vector_cosine_ops);


-- ===========================================================================
-- TABELA 2: DATA WAREHOUSE (TABELA ALVO)
-- Tabela contendo os registros do dataset com as lacunas a serem preenchidas.
-- ===========================================================================
CREATE TABLE IF NOT EXISTS dw_charity_data (
    id SERIAL PRIMARY KEY,
    
    -- Chave de vínculo com os arquivos PDF na Knowledge Base
    document_id VARCHAR(255) NOT NULL UNIQUE,
    
    -- Metadados de contexto (podem ser usados para enriquecer o prompt da LLM)
    charity_name VARCHAR(255),
    charity_number NUMERIC,
    address__post_town VARCHAR(255),
    address__postcode VARCHAR(50),
    address__street_line TEXT,
    report_date DATE,
    
    -- ==========================================
    -- COLUNAS ALVO (Onde os NULLs existirão)
    -- ==========================================
    income_annually_in_british_pounds NUMERIC(15, 2),
    spending_annually_in_british_pounds NUMERIC(15, 2),

    -- ==========================================
    -- COLUNAS DE LINHAGEM E AUDITORIA
    -- ==========================================
    imputado_por_ia BOOLEAN DEFAULT FALSE,
    fonte_documento VARCHAR(255),       -- Confirma o documento lido (Redundância de segurança)
    pagina INT,                         -- A página exata onde o valor financeiro foi encontrado
    trecho_evidenciado TEXT,            -- O texto extraído do PDF que prova o valor numérico
    confianca FLOAT,                    -- Grau de certeza retornado pela estruturação da IA
    data_imputacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índice auxiliar para varredura de lacunas no DW
CREATE INDEX IF NOT EXISTS idx_dw_missing_data 
ON dw_charity_data (document_id) 
WHERE income_annually_in_british_pounds IS NULL OR spending_annually_in_british_pounds IS NULL;