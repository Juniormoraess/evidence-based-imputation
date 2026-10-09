# Constituição do Projeto: Evidence-Based Imputation

## 1. Visão Geral
Este projeto é uma Prova de Conceito (PoC) de um framework de Engenharia de Dados focado em Imputação Baseada em Evidências. O objetivo é preencher valores nulos (NULL) em tabelas estruturadas buscando fatos em documentos não estruturados, garantindo total rastreabilidade (lineage) e auditabilidade do dado imputado.

## 2. Arquitetura e Stack Tecnológica (Inegociável)
A IA assistente DEVE respeitar rigorosamente as seguintes escolhas, não sugerindo alternativas a menos que explicitamente solicitado:
* **Abordagem:** Raw-RAG (fragmentação de documentos brutos e busca em tempo de execução).
* **Linguagem:** Python 3.10+ (Scripts modulares, sem frameworks pesados de orquestração como Airflow).
* **Banco de Dados:** PostgreSQL local com a extensão `pgvector`. O Postgres atuará simultaneamente como Data Warehouse e Vector DB.
* **Modelos de IA:**
  * **LLM:** Deepseek V4 (via API).
  * **Embedding:** A ser definido no módulo de ingestão (modelos leves e locais).
* **Validação:** `Pydantic` será usado para forçar a saída estruturada da LLM.
* **Testes:** `pytest` como framework padrão.

## 3. Padrões de Código e Qualidade
1. **Tipagem Estrita (Type Hints):** Todo código Python DEVE conter type hints explícitos (PEP 484). É proibido o uso irrestrito de `Any` ou funções sem assinatura de retorno.
2. **Programação Defensiva:** Antecipe falhas. Falhas de parse do LLM, limites de API ou chaves ausentes devem gerar exceções tratadas ou logs de erro, NUNCA falhas silenciosas.
3. **Mocking Obrigatório em Testes:** Nenhum teste automatizado deve realizar chamadas reais à API do Deepseek. Respostas da LLM devem ser simuladas (mockadas) para testar a resiliência do framework.

## 4. Regras de Comportamento da IA Assistente
1. **Atuação Restrita:** A IA atua como executora de especificações. Não tome decisões arquiteturais, não mude a stack e não crie abstrações prematuras.
2. **Foco na Linhagem (Lineage):** Todo código de imputação gerado deve obrigatoriamente incluir a atualização das colunas de auditoria (ex: `fonte_documento`, `pagina`, `trecho_evidenciado`, `confianca`).
3. **Simplicidade (KISS):** Escreva código limpo, procedural ou funcional. Evite hierarquias de classes complexas para tarefas que um script linear resolve bem.
4. **Pare e Pergunte (Stop and Ask):** Se uma especificação (spec) fornecida for ambígua, contraditória ou faltarem definições de infraestrutura, PARE a geração de código e faça uma pergunta direta. Não presuma decisões estruturais críticas.
5. **Respostas Diretas:** Ao gerar código, forneça o bloco completo e funcional. Evite explicações teóricas longas.