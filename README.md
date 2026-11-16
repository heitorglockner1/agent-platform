# agent-platform

Referência de arquitetura para IA generativa em produção: RAG com citações, agente com tetos de custo, portas e adaptadores, observabilidade. Python 3.11+, FastAPI, LangChain.

Destila padrões que aprendi construindo o [Caleryn](https://caleryn.com), um SaaS multi-tenant em produção que usa agentes e RAG. Este repositório não contém código do produto: é uma versão pequena, testável e sem chave de API que mostra as decisões.

## Rodar

```bash
pip install -e ".[dev]"
pytest                                                                             # testes
pytest --cov=aiplatform --cov-report=term-missing:skip-covered --cov-fail-under=85  # como no CI
uvicorn aiplatform.main:app                                                        # sem chave usa adaptadores locais
```

Sem nenhuma variável de ambiente, o servidor usa um modelo de demonstração (um roteiro fixo, não um LLM), embeddings por hashing e índice em memória:

```bash
curl -X POST localhost:8000/documents -H 'content-type: application/json' \
  -d '{"id":"pol","text":"O prazo de reembolso é de sete dias corridos.","source":"politica.md"}'
```

```json
{"document_id":"pol","chunks":1}
```

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"qual o prazo de reembolso?"}'
```

```json
{"answer":"O prazo de reembolso é de sete dias corridos.","grounded":true,"citations":[{"chunk_id":"pol#0","source":"politica.md","score":0.5164}]}
```

```bash
curl -X POST localhost:8000/agent -H 'content-type: application/json' \
  -d '{"task":"qual o prazo de reembolso?"}'
```

```json
{"answer":"O prazo de reembolso é de sete dias corridos.\n\n[executado de fato: search_documents]","stopped_by":"concluded","iterations":2,"tool_calls_executed":1,"tokens_used":136,"errors":[]}
```

`/ask` é um pipeline: recupera, e só chama o modelo se algo passou do limiar. `/agent` deixa o modelo decidir o que buscar, dentro de tetos de voltas, tokens e tamanho de retorno. O rodapé `[executado de fato: ...]` é gerado por código, não pelo modelo.

### Configuração

Lida só em `settings.py`; valor inválido derruba a inicialização com o nome da variável.

| Variável | Padrão | Efeito |
|---|---|---|
| `OPENAI_API_KEY` | ausente | presente: o modelo vira `ChatOpenAI` via LangChain (extra `langchain`) |
| `LLM_MODEL` | `gpt-4o-mini` | modelo de chat |
| `LLM_TIMEOUT_S` | `30` | timeout por chamada; estouro vira HTTP 502 |
| `EMBEDDINGS` | `hashing` | `openai` usa `OpenAIEmbeddings` (extra `langchain`) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | modelo de embeddings |
| `VECTOR_STORE` | `memory` | `chroma` usa ChromaDB (extra `chroma`) |
| `CHROMA_PATH` | ausente | diretório do Chroma persistente; ausente usa o efêmero |
| `RAG_TOP_K` | `4` | trechos recuperados por busca |
| `RAG_MIN_SCORE` | `0.2` | limiar de similaridade; abaixo dele a resposta é "não encontrei" |
| `AGENT_MAX_ITERATIONS` | `4` | voltas do agente |
| `AGENT_MAX_TOKENS` | `20000` | orçamento de tokens por execução do agente |

### Docker

```bash
docker build -t agent-platform .                                       # núcleo + langchain
docker build -t agent-platform --build-arg EXTRAS=langchain,chroma .   # com Chroma
docker run -p 8000:8000 agent-platform
```

## Arquitetura

```
domain/          dataclasses puras (Document, Chunk, Answer, ToolSpec, AgentMessage...)
application/     ports.py (interfaces), rag.py, agent_loop.py, agentic_rag.py, budget.py
infrastructure/  adaptadores: local (memória), demo, Chroma, LangChain (chat e embeddings)
web/             FastAPI: rotas finas, middleware de métricas e request id
settings.py      configuração do ambiente
observability.py log JSON e tracer
main.py          composição: escolhe os adaptadores a partir de Settings
```

Dependências apontam para dentro: `application` nunca importa `infrastructure`. Trocar Chroma por OpenSearch, ou OpenAI por Bedrock, é escrever um adaptador novo, sem tocar em regra de negócio. Portas que fazem I/O de rede são assíncronas.

## Onde está cada decisão

| Decisão | Código | Teste |
|---|---|---|
| Tetos do agente: voltas, tokens, tamanho de retorno | `application/agent_loop.py` | `test_agent_loop.py` |
| Última volta sem ferramentas; pedido nela é ignorado | `agent_loop.py` | `test_last_turn_has_no_tools_so_model_must_conclude` |
| Chamada repetida não reexecuta | `agent_loop.py` | `test_repeated_call_does_not_reexecute` |
| Rodapé factual em vez de confiar no texto do modelo | `agent_loop.py` | `test_model_claim_is_not_trusted_when_tool_failed` |
| Orçamento consome o uso informado pelo provedor, ou estimado | `agent_loop.py`, `langchain_chat.py` | `test_reported_usage_consumes_the_budget` |
| Abaixo do limiar, recusa sem chamar o modelo | `application/rag.py` | `test_refuses_instead_of_inventing_when_nothing_relevant` |
| Citações vêm da recuperação, não do modelo | `rag.py` | `test_answer_has_citations_from_retrieval` |
| Reindexar substitui a versão antiga, sem perdê-la se o embedder falhar | `rag.py` | `test_reindex_keeps_old_version_when_embedding_fails` |
| Chunking com sobreposição limitada, sem trecho redundante | `rag.py` | `test_every_chunk_adds_new_words_and_rebuilds_the_text` |
| Mesmo contrato para todo banco vetorial | `infrastructure/local.py`, `chroma_store.py` | `test_vector_store_contract.py` |
| Qualquer provedor LangChain, inclusive conteúdo em blocos | `infrastructure/langchain_chat.py` | `test_langchain_adapter.py` |
| Falha ou timeout do provedor vira 502; erro inesperado vira 500 contado | `web/app.py` | `test_api_errors.py` |
| Validação de entrada e schema de resposta no OpenAPI | `web/app.py` | `test_openapi_describes_the_ask_response` |
| Log JSON com request id; métrica por rota e por desfecho do agente | `observability.py`, `web/app.py` | `test_api_flow_metrics_and_request_id`, `test_agent_api.py` |
| Eventos de LLM (`agent.turn`, `rag.retrieval`) pela porta `Tracer` | `observability.py` | `test_agent_turns_and_retrieval_are_traced` |
| Eventos de domínio | `EventPublisherPort` publica `document.indexed`; só há um barramento em memória, sem consumidor | `test_reindex_removes_old_chunks_and_publishes_event` |
| CI: lint, tipos, testes em 3.11 e 3.12, smoke da imagem | `.github/workflows/ci.yml`, `Dockerfile` | |

## O que este repositório não cobre

Transparência vale mais que checklist inflado:

- **AWS:** nenhum deploy aqui. No Caleryn o deploy roda em Fly.io com Docker e CI; os adaptadores de nuvem (Bedrock, SQS, OpenSearch) são o próximo passo natural e cabem nas portas existentes.
- **MongoDB e brokers (Kafka, RabbitMQ, SQS):** só existe a porta e um barramento em memória.
- **Migração monolito para microsserviços:** o Caleryn é um monolito modular com worker separado; não há microsserviços aqui.
- **Autenticação e isolamento por cliente:** não há.
- **Persistência:** o store padrão é em memória e perde os dados ao reiniciar; use `VECTOR_STORE=chroma` com `CHROMA_PATH`.
- **Chunking estrutural:** o corte é por palavras e descarta quebras de linha, então tabelas e código perdem a estrutura.
- **Calibração:** `RAG_MIN_SCORE` foi calibrado para o embedder por hashing e precisa ser recalibrado ao trocar de embedder.
- **Lockfile:** as dependências têm só limite inferior de versão.

## Aprendizados

Veja [docs/LEARNINGS.md](docs/LEARNINGS.md): o que o uso real com LLMs ensinou e que teste de unidade não pega.
