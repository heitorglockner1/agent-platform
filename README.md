# ai-agent-platform

Referência de arquitetura para IA generativa em produção: RAG com citações, agentes com tetos de custo, portas e adaptadores, observabilidade. Python 3.12, FastAPI, LangChain.

Destila padrões que aprendi construindo o [Caleryn](https://caleryn.com), um SaaS multi-tenant em produção que usa agentes e RAG. Este repositório não contém código do produto: é uma versão pequena, testável e sem chave de API que mostra as decisões.

## Rodar

```bash
pip install -e ".[dev]"
pytest                      # testes
pytest --cov=aiplatform --cov-report=term-missing:skip-covered --cov-fail-under=85  # como no CI
uvicorn aiplatform.main:app # sem OPENAI_API_KEY usa adaptadores locais
```

```bash
curl -X POST localhost:8000/documents -H 'content-type: application/json' \
  -d '{"id":"pol","text":"O prazo de reembolso é de sete dias corridos.","source":"politica.md"}'
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"qual o prazo de reembolso?"}'
```

Com `OPENAI_API_KEY` o modelo passa a ser um `ChatOpenAI` via LangChain. Com `VECTOR_STORE=chroma` (extra `.[chroma]`) o índice vai para o ChromaDB.

## Arquitetura

```
domain/          dataclasses puras (Document, Chunk, Answer, ToolSpec...)
application/     ports.py (interfaces), rag.py, agent_loop.py
infrastructure/  adaptadores: local (memória), Chroma, LangChain, fakes
web/             FastAPI, métricas Prometheus, logs JSON com request_id
```

Dependências apontam para dentro: `application` nunca importa `infrastructure`. Trocar Chroma por OpenSearch, ou OpenAI por Bedrock, é escrever um adaptador novo, sem tocar em regra de negócio.

## Requisitos de IA cobertos

| Tema | Onde |
|---|---|
| Python, APIs resilientes | `web/app.py`: validação Pydantic, request id, métricas por rota |
| Clean Architecture, SOLID, portas e adaptadores | `application/ports.py` e `infrastructure/` |
| IA Generativa com LLMs via API | `infrastructure/langchain_chat.py`: qualquer `BaseChatModel` (OpenAI, Bedrock via `ChatBedrockConverse`, open-source) |
| RAG: indexação e recuperação | `application/rag.py`: chunking com sobreposição, limiar de similaridade, citações vindas do código, reindexação sem chunks fantasmas |
| LangChain | adaptador de provedor com tool calling; o loop do agente é próprio (ver LEARNINGS) |
| Banco vetorial | porta `VectorStorePort`; adaptadores em memória e ChromaDB |
| Agentes de IA | `application/agent_loop.py`: tetos de iterações, orçamento e retorno; deduplicação de chamadas; rodapé factual |
| Event-driven | `EventPublisherPort` + `document.indexed` publicado na indexação |
| Observabilidade | logs JSON, `/metrics` Prometheus, porta `Tracer` para Langfuse/OpenTelemetry |
| CI/CD, Docker | `.github/workflows/ci.yml`, `Dockerfile` (usuário não-root, healthcheck) |

## O que este repositório não cobre

Transparência vale mais que checklist inflado:

- **AWS:** nenhum deploy aqui. No Caleryn o deploy roda em Fly.io com Docker e CI; os adaptadores de nuvem (Bedrock, SQS, OpenSearch) são o próximo passo natural e cabem nas portas existentes.
- **MongoDB e brokers (Kafka, RabbitMQ, SQS):** só existe a porta e um barramento em memória.
- **Migração monolito para microsserviços:** o Caleryn é um monolito modular com worker separado; não há microsserviços aqui.

## Aprendizados

Veja [docs/LEARNINGS.md](docs/LEARNINGS.md): o que o uso real com LLMs ensinou e que teste de unidade não pega.
