# Aprendizados de IA em produção

Os itens com referência viraram código e teste neste repositório. Os marcados como aprendizado do Caleryn vieram do produto e não têm código aqui.

## Agentes

**Agente só quando a segunda pergunta depende da primeira.** Se o passo seguinte é conhecido, é pipeline: previsível, testável e mais barato. Agente é exceção. (`/ask` é pipeline, `/agent` é a exceção: `web/app.py`)

**Três tetos, sempre** (`agent_loop.py`):
1. `max_iterations`.
2. Orçamento reconferido a cada volta. O custo total é decisão do modelo, não sua.
3. Tamanho máximo do retorno de ferramenta. Ele é reenviado em toda volta seguinte.

(`test_reported_usage_consumes_the_budget`, `test_tool_result_is_capped`)

**Última volta sem ferramentas.** Sem isso o teto de iterações vira final sem saída: o modelo pede mais uma coisa em vez de concluir. Com `tools=[]` ele é obrigado a responder (`test_last_turn_has_no_tools...`). Detalhe de API: várias recusam lista vazia, então o adaptador omite o parâmetro (`test_empty_tools_does_not_bind`).

**Chamada repetida não reexecuta.** Mesma ferramenta e mesmos argumentos devolvem um lembrete. Em um caso real, derrubou 17 chamadas para 8 (`test_repeated_call_does_not_reexecute`).

**O modelo afirma o que não fez.** Ele escreveu "criei três rascunhos" com todas as tentativas recusadas. O texto final leva um rodapé gerado por código com o que de fato executou (`test_model_claim_is_not_trusted...`).

**Limite de escrita mora no handler, nunca no prompt.** Prompt é pedido, handler é lei. Teto de itens, janela de datas e bloqueios são regras de negócio com teste. (`ToolSpec` em `domain/models.py`; erro do handler volta ao modelo como texto: `test_model_claim_is_not_trusted...`)

**LLM não tem relógio.** Sem a data de hoje e a janela válida no prompt, ele chuta uma data plausível, a ferramenta recusa, e ele chuta outra. Ferramenta que valida data exige a data no contexto. *(Aprendizado do Caleryn, sem código neste repositório.)*

**Por que o loop é próprio e o LangChain é só provedor.** O loop precisa dos tetos acima, de dedupe e de rodapé factual, e quer ser legível e testável com um fake de dez linhas. O LangChain entra onde agrega: uma interface única para vários provedores (`infrastructure/langchain_chat.py`, `langchain_embeddings.py`).

## RAG

**Recusar é melhor que inventar.** Abaixo do limiar de similaridade a resposta é "não encontrei" e `grounded=False`. O modelo nem é chamado (`test_refuses_instead_of_inventing_when_nothing_relevant`).

**Citações vêm do código.** São o que foi recuperado, não o que o modelo diz ter usado (`test_answer_has_citations_from_retrieval`).

**Reindexar apaga antes.** Sem isso, versões antigas do documento continuam respondendo (`test_reindex_removes_old_chunks_and_publishes_event`).

**Teste achou um bug real aqui:** o embedder de teste casava palavras funcionais ("de", "o") e fazia uma pergunta sobre bolo recuperar a política de reembolso. Tokens curtos agora são ignorados. Em produção o mesmo problema aparece como limiar mal calibrado: calibre com perguntas fora do domínio, não só com as que deveriam acertar. O evento `rag.retrieval` registra o `top_score` antes do limiar para isso (`application/rag.py`).

## Geração

**Instrução no meio de um prompt gigante não pega.** Requisito que precisa aparecer no texto (estilo, nível de linguagem) funciona melhor como etapa própria e curta, em chamada dedicada. Teste de unidade só prova que a string chegou ao prompt. Para saber se o modelo obedece, rode um smoke com chave real. *(Aprendizado do Caleryn, sem código neste repositório.)*

**Níveis de modelo, não ids.** O código pede `light`, `creative`, `critical`, e uma camada traduz para o modelo configurado. Trocar de modelo vira configuração. *(Aprendizado do Caleryn, sem código neste repositório.)*

## Operação

**Teste que só roda em SQLite não prova o ramo de produção.** Se a produção usa outro banco, o E2E precisa passar por ele. *(Aprendizado do Caleryn, sem código neste repositório.)*

**Observabilidade desde o primeiro endpoint:** request id propagado, log em JSON, métrica por rota (`observability.py`, `web/app.py`, `test_api_flow_metrics_and_request_id`).
