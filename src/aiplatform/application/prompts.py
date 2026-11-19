"""Todo texto que o modelo recebe, num lugar só.

Prompt é artefato versionado: mudar um texto daqui muda o comportamento do
modelo, então a mudança deve vir com o motivo no commit. Os prompts são curtos
de propósito: o que garante o comportamento é o código (limiar, citações,
tetos, rodapé factual), não o texto. Teste de unidade só prova que a string
chegou ao modelo; para saber se ele obedece, rode um smoke com chave real.
"""

from typing import Any

# --- RAG do /ask (application/rag.py) -------------------------------------------

# Instrução de sistema. A recusa ("diga que não sabe") é a segunda barreira: a
# primeira é o limiar de similaridade, que nem chama o modelo quando não há contexto.
GROUNDING_INSTRUCTION = "Responda só com base no CONTEXTO. Se não estiver nele, diga que não sabe."

# Cabeçalhos da mensagem do usuário. Cada trecho vai prefixado com o seu `[id]`
# para o modelo poder citá-lo; as citações da resposta, porém, vêm do código.
CONTEXT_HEADER = "CONTEXTO:"
QUESTION_HEADER = "PERGUNTA:"

# --- Agente do /agent (application/agentic_rag.py) ------------------------------

# "Busque antes de responder" evita resposta de memória do modelo; "não invente"
# repete a regra do RAG para o caso de a busca voltar vazia.
AGENT_SYSTEM_PROMPT = (
    "Você responde perguntas usando a ferramenta search_documents. Busque antes de "
    "responder. Se a busca não trouxer nada relevante, diga que não encontrou. Não invente."
)

# Descrição e schema da ferramenta também são prompt: é por eles que o modelo
# decide quando chamá-la e com que argumentos.
SEARCH_TOOL_NAME = "search_documents"
SEARCH_TOOL_DESCRIPTION = "Busca trechos relevantes nos documentos indexados."
SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Termos a buscar nos documentos indexados."}
    },
    "required": ["query"],
    "additionalProperties": False,
}

# --- Loop do agente (application/agent_loop.py) ---------------------------------

# Devolvido no lugar do resultado quando o modelo repete ferramenta e argumentos.
# Manda concluir porque repetição costuma ser o modelo andando em círculo.
REPEATED_CALL_REPLY = "Chamada repetida: o resultado já está acima. Conclua com o que tem."
