"""Cadeia de geração de respostas (RAG) usando LangChain (LCEL).

Recebe uma pergunta, recupera os chunks mais relevantes (via
`HybridRetriever`) e usa um LLM para gerar uma resposta baseada apenas no
contexto recuperado, citando as fontes (documento e página) utilizadas.
"""
from __future__ import annotations

from typing import Any, List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import RerankingHybridRetriever

SYSTEM_PROMPT = """Você é um assistente que responde perguntas EXCLUSIVAMENTE com base no \
CONTEXTO fornecido abaixo, extraído de documentos PDF.

Regras:
- Se a resposta não estiver no contexto, diga claramente que não encontrou \
a informação nos documentos indexados. Não invente informações.
- Sempre que possível, cite a fonte (nome do documento e número da página) \
usada para embasar cada afirmação.
- Responda de forma objetiva e no mesmo idioma da pergunta.

CONTEXTO:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


class SourceInfo(TypedDict):
    document_name: Any
    page: Any
    date: Any
    origin: Any


class RAGResult(TypedDict):
    question: str
    answer: str
    context: str
    sources: List[SourceInfo]


def _get_llm():
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_chat_model, temperature=settings.llm_temperature)

    raise ValueError(
        f"Provedor de LLM não suportado: '{provider}'. Use 'openai' na variável LLM_PROVIDER."
    )


def _format_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("document_name", "desconhecido")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Fonte: {source}, página {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _to_source_info(doc: Document) -> SourceInfo:
    return {
        "document_name": doc.metadata.get("document_name"),
        "page": doc.metadata.get("page"),
        "date": doc.metadata.get("date"),
        "origin": doc.metadata.get("origin"),
    }


class RAGChain:
    """Orquestra recuperação híbrida (+ reranking via LLM, se habilitado)
    e geração de resposta via LLM."""

    def __init__(self) -> None:
        self.hybrid_retriever = HybridRetriever()
        self.llm = _get_llm()

        # Quando habilitado (RERANK_ENABLED=true), busca um pool maior de
        # candidatos via RRF e os reordena por relevância real à pergunta
        # usando o próprio LLM, antes de montar o contexto final.
        if settings.rerank_enabled:
            self.retriever = RerankingHybridRetriever(self.hybrid_retriever, self.llm)
        else:
            self.retriever = self.hybrid_retriever

        self.chain = PROMPT | self.llm | StrOutputParser()

    def answer(self, question: str) -> RAGResult:
        docs = self.retriever.retrieve(question)
        context = _format_context(docs)

        answer_text = self.chain.invoke({"context": context, "question": question})

        return {
            "question": question,
            "answer": answer_text,
            "context": context,
            "sources": [_to_source_info(doc) for doc in docs],
        }
