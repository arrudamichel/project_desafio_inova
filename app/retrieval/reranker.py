"""Reranking dos documentos recuperados usando um LLM como juiz de relevância.

Etapa opcional entre a recuperação híbrida (RRF) e a geração da resposta:
um pool maior de candidatos (ver `HybridRetriever.retrieve_candidates`) é
reordenado por um LLM, que atribui uma nota de relevância real (semântica
e contextual) a cada chunk em relação à pergunta — algo que o RRF sozinho
não consegue capturar, pois combina apenas posições de ranking.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval.hybrid_retriever import HybridRetriever


class _ChunkRelevance(BaseModel):
    chunk_id: int = Field(description="Identificador do chunk (chunk_id) avaliado.")
    relevance: float = Field(
        ge=0.0,
        le=1.0,
        description="Nota de relevância do chunk em relação à pergunta: 0.0 = irrelevante, 1.0 = totalmente relevante.",
    )


class _RerankJudgement(BaseModel):
    scores: List[_ChunkRelevance] = Field(
        description="Uma nota de relevância para cada chunk candidato fornecido, identificado pelo seu chunk_id."
    )


_RERANK_SYSTEM_PROMPT = (
    "Você é um avaliador de relevância para um sistema de busca (RAG). Dada "
    "uma pergunta e uma lista de trechos (chunks) candidatos, atribua a "
    "CADA chunk uma nota de relevância de 0.0 a 1.0, indicando o quanto "
    "aquele trecho especificamente ajuda a responder à pergunta. Julgue "
    "apenas pelo conteúdo do texto, ignorando a ordem em que os chunks "
    "foram apresentados. Retorne uma nota para todos os chunk_id recebidos."
)

_RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _RERANK_SYSTEM_PROMPT),
        (
            "human",
            "PERGUNTA:\n{question}\n\n"
            "TRECHOS CANDIDATOS:\n{candidates}",
        ),
    ]
)


def _format_candidates(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", -1)
        source = doc.metadata.get("document_name", "desconhecido")
        parts.append(f"[chunk_id={chunk_id} | fonte={source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class LLMReranker:
    """Reordena uma lista de documentos candidatos por relevância à
    pergunta, usando um LLM com saída estruturada (Pydantic)."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = _RERANK_PROMPT | llm.with_structured_output(_RerankJudgement)

    def rerank(self, question: str, docs: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Reordena `docs` por relevância à `question` e retorna os
        `top_k` melhores. Se `docs` estiver vazio, retorna lista vazia.
        Em caso de falha ao pontuar algum chunk, ele recebe nota 0.0
        (vai para o fim do ranking), sem interromper o processo.
        """
        if not docs:
            return []

        candidates_text = _format_candidates(docs)
        judgement: _RerankJudgement = self._chain.invoke(
            {"question": question, "candidates": candidates_text}
        )

        score_by_chunk_id = {item.chunk_id: item.relevance for item in judgement.scores}

        def _score(doc: Document) -> float:
            return score_by_chunk_id.get(doc.metadata.get("chunk_id", -1), 0.0)

        ranked = sorted(docs, key=_score, reverse=True)
        k = top_k or settings.top_k_final
        return ranked[:k]


class RerankingHybridRetriever:
    """Combina a recuperação híbrida (RRF) com um reranking via LLM:
    busca um pool maior de candidatos e os reordena por relevância real
    à pergunta antes de cortar para o `top_k_final`.
    """

    def __init__(self, retriever: HybridRetriever, llm: BaseChatModel) -> None:
        self.retriever = retriever
        self.reranker = LLMReranker(llm)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        k = k or settings.top_k_final
        candidates = self.retriever.retrieve_candidates(query, pool_size=settings.rerank_pool_size)
        return self.reranker.rerank(query, candidates, top_k=k)
