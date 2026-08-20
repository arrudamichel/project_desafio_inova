"""Recuperação híbrida: combina o índice léxico e o índice semântico do
ChromaDB via Reciprocal Rank Fusion (RRF).

Etapa de "Recuperação das informações" do projeto:
- Busca os `top_k_semantic` chunks mais similares semanticamente.
- Busca os `top_k_lexical` chunks mais similares lexicalmente.
- Combina os dois rankings em um único ranking final via RRF, que
  favorece documentos bem posicionados em qualquer uma das estratégias.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.embeddings.lexical import LexicalEmbeddings
from app.embeddings.semantic import get_semantic_embeddings

DocKey = Tuple[str, int]


class HybridRetriever:
    """Recuperador híbrido léxico + semântico sobre o ChromaDB."""

    def __init__(self) -> None:
        self.semantic_store = Chroma(
            collection_name=settings.semantic_collection,
            embedding_function=get_semantic_embeddings(),
            persist_directory=settings.chroma_dir,
        )
        self.lexical_store = Chroma(
            collection_name=settings.lexical_collection,
            embedding_function=LexicalEmbeddings(n_features=settings.lexical_vector_size),
            persist_directory=settings.chroma_dir,
        )

    @staticmethod
    def _doc_key(doc: Document) -> DocKey:
        """Chave usada para identificar um chunk de forma única entre as
        duas coleções (mesmo chunk, embeddings diferentes)."""
        return (doc.metadata.get("document_name", "desconhecido"), doc.metadata.get("chunk_id", -1))

    def retrieve_semantic(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Busca apenas no índice semântico."""
        return self.semantic_store.similarity_search(query, k=k or settings.top_k_semantic)

    def retrieve_lexical(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Busca apenas no índice léxico."""
        return self.lexical_store.similarity_search(query, k=k or settings.top_k_lexical)

    def _fuse(self, query: str, pool_size: int) -> List[Document]:
        """Executa a fusão RRF entre os rankings léxico e semântico e
        retorna até `pool_size` documentos ordenados pelo score combinado.
        """
        semantic_hits = self.retrieve_semantic(query, k=max(pool_size, settings.top_k_semantic))
        lexical_hits = self.retrieve_lexical(query, k=max(pool_size, settings.top_k_lexical))

        scores: Dict[DocKey, float] = {}
        docs_by_key: Dict[DocKey, Document] = {}

        for rank, doc in enumerate(semantic_hits):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            docs_by_key[key] = doc

        for rank, doc in enumerate(lexical_hits):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            docs_by_key[key] = doc

        ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [docs_by_key[key] for key in ranked_keys[:pool_size]]

    def retrieve_candidates(self, query: str, pool_size: Optional[int] = None) -> List[Document]:
        """Retorna um pool de candidatos (maior que o `top_k_final`) já
        fundidos via RRF, tipicamente usado como entrada para um
        reranking posterior (ex.: via LLM)."""
        pool_size = pool_size or settings.rerank_pool_size
        return self._fuse(query, pool_size)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Busca híbrida: combina os resultados léxico e semântico via
        Reciprocal Rank Fusion (RRF).

        RRF score de um documento = soma, para cada ranking em que ele
        aparece, de `1 / (rrf_k + posição_no_ranking)`. Isso evita
        depender de escalas de similaridade incompatíveis entre os dois
        índices (distância cosseno semântica vs. hashing léxico).
        """
        k = k or settings.top_k_final
        return self._fuse(query, k)
