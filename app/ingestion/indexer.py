"""Construção dos índices léxico e semântico no ChromaDB.

Passos 4 e 5 do pipeline de indexação: os mesmos chunks são inseridos em
duas coleções distintas do ChromaDB, cada uma com uma função de embedding
diferente:

- `lexical_index`  -> vetores léxicos (HashingVectorizer / bag-of-words).
- `semantic_index`  -> vetores semânticos (modelo de embedding).

Os metadados de cada chunk (document_name, page, date, origin, chunk_id)
são preservados em ambas as coleções.
"""
from __future__ import annotations

from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.embeddings.lexical import LexicalEmbeddings
from app.embeddings.semantic import get_semantic_embeddings


def build_indexes(chunks: List[Document]) -> None:
    """Cria/atualiza as coleções léxica e semântica do ChromaDB a partir
    da lista de chunks fornecida."""
    if not chunks:
        raise ValueError("Nenhum chunk fornecido para indexação.")

    semantic_embeddings = get_semantic_embeddings()
    lexical_embeddings = LexicalEmbeddings(n_features=settings.lexical_vector_size)

    # Índice semântico: significado do texto (embedding neural).
    Chroma.from_documents(
        documents=chunks,
        embedding=semantic_embeddings,
        collection_name=settings.semantic_collection,
        persist_directory=settings.chroma_dir,
    )

    # Índice léxico: sobreposição literal de termos (hashing/bag-of-words).
    Chroma.from_documents(
        documents=chunks,
        embedding=lexical_embeddings,
        collection_name=settings.lexical_collection,
        persist_directory=settings.chroma_dir,
    )
