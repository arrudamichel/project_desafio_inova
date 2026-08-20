"""Fábrica de embeddings semânticos.

Passo 5 do pipeline de indexação: aplica um modelo de embedding (rede
neural treinada) para gerar vetores que capturam o significado do texto,
usados para popular o índice semântico no ChromaDB.

Dois provedores são suportados:
- "huggingface": modelo local via `sentence-transformers` (padrão, sem
  necessidade de API key).
- "openai": embeddings via API da OpenAI (requer OPENAI_API_KEY).
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.config import settings


def get_semantic_embeddings() -> Embeddings:
    provider = settings.embedding_provider.lower()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    raise ValueError(
        f"Provedor de embedding semântico não suportado: '{provider}'. "
        "Use 'huggingface' ou 'openai' na variável EMBEDDING_PROVIDER."
    )
