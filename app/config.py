"""Configurações centrais da aplicação.

Todas as configurações podem ser sobrescritas por variáveis de ambiente
(carregadas automaticamente de um arquivo `.env` na raiz do projeto).
Veja `.env.example` para a lista completa de opções disponíveis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Carrega variáveis do arquivo .env (se existir) para o ambiente do processo.
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "sim"}


@dataclass(frozen=True)
class Settings:
    # --- Diretórios ---
    pdf_source_dir: str = os.getenv("PDF_SOURCE_DIR", "data/pdfs")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma_db")
    eval_dataset_path: str = os.getenv("EVAL_DATASET_PATH", "data/eval/qa_dataset.json")
    eval_report_path: str = os.getenv("EVAL_REPORT_PATH", "data/eval/eval_report.json")

    # --- Coleções do ChromaDB ---
    # Duas coleções são criadas no mesmo banco vetorial: uma com vetores
    # léxicos (hashing/bag-of-words) e outra com vetores semânticos (modelo
    # de embedding). Isso permite comparar e combinar as duas estratégias.
    semantic_collection: str = os.getenv("SEMANTIC_COLLECTION", "semantic_index")
    lexical_collection: str = os.getenv("LEXICAL_COLLECTION", "lexical_index")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- Embeddings semânticos ---
    # "huggingface" -> roda localmente, sem custo, sem API key (padrão).
    # "openai" -> usa a API de embeddings da OpenAI (requer OPENAI_API_KEY).
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Embedding léxico ---
    # Dimensão do vetor gerado pelo HashingVectorizer (representação léxica).
    lexical_vector_size: int = int(os.getenv("LEXICAL_VECTOR_SIZE", "512"))

    # --- LLM de geração ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # --- Recuperação ---
    top_k_semantic: int = int(os.getenv("TOP_K_SEMANTIC", "5"))
    top_k_lexical: int = int(os.getenv("TOP_K_LEXICAL", "5"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "4"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))  # constante do Reciprocal Rank Fusion

    # --- Reranking via LLM ---
    # Após a fusão RRF, um pool maior de candidatos pode ser reordenado por
    # um LLM (relevância semântica real à pergunta) antes de cortar para
    # `top_k_final`. Isso melhora a precisão do contexto final, ao custo
    # de uma chamada extra ao LLM por pergunta.
    rerank_enabled: bool = _get_bool("RERANK_ENABLED", True)
    rerank_pool_size: int = int(os.getenv("RERANK_POOL_SIZE", "15"))

    # --- Avaliação ---
    # K usado no cálculo de Precision@K, Recall@K e NDCG@K.
    eval_top_k: int = int(os.getenv("EVAL_TOP_K", "5"))


settings = Settings()
