"""Embedding léxico: representa cada chunk como um vetor esparso de termos
(hashing trick / bag-of-words), sem nenhum modelo de linguagem treinado.

Isso simula um "índice léxico" clássico (estilo TF-IDF / BM25) armazenado
como vetores densos dentro do ChromaDB, complementando o índice semântico
baseado em modelo de embedding (ver `app/embeddings/semantic.py`).
"""
from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import HashingVectorizer


class LexicalEmbeddings(Embeddings):
    """Implementação de `Embeddings` (interface do LangChain) baseada em
    `HashingVectorizer` do scikit-learn.

    Vantagens para fins didáticos/demonstrativos:
    - Não precisa de treinamento nem de modelo pré-treinado.
    - É determinística e rápida.
    - Captura sobreposição literal de termos (léxico), diferente do
      embedding semântico, que captura similaridade de significado.
    """

    def __init__(self, n_features: int = 512) -> None:
        self.n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
            lowercase=True,
        )

    def _vectorize(self, texts: List[str]) -> List[List[float]]:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._vectorize(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize([text])[0]
