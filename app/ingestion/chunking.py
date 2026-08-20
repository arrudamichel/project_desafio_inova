"""Chunking (segmentação) dos documentos extraídos dos PDFs.

Passo 3 do pipeline de indexação: divide o texto de cada página em pedaços
(chunks) menores e sobrepostos, adequados para embedding e recuperação.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Divide os documentos em chunks menores, preservando os metadados
    originais (document_name, page, date, origin) e adicionando um
    `chunk_id` sequencial único para cada pedaço gerado.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks
