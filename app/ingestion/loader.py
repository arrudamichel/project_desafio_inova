"""Leitura de arquivos PDF e extração de texto + metadados.

Passos 1, 2 e 6 do pipeline de indexação:
1. Lê arquivos PDF de um diretório.
2. Extrai o conteúdo textual de cada página (via `PyPDFLoader`).
6. Adiciona metadados: nome do documento, página, data e origem.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdfs(source_dir: str) -> List[Document]:
    """Lê recursivamente todos os PDFs de `source_dir` e retorna um
    `Document` do LangChain por página, já com metadados enriquecidos.

    Metadados adicionados a cada página:
    - document_name: nome do arquivo PDF de origem.
    - page: número da página (1-indexado).
    - date: data de modificação do arquivo (proxy para "data do documento").
    - origin: caminho da pasta de onde o arquivo foi lido.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Diretório de PDFs não encontrado: '{source_dir}'. "
            "Crie a pasta e adicione seus arquivos PDF antes de indexar."
        )

    pdf_files = sorted(source_path.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum arquivo .pdf encontrado em '{source_dir}'. "
            "Adicione ao menos um PDF antes de rodar a indexação."
        )

    documents: List[Document] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        file_stat = pdf_path.stat()
        modified_at = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d")

        for page in pages:
            raw_page_number = page.metadata.get("page", 0)
            page.metadata.update(
                {
                    "document_name": pdf_path.name,
                    "page": raw_page_number + 1,  # PyPDFLoader é 0-indexado
                    "date": modified_at,
                    "origin": str(pdf_path.parent),
                }
            )
            documents.append(page)

    return documents
