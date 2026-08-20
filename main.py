"""Ponto de entrada principal da aplicação.

Uso:
    python main.py "Qual o prazo de entrega definido no contrato?"
    python main.py --index                         # (re)indexa os PDFs
    python main.py --index "Qual é o objetivo do projeto?"
    python main.py --evaluate                       # roda a avaliação
    python main.py                                  # modo interativo
"""
from __future__ import annotations

import argparse

from app.config import settings
from app.evaluation.evaluate import run_full_evaluation
from app.generation.qa_chain import RAGChain
from app.ingestion.chunking import split_documents
from app.ingestion.indexer import build_indexes
from app.ingestion.loader import load_pdfs


def run_indexing(source_dir: str) -> None:
    """Executa o pipeline completo de indexação: leitura de PDFs,
    chunking e criação dos índices léxico e semântico no ChromaDB."""
    print(f"[1/3] Lendo PDFs em: {source_dir}")
    documents = load_pdfs(source_dir)
    print(f"      {len(documents)} página(s) extraída(s).")

    print("[2/3] Gerando chunks...")
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"      {len(chunks)} chunk(s) gerado(s).")

    print("[3/3] Construindo índices léxico e semântico no ChromaDB...")
    build_indexes(chunks)
    print(f"      Índices salvos em: {settings.chroma_dir}\n")
    print("Indexação concluída com sucesso.\n")


def ask_question(question: str) -> None:
    """Responde a uma pergunta usando o pipeline de RAG (recuperação
    híbrida + geração via LLM) e imprime a resposta com as fontes."""
    rag = RAGChain()
    result = rag.answer(question)

    print("\nResposta:")
    print(result["answer"])

    print("\nFontes utilizadas:")
    for source in result["sources"]:
        print(
            f"  - {source['document_name']} "
            f"(página {source['page']}, data {source['date']}, origem {source['origin']})"
        )
    print()


def _run_interactive_loop() -> None:
    print("Modo interativo. Digite sua pergunta ou 'sair' para encerrar.\n")
    while True:
        try:
            question = input("Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"sair", "exit", "quit"}:
            break
        if not question:
            continue

        ask_question(question)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sistema de Perguntas e Respostas sobre documentos PDF usando "
            "RAG (LangChain + ChromaDB)."
        )
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Pergunta a ser respondida com base nos documentos indexados.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="(Re)indexa os PDFs (leitura, chunking e criação dos índices) antes de responder.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Executa a avaliação de recuperação e geração do pipeline.",
    )
    parser.add_argument(
        "--source-dir",
        default=settings.pdf_source_dir,
        help=f"Diretório com os arquivos PDF a indexar (padrão: {settings.pdf_source_dir}).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    
    if args.index:
        run_indexing(args.source_dir)

    if args.evaluate:
        run_full_evaluation()
        return
        
    if args.question:
        ask_question(args.question)
        return

    _run_interactive_loop()


if __name__ == "__main__":
    main()
