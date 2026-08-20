"""Pacote principal da aplicação de RAG (Retrieval-Augmented Generation).

Submódulos:
- ingestion: leitura de PDFs, chunking e indexação.
- embeddings: modelos de vetorização léxica e semântica.
- retrieval: recuperação híbrida (léxica + semântica) via ChromaDB.
- generation: cadeia de geração de respostas (LLM) com LangChain.
- evaluation: avaliação de recuperação e geração.
"""
