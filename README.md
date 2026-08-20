# RAG sobre PDFs com LangChain + ChromaDB (índice híbrido léxico + semântico)

Sistema de **Perguntas e Respostas (Q&A)** sobre uma base de documentos PDF,
construído com **LangChain**, usando **ChromaDB** como banco vetorial e um
mecanismo de **recuperação híbrida** que combina um **índice léxico**
(baseado em hashing/bag-of-words) e um **índice semântico** (baseado em
modelo de embedding), fundidos via *Reciprocal Rank Fusion* (RRF).

O projeto foi dividido em três etapas, conforme o diagrama
[indexacao.png](indexacao.png):

1. **Indexação dos documentos** — leitura, chunking e criação dos índices.
2. **Recuperação das informações** — busca híbrida sobre o ChromaDB.
3. **Avaliação** — métricas de qualidade da recuperação e da geração.

---

## Sumário

- [Visão geral do pipeline](#visão-geral-do-pipeline)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Como executar](#como-executar)
  - [1. Indexar os PDFs](#1-indexar-os-pdfs)
  - [2. Fazer perguntas](#2-fazer-perguntas)
  - [3. Avaliar o pipeline](#3-avaliar-o-pipeline)
- [Detalhes de arquitetura](#detalhes-de-arquitetura)
- [Limitações e próximos passos](#limitações-e-próximos-passos)

---

## Visão geral do pipeline

### Etapa 1 — Indexação

| Passo | Descrição | Módulo |
|---|---|---|
| 1 | Leitura dos arquivos PDF de um diretório | [app/ingestion/loader.py](app/ingestion/loader.py) |
| 2 | Extração do texto de cada página | [app/ingestion/loader.py](app/ingestion/loader.py) (`PyPDFLoader`) |
| 3 | Chunking (segmentação em pedaços menores) | [app/ingestion/chunking.py](app/ingestion/chunking.py) |
| 4 | Criação do **índice léxico** no ChromaDB | [app/embeddings/lexical.py](app/embeddings/lexical.py) + [app/ingestion/indexer.py](app/ingestion/indexer.py) |
| 5 | Criação do **índice semântico** no ChromaDB (após embedding) | [app/embeddings/semantic.py](app/embeddings/semantic.py) + [app/ingestion/indexer.py](app/ingestion/indexer.py) |
| 6 | Adição de metadados (documento, página, data, origem) | [app/ingestion/loader.py](app/ingestion/loader.py) |

### Etapa 2 — Recuperação

A busca combina dois rankings independentes (léxico e semântico) do
ChromaDB usando **Reciprocal Rank Fusion (RRF)**, implementada em
[app/retrieval/hybrid_retriever.py](app/retrieval/hybrid_retriever.py).

Opcionalmente (habilitado por padrão via `RERANK_ENABLED=true`), um pool
maior de candidatos gerado pelo RRF (`RERANK_POOL_SIZE`) é reordenado por
um **LLM reranker** ([app/retrieval/reranker.py](app/retrieval/reranker.py)),
que atribui uma nota de relevância real à pergunta para cada chunk antes
de cortar para o `TOP_K_FINAL` usado na geração da resposta.

### Etapa 3 — Avaliação

Implementada em [app/evaluation/evaluate.py](app/evaluation/evaluate.py),
combinando métricas clássicas de ranking com avaliação via LLM-as-judge:

- **Recuperação**: Precision@K, Recall@K e NDCG@K, comparando as
  estratégias `lexical`, `semantic`, `hybrid` (RRF) e `hybrid_reranked`
  (RRF + reranking via LLM).
- **Geração**: Context Recall, Faithfulness/Groundedness e Answer
  Relevance, avaliadas pelo próprio LLM a partir do contexto recuperado
  (já com reranking, se habilitado), da resposta gerada e da resposta de
  referência (gabarito).

---

## Estrutura do projeto

```
desafio-tecnico-inova/
├── main.py                        # Ponto de entrada (CLI)
├── requirements.txt
├── .env.example
├── indexacao.png                  # Diagrama do pipeline de indexação
├── data/
│   ├── pdfs/                      # Coloque aqui os PDFs a indexar
│   └── eval/
│       ├── qa_dataset.json        # Dataset de avaliação (perguntas/respostas)
│       └── eval_report.json       # Gerado após rodar a avaliação
├── chroma_db/                     # Banco vetorial persistido (gerado em runtime)
└── app/
    ├── config.py                  # Configurações centrais (via .env)
    ├── ingestion/
    │   ├── loader.py               # Leitura de PDF + metadados
    │   ├── chunking.py             # Divisão em chunks
    │   └── indexer.py              # Criação dos índices léxico/semântico
    ├── embeddings/
    │   ├── lexical.py               # Embedding léxico (HashingVectorizer)
    │   └── semantic.py              # Embedding semântico (HuggingFace/OpenAI)
    ├── retrieval/
    │   ├── hybrid_retriever.py      # Recuperação híbrida (RRF)
    │   └── reranker.py              # Reranking dos candidatos via LLM
    ├── generation/
    │   └── qa_chain.py              # Cadeia RAG (LCEL) de geração de resposta
    └── evaluation/
        └── evaluate.py               # Avaliação de retrieval e geração
```

---

## Instalação

**Pré-requisitos:** Python 3.10+ e `pip`.

```bash
# 1. Clone/acesse a pasta do projeto
cd desafio-tecnico-inova

# 2. Crie e ative um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 3. Instale as dependências
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuração

Copie o arquivo de exemplo e ajuste conforme necessário:

```bash
cp .env.example .env
```

Principais variáveis:

| Variável | Padrão | Descrição |
|---|---|---|
| `EMBEDDING_PROVIDER` | `huggingface` | Provedor do embedding semântico: `huggingface` (local, gratuito) ou `openai`. |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Modelo local usado quando `EMBEDDING_PROVIDER=huggingface`. |
| `LLM_PROVIDER` | `openai` | Provedor do LLM de geração das respostas. |
| `OPENAI_API_KEY` | — | Necessária para o LLM (`openai`) e, opcionalmente, para embeddings `openai`. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Modelo de chat usado na geração da resposta. |
| `PDF_SOURCE_DIR` | `data/pdfs` | Diretório com os PDFs a indexar. |
| `CHROMA_DIR` | `chroma_db` | Diretório onde o ChromaDB persiste os índices. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Tamanho e sobreposição dos chunks (em caracteres). |
| `TOP_K_SEMANTIC` / `TOP_K_LEXICAL` / `TOP_K_FINAL` | `5` / `5` / `4` | Quantidade de resultados considerados em cada etapa da busca híbrida. |
| `RRF_K` | `60` | Constante de suavização do Reciprocal Rank Fusion. |
| `RERANK_ENABLED` | `true` | Habilita o reranking via LLM entre a fusão RRF e a geração da resposta. |
| `RERANK_POOL_SIZE` | `15` | Quantidade de candidatos (pós-RRF) enviados ao LLM para reranking, antes de cortar para `TOP_K_FINAL`. |
| `EVAL_TOP_K` | `5` | K usado no cálculo de Precision@K, Recall@K e NDCG@K durante a avaliação. |

> **Importante:** mesmo usando embeddings locais (`huggingface`), a etapa
> de **geração** da resposta atualmente depende de um LLM da OpenAI
> (`OPENAI_API_KEY`). Para rodar 100% local, adapte
> [app/generation/qa_chain.py](app/generation/qa_chain.py) para usar outro
> provedor (ex.: Ollama).

---

## Como executar

### 1. Indexar os PDFs

Coloque seus arquivos `.pdf` em [data/pdfs/](data/pdfs) e rode:

```bash
python main.py --index
```

Isso executa: leitura dos PDFs → extração de texto → chunking → criação
dos índices léxico e semântico no ChromaDB (persistidos em `chroma_db/`).

### 2. Fazer perguntas

**Pergunta única (linha de comando):**

```bash
python main.py "Qual é o objetivo principal do documento?"
```

**Indexar e perguntar em um único comando:**

```bash
python main.py --index "Qual é o objetivo principal do documento?"
```

**Modo interativo** (sem argumento de pergunta):

```bash
python main.py
```

```
Modo interativo. Digite sua pergunta ou 'sair' para encerrar.

Pergunta> Qual é o prazo mencionado no contrato?
```

A resposta é exibida junto com as fontes utilizadas (documento, página,
data e origem), conforme os metadados extraídos na indexação.

### 3. Avaliar o pipeline

Edite [data/eval/qa_dataset.json](data/eval/qa_dataset.json) com perguntas
reais sobre os PDFs indexados, informando o documento e a resposta
esperados. Depois rode:

```bash
python main.py --evaluate
```

Isso imprime no console e salva em `data/eval/eval_report.json`:

**Métricas de recuperação** (para as estratégias `lexical`, `semantic`,
`hybrid` e `hybrid_reranked`, no top-K definido por `EVAL_TOP_K`):

- **Precision@K**: proporção de chunks relevantes (do documento esperado)
  entre os K primeiros resultados recuperados.
- **Recall@K**: proporção dos chunks relevantes existentes no índice que
  foram efetivamente recuperados entre os top-K.
- **NDCG@K**: qualidade da ordenação, penalizando quando os chunks
  relevantes aparecem em posições piores do ranking.

**Métricas de geração** (avaliadas pelo próprio LLM como "juiz"):

- **Context Recall**: verifica se o contexto recuperado contém as
  informações necessárias para responder à pergunta, comparando com a
  resposta de referência (`expected_answer`).
- **Faithfulness / Groundedness**: verifica se as afirmações da resposta
  gerada estão fundamentadas no contexto recuperado, reduzindo o risco de
  alucinações.
- **Answer Relevance**: avalia se a resposta gerada responde de forma
  direta e completa à pergunta feita.

Cada métrica de geração é calculada decompondo a resposta relevante em
afirmações atômicas (via LLM com saída estruturada — Pydantic) e medindo
a proporção delas que é sustentada pelo contexto ou pela resposta, com uma
justificativa (`reasoning`) registrada no relatório para auditoria.

---

## Detalhes de arquitetura

### Por que dois índices (léxico + semântico) no mesmo ChromaDB?

- **Índice léxico** ([app/embeddings/lexical.py](app/embeddings/lexical.py)):
  usa `HashingVectorizer` (scikit-learn) para gerar vetores densos a
  partir da contagem/hash de termos — sem nenhum modelo neural. Captura
  bem correspondências **exatas de palavras** (siglas, nomes próprios,
  números de cláusulas, etc.), coisas em que embeddings semânticos às
  vezes "generalizam demais" e perdem precisão.
- **Índice semântico** ([app/embeddings/semantic.py](app/embeddings/semantic.py)):
  usa um modelo de embedding (local via `sentence-transformers` ou via
  API da OpenAI) para captar **similaridade de significado**, mesmo
  quando a pergunta usa palavras diferentes das do texto original.
- Ambos são armazenados como **coleções separadas** dentro do mesmo
  `persist_directory` do ChromaDB (`lexical_index` e `semantic_index`),
  reaproveitando os mesmos chunks e metadados.

### Recuperação híbrida via RRF

Em vez de tentar normalizar e comparar diretamente escalas de similaridade
incompatíveis (distância cosseno semântica vs. similaridade de hashing
léxico), o [HybridRetriever](app/retrieval/hybrid_retriever.py) usa
**Reciprocal Rank Fusion**: cada documento recebe uma pontuação baseada na
sua **posição** em cada ranking (léxico e semântico), somando
$1 / (k_{rrf} + rank)$ para cada lista em que aparece. Documentos bem
posicionados em ambos os rankings sobem para o topo do resultado final.

### Reranking via LLM

O RRF é puramente posicional: ele não "lê" o conteúdo dos chunks, apenas
combina rankings. Para refinar a precisão do contexto final, o
[LLMReranker](app/retrieval/reranker.py) adiciona uma etapa opcional
(`RERANK_ENABLED=true`, padrão):

1. O [HybridRetriever.retrieve_candidates](app/retrieval/hybrid_retriever.py)
   busca um **pool maior** de candidatos via RRF (`RERANK_POOL_SIZE`,
   padrão 15) — mais do que o `TOP_K_FINAL` usado na resposta.
2. O LLM recebe a pergunta e todos os candidatos (identificados por
   `chunk_id`) e retorna, via **saída estruturada** (Pydantic), uma nota
   de relevância de 0.0 a 1.0 para cada chunk.
3. Os candidatos são reordenados por essa nota e cortados para
   `TOP_K_FINAL` antes de montar o contexto passado ao LLM de geração.

Isso captura relevância semântica/contextual real (ex.: sinônimos,
negações, relações entre entidades) que a fusão de rankings sozinha não
enxerga, ao custo de uma chamada extra ao LLM por pergunta. Pode ser
desativado (`RERANK_ENABLED=false`) para reduzir custo/latência, caso a
recuperação híbrida (RRF) já seja suficiente para o seu domínio.

### Metadados

Cada chunk carrega os metadados definidos no passo 6 da indexação:

- `document_name`: nome do arquivo PDF de origem.
- `page`: número da página (1-indexado).
- `date`: data de modificação do arquivo (proxy da data do documento).
- `origin`: pasta de onde o PDF foi lido.
- `chunk_id`: identificador sequencial único do chunk.

Esses metadados são devolvidos junto com a resposta final, permitindo
rastrear exatamente de onde veio cada trecho usado pelo LLM.

### Geração (RAG chain)

[app/generation/qa_chain.py](app/generation/qa_chain.py) implementa uma
cadeia LCEL simples: os chunks recuperados são formatados como contexto
(com citação de fonte e página) e passados a um `ChatPromptTemplate` que
instrui o LLM a responder **apenas** com base no contexto fornecido,
citando as fontes e admitindo quando não sabe a resposta.

---

## Limitações e próximos passos

- O reranking via LLM (`RERANK_ENABLED=true`) adiciona uma chamada extra
  ao LLM por pergunta (recuperação) e por avaliação de recuperação, o que
  aumenta custo e latência; desative-o (`RERANK_ENABLED=false`) se o
  ganho de precisão não justificar o custo no seu caso de uso, ou troque
  por um cross-encoder local (ex.: `BAAI/bge-reranker-base`) para reduzir
  a dependência de chamadas de API.
- As métricas de geração (Context Recall, Faithfulness, Answer Relevance)
  usam o próprio LLM como juiz (LLM-as-judge), o que introduz custo
  (chamadas extras de API) e alguma variabilidade entre execuções; 
- O relatório de avaliação (`data/eval/eval_report.json`) inclui a
  justificativa (`reasoning`) de cada julgamento, útil para auditar
  manualmente decisões inesperadas do LLM-juiz.
- O provedor de LLM atualmente suportado é a OpenAI; para uso 100% local,
  é possível estender `app/generation/qa_chain.py` para outros provedores
  compatíveis com LangChain (ex.: Ollama, llama.cpp).
- O índice léxico via hashing é uma aproximação didática de um índice
  léxico "clássico"; para produção, avaliar uma implementação BM25 real
  (ex.: `rank_bm25` ou busca full-text nativa) como alternativa/complemento.
