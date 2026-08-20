"""Avaliação do modelo e da arquitetura escolhida.

Etapa de "Avaliação" do projeto: mede quantitativamente a qualidade de
duas partes do pipeline, usando um dataset de perguntas com respostas e
documento esperados (`data/eval/qa_dataset.json`):

1. Recuperação (retrieval) — compara as estratégias léxica, semântica,
   híbrida (RRF) e híbrida com reranking via LLM, usando métricas
   clássicas de ranking:
   - **Precision@K**: proporção de documentos relevantes entre os
     primeiros K resultados recuperados.
   - **Recall@K**: proporção dos documentos relevantes existentes (no
     índice) que foram efetivamente recuperados entre os top-K.
   - **NDCG@K**: qualidade da ordenação dos documentos relevantes,
     penalizando resultados relevantes que aparecem em posições piores.

2. Geração (generation) — usa o próprio LLM como "juiz" (LLM-as-judge)
   para avaliar o pipeline RAG completo:
   - **Context Recall**: verifica se o contexto recuperado contém as
     informações necessárias para responder à pergunta (comparando com a
     resposta de referência/gabarito).
   - **Faithfulness / Groundedness**: verifica se as afirmações da
     resposta gerada estão fundamentadas no contexto recuperado,
     reduzindo o risco de alucinações.
   - **Answer Relevance**: avalia se a resposta gerada realmente responde
     de forma direta e completa à pergunta feita.

O resultado é impresso no console e salvo em `data/eval/eval_report.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.generation.qa_chain import RAGChain, _get_llm
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import RerankingHybridRetriever

RetrieveFn = Callable[[str], List[Document]]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset de avaliação não encontrado em '{path}'. "
            "Crie o arquivo com perguntas, documento e resposta esperados "
            "(veja data/eval/qa_dataset.json como exemplo)."
        )
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Métricas de recuperação: Precision@K, Recall@K, NDCG@K
# ---------------------------------------------------------------------------

def _relevance_vector(docs: List[Document], expected_document: str) -> List[int]:
    """Vetor binário de relevância na ordem em que os documentos foram
    recuperados: 1 se o chunk pertence ao documento esperado, 0 caso
    contrário."""
    return [1 if doc.metadata.get("document_name") == expected_document else 0 for doc in docs]


def _count_total_relevant(retriever: HybridRetriever, expected_document: str) -> int:
    """Conta quantos chunks do documento esperado existem de fato no
    índice (usado como denominador do Recall@K). Como léxico e semântico
    indexam os mesmos chunks, basta consultar uma das duas coleções."""
    try:
        result = retriever.semantic_store.get(where={"document_name": expected_document})
        return len(result.get("ids", []))
    except Exception:
        return 0


def _precision_at_k(relevances: List[int], k: int) -> float:
    top_k = relevances[:k]
    return sum(top_k) / len(top_k) if top_k else 0.0


def _recall_at_k(relevances: List[int], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    top_k = relevances[:k]
    return min(sum(top_k) / total_relevant, 1.0)


def _ndcg_at_k(relevances: List[int], k: int) -> float:
    top_k = relevances[:k]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(top_k))

    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    dataset_path: Optional[str] = None,
    k: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Avalia e compara as quatro estratégias de recuperação disponíveis
    (léxica, semântica, híbrida e híbrida com reranking via LLM) usando
    Precision@K, Recall@K e NDCG@K."""
    dataset = _load_dataset(dataset_path or settings.eval_dataset_path)
    k = k or settings.eval_top_k
    retriever = HybridRetriever()
    reranking_retriever = RerankingHybridRetriever(retriever, _get_llm())

    strategies: Dict[str, RetrieveFn] = {
        "lexical": lambda q: retriever.retrieve_lexical(q, k=k),
        "semantic": lambda q: retriever.retrieve_semantic(q, k=k),
        "hybrid": lambda q: retriever.retrieve(q, k=k),
        "hybrid_reranked": lambda q: reranking_retriever.retrieve(q, k=k),
    }

    accumulators: Dict[str, Dict[str, List[float]]] = {
        name: {"precision": [], "recall": [], "ndcg": []} for name in strategies
    }

    for item in dataset:
        expected_document = item["expected_document"]
        total_relevant = _count_total_relevant(retriever, expected_document)

        for name, retrieve_fn in strategies.items():
            docs = retrieve_fn(item["question"])
            relevances = _relevance_vector(docs, expected_document)

            accumulators[name]["precision"].append(_precision_at_k(relevances, k))
            accumulators[name]["recall"].append(_recall_at_k(relevances, k, total_relevant))
            accumulators[name]["ndcg"].append(_ndcg_at_k(relevances, k))

    total = len(dataset) or 1
    return {
        name: {
            f"precision@{k}": sum(values["precision"]) / total,
            f"recall@{k}": sum(values["recall"]) / total,
            f"ndcg@{k}": sum(values["ndcg"]) / total,
        }
        for name, values in accumulators.items()
    }


# ---------------------------------------------------------------------------
# Métricas de geração via LLM-as-judge: Context Recall, Faithfulness,
# Answer Relevance
# ---------------------------------------------------------------------------

class ContextRecallJudgement(BaseModel):
    """Julgamento de Context Recall: o contexto recuperado contém as
    informações necessárias para responder à pergunta?"""

    total_statements: int = Field(
        description="Número de afirmações factuais atômicas identificadas na resposta de referência (gabarito)."
    )
    attributed_statements: int = Field(
        description="Quantas dessas afirmações podem ser confirmadas/encontradas no contexto recuperado."
    )
    reasoning: str = Field(description="Justificativa breve e objetiva do julgamento.")


class FaithfulnessJudgement(BaseModel):
    """Julgamento de Faithfulness/Groundedness: as afirmações da resposta
    gerada estão fundamentadas no contexto recuperado?"""

    total_claims: int = Field(
        description="Número de afirmações factuais atômicas identificadas na resposta gerada."
    )
    supported_claims: int = Field(
        description="Quantas dessas afirmações estão de fato fundamentadas no contexto recuperado."
    )
    reasoning: str = Field(description="Justificativa breve e objetiva do julgamento.")


class AnswerRelevanceJudgement(BaseModel):
    """Julgamento de Answer Relevance: o quanto a resposta gerada
    realmente responde à pergunta feita."""

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "0.0 = não responde à pergunta / foge do assunto; "
            "1.0 = responde de forma direta e completa à pergunta."
        ),
    )
    reasoning: str = Field(description="Justificativa breve e objetiva do julgamento.")


_JUDGE_SYSTEM_PROMPT = (
    "Você é um avaliador criterioso e imparcial de sistemas de RAG "
    "(Retrieval-Augmented Generation). Baseie-se estritamente nos textos "
    "fornecidos, sem usar conhecimento externo, e responda no formato "
    "estruturado solicitado."
)

_CONTEXT_RECALL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "Tarefa: avaliar CONTEXT RECALL.\n"
            "Decomponha a RESPOSTA DE REFERÊNCIA em afirmações factuais atômicas. "
            "Para cada afirmação, verifique se ela pode ser confirmada/encontrada "
            "no CONTEXTO RECUPERADO. Informe o total de afirmações e quantas delas "
            "são sustentadas pelo contexto.\n\n"
            "PERGUNTA:\n{question}\n\n"
            "RESPOSTA DE REFERÊNCIA (gabarito):\n{reference_answer}\n\n"
            "CONTEXTO RECUPERADO:\n{context}",
        ),
    ]
)

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "Tarefa: avaliar FAITHFULNESS/GROUNDEDNESS.\n"
            "Decomponha a RESPOSTA GERADA em afirmações factuais atômicas. Para cada "
            "afirmação, verifique se ela é sustentada pelo CONTEXTO RECUPERADO "
            "(evidência de que não é uma alucinação). Informe o total de afirmações "
            "e quantas delas estão de fato fundamentadas no contexto.\n\n"
            "PERGUNTA:\n{question}\n\n"
            "RESPOSTA GERADA:\n{generated_answer}\n\n"
            "CONTEXTO RECUPERADO:\n{context}",
        ),
    ]
)

_ANSWER_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "Tarefa: avaliar ANSWER RELEVANCE.\n"
            "Julgue o quanto a RESPOSTA GERADA responde direta e completamente à "
            "PERGUNTA, independentemente de estar correta ou fundamentada em algum "
            "contexto. Atribua uma nota de 0.0 a 1.0.\n\n"
            "PERGUNTA:\n{question}\n\n"
            "RESPOSTA GERADA:\n{generated_answer}",
        ),
    ]
)


def _context_recall_score(
    judge_llm: BaseChatModel, question: str, reference_answer: str, context: str
) -> Tuple[float, str]:
    chain = _CONTEXT_RECALL_PROMPT | judge_llm.with_structured_output(ContextRecallJudgement)
    result: ContextRecallJudgement = chain.invoke(
        {"question": question, "reference_answer": reference_answer, "context": context}
    )
    if result.total_statements <= 0:
        return 0.0, result.reasoning
    score = min(result.attributed_statements / result.total_statements, 1.0)
    return score, result.reasoning


def _faithfulness_score(
    judge_llm: BaseChatModel, question: str, generated_answer: str, context: str
) -> Tuple[float, str]:
    chain = _FAITHFULNESS_PROMPT | judge_llm.with_structured_output(FaithfulnessJudgement)
    result: FaithfulnessJudgement = chain.invoke(
        {"question": question, "generated_answer": generated_answer, "context": context}
    )
    if result.total_claims <= 0:
        return 0.0, result.reasoning
    score = min(result.supported_claims / result.total_claims, 1.0)
    return score, result.reasoning


def _answer_relevance_score(
    judge_llm: BaseChatModel, question: str, generated_answer: str
) -> Tuple[float, str]:
    chain = _ANSWER_RELEVANCE_PROMPT | judge_llm.with_structured_output(AnswerRelevanceJudgement)
    result: AnswerRelevanceJudgement = chain.invoke(
        {"question": question, "generated_answer": generated_answer}
    )
    return result.relevance_score, result.reasoning


def evaluate_generation(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """Avalia a qualidade do pipeline RAG completo (recuperação + geração)
    usando o próprio LLM como juiz para Context Recall, Faithfulness e
    Answer Relevance."""
    dataset = _load_dataset(dataset_path or settings.eval_dataset_path)
    rag = RAGChain()
    judge_llm = _get_llm()

    context_recall_scores: List[float] = []
    faithfulness_scores: List[float] = []
    answer_relevance_scores: List[float] = []
    details: List[Dict[str, Any]] = []

    for item in dataset:
        question = item["question"]
        reference_answer = item.get("expected_answer", "")

        result = rag.answer(question)
        generated_answer = result["answer"]
        context = result["context"]

        context_recall, cr_reasoning = _context_recall_score(
            judge_llm, question, reference_answer, context
        )
        faithfulness, f_reasoning = _faithfulness_score(
            judge_llm, question, generated_answer, context
        )
        answer_relevance, ar_reasoning = _answer_relevance_score(
            judge_llm, question, generated_answer
        )

        context_recall_scores.append(context_recall)
        faithfulness_scores.append(faithfulness)
        answer_relevance_scores.append(answer_relevance)

        details.append(
            {
                "question": question,
                "generated_answer": generated_answer,
                "expected_answer": reference_answer,
                "context_recall": round(context_recall, 3),
                "faithfulness": round(faithfulness, 3),
                "answer_relevance": round(answer_relevance, 3),
                "reasoning": {
                    "context_recall": cr_reasoning,
                    "faithfulness": f_reasoning,
                    "answer_relevance": ar_reasoning,
                },
                "sources": result["sources"],
            }
        )

    total = len(dataset) or 1
    return {
        "average_context_recall": sum(context_recall_scores) / total,
        "average_faithfulness": sum(faithfulness_scores) / total,
        "average_answer_relevance": sum(answer_relevance_scores) / total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Execução completa
# ---------------------------------------------------------------------------

def run_full_evaluation(dataset_path: Optional[str] = None) -> None:
    """Executa a avaliação completa (recuperação + geração) e salva um
    relatório em disco."""
    dataset_path = dataset_path or settings.eval_dataset_path
    k = settings.eval_top_k

    print(f"Avaliando qualidade da recuperação (retrieval) @ K={k}...")
    retrieval_report = evaluate_retrieval(dataset_path, k=k)
    for strategy, metrics in retrieval_report.items():
        print(
            f"  {strategy:16s} | "
            f"precision@{k}={metrics[f'precision@{k}']:.2f} | "
            f"recall@{k}={metrics[f'recall@{k}']:.2f} | "
            f"ndcg@{k}={metrics[f'ndcg@{k}']:.2f}"
        )

    print(f"\nAvaliando qualidade da geração (RAG) com LLM-as-judge (RERANK_ENABLED={settings.rerank_enabled})...")
    generation_report = evaluate_generation(dataset_path)
    print(f"  Context Recall médio  : {generation_report['average_context_recall']:.2f}")
    print(f"  Faithfulness médio    : {generation_report['average_faithfulness']:.2f}")
    print(f"  Answer Relevance médio: {generation_report['average_answer_relevance']:.2f}")

    output_path = Path(settings.eval_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"retrieval": retrieval_report, "generation": generation_report},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nRelatório completo salvo em: {output_path}")
