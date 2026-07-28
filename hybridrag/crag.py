"""Agent 2 — the Corrective-RAG (CRAG) query loop, as a LangGraph StateGraph.

    route -> retrieve_primary -> evaluate --(conditional)-->
        correct   : refine(if noisy) -> generate                 (early exit)
        ambiguous : primary(k_in) + vectorless(k_ex) -> fuse -> generate
        incorrect : vectorless(k_ex)  [FULLY INTERNAL, no web] -> generate

Cost discipline: evaluator / navigation / refine run on the CHEAP model; only the
final generator uses the STRONG model. Refinement runs only when context is noisy.

The graph is built per-agent with the clients captured in closures, so it stays
dependency-injectable (real clients or offline fakes) and unit-testable. The
node functions (evaluate / refine / generate) are module-level and reusable.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from hybridrag import config
from hybridrag.catalog import RoutingCatalog
from hybridrag.common import QueryResult, RetrievalResult, Verdict
from hybridrag.retrieval import graph_retrieve, vector_retrieve, vectorless_retrieve

logger = logging.getLogger(__name__)

_RELATIONAL_HINTS = ("relationship", "related", "depend", "connect", "between",
                     "how does", "linked", "cause", "impact")

_EVAL_SYSTEM = """You judge whether retrieved context can answer a question. Return ONLY JSON:
{"score": <float 0..1>, "reason": "<short>"}
- 1.0 = fully answers; 0.0 = irrelevant/empty. Judge relevance & sufficiency, not style.
"""

_REFINE_SYSTEM = """You clean retrieved context for a question. Decompose the context into facts,
DROP anything irrelevant to the question, and recompose the kept facts into tight prose.
Return ONLY the cleaned text (no preamble). Keep only what helps answer the question.
"""

_GEN_SYSTEM = """You answer the question using ONLY the provided knowledge. Be accurate and concise.
If the knowledge is insufficient, say what is missing. Do not invent facts.
"""


# --- reusable node logic (pure functions) -----------------------------------

def evaluate(lmaas, query: str, result: RetrievalResult) -> Verdict:
    if result.is_empty():
        return Verdict(label="incorrect", score=0.0, reason="no context retrieved")
    try:
        out = lmaas.complete_json(
            _EVAL_SYSTEM, f"QUESTION: {query}\n\nCONTEXT:\n{result.text[:4000]}", cheap=True)
        score = float(out.get("score", 0.0)) if isinstance(out, dict) else 0.0
        reason = out.get("reason", "") if isinstance(out, dict) else ""
    except Exception as exc:  # noqa: BLE001 — evaluator failure -> ambiguous (safe)
        logger.warning("evaluator failed (%s)", exc)
        return Verdict(label="ambiguous", score=0.5, reason="evaluator error")
    score = max(0.0, min(1.0, score))
    if score >= config.CRAG_CORRECT_THRESHOLD:
        label = "correct"
    elif score < config.CRAG_INCORRECT_THRESHOLD:
        label = "incorrect"
    else:
        label = "ambiguous"
    return Verdict(label=label, score=score, reason=reason)


def refine(lmaas, query: str, text: str) -> tuple[str, bool]:
    """Returns (refined_text, called). Skips the LLM call for small context."""
    if len(text.split()) < config.REFINE_MIN_TOKENS:
        return text, False
    try:
        return lmaas.complete_text(_REFINE_SYSTEM, f"QUESTION: {query}\n\nCONTEXT:\n{text}",
                                   cheap=True, max_tokens=1024), True
    except Exception as exc:  # noqa: BLE001
        logger.warning("refine failed (%s); using raw context", exc)
        return text, False


def generate(lmaas, query: str, knowledge: str) -> str:
    return lmaas.complete_text(
        _GEN_SYSTEM, f"QUESTION: {query}\n\nKNOWLEDGE:\n{knowledge}", cheap=False, max_tokens=1024)


def _fuse(k_in: str, k_ex: str) -> str:
    return "\n\n---\n\n".join(p for p in (k_in, k_ex) if p and p.strip())


# --- LangGraph state ---------------------------------------------------------

class QueryState(TypedDict, total=False):
    query: str
    primary_source: str
    primary: RetrievalResult
    verdict: Verdict
    knowledge: str
    answer: str
    sources: list[str]
    calls: int
    trace: list[str]


class CRAGQueryAgent:
    """Agent 2 as a compiled LangGraph app. Inject real or fake clients."""

    def __init__(self, lmaas, dis, graph, catalog: RoutingCatalog | None = None,
                 collection: str | None = None) -> None:
        self.lmaas = lmaas
        self.dis = dis
        self.graph = graph
        self.catalog = catalog or RoutingCatalog()
        self.collection = collection or config.DIS_COLLECTION
        self.app = self._build_app()

    # -- retrieval helpers ---------------------------------------------------

    def _primary_source(self, query: str) -> str:
        cand = self.catalog.candidates(query)
        relational = any(h in query.lower() for h in _RELATIONAL_HINTS)
        if relational and cand.get("graph"):
            return "graph"
        if cand.get("vector"):
            return "vector"
        if cand.get("graph"):
            return "graph"
        return "vector"

    def _retrieve(self, source: str, query: str) -> RetrievalResult:
        if source == "graph":
            return graph_retrieve(self.graph, query)
        return vector_retrieve(self.dis, query, self.collection)

    def _vectorless(self, query: str) -> RetrievalResult:
        # scope navigation to the documents the catalog says are relevant, so the
        # LLM reads a focused table-of-contents instead of the whole corpus
        doc_ids = self.catalog.candidate_doc_ids(query)
        return vectorless_retrieve(self.lmaas, self.graph, query, doc_ids=doc_ids)

    # -- graph nodes ---------------------------------------------------------

    def _n_route(self, state: QueryState) -> QueryState:
        return {"primary_source": self._primary_source(state["query"]),
                "calls": 0, "trace": [], "sources": []}

    def _n_retrieve(self, state: QueryState) -> QueryState:
        pr = self._retrieve(state["primary_source"], state["query"])
        note = "empty" if pr.is_empty() else f"{len(pr.chunks)} chunks"
        return {"primary": pr,
                "trace": state["trace"] + [f"primary retrieve via {state['primary_source']}: {note}"]}

    def _n_evaluate(self, state: QueryState) -> QueryState:
        pr = state["primary"]
        verdict = evaluate(self.lmaas, state["query"], pr)
        calls = state.get("calls", 0) + (0 if pr.is_empty() else 1)
        return {"verdict": verdict, "calls": calls,
                "trace": state["trace"] + [f"evaluator: {verdict.label} (score={verdict.score:.2f})"]}

    def _route_after_eval(self, state: QueryState) -> str:
        return state["verdict"].label  # "correct" | "ambiguous" | "incorrect"

    def _n_correct(self, state: QueryState) -> QueryState:
        text, called = refine(self.lmaas, state["query"], state["primary"].text)
        return {"knowledge": text, "sources": [state["primary_source"]],
                "calls": state.get("calls", 0) + (1 if called else 0)}

    def _n_ambiguous(self, state: QueryState) -> QueryState:
        k_in, called = refine(self.lmaas, state["query"], state["primary"].text)
        k_ex = self._vectorless(state["query"])
        calls = state.get("calls", 0) + (1 if called else 0) + 1  # +1 vectorless nav
        note = "empty" if k_ex.is_empty() else f"{len(k_ex.chunks)} sections"
        sources = [state["primary_source"]] + (["vectorless"] if not k_ex.is_empty() else [])
        return {"knowledge": _fuse(k_in, k_ex.text), "sources": sources, "calls": calls,
                "trace": state["trace"] + [f"corrective (vectorless): {note}"]}

    def _n_incorrect(self, state: QueryState) -> QueryState:
        k_ex = self._vectorless(state["query"])
        calls = state.get("calls", 0) + 1
        trace = state["trace"] + [f"corrective (vectorless): "
                                  f"{'empty' if k_ex.is_empty() else str(len(k_ex.chunks)) + ' sections'}"]
        if k_ex.is_empty():
            other = "graph" if state["primary_source"] == "vector" else "vector"
            alt = self._retrieve(other, state["query"])
            trace.append(f"fallback to {other}: {'empty' if alt.is_empty() else str(len(alt.chunks)) + ' chunks'}")
            return {"knowledge": alt.text, "sources": [other] if not alt.is_empty() else [],
                    "calls": calls, "trace": trace}
        return {"knowledge": k_ex.text, "sources": ["vectorless"], "calls": calls, "trace": trace}

    def _n_generate(self, state: QueryState) -> QueryState:
        knowledge = state.get("knowledge", "") or ""
        if not knowledge.strip():
            return {"answer": "I couldn't find enough internal information to answer this confidently.",
                    "trace": state["trace"] + ["no knowledge assembled — abstaining"]}
        answer = generate(self.lmaas, state["query"], knowledge)
        return {"answer": answer, "calls": state.get("calls", 0) + 1}

    # -- graph assembly ------------------------------------------------------

    def _build_app(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(QueryState)
        g.add_node("route", self._n_route)
        g.add_node("retrieve", self._n_retrieve)
        g.add_node("evaluate", self._n_evaluate)
        g.add_node("correct", self._n_correct)
        g.add_node("ambiguous", self._n_ambiguous)
        g.add_node("incorrect", self._n_incorrect)
        g.add_node("generate", self._n_generate)

        g.add_edge(START, "route")
        g.add_edge("route", "retrieve")
        g.add_edge("retrieve", "evaluate")
        g.add_conditional_edges("evaluate", self._route_after_eval,
                                {"correct": "correct", "ambiguous": "ambiguous",
                                 "incorrect": "incorrect"})
        for branch in ("correct", "ambiguous", "incorrect"):
            g.add_edge(branch, "generate")
        g.add_edge("generate", END)
        return g.compile()

    # -- public API (unchanged) ----------------------------------------------

    def run(self, query: str) -> QueryResult:
        final = self.app.invoke({"query": query})
        verdict = final.get("verdict") or Verdict(label="incorrect", score=0.0)
        return QueryResult(query=query, answer=final.get("answer", ""), verdict=verdict,
                           sources=final.get("sources", []), llm_calls=final.get("calls", 0),
                           trace=final.get("trace", []))
