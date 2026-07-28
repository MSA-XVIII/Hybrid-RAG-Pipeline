"""The three retrieval modes, each returning a common RetrievalResult.

  * vector_retrieve     — DIS managed RAG (semantic passage lookup)
  * graph_retrieve      — Neptune relationship/keyword search over the KG
  * vectorless_retrieve — LLM reasons over the section-heading tree (PageIndex-style),
                          then pulls the chosen sections' text. This is the CRAG
                          corrective route (Ambiguous / Incorrect). No embeddings.
"""

from __future__ import annotations

import logging

from hybridrag import config
from hybridrag.common import RetrievalResult

logger = logging.getLogger(__name__)


def vector_retrieve(dis, query: str, collection: str, top_k: int | None = None) -> RetrievalResult:
    top_k = top_k or config.RETRIEVE_TOP_K
    passages = dis.retrieve(query, collection, top_k=top_k) or []
    text = "\n\n".join(p.get("text", "") for p in passages if p.get("text"))
    return RetrievalResult(source="vector", text=text, chunks=passages,
                           meta={"collection": collection, "k": top_k})


def graph_retrieve(graph, query: str, limit: int | None = None) -> RetrievalResult:
    limit = limit or config.RETRIEVE_TOP_K
    hits = graph.keyword_search(query, limit=limit) or []
    text = "\n\n".join(h.get("text", "") for h in hits if h.get("text"))
    return RetrievalResult(source="graph", text=text, chunks=hits,
                           meta={"limit": limit})


_NAV_SYSTEM = """You navigate a document by its table of contents. Given a question and a
list of section headings (each with an id), pick the section ids most likely to contain the
answer — reason about meaning, not keywords. Return ONLY JSON:
{"section_ids": ["<id>", "..."]}
Rules:
- Choose few, high-precision sections (usually 1-4). Empty list if none look relevant.
"""


def vectorless_retrieve(lmaas, graph, query: str, doc_ids: list[str] | None = None,
                        max_sections: int = 4) -> RetrievalResult:
    """Reasoning-based retrieval over the section tree (no embeddings).

    1) pull headings (tiny), 2) LLM picks relevant section ids, 3) fetch their text.
    """
    sections = graph.get_sections(doc_ids) or []
    if not sections:
        return RetrievalResult(source="vectorless", text="", chunks=[])

    toc = "\n".join(f"- {s['section_id']}: {s.get('heading', '')}" for s in sections)
    try:
        out = lmaas.complete_json(_NAV_SYSTEM, f"QUESTION: {query}\n\nHEADINGS:\n{toc}", cheap=True)
    except Exception as exc:  # noqa: BLE001 — fall back to no selection
        logger.warning("vectorless navigation failed (%s)", exc)
        out = {}
    chosen = (out.get("section_ids") if isinstance(out, dict) else None) or []
    chosen = chosen[:max_sections]
    if not chosen:
        return RetrievalResult(source="vectorless", text="", chunks=[],
                               meta={"navigated": True, "chosen": []})

    texts = graph.get_section_texts(chosen) or []
    chunks = []
    for t in texts:
        body = t.get("text")
        body = " ".join(body) if isinstance(body, list) else (body or "")
        chunks.append({"section_id": t.get("section_id"), "heading": t.get("heading"), "text": body})
    text = "\n\n".join(f"## {c['heading']}\n{c['text']}" for c in chunks if c["text"])
    return RetrievalResult(source="vectorless", text=text, chunks=chunks,
                           meta={"navigated": True, "chosen": chosen})
