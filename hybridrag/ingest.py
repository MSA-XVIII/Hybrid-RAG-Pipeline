"""Agent 1 — the ingest router.

Given a document, decide how to store it:
  * vector (DIS)  — ALWAYS the baseline
  * graph (Neptune) — ADDITIONALLY when the doc has enough DISTINCT entities to
    make relationships worthwhile.

The graph decision is made by scanning the WHOLE document (windowed, cheap model),
counting distinct entities, and comparing to INGEST_MIN_GRAPH_ENTITIES — not by
eyeballing a 2000-char sample. Then it writes the routing catalog so Agent 2 can
route queries without guessing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from hybridrag import config
from hybridrag.catalog import RoutingCatalog
from hybridrag.common import CatalogEntry

logger = logging.getLogger(__name__)

_DOMAIN_SYSTEM = """Assign a short domain label to the document. Return ONLY JSON:
{"domain": "<short label, e.g. service-manual, policy, faq, spec>"}
"""

_ENTITY_SYSTEM = """List every DISTINCT salient entity in the text — named components, features,
tools, procedures, parameters, error codes, roles. Return ONLY JSON:
{"entities": ["<canonical name>", "..."]}
Rules:
- Canonical names only; do NOT list the same entity twice.
- Ignore document furniture (page numbers, headers/footers, generic filler).
- Return an empty list if the text names nothing of substance.
"""


# --- document structuring (self-contained stand-in for pdf-to-kg-agent) ------

# Pure document furniture (footer codes / bare page numbers) — never a heading
# or chunk. Mirrors loader._is_boilerplate so .md/.txt get the same cleanup.
_BOILER_RE = re.compile(
    r"^(?:\d{1,4}|\d{3,}[A-Za-z]?\s+EN\s+\d{4,}|\d{3,}[A-Za-z]?|page\s+\d+)$", re.I)


def _is_boilerplate(line: str) -> bool:
    return bool(_BOILER_RE.match(line.strip()))


def _is_heading(line: str) -> bool:
    """Trust only strong heading signals: a markdown ``#`` (native to .md, and
    emitted by the PDF loader from font size) or dotted multi-level numbering
    (``2.4.3 Foo``). Deliberately NOT single-number list steps (``1 Do X``) or
    short ALL-CAPS lines (UI labels) — those flooded the old tree with noise.
    """
    s = line.strip()
    if not s:
        return False
    return s.startswith("#") or bool(re.match(r"^\d+\.\d+", s))


def _naive_sections(text: str) -> list[dict]:
    """Split structured text into a section tree (heading + chunks).

    Relies on :func:`_is_heading`; boilerplate lines are dropped outright. A
    heading with no body (e.g. a chapter title page) is kept as a structural
    node, but a body-less leading "Introduction" placeholder is discarded.
    """
    sections: list[dict] = []
    cur = {"order": 0, "heading": "Introduction", "body": []}
    for raw in text.splitlines():
        s = raw.strip()
        if not s or _is_boilerplate(s):
            continue
        if _is_heading(s):
            if cur["body"] or cur["heading"] != "Introduction":
                sections.append(cur)
            heading = re.sub(r"^#{1,6}\s*", "", s).strip()
            cur = {"order": len(sections), "heading": heading, "body": []}
        else:
            cur["body"].append(s)
    if cur["body"] or cur["heading"] != "Introduction":
        sections.append(cur)
    for s in sections:
        s["chunks"] = _chunk(" ".join(s.pop("body")))
    return sections or [{"order": 0, "heading": "Document", "chunks": _chunk(text)}]


def _entity_mentions(sections: list[dict], entities: list[str],
                     max_per_entity: int = 12) -> list[dict]:
    """Link each extracted entity to the sections whose text mentions it
    (case-insensitive substring). An entity that appears in several sections
    becomes a shared hub node — this is what interconnects the graph. Entities
    that match no section text are skipped (no orphan nodes).
    """
    haystacks = [(s.get("order", 0),
                  (s.get("heading", "") + " " + " ".join(s.get("chunks", []))).lower())
                 for s in sections]
    mentions: list[dict] = []
    for ent in entities or []:
        e = str(ent).strip()
        if len(e) < 3:
            continue
        el = e.lower()
        hits = 0
        for order, text in haystacks:
            if el in text:
                mentions.append({"order": order, "entity": e})
                hits += 1
                if hits >= max_per_entity:
                    break
    return mentions


def _chunk(text: str, target_words: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + target_words]) for i in range(0, len(words), target_words)]


# --- full-document entity scan ----------------------------------------------

def _windows(text: str, window_words: int, max_windows: int) -> list[str]:
    """Split the FULL text into word-windows; if there are more than max_windows,
    sample them EVENLY across the document so coverage stays whole-document while
    the number of LLM calls stays bounded."""
    words = text.split()
    if not words:
        return []
    wins = [" ".join(words[i:i + window_words]) for i in range(0, len(words), window_words)]
    if len(wins) <= max_windows:
        return wins
    idxs = sorted({round(i * (len(wins) - 1) / (max_windows - 1)) for i in range(max_windows)})
    return [wins[i] for i in idxs]


def _norm_entity(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


class IngestState(TypedDict, total=False):
    doc_id: str
    title: str
    text: str
    assessment: dict
    in_graph: bool
    entry: CatalogEntry


class IngestAgent:
    """Agent 1 as a compiled LangGraph app. Inject real or fake clients."""

    def __init__(self, lmaas, dis, graph, catalog: RoutingCatalog | None = None,
                 collection: str | None = None, min_graph_entities: int | None = None) -> None:
        self.lmaas = lmaas
        self.dis = dis
        self.graph = graph
        self.catalog = catalog or RoutingCatalog()
        self.collection = collection or config.DIS_COLLECTION
        self.min_graph_entities = (config.INGEST_MIN_GRAPH_ENTITIES
                                   if min_graph_entities is None else min_graph_entities)
        self.app = self._build_app()

    def _domain(self, title: str, text: str) -> str:
        try:
            out = self.lmaas.complete_json(_DOMAIN_SYSTEM, f"TITLE: {title}\n\n{text[:1500]}", cheap=True)
            return out.get("domain", "unknown") if isinstance(out, dict) else "unknown"
        except Exception as exc:  # noqa: BLE001
            logger.warning("domain classify failed (%s)", exc)
            return "unknown"

    def assess_document(self, title: str, text: str) -> dict:
        """Scan the WHOLE document and decide graph-worthiness by distinct-entity count.

        Returns {domain, entities, entity_count, word_count, density, build_graph}.
        """
        windows = _windows(text, config.INGEST_ENTITY_SCAN_WINDOW_WORDS,
                           config.INGEST_ENTITY_SCAN_MAX_WINDOWS)
        seen: dict[str, str] = {}   # normalized -> display form (dedupe across windows)
        for w in windows:
            try:
                out = self.lmaas.complete_json(_ENTITY_SYSTEM, w, cheap=True)
            except Exception as exc:  # noqa: BLE001 — a bad window shouldn't abort ingest
                logger.warning("entity scan window failed (%s)", exc)
                continue
            for e in (out.get("entities") if isinstance(out, dict) else []) or []:
                key = _norm_entity(e)
                if key and key not in seen:
                    seen[key] = str(e).strip()

        entities = list(seen.values())
        word_count = len(text.split())
        entity_count = len(entities)
        density = round(entity_count / max(1.0, word_count / 1000.0), 2)  # entities per 1k words
        build_graph = entity_count >= self.min_graph_entities
        logger.info("assessed %s: %d distinct entities across %d words (density %.2f/1k) -> graph=%s",
                    title, entity_count, word_count, density, build_graph)
        return {"domain": self._domain(title, text), "entities": entities,
                "entity_count": entity_count, "word_count": word_count,
                "density": density, "build_graph": build_graph}

    # -- graph nodes ---------------------------------------------------------

    def _n_assess(self, state: IngestState) -> IngestState:
        return {"assessment": self.assess_document(state["title"], state["text"])}

    def _route_store(self, state: IngestState) -> str:
        # entity-dense -> graph path; otherwise -> vector path
        return "graph" if state["assessment"]["build_graph"] else "vector"

    def _upload_vector(self, state: IngestState) -> None:
        self.dis.upload_document(self.collection, state["doc_id"], state["text"],
                                 metadata={"title": state["title"],
                                           "domain": state["assessment"]["domain"]})

    def _n_store_vector(self, state: IngestState) -> IngestState:
        self._upload_vector(state)  # side effect; flags computed in write_catalog
        return {}

    def _n_store_graph(self, state: IngestState) -> IngestState:
        sections = _naive_sections(state["text"])
        self.graph.upsert_document_tree(state["doc_id"], state["title"], sections)
        # Link the entities found during assessment to the sections that mention
        # them, so the graph gets Entity nodes + cross-references (interconnections)
        # instead of a bare Document->Section->Chunk star.
        mentions = _entity_mentions(sections, state["assessment"]["entities"])
        if mentions:
            self.graph.upsert_entities(state["doc_id"], mentions)
        # entity-dense docs are graph-only by default; optionally dual-store.
        if config.STORE_GRAPH_DOCS_IN_VECTOR:
            self._upload_vector(state)
        return {}

    def _n_write_catalog(self, state: IngestState) -> IngestState:
        a = state["assessment"]
        # Flags are a deterministic function of the routing decision + config,
        # so compute them here rather than relying on cross-node state merges.
        build_graph = a["build_graph"]
        in_graph = build_graph
        in_vector = (not build_graph) or config.STORE_GRAPH_DOCS_IN_VECTOR
        entry = CatalogEntry(
            doc_id=state["doc_id"], title=state["title"], in_vector=in_vector, in_graph=in_graph,
            collection=self.collection, domain=a["domain"],
            entities=a["entities"][:40],
            section_tree_ref="neptune" if in_graph else "",
            meta={"entity_count": a["entity_count"], "density": a["density"],
                  "word_count": a["word_count"], "min_graph_entities": self.min_graph_entities},
        )
        self.catalog.put(entry)
        logger.info("ingested %s -> vector=%s graph=%s (%s, %d entities)",
                    state["doc_id"], in_vector, in_graph, a["domain"], a["entity_count"])
        return {"entry": entry}

    # -- graph assembly ------------------------------------------------------

    def _build_app(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(IngestState)
        g.add_node("assess", self._n_assess)
        g.add_node("store_vector", self._n_store_vector)
        g.add_node("store_graph", self._n_store_graph)
        g.add_node("write_catalog", self._n_write_catalog)

        g.add_edge(START, "assess")
        # the router: entity-dense docs take the graph path, others the vector path
        g.add_conditional_edges("assess", self._route_store,
                                {"graph": "store_graph", "vector": "store_vector"})
        g.add_edge("store_graph", "write_catalog")
        g.add_edge("store_vector", "write_catalog")
        g.add_edge("write_catalog", END)
        return g.compile()

    # -- public API (unchanged) ----------------------------------------------

    def ingest(self, doc_id: str, text: str, title: str = "") -> CatalogEntry:
        final = self.app.invoke({"doc_id": doc_id, "title": title or doc_id, "text": text})
        return final["entry"]

    def ingest_documents(self, docs: list[dict]) -> list[CatalogEntry]:
        """Ingest many documents in one call.

        ``docs`` are loader-style records: ``{"doc_id", "text", "title"}``
        (``title`` optional). Returns the list of :class:`CatalogEntry`.
        """
        entries = [self.ingest(d["doc_id"], d["text"], title=d.get("title", ""))
                   for d in docs]
        n_graph = sum(e.in_graph for e in entries)
        logger.info("ingested %d document(s): %d -> graph, %d -> vector",
                    len(entries), n_graph, len(entries) - n_graph)
        return entries
