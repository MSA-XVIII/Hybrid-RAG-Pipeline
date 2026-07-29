"""LMaaS-driven knowledge-graph extraction.

Unlike Agent 1's ingest gate (which asks LMaaS only for a FLAT entity list and
builds a Document->Section->Chunk skeleton by regex), this module has LMaaS build
the actual knowledge graph: it extracts subject-relation-object TRIPLES with typed
endpoints, so the graph gets

  * typed nodes  — each entity carries an LLM-assigned label (Component, ErrorCode,
                   Procedure, Role, System, ...),
  * typed edges  — real entity->entity relationships (CAUSES, PART_OF, RESOLVED_BY,
                   PERFORMS, ...), not just structural HAS_SECTION/MENTIONS links.

Entities merge by name-derived id (see common.entity_id), so the same entity named
across windows/documents collapses to ONE node — that shared node is what turns the
extracted triples into a connected graph.

Only stdlib + hybridrag.common are imported, so importing this module is free; the
LMaaS calls happen when you run an extractor with a real (or fake) client.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from hybridrag.common import entity_id

_PAGEINDEX_SYSTEM = """You are building a hierarchical PageIndex of a document — the
table-of-contents-like tree a navigator uses to find where an answer lives.
Read the document and return ONLY JSON of this exact shape:
{"title": "<document title>",
 "children": [
   {"title": "<section title>",
    "summary": "<one short sentence: what this section covers>",
    "children": [ ...same shape, nested... ]}
 ]}
Rules:
- Reflect the document's ACTUAL structure and nesting (chapters -> sub-sections).
- Titles are concise; summaries are one factual sentence.
- Include every substantive section; omit page furniture (page numbers, footers).
- Nest as deep as the document warrants; leaf sections have "children": [].
"""

_TRIPLES_SYSTEM = """You are building a knowledge graph from a technical document.
Extract the factual relationships it states as subject-relation-object triples.
Return ONLY JSON of this exact shape:
{"triples": [
  {"subject": "<entity>", "subject_type": "<Type>",
   "relation": "<RELATION>",
   "object": "<entity>", "object_type": "<Type>"}
]}
Rules:
- Entities are concrete named things: components, error codes, procedures, roles,
  systems, parameters. Use canonical names (e.g. "Error Code E-204", "Collimator").
- Types are short PascalCase labels you choose, e.g. Component, ErrorCode,
  Procedure, Role, System, Parameter.
- Relations are short UPPER_SNAKE_CASE verbs, e.g. CAUSES, INDICATES, PART_OF,
  RESOLVED_BY, PERFORMS, RUNS_ON, REQUIRES, POSITIONS.
- Only include relationships the text explicitly supports. Do NOT invent facts.
- Ignore document furniture (page numbers, bare headings, generic filler).
- Return an empty list if the text asserts no concrete relationships.
"""


def _windows(text: str, window_words: int, max_windows: int) -> list[str]:
    """Split text into word-windows; if there are more than max_windows, sample
    them EVENLY so coverage stays whole-document while LLM calls stay bounded."""
    words = text.split()
    if not words:
        return []
    wins = [" ".join(words[i:i + window_words]) for i in range(0, len(words), window_words)]
    if len(wins) <= max_windows:
        return wins
    idxs = sorted({round(i * (len(wins) - 1) / (max_windows - 1)) for i in range(max_windows)})
    return [wins[i] for i in idxs]


def _clean(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass
class KnowledgeGraph:
    """Typed entity graph assembled from LMaaS triples.

    nodes: entity_id -> {"id", "name", "type", "docs": set[str]}
    edges: (subject_id, relation, object_id) -> {"source", "target", "relation",
                                                 "docs": set[str], "count": int}
    """

    nodes: dict[str, dict] = field(default_factory=dict)
    edges: dict[tuple, dict] = field(default_factory=dict)
    _types: dict[str, Counter] = field(default_factory=dict)  # entity_id -> type votes

    def add_triple(self, t: dict, doc_id: str = "") -> bool:
        s, sty = _clean(t.get("subject")), _clean(t.get("subject_type")) or "Entity"
        o, oty = _clean(t.get("object")), _clean(t.get("object_type")) or "Entity"
        rel = _clean(t.get("relation")).upper().replace(" ", "_")
        if not s or not o or not rel or len(s) < 2 or len(o) < 2:
            return False
        sid, oid = entity_id(s), entity_id(o)
        if sid == oid:            # self-loop from a name collision — skip
            return False
        self._node(sid, s, sty, doc_id)
        self._node(oid, o, oty, doc_id)
        key = (sid, rel, oid)
        edge = self.edges.get(key)
        if edge is None:
            self.edges[key] = {"source": sid, "target": oid, "relation": rel,
                               "docs": {doc_id} if doc_id else set(), "count": 1}
        else:
            edge["count"] += 1
            if doc_id:
                edge["docs"].add(doc_id)
        return True

    def _node(self, eid: str, name: str, etype: str, doc_id: str) -> None:
        node = self.nodes.get(eid)
        if node is None:
            node = {"id": eid, "name": name, "type": etype, "docs": set()}
            self.nodes[eid] = node
            self._types[eid] = Counter()
        self._types[eid][etype] += 1
        # keep the most-voted type as the node label
        node["type"] = self._types[eid].most_common(1)[0][0]
        if doc_id:
            node["docs"].add(doc_id)

    # -- read helpers --------------------------------------------------------

    def triples(self) -> list[dict]:
        """Flat [{subject, subject_type, relation, object, object_type}] view."""
        out = []
        for (sid, rel, oid), e in self.edges.items():
            out.append({"subject": self.nodes[sid]["name"], "subject_type": self.nodes[sid]["type"],
                        "relation": rel,
                        "object": self.nodes[oid]["name"], "object_type": self.nodes[oid]["type"]})
        return out

    def type_counts(self) -> dict[str, int]:
        c: Counter = Counter(n["type"] for n in self.nodes.values())
        return dict(c.most_common())

    def to_store(self) -> tuple[list[dict], list[dict]]:
        """(nodes, edges) as plain JSON-safe dicts, for a graph store writer.

        nodes = [{id, name, type, docs}]  edges = [{source, target, relation, count, docs}]
        """
        nodes = [{"id": n["id"], "name": n["name"], "type": n["type"],
                  "docs": sorted(n["docs"])} for n in self.nodes.values()]
        edges = [{"source": e["source"], "target": e["target"], "relation": e["relation"],
                  "count": e["count"], "docs": sorted(e["docs"])} for e in self.edges.values()]
        return nodes, edges

    def to_vis(self) -> tuple[list[dict], list[dict]]:
        """(nodes, edges) shaped for visualize.entity_kg_html / vis-network."""
        nodes = [{"id": n["id"], "label": n["name"], "group": n["type"],
                  "title": f"{n['type']} · {len(n['docs'])} doc(s)"} for n in self.nodes.values()]
        edges = [{"from": e["source"], "to": e["target"], "label": e["relation"]}
                 for e in self.edges.values()]
        return nodes, edges


class KGExtractor:
    """Run LMaaS over documents and assemble a typed KnowledgeGraph.

    Inject a real ``LMaaSClient`` (or any object exposing ``complete_json``). Uses
    the STRONG deployment by default — relationship extraction rewards quality; set
    ``cheap=True`` to use the mini deployment.
    """

    def __init__(self, lmaas, *, window_words: int = 350, max_windows: int = 6,
                 cheap: bool = False) -> None:
        self.lmaas = lmaas
        self.window_words = window_words
        self.max_windows = max_windows
        self.cheap = cheap

    def extract_triples(self, text: str) -> list[dict]:
        """Windowed triple extraction over one document's text."""
        triples: list[dict] = []
        for w in _windows(text, self.window_words, self.max_windows):
            try:
                out = self.lmaas.complete_json(_TRIPLES_SYSTEM, w, cheap=self.cheap)
            except Exception:  # noqa: BLE001 — a bad window shouldn't abort the doc
                continue
            got = out.get("triples") if isinstance(out, dict) else out
            if isinstance(got, list):
                triples.extend(t for t in got if isinstance(t, dict))
        return triples

    def build(self, docs: list[dict], *, on_doc=None) -> KnowledgeGraph:
        """Build one merged graph across ``docs`` ({doc_id, text, title}).

        ``on_doc(doc_id, n_triples)`` is called after each document for progress.
        """
        kg = KnowledgeGraph()
        for d in docs:
            triples = self.extract_triples(d["text"])
            added = sum(kg.add_triple(t, doc_id=d["doc_id"]) for t in triples)
            if on_doc:
                on_doc(d["doc_id"], added)
        return kg


def _count_nodes(tree: dict) -> int:
    return 1 + sum(_count_nodes(c) for c in (tree.get("children") or []))


class PageIndexExtractor:
    """Ask LMaaS for a document's hierarchical PageIndex (the TOC-like tree the
    vectorless navigator reasons over) — LLM-built nesting, not regex heading
    numbers. Inject a real ``LMaaSClient`` (or any object exposing ``complete_json``).
    """

    def __init__(self, lmaas, *, max_words: int = 6000, cheap: bool = False) -> None:
        self.lmaas = lmaas
        self.max_words = max_words
        self.cheap = cheap

    def build_one(self, doc: dict) -> dict:
        """Return a nested {title, children:[{title, summary, children}]} tree for
        one document ({doc_id, text, title}). On failure returns an empty tree."""
        text = " ".join(doc["text"].split()[:self.max_words])
        user = f"TITLE: {doc.get('title') or doc['doc_id']}\n\n{text}"
        try:
            out = self.lmaas.complete_json(_PAGEINDEX_SYSTEM, user, cheap=self.cheap)
        except Exception as exc:  # noqa: BLE001
            out = {"title": doc.get("title") or doc["doc_id"], "children": [], "error": str(exc)}
        if not isinstance(out, dict):
            out = {"title": doc.get("title") or doc["doc_id"], "children": []}
        out.setdefault("title", doc.get("title") or doc["doc_id"])
        out.setdefault("children", [])
        out["doc_id"] = doc["doc_id"]
        return out

    def build(self, docs: list[dict], *, on_doc=None) -> dict[str, dict]:
        """Build a PageIndex per document; returns {doc_id: tree}."""
        trees: dict[str, dict] = {}
        for d in docs:
            tree = self.build_one(d)
            trees[d["doc_id"]] = tree
            if on_doc:
                on_doc(d["doc_id"], _count_nodes(tree) - 1)  # minus the doc root
        return trees
