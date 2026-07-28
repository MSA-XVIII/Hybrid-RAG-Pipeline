"""Routing catalog — the linchpin that records which store each document went to.

Written by Agent 1 at ingest, read by Agent 2 at query time so routing is never
blind. Backed by a JSON file for the prototype; swap for DynamoDB / a Neptune
subgraph in production without changing the interface.
"""

from __future__ import annotations

import json
import os
import threading

from hybridrag import config
from hybridrag.common import CatalogEntry


class RoutingCatalog:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or config.CATALOG_PATH
        self._lock = threading.Lock()
        self._entries: dict[str, CatalogEntry] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._entries = {d["doc_id"]: CatalogEntry.from_dict(d) for d in data}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries.values()], f, indent=2, ensure_ascii=False)

    def put(self, entry: CatalogEntry) -> None:
        with self._lock:
            self._entries[entry.doc_id] = entry
            self._save()

    def get(self, doc_id: str) -> CatalogEntry | None:
        return self._entries.get(doc_id)

    def all(self) -> list[CatalogEntry]:
        return list(self._entries.values())

    # --- query-time routing helpers -----------------------------------------

    def has_vector(self) -> bool:
        return any(e.in_vector for e in self._entries.values())

    def has_graph(self) -> bool:
        return any(e.in_graph for e in self._entries.values())

    def candidates(self, query: str) -> dict[str, bool]:
        """Which stores plausibly hold docs relevant to the query.

        Prototype heuristic: overlap the query terms with catalogued entities /
        domains; if nothing matches, fall back to whatever stores exist. Keeps
        Agent 2 from calling a store that has no relevant content.
        """
        q = query.lower()
        vector = graph = False
        matched = False
        for e in self._entries.values():
            hay = " ".join([e.title, e.domain, " ".join(e.entities)]).lower()
            if any(tok for tok in _terms(q) if tok in hay):
                matched = True
                vector = vector or e.in_vector
                graph = graph or e.in_graph
        if not matched:  # no strong signal -> allow any store that exists
            vector, graph = self.has_vector(), self.has_graph()
        return {"vector": vector, "graph": graph}

    def candidate_doc_ids(self, query: str, *, in_graph: bool = True) -> list[str]:
        """doc_ids whose title/domain/entities overlap the query.

        Used to SCOPE vectorless navigation to the relevant documents (so it reads
        a focused table-of-contents, not the whole corpus). Falls back to every
        graph doc when nothing matches, so vectorless still has something to read.
        ``in_graph=True`` restricts to docs that actually have a graph tree.
        """
        q_terms = _terms(query.lower())
        pool = [e for e in self._entries.values() if (e.in_graph or not in_graph)]
        matched = []
        for e in pool:
            hay = " ".join([e.title, e.domain, " ".join(e.entities)]).lower()
            if any(t in hay for t in q_terms):
                matched.append(e.doc_id)
        return matched or [e.doc_id for e in pool]


def _terms(text: str) -> list[str]:
    return [t for t in text.replace(",", " ").replace("?", " ").split() if len(t) >= 4]
