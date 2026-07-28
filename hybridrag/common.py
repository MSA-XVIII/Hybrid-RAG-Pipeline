"""Shared data shapes passed between the agents, tools, and clients."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def entity_id(name: str) -> str:
    """Global (doc-independent) node id for an entity.

    The id is derived only from the entity NAME, so the same entity mentioned
    in different sections or different documents collapses to ONE graph node —
    that shared node is what turns the graph from a star into a web.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return f"ent_{slug}" if slug else "ent_unknown"


@dataclass
class RetrievalResult:
    """What any retriever (vector / graph / vectorless) returns."""

    source: str                      # "vector" | "graph" | "vectorless"
    text: str                        # concatenated context for the generator
    chunks: list[dict] = field(default_factory=list)  # [{doc_id, text, score, ...}]
    meta: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.text and self.text.strip())


@dataclass
class Verdict:
    """CRAG retrieval-evaluator output."""

    label: str                       # "correct" | "ambiguous" | "incorrect"
    score: float                     # 0..1 relevance of retrieval to the query
    reason: str = ""


@dataclass
class CatalogEntry:
    """One document's routing record."""

    doc_id: str
    title: str = ""
    in_vector: bool = False
    in_graph: bool = False
    collection: str = ""
    domain: str = ""
    entities: list[str] = field(default_factory=list)
    section_tree_ref: str = ""       # "neptune" when the graph holds the tree
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id, "title": self.title,
            "in_vector": self.in_vector, "in_graph": self.in_graph,
            "collection": self.collection, "domain": self.domain,
            "entities": self.entities, "section_tree_ref": self.section_tree_ref,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CatalogEntry":
        return cls(
            doc_id=d["doc_id"], title=d.get("title", ""),
            in_vector=d.get("in_vector", False), in_graph=d.get("in_graph", False),
            collection=d.get("collection", ""), domain=d.get("domain", ""),
            entities=d.get("entities", []) or [],
            section_tree_ref=d.get("section_tree_ref", ""), meta=d.get("meta", {}) or {},
        )


@dataclass
class QueryResult:
    """Final output of Agent 2 for one query."""

    query: str
    answer: str
    verdict: Verdict
    sources: list[str] = field(default_factory=list)   # which retrievers contributed
    llm_calls: int = 0
    trace: list[str] = field(default_factory=list)     # human-readable step log
