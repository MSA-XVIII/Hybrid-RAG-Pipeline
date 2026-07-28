"""Offline fakes for LMaaS / DIS / Graph so the whole system runs with NO
credentials. They duck-type the real clients (same method signatures), so
demo.py / tests swap them in by construction.

Design intent: model just enough behaviour to exercise the real orchestration —
  * FakeLMaaS   — recognises each prompt (classify / evaluate / navigate / refine /
                  generate) and returns deterministic output driven by term overlap,
                  so CRAG verdicts depend on retrieval quality (not randomness).
  * FakeDIS     — in-memory vector store with SHALLOW coverage (indexes only the
                  first `coverage_words` of each doc). This makes the corrective
                  path demonstrable: deep-section queries miss the vector store and
                  get recovered by vectorless navigation over the full graph tree.
  * FakeGraph   — in-memory Document/Section/Chunk tree.
"""

from __future__ import annotations


def _terms(text: str) -> list[str]:
    return [t for t in text.lower().replace(",", " ").replace("?", " ").replace(".", " ").split()
            if len(t) >= 4]


def _recall(query: str, context: str) -> float:
    q = set(_terms(query))
    if not q:
        return 0.0
    ctx = set(_terms(context))
    return len(q & ctx) / len(q)


def _section(user: str, marker: str, end_markers: tuple[str, ...] = ()) -> str:
    """Extract text after `marker` up to the next end marker."""
    if marker not in user:
        return ""
    tail = user.split(marker, 1)[1]
    for em in end_markers:
        if em in tail:
            tail = tail.split(em, 1)[0]
    return tail.strip()


class FakeLMaaS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    # -- JSON prompts: domain / entity-scan / evaluate / navigate ------------
    def complete_json(self, system: str, user: str, *, cheap: bool = False, max_tokens: int = 4096):
        if "Assign a short domain label" in system:
            self.calls.append("domain")
            return {"domain": _guess_domain(user.lower())}
        if "DISTINCT salient entity" in system:
            self.calls.append("entities")
            # extract capitalized entity phrases from THIS window (deduped later)
            return {"entities": _caps_phrases(user, limit=20)}
        if "judge whether retrieved context" in system:
            self.calls.append("evaluate")
            q = _section(user, "QUESTION:", ("CONTEXT:",))
            ctx = _section(user, "CONTEXT:")
            return {"score": round(_recall(q, ctx), 2), "reason": "term-overlap (fake)"}
        if "navigate a document" in system:
            self.calls.append("navigate")
            q = _section(user, "QUESTION:", ("HEADINGS:",))
            toc = _section(user, "HEADINGS:")
            chosen = []
            for line in toc.splitlines():
                line = line.strip().lstrip("- ").strip()
                if not line or ":" not in line:
                    continue
                sid, heading = line.split(":", 1)
                if _recall(q, heading) > 0:
                    chosen.append(sid.strip())
            return {"section_ids": chosen[:4]}
        return {}

    # -- text prompts: refine / generate -------------------------------------
    def complete_text(self, system: str, user: str, *, cheap: bool = False,
                      max_tokens: int = 1024, temperature: float = 0.0) -> str:
        if "clean retrieved context" in system:
            self.calls.append("refine")
            ctx = _section(user, "CONTEXT:")
            return " ".join(ctx.split()[:80])  # pretend to keep the salient part
        if "answer the question using ONLY" in system:
            self.calls.append("generate")
            know = _section(user, "KNOWLEDGE:")
            snippet = " ".join(know.split()[:50])
            return f"[fake answer] Based on the retrieved knowledge: {snippet}"
        return ""


_STOP = {"this", "that", "these", "those", "the", "and", "nothing", "none",
         "however", "later", "it", "they", "there", "here", "records"}


def _caps_phrases(text: str, limit: int = 5) -> list[str]:
    import re
    # single-space connector so phrases never span line breaks
    phrases = re.findall(r"\b([A-Z][a-z]+(?:[ ][A-Z][a-z]+){0,3})\b", text)
    seen, out = set(), []
    for p in phrases:
        key = p.lower()
        # drop single filler words; keep real multi-word or domain-y names
        if key in seen or len(p) <= 3 or (" " not in p and key in _STOP):
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def _guess_domain(body: str) -> str:
    if "error code" in body or "manual" in body:
        return "service-manual"
    if "policy" in body or "retention" in body:
        return "policy"
    return "general"


class FakeDIS:
    """In-memory vector store with shallow coverage (first `coverage_words`)."""

    def __init__(self, coverage_words: int = 60) -> None:
        self.coverage_words = coverage_words
        self.collections: dict[str, list[dict]] = {}

    def create_collection(self, name: str) -> str:
        self.collections.setdefault(name, [])
        return name

    def upload_document(self, collection: str, doc_id: str, text: str, metadata=None) -> None:
        self.collections.setdefault(collection, [])
        head = " ".join(text.split()[: self.coverage_words])  # shallow index
        self.collections[collection].append(
            {"doc_id": doc_id, "text": head, "metadata": metadata or {}})

    def retrieve(self, query: str, collection: str, top_k: int = 5) -> list[dict]:
        docs = self.collections.get(collection, [])
        scored = []
        for d in docs:
            score = _recall(query, d["text"])
            if score > 0:
                scored.append({"doc_id": d["doc_id"], "text": d["text"], "score": round(score, 2)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class FakeGraph:
    """In-memory Document -> Section -> Chunk tree."""

    def __init__(self) -> None:
        # doc_id -> {"title", "sections":[{section_id, order, heading, chunks[]}]}
        self.docs: dict[str, dict] = {}
        self._mentions: dict[str, dict[str, set]] = {}  # doc_id -> {section_id: {entity_id}}
        self._entity_names: dict[str, str] = {}         # entity_id -> display name

    def upsert_document_tree(self, doc_id: str, title: str, sections: list[dict]) -> None:
        secs = []
        for i, s in enumerate(sections):
            order = s.get("order", i)
            secs.append({"section_id": f"sec_{doc_id}_{order}", "order": order,
                         "heading": s.get("heading", ""), "chunks": list(s.get("chunks", []) or [])})
        self.docs[doc_id] = {"title": title, "sections": secs}

    def upsert_entities(self, doc_id: str, mentions: list[dict]) -> None:
        from hybridrag.common import entity_id
        dm = self._mentions.setdefault(doc_id, {})
        for m in mentions:
            eid = entity_id(m["entity"])
            self._entity_names[eid] = m["entity"]
            sid = f"sec_{doc_id}_{m.get('order', 0)}"
            dm.setdefault(sid, set()).add(eid)

    def _iter_sections(self, doc_ids=None):
        for did, d in self.docs.items():
            if doc_ids and did not in doc_ids:
                continue
            for s in d["sections"]:
                yield did, s

    def get_sections(self, doc_ids=None) -> list[dict]:
        return [{"section_id": s["section_id"], "heading": s["heading"], "doc_id": did}
                for did, s in self._iter_sections(doc_ids)]

    def get_section_texts(self, section_ids: list[str]) -> list[dict]:
        want = set(section_ids)
        return [{"section_id": s["section_id"], "heading": s["heading"], "text": " ".join(s["chunks"])}
                for _, s in self._iter_sections() if s["section_id"] in want]

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        hits = []
        for did, s in self._iter_sections():
            body = " ".join(s["chunks"])
            score = _recall(query, body)
            if score > 0:
                hits.append({"doc_id": did, "section_id": s["section_id"], "text": body,
                             "score": round(score, 2)})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:limit]

    def get_tree(self, doc_ids=None) -> list[dict]:
        """Full Document -> Section -> Chunk tree (+ Entity mentions) for viz."""
        out = []
        for did, d in self.docs.items():
            if doc_ids and did not in doc_ids:
                continue
            dm = self._mentions.get(did, {})
            doc_eids = sorted({e for eids in dm.values() for e in eids})
            out.append({"doc_id": did, "title": d["title"],
                        "entities": [{"entity_id": e, "name": self._entity_names.get(e, e)}
                                     for e in doc_eids],
                        "sections": [
                {"section_id": s["section_id"], "order": s["order"], "heading": s["heading"],
                 "chunks": [{"chunk_id": f"{s['section_id']}_c{i}", "text": c}
                            for i, c in enumerate(s["chunks"])],
                 "entities": sorted(dm.get(s["section_id"], set()))}
                for s in d["sections"]]})
        return out
