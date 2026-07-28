"""End-to-end OFFLINE demo — no credentials needed.

Wires the fake LMaaS/DIS/Graph clients, ingests a couple of documents through
Agent 1 (ingest router), then runs queries through Agent 2 (CRAG loop), printing
the routing decision, the CRAG verdict, the corrective path, and the answer.

Run:  python demo.py
Switch use_fakes=False (and fill config REPLACE_MEs) to hit the real services.
"""

from __future__ import annotations

import os
import tempfile

from hybridrag.catalog import RoutingCatalog
from hybridrag.crag import CRAGQueryAgent
from hybridrag.ingest import IngestAgent

USE_FAKES = True

# --- sample documents --------------------------------------------------------
# The manual's INTRO names its topics but doesn't detail them; deep detail lives
# The manual is entity-dense -> Agent 1 routes it to the GRAPH (not the vector
# store). Section "2 Calibration Error Recovery" has a heading whose terms its
# body does NOT repeat, so a graph keyword-search can miss while vectorless
# heading-navigation recovers — that's the corrective path.
MANUAL = """View User Manual Overview
This manual covers the Imaging Workstation, its Viewport, and the Field Engineer workflow.
1 Imaging Features
The Smart Reading Protocol and Hanging Protocol tools arrange studies on the Display.
1.1 Smart Reading Protocol
It employs machine learning and learns preferences via a teach action snapshot.
2 Calibration Error Recovery
Restart the unit and rerun the routine to clear fault code 3455rr.
"""

# Prose, few entities -> Agent 1 routes it to the VECTOR store only.
POLICY = """Data Retention Policy
This policy defines how long records are retained and the approval needed for deletion.
Records are kept for seven years unless legal hold applies. Nothing here concerns imaging.
"""


def build_agents(use_fakes: bool):
    catalog_path = os.path.join(tempfile.gettempdir(), "hybridrag_demo_catalog.json")
    if os.path.exists(catalog_path):
        os.remove(catalog_path)
    catalog = RoutingCatalog(path=catalog_path)

    if use_fakes:
        from hybridrag.fakes import FakeDIS, FakeGraph, FakeLMaaS
        lmaas, dis, graph = FakeLMaaS(), FakeDIS(), FakeGraph()
    else:  # real services — requires config REPLACE_MEs to be filled
        from hybridrag.clients import DISClient, GraphClient, LMaaSClient
        lmaas, dis, graph = LMaaSClient(), DISClient(), GraphClient()

    ingest = IngestAgent(lmaas, dis, graph, catalog=catalog, collection="demo")
    query = CRAGQueryAgent(lmaas, dis, graph, catalog=catalog, collection="demo")
    return ingest, query


def main() -> None:
    ingest, query = build_agents(USE_FAKES)

    print("=" * 78)
    print("AGENT 1 — ingest routing")
    print("=" * 78)
    for doc_id, title, text in [("view_um", "View User Manual", MANUAL),
                                 ("policy", "Data Retention Policy", POLICY)]:
        entry = ingest.ingest(doc_id, text, title=title)
        print(f"  {doc_id:10s} vector={entry.in_vector} graph={entry.in_graph} "
              f"entities={entry.meta.get('entity_count')} "
              f"density={entry.meta.get('density')}/1k words "
              f"(bar={entry.meta.get('min_graph_entities')}) domain={entry.domain}")
        print(f"             found: {entry.entities}")

    print("\n" + "=" * 78)
    print("AGENT 2 — CRAG query loop")
    print("=" * 78)
    queries = [
        "Smart Reading Protocol Hanging Protocol Display",   # graph body hit -> CORRECT
        "calibration error recovery",                         # body misses, heading matches -> INCORRECT -> vectorless
        "Smart Reading Protocol calibration recovery",        # partial body + heading -> AMBIGUOUS -> fuse
        "data retention period",                              # prose doc, vector-only
    ]
    for q in queries:
        res = query.run(q)
        print(f"\nQ: {q}")
        print(f"   verdict : {res.verdict.label} (score={res.verdict.score:.2f})")
        print(f"   sources : {res.sources}")
        print(f"   LLM calls: {res.llm_calls}")
        for step in res.trace:
            print(f"     - {step}")
        print(f"   answer  : {res.answer[:160]}")


if __name__ == "__main__":
    main()
