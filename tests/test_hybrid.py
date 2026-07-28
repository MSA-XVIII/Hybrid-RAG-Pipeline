"""Offline tests for the hybrid RAG orchestration — no credentials, all fakes.

Run:  pytest -q   (from the hybrid-rag/ directory)
"""

from __future__ import annotations

import os
import tempfile

import pytest

from hybridrag.catalog import RoutingCatalog
from hybridrag.common import CatalogEntry
from hybridrag.crag import CRAGQueryAgent
from hybridrag.fakes import FakeDIS, FakeGraph, FakeLMaaS
from hybridrag.ingest import IngestAgent


def _catalog(tmp_name: str) -> RoutingCatalog:
    path = os.path.join(tempfile.gettempdir(), tmp_name)
    if os.path.exists(path):
        os.remove(path)
    return RoutingCatalog(path=path)


# --- catalog ----------------------------------------------------------------

def test_catalog_roundtrip():
    cat = _catalog("t_catalog.json")
    cat.put(CatalogEntry(doc_id="d1", title="Manual", in_vector=True, in_graph=True,
                         domain="service-manual", entities=["SRP"]))
    got = cat.get("d1")
    assert got and got.in_graph and got.entities == ["SRP"]
    # reload from disk
    cat2 = RoutingCatalog(path=cat.path)
    assert cat2.get("d1").title == "Manual"


# --- Agent 1: ingest routing ------------------------------------------------

def test_ingest_routes_dense_doc_to_graph_only():
    cat = _catalog("t_ing1.json")
    dis, graph = FakeDIS(), FakeGraph()
    # bar=3; the doc names 4 distinct entities -> entity-dense
    agent = IngestAgent(FakeLMaaS(), dis, graph, catalog=cat, collection="c", min_graph_entities=3)
    entry = agent.ingest("m1", "Service Manual install steps. Error Codes reference. "
                               "Detector Calibration. Smart Reading Protocol feature.",
                         title="Manual")
    assert entry.in_graph is True                   # >= 3 entities -> graph
    assert entry.in_vector is False                 # NEW policy: graph-only, not vectorized
    assert entry.meta["entity_count"] >= 3
    assert graph.get_sections(["m1"])               # tree was written
    assert not dis.collections.get("c")             # nothing uploaded to the vector store


def test_ingest_routes_prose_to_vector_only():
    cat = _catalog("t_ing2.json")
    dis, graph = FakeDIS(), FakeGraph()
    agent = IngestAgent(FakeLMaaS(), dis, graph, catalog=cat, collection="c", min_graph_entities=3)
    entry = agent.ingest("p1", "a short narrative about our company values and history.",
                         title="about us")
    assert entry.in_vector is True                  # too few entities -> vector
    assert entry.in_graph is False
    assert not graph.get_sections(["p1"])
    assert dis.collections.get("c")                 # uploaded to the vector store


def test_full_document_entity_gate():
    """The graph gate is driven by DISTINCT entities across the whole doc."""
    cat = _catalog("t_gate.json")
    dis, graph = FakeDIS(), FakeGraph()
    agent = IngestAgent(FakeLMaaS(), dis, graph, catalog=cat, collection="c")  # default bar=8
    rich = ("The Detector, Calibration Routine, Field Engineer, Hanging Protocol, "
            "Image Display, Viewport, Error Codes, Smart Reading Protocol, and Recovery "
            "Steps are all covered in detail.")
    sparse = "The Detector was serviced."
    r = agent.ingest("rich", rich, title="Rich Manual")
    s = agent.ingest("sparse", sparse, title="Note")
    assert r.meta["entity_count"] >= 8 and r.in_graph is True and r.in_vector is False
    assert s.meta["entity_count"] < 8 and s.in_graph is False and s.in_vector is True


def test_dual_store_toggle(monkeypatch):
    """With STORE_GRAPH_DOCS_IN_VECTOR on, a dense doc goes to BOTH stores."""
    from hybridrag import config as cfg
    monkeypatch.setattr(cfg, "STORE_GRAPH_DOCS_IN_VECTOR", True)
    cat = _catalog("t_dual.json")
    dis, graph = FakeDIS(), FakeGraph()
    agent = IngestAgent(FakeLMaaS(), dis, graph, catalog=cat, collection="c", min_graph_entities=3)
    e = agent.ingest("m1", "Service Manual. Error Codes. Detector Calibration. "
                           "Smart Reading Protocol.", title="Manual")
    assert e.in_graph is True and e.in_vector is True
    assert dis.collections.get("c")


# --- Agent 2: CRAG verdict branches -----------------------------------------

def _ingested_agents(tmp_name: str):
    cat = _catalog(tmp_name)
    dis, graph, lmaas = FakeDIS(), FakeGraph(), FakeLMaaS()
    ing = IngestAgent(lmaas, dis, graph, catalog=cat, collection="c", min_graph_entities=2)
    # Entity-dense -> graph-only. A section HEADING ("Calibration Error Recovery")
    # carries terms its BODY lacks, so graph keyword-search can miss while
    # vectorless heading-navigation recovers. Headings on their own lines.
    manual = (
        "View User Manual covers image display and layout tools.\n"
        "## 1.1 Imaging Features\n"
        "The system shows studies using the configured viewport.\n"
        "## 1.2 Calibration Error Recovery\n"
        "Restart the unit and rerun the routine to clear fault code 3455rr.\n"
    )
    ing.ingest("view_um", manual, title="View User Manual")
    q = CRAGQueryAgent(lmaas, dis, graph, catalog=cat, collection="c")
    return q, lmaas


def test_correct_path_early_exits():
    q, _ = _ingested_agents("t_correct.json")
    # intro body has all these terms -> strong graph hit -> correct
    res = q.run("image display layout tools")
    assert res.verdict.label == "correct"
    assert "vectorless" not in res.sources        # no correction needed
    assert res.answer


def test_incorrect_path_recovers_via_vectorless():
    q, _ = _ingested_agents("t_incorrect.json")
    # no BODY contains these terms -> graph retrieval empty (incorrect) -> the
    # "Calibration Error Recovery" HEADING matches -> vectorless recovers.
    res = q.run("calibration error recovery")
    assert res.verdict.label == "incorrect"
    assert res.sources == ["vectorless"]
    assert "3455rr" in res.answer or "restart" in res.answer.lower()


def test_ambiguous_path_fuses_primary_and_vectorless():
    q, _ = _ingested_agents("t_ambiguous.json")
    # "image display" hits the intro body (partial); "calibration recovery" hits a
    # heading -> mid score (ambiguous) -> fuse graph + vectorless.
    res = q.run("image display calibration recovery")
    assert res.verdict.label == "ambiguous"
    assert "vectorless" in res.sources


def test_abstains_when_nothing_internal_matches():
    q, _ = _ingested_agents("t_abstain.json")
    res = q.run("quarterly revenue forecast for the finance department")
    # nothing internal matches -> incorrect -> vectorless empty -> abstain
    assert res.verdict.label == "incorrect"
    assert res.answer


# --- multi-document features -------------------------------------------------

def test_ingest_documents_batch():
    cat = _catalog("t_batch.json")
    dis, graph = FakeDIS(), FakeGraph()
    agent = IngestAgent(FakeLMaaS(), dis, graph, catalog=cat, collection="c", min_graph_entities=3)
    docs = [
        {"doc_id": "m1", "title": "Manual",
         "text": "Service Manual. Error Codes. Detector Calibration. Smart Reading Protocol."},
        {"doc_id": "p1", "title": "Notes", "text": "a short note with no entities."},
    ]
    entries = agent.ingest_documents(docs)
    assert len(entries) == 2
    assert {e.doc_id for e in entries} == {"m1", "p1"}
    assert cat.get("m1").in_graph is True          # dense -> graph
    assert cat.get("p1").in_graph is False         # sparse -> vector


def test_vectorless_scopes_to_candidate_docs():
    cat = _catalog("t_scope.json")
    dis, graph, lmaas = FakeDIS(), FakeGraph(), FakeLMaaS()
    ing = IngestAgent(lmaas, dis, graph, catalog=cat, collection="c", min_graph_entities=2)
    ing.ingest("aurora",
               "Aurora Imaging Manual\n## 1.1 Detector Calibration\nCalibrate the Detector and Collimator.\n",
               title="Aurora Imaging Manual")
    ing.ingest("handbook",
               "Field Handbook\n## 1.1 Escalation Rotation\nContact the Duty Lead after two hours.\n",
               title="Field Handbook")

    # scoping: the query matches only the handbook -> just that doc_id
    assert cat.candidate_doc_ids("Escalation Rotation Duty Lead") == ["handbook"]

    # end-to-end: graph body-search misses, vectorless recovers from the handbook
    q = CRAGQueryAgent(lmaas, dis, graph, catalog=cat, collection="c")
    res = q.run("Escalation Rotation")
    assert "vectorless" in res.sources
    assert "duty lead" in res.answer.lower()       # recovered handbook content, not aurora


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
