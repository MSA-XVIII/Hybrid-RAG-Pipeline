# hybrid-rag

Adaptive **hybrid RAG** prototype: two agents over three retrieval modes
(vector / graph / vectorless), with a Corrective-RAG (CRAG) query loop.

This is a **self-contained, independently testable** sibling of `pdf-to-kg-agent`.
It runs **fully offline with fakes** (no credentials), and switches to the real
GE HealthCare services (LMaaS, DIS, Neptune) by filling the `REPLACE_ME` values
in `hybridrag/config.py` (or the matching env vars).

See `../pdf-to-kg-agent/HYBRID_RAG_PLAN.md` for the full design.

## What's here

```
hybridrag/
  config.py        all settings — REPLACE_ME placeholders for every secret/endpoint
  common.py        shared dataclasses (RetrievalResult, Verdict, QueryResult)
  clients.py       LMaaSClient (Azure OpenAI + IDAM client-creds)
                   DISClient   (DIS + IDAM token-exchange)   -- REPLACE_ME endpoints
                   GraphClient (Neptune Gremlin)             -- REPLACE_ME endpoint
  catalog.py       RoutingCatalog — records which store each doc went to
  ingest.py        Agent 1 (LangGraph StateGraph): assess -> route entity-dense docs to
                     Neptune(graph), others to DIS(vector) -> write catalog
  retrieval.py     vector / graph / vectorless retrievers
  crag.py          Agent 2 (LangGraph StateGraph): route -> retrieve -> evaluate ->
                     correct | ambiguous | incorrect -> generate
  visualize.py     knowledge-graph + PageIndex-tree HTML views
  fakes.py         FakeLMaaS / FakeDIS / FakeGraph for offline runs
demo.py            end-to-end offline demo (no credentials needed)
notebooks/         demo_hybrid_rag.ipynb — offline demo with inline visualizations
tests/             offline tests for routing, CRAG verdicts, catalog
```

Both agents are **LangGraph `StateGraph`s** (deployment-ready orchestration); they run
locally, so the offline demo/tests still need no credentials.

## Run it offline (no credentials)

```bash
cd hybrid-rag
pip install -r requirements.txt          # langgraph + pytest; no credentials needed
python demo.py                           # ingest fake docs + run queries through both agents
pytest -q                                # exercise Agent 1 routing + Agent 2 CRAG paths
```

`demo.py` wires the **fake** clients, so it exercises the real orchestration
logic (routing, CRAG verdicts, fusion, vectorless correction) with canned model
responses — proving the architecture end to end without hitting any service.

## Switch to the real services

1. Fill the `REPLACE_ME` values in `hybridrag/config.py` (or set the env vars):
   - **LMaaS** (Azure OpenAI + IDAM client-credentials) — same as `pdf-to-kg-agent`.
   - **DIS** (IDAM **token-exchange** + collection/upload/retrieve endpoints) — the
     exact endpoint paths come from the "DIS API Collection" Bruno export; they're
     marked `REPLACE_ME` in `clients.py`.
   - **Neptune** endpoint for the graph + vectorless tools.
2. Build the real clients instead of the fakes (see `demo.py` — swap
   `use_fakes=True` to `False`).

## Status / gates

- Orchestration (both agents, CRAG, catalog, fusion): **implemented + offline-tested**.
- Real LMaaS client: **implemented** (mirrors `pdf-to-kg-agent`), needs creds.
- Real DIS client: **structure implemented**, endpoint paths are `REPLACE_ME`
  pending the DIS Bruno collection.
- Real Graph client: thin Gremlin wrapper, needs the Neptune endpoint.
