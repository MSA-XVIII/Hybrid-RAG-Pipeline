# Session Context — hybrid-rag

**Purpose:** Hand-off file so a fresh Claude Code session can resume the hybrid-RAG work without re-deriving anything.
**Workspace:** `c:\Users\250029286\Downloads\Knowledge Graphs\hybrid-rag`
**Last updated:** 2026-07-21
**Sibling projects:** `../pdf-to-kg-agent` (the PDF→KG LangGraph agent this builds on) and `../service-manual-kg`, `../patient-kg`, `../dis-kg-prototype`.

---

## 1. What this project is

An **adaptive hybrid RAG** prototype: two **LangGraph** agents over **three retrieval modes**, choosing the right store per document and per query, with Corrective-RAG (CRAG) self-checking.

- **Agent 1 — ingest router** (`hybridrag/ingest.py`): scans a whole document for **distinct entities**; routes **entity-dense docs → graph (Neptune)**, everything else → **vector (DIS)**. Writes a **routing catalog**. Default is mutually-exclusive (graph docs are NOT also vectorized) — toggle `STORE_GRAPH_DOCS_IN_VECTOR` to dual-store.
- **Agent 2 — CRAG query loop** (`hybridrag/crag.py`): route → retrieve → **evaluate** → `correct | ambiguous | incorrect` → generate.
  - correct → refine + generate (early exit)
  - ambiguous → primary + **vectorless** corrective, fuse
  - incorrect → **vectorless** corrective (FULLY INTERNAL — no web search, by decision)
- **Three retrieval modes:** vector (DIS), graph (Neptune Gremlin keyword search), **vectorless** (PageIndex-style: LLM reasons over the section-heading tree, no embeddings) — vectorless is the **corrective** route, scoped to the catalog's candidate docs.

Full design: `HYBRID_RAG_PLAN.md` (in `../pdf-to-kg-agent/`). Function-by-function guide: `CODEBASE_GUIDE.md` (this folder).

---

## 2. Current status

- **Works fully offline with fakes — no credentials.** `python demo.py`, `pytest -q` (11 tests pass), and `notebooks/demo_hybrid_rag.ipynb` all run credential-free.
- **Both agents are real LangGraph `StateGraph`s** (deployment-ready). `langgraph` is a required dep and runs locally.
- **Real clients implemented** with `REPLACE_ME` placeholders (LMaaS chat, DIS, Neptune graph). The public API is identical fake-vs-real, so swapping is just which client objects you pass in.
- **The graph build against real Neptune is ready to run** (user has a Neptune endpoint). Notebook §11–§12 do it.

**The gate:** live LMaaS/IDAM + DIS credentials. Neptune the user has.

---

## 3. File map (`hybridrag/`)

| File | Role |
|---|---|
| `config.py` | all settings; `REPLACE_ME` for every secret/endpoint; thresholds (see §7) |
| `common.py` | dataclasses: `RetrievalResult`, `Verdict`, `CatalogEntry`, `QueryResult` |
| `clients.py` | `LMaaSClient` (Azure OpenAI + IDAM **client-credentials**), `DISClient` (DIS + IDAM **token-exchange**), `GraphClient` (Neptune Gremlin + SigV4). Third-party imports are lazy. |
| `catalog.py` | `RoutingCatalog` (JSON file): `put/get/all`, `candidates(query)`, `candidate_doc_ids(query)` (scopes vectorless) |
| `ingest.py` | Agent 1 StateGraph: `assess → (route) → store_graph \| store_vector → write_catalog`. `assess_document` = windowed full-doc entity scan. `ingest_documents(docs)` = batch helper. |
| `retrieval.py` | `vector_retrieve`, `graph_retrieve`, `vectorless_retrieve` |
| `crag.py` | Agent 2 StateGraph: `route → retrieve → evaluate →(cond)→ correct\|ambiguous\|incorrect → generate`. Reusable `evaluate/refine/generate` fns. |
| `visualize.py` | `graph_html` (force-directed KG) + `pageindex_html` (hierarchical tree) → self-contained vis-network HTML; `iframe_srcdoc` for notebook |
| `loader.py` | `load_documents("docs")` — reads `.md`/`.txt` directly, `.pdf` via pdfplumber |
| `fakes.py` | `FakeLMaaS` / `FakeDIS` / `FakeGraph` — in-memory stand-ins (term-overlap heuristics) for offline runs |
| `demo.py` (root) | offline end-to-end demo |
| `notebooks/demo_hybrid_rag.ipynb` | comprehensive feature-labeled demo (§0 install → §10 offline features → §11–§13 real Neptune) |
| `tests/test_hybrid.py` | 11 offline tests (catalog, routing, CRAG verdicts, batch ingest, scoping, dual-store toggle) |
| `docs/` | sample docs: `imaging_service_manual.md`, `field_service_handbook.md`, `centricity_mp3510_reference_manual.md` (drop your own `.md`/`.txt`/`.pdf` here) |

---

## 4. Key decisions (do NOT relitigate)

1. **LangGraph for both agents** — user wants this for eventual deployment; runs locally so offline path stays credential-free.
2. **Entity-dense docs are graph-ONLY by default** (not also vectorized). Trade-off: they lose fuzzy vector recall, served by graph keyword + vectorless instead. Reversible via `STORE_GRAPH_DOCS_IN_VECTOR`.
3. **Graph-worthiness gate = distinct-entity count across the WHOLE document** (not a 2000-char sample). Default `INGEST_MIN_GRAPH_ENTITIES=8`. (The user pushed back that the old "3 from a sample" was meaningless — this replaced it.)
4. **Vectorless = corrective route** for Ambiguous/Incorrect (not a third pipeline); reuses the Neptune section tree; **no embeddings** (dropped for cost/simplicity).
5. **Incorrect branch is FULLY INTERNAL — no web search** (VPC constraint; user decision).
6. **Model tiering:** evaluator/navigation/refine use the CHEAP LMaaS deployment; only the final generator uses the STRONG one. Don't cut quality calls — economize via tier + conditional execution + parallelism + caching.
7. **Fusion over selection** — the generator sees both primary + vectorless context.
8. **Orchestration platform = LangGraph, or the org's AWaaS — NOT n8n.**
9. **Flags in `write_catalog` are computed deterministically** from (build_graph + dual-store config), not propagated as node state — a LangGraph conditional-branch state-merge quirk made propagation unreliable.
10. **Naming:** clients are still `Fake*`. A rename to `Offline*` was proposed for demo polish but NOT done (deferred — ask before doing).

---

## 5. The two service contracts (confirmed from Confluence — see memory)

- **LMaaS** = Azure OpenAI gateway + **IDAM client-credentials** auth. `LMaaSClient` uses `openai.AzureOpenAI` with an `azure_ad_token_provider`. Config: `LMAAS_ENDPOINT`, `LMAAS_DEPLOYMENT_STRONG`, `LMAAS_DEPLOYMENT_CHEAP`, `LMAAS_API_VERSION`, IDAM `*_CLIENT_ID/SECRET`, `LMAAS_AUDIENCE`. (Memory: `lmaas-api-contract`.)
- **DIS** = managed vector-RAG platform + **IDAM token-exchange** auth (different from LMaaS): app token → exchange (`grant_type=token_exchange`, `audience=<DIS app id>`) → DIS-scoped token carrying RBAC roles (Admin=ingest, User=query). Services: Upload, Collection Management, Retrieval. **Exact endpoint paths are `REPLACE_ME`** — pull from the Bruno export on Confluence "DIS API Collection - 1.0 Rev 2" (id 2483716225). `keyword_search` uses `TextP.containing`, which needs Neptune full-text search (OpenSearch) enabled to work for real. (Memory: `dis-api-contract`.)

Confluence access is via the claude.ai Atlassian Rovo MCP (cloudId `9b6acdc1-bbcb-48da-acfc-bdcbe5db718c`).

---

## 6. How to run

**Offline (no creds):**
```bash
cd hybrid-rag
pip install -r requirements.txt      # langgraph + pytest (+ openai/requests/boto3/pdfplumber for real)
python demo.py
pytest -q                            # 11 pass
jupyter notebook notebooks/demo_hybrid_rag.ipynb   # §0 install cell; restart kernel if langgraph import fails
```

**Real Neptune (user has the endpoint):** in the notebook, run §0–§1, then §11 (set `config.NEPTUNE_ENDPOINT`, connectivity check), §12 (Agent 1 ingests `docs/` into real Neptune — graph build is REAL; LMaaS/DIS stay faked since no creds), §12.1 (visualize live), §13 (vectorless over live Neptune). Requires the SageMaker notebook to be in the Neptune VPC with an IAM role allowed to reach Neptune (SigV4). If TLS errors: `config.DIS_VERIFY_SSL = False`.

---

## 7. Config thresholds (defaults in `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `CRAG_CORRECT_THRESHOLD` | 0.7 | eval score ≥ → correct |
| `CRAG_INCORRECT_THRESHOLD` | 0.3 | eval score < → incorrect (between = ambiguous) |
| `REFINE_MIN_TOKENS` | 180 | skip refine call below this context size |
| `INGEST_MIN_GRAPH_ENTITIES` | 8 | distinct whole-doc entities to build a graph |
| `INGEST_ENTITY_SCAN_WINDOW_WORDS` / `_MAX_WINDOWS` | 400 / 8 | full-doc entity-scan windows (cost cap) |
| `STORE_GRAPH_DOCS_IN_VECTOR` | false | dual-store graph docs if true |
| `RETRIEVE_TOP_K` | 5 | passages/chunks per retriever |
| `EXTRACT_CONCURRENCY` | 5 | (carried over concept) |

---

## 8. Known limitations / caveats

- **Fakes are crude** (term-overlap heuristics) — verdicts skew (e.g. "ambiguous" where the real LMaaS would say "correct"). Logic/paths are real; only the model responses are simulated.
- **DIS endpoint paths + response shape are `REPLACE_ME`** pending the Bruno collection.
- **Graph keyword-search (`GraphClient.keyword_search`) needs Neptune full-text search** — until then, real graph docs lean on the vectorless path. The write→read→visualize→vectorless path uses only standard Gremlin and works as-is.
- **Entity-gate + vectorless navigator use `FakeLMaaS` in the notebook's real-Neptune section** (no LMaaS creds). Structuring + Neptune writes are 100% real. Swap `FakeLMaaS()` → `LMaaSClient()` when creds arrive.

---

## 9. Open TODOs / next steps

1. **Get LMaaS + DIS credentials** (LMaaS: endpoint + strong/cheap deployment names + IDAM client id/secret/audience; DIS: subscribe app for token-exchange + the Bruno API collection). This is the gate.
2. Fill DIS endpoint paths in `config.py` / `clients.py` from the Bruno collection; smoke-test `DISClient`.
3. Enable/verify **Neptune full-text search** so `graph_retrieve` works for real (or swap in a text index).
4. Swap `FakeLMaaS` → `LMaaSClient` in the notebook once LMaaS creds land → real LLM-driven entity gate, evaluator, vectorless navigator, generator.
5. (Optional, deferred) rename `Fake*` → `Offline*`; add LangGraph checkpointing (`MemorySaver`) so a failed real run resumes without re-calling the LLM.
6. (Optional) confidence-based fan-out, caching, streaming generation.
