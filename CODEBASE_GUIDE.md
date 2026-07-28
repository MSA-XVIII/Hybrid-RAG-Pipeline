# hybrid-rag — Codebase Guide

A function-by-function walkthrough of the whole system, so you can explain the
workflow end to end: what each piece does, its parameters, and every threshold.

Trivial helpers (dataclass serializers, one-line getters) are noted briefly;
everything with real behavior is explained in full.

---

## 0. The 30-second mental model

Two agents over three retrieval modes, sharing a routing catalog:

```
INGEST  (Agent 1) : document → assess → ROUTE: entity-dense → Neptune(graph) ; else → DIS(vector) → write catalog
QUERY   (Agent 2) : query → route → retrieve → EVALUATE → correct | ambiguous | incorrect → generate
                                                            │           │            │
                                                      early exit   fuse+vectorless  vectorless (internal only)
```

Both agents are **LangGraph `StateGraph`s** (deployment-ready), so each step is an
explicit node with typed state and conditional edges. LangGraph runs locally, so the
offline path stays credential-free.

Every external service (LMaaS, DIS, Neptune) sits behind a **client object**. The
agents call methods on that object and don't care if it's the **real** client
([clients.py](hybridrag/clients.py)) or a **fake** ([fakes.py](hybridrag/fakes.py)).
That's what makes the whole thing testable offline.

---

## 1. `config.py` — every setting and threshold

All values come from env vars, defaulting to `REPLACE_ME` for anything secret.
`is_configured(value)` returns `False` for empty or `REPLACE_ME` strings; the real
clients call it to fail fast with a clear message instead of a confusing 401.

### Thresholds you will be asked about

| Setting | Default | What it controls |
|---|---|---|
| `CRAG_CORRECT_THRESHOLD` | **0.7** | evaluator score ≥ this → verdict **correct** (retrieval trusted, early exit) |
| `CRAG_INCORRECT_THRESHOLD` | **0.3** | evaluator score < this → verdict **incorrect** (retrieval discarded, correct via vectorless) |
| (between the two) | 0.3–0.7 | verdict **ambiguous** (keep primary **and** add vectorless, fuse) |
| `REFINE_MIN_TOKENS` | **180** | context smaller than this skips the refine LLM call (not worth it) |
| `INGEST_MIN_GRAPH_ENTITIES` | **8** | Agent 1 builds a graph only if the **whole-document** scan finds ≥ this many **distinct** entities |
| `INGEST_ENTITY_SCAN_WINDOW_WORDS` | **400** | words per entity-scan window (one cheap LLM call each) |
| `INGEST_ENTITY_SCAN_MAX_WINDOWS` | **8** | cost cap on scan calls per doc; more windows than this are sampled evenly across the doc |
| `STORE_GRAPH_DOCS_IN_VECTOR` | **False** | if true, entity-dense docs go to **both** stores; default keeps them **graph-only** |
| `RETRIEVE_TOP_K` | **5** | how many passages/chunks each retriever returns |
| `LMAAS_DEPLOYMENT_STRONG` / `_CHEAP` | REPLACE_ME | model tiering — strong model for generation, cheap model for evaluator/nav/refine |

The score bands are the heart of CRAG: **0.7** and **0.3** are the two dials that
decide how often the system trusts, fuses, or corrects. Raise `CORRECT` toward 0.8
to make it more skeptical (more corrections, higher cost); lower `INCORRECT` toward
0.2 to fuse more and discard less.

---

## 2. `common.py` — the data shapes passed around

Four dataclasses. These are the "nouns" every function speaks in.

- **`RetrievalResult`** — what any retriever returns.
  - `source`: `"vector"` | `"graph"` | `"vectorless"`
  - `text`: the concatenated context handed to the evaluator/generator
  - `chunks`: the raw hits `[{doc_id, text, score, ...}]`
  - `meta`: retriever-specific info (collection, chosen section ids, …)
  - `is_empty()`: True when there's no usable text — the trigger for an automatic "incorrect" verdict.
- **`Verdict`** — the CRAG evaluator's output: `label` (correct/ambiguous/incorrect), `score` (0–1), `reason`.
- **`CatalogEntry`** — one document's routing record (see §4). `to_dict`/`from_dict` are trivial JSON serializers.
- **`QueryResult`** — Agent 2's final output: `answer`, `verdict`, `sources` (which retrievers contributed), `llm_calls` (cost counter), `trace` (human-readable step log for the demo).

---

## 3. `clients.py` — the real backends (all `REPLACE_ME` until configured)

Third-party imports (`openai`, `requests`, `boto3`) are **lazy** (inside methods),
so importing this module costs nothing and the offline path never needs them.

### Helpers
- `_lit(value)` → quotes a string as a Gremlin literal (escapes quotes/backslashes).
- `_values(resp)` → pulls `resp["result"]["data"]` out of a Gremlin HTTP response.
- `_require(*pairs)` → raises `NotConfigured` listing any settings still `REPLACE_ME`.
- `_parse_json(content)` → parses model output as JSON, tolerating a fenced ```json block.

### `IDAMClientCredentials` — LMaaS auth
- `token()` → returns a cached OAuth2 **client-credentials** bearer token, refreshing when it's within 60 s of expiry (`_TOKEN_SKEW_S`). Thread-safe (double-checked lock).
- Grant sent to `IDAM_TOKEN_ENDPOINT`: `grant_type=client_credentials` + `client_id`/`client_secret`/`audience`.

### `IDAMTokenExchange` — DIS auth (the different one)
- `_app_token()` → gets a base app token from IDAM (client-credentials, scope `openid`).
- `token()` → performs the **token-exchange** grant: `grant_type=token_exchange`, `audience=<DIS app id>`, `token=<app token>`, with HTTP Basic auth of the client id/secret. Returns a **DIS-scoped** token that carries the RBAC roles claim. Cached + refreshed like above.
- **Why two auth classes:** LMaaS accepts a plain client-credentials token; DIS requires the exchanged, role-bearing token. This is confirmed from the DIS Confluence docs, not assumed.

### `LMaaSClient` — the LLM
- `__init__(token_gen=None)` — injectable token generator (defaults to `IDAMClientCredentials`).
- `_get_client()` — lazily builds the `openai.AzureOpenAI` client, passing `azure_ad_token_provider=self._token_gen.token` so every request auto-attaches a fresh IDAM token. Cached.
- `_deployment(cheap)` — returns the cheap or strong deployment name; this is the **model-tiering switch**.
- `complete_text(system, user, *, cheap=False, max_tokens=1024, temperature=0.0)` → returns the model's text. Used by **refine** and **generate**.
- `complete_json(system, user, *, cheap=False, max_tokens=4096)` → forces JSON mode and parses the result. Used by **domain-classification**, **entity-scan**, **evaluate**, **navigate**.
- The `cheap` flag is how the cost strategy is enforced at the call site: evaluator/nav/refine pass `cheap=True`; only `generate` passes `cheap=False`.

### `DISClient` — the vector store (managed RAG)
- `_headers()` → `Authorization: Bearer <DIS-scoped token>`.
- `_url(path, **fmt)` → builds `DIS_BASE_URL + path`, format-filling `{collection}` etc. Both are `REPLACE_ME` until you have the Bruno API collection.
- `create_collection(name)` → creates a namespace (needs the **Admin** role). Returns the collection id.
- `upload_document(collection, doc_id, text, metadata=None)` → ingests one document (Admin role). DIS does its own chunking/embedding server-side.
- `retrieve(query, collection, top_k=5)` → the query call (**User** role) → returns `[{doc_id, text, score, ...}]`. Tolerates a few response shapes (`results`/`passages`/`chunks`) since the exact one comes from the Bruno collection.

### `GraphClient` — Neptune (graph + the vectorless section tree)
Higher-level methods (mirrored by `FakeGraph`) so the retrievers don't write raw Gremlin:
- `upsert_document_tree(doc_id, title, sections)` → idempotent Gremlin upsert of `Document→HAS_SECTION→Section→HAS_CHUNK→Chunk`. `sections = [{"order", "heading", "chunks":[...]}]`. This is the structural tree that powers both graph and vectorless retrieval.
- `get_sections(doc_ids=None)` → **headings only** (cheap) — the "table of contents" the vectorless navigator reads.
- `get_section_texts(section_ids)` → full text (joined chunks) for the sections the navigator chose.
- `keyword_search(query, limit=5)` → containment search over `Chunk` text. **Caveat:** uses `TextP.containing`, which plain Neptune doesn't support without full-text search enabled — this method needs reworking against your real cluster.
- `run_gremlin(gremlin)` → raw POST to `/gremlin`, SigV4-signed via `_sigv4_headers()` when `NEPTUNE_USE_IAM` is on.

---

## 4. `catalog.py` — the routing catalog (the linchpin)

A JSON-file store recording which store each document went to. Written by Agent 1,
read by Agent 2 so routing is never blind.

- `RoutingCatalog(path=None)` → loads existing entries from disk on construction.
- `put(entry)` / `get(doc_id)` / `all()` → CRUD; `put` also saves to disk. Thread-safe.
- `has_vector()` / `has_graph()` → whether *any* catalogued doc is in that store.
- **`candidates(query)`** → the query-time router's key input. Returns `{"vector": bool, "graph": bool}`:
  1. For each catalogued doc, overlap the query's terms (length ≥ 4) against the doc's `title + domain + entities`.
  2. If anything matches, enable the stores those matching docs live in.
  3. If **nothing** matches (no signal), fall back to whatever stores exist — so a query is never starved of a route.
- `_terms(text)` → tokenizer: words of length ≥ 4 (drops short/noise words). The same overlap primitive the fakes use.

---

## 5. `ingest.py` — Agent 1 (the ingest router)

**Built as a LangGraph `StateGraph`** (`IngestState`) that is a genuine **router**:
`assess →` *(conditional `_route_store`)* `→ store_graph | store_vector → write_catalog → END`.
An **entity-dense** document goes down the **graph** path (Neptune only); everything
else goes down the **vector** path (DIS). `assess_document` (below) is the reusable
logic; the `_n_*` node methods wrap it and call the injected clients.
`IngestAgent.ingest()` compiles the graph once and `invoke`s it, returning the
`CatalogEntry` (unchanged public API).

> **Routing policy:** by default, entity-dense docs are **graph-only** — they are *not*
> also stored as vectors (avoids duplicating entity-rich content; the graph +
> vectorless retrieval serve them). Set `STORE_GRAPH_DOCS_IN_VECTOR=true` to dual-store.
> The trade-off: graph-only docs lose fuzzy semantic (vector) recall; they rely on graph
> keyword-search + vectorless structural retrieval, with CRAG correcting weak hits.

### `_naive_sections(text)`
Splits raw text into a `[{order, heading, chunks}]` tree — a self-contained stand-in
for the `pdf-to-kg-agent` structuring step. A line is treated as a **heading** if it's
a numbered heading (`2.4.3 ...`), a short ALL-CAPS line (≤ 8 words), or a markdown
`#` heading. Everything else is body, grouped under the current heading. Each
section's body is then chunked.
> **Gotcha worth knowing:** headings must be on their **own line** to be detected — inline "2.4.3 Foo" mid-paragraph won't split. (This bit us once in tests.)

### `_chunk(text, target_words=120)`
Splits text into ~120-word chunks (no overlap — deliberately simple for the prototype).

### `_windows(text, window_words, max_windows)`
Splits the **whole** document into ~400-word windows. If there are more windows than
`max_windows` (8), it **samples them evenly across the document** — so a very long doc
is still assessed end-to-end, but the number of LLM calls stays bounded. `_norm_entity`
lowercases/space-collapses a name so the same entity in two windows dedupes to one.

### `IngestAgent`
- `__init__(lmaas, dis, graph, catalog=None, collection=None, min_graph_entities=None)` — inject clients; `min_graph_entities` defaults to `INGEST_MIN_GRAPH_ENTITIES` (8) but can be overridden (tests use a low value).
- `_domain(title, text)` → one **cheap** LLM call to label the domain (service-manual / policy / …).
- **`assess_document(title, text)`** → the graph-worthiness gate. This is the part you asked about:
  1. `_windows(...)` splits the **entire** document (not a 2000-char sample).
  2. For each window, a **cheap** LLM call extracts distinct entities; results are **unioned + deduped** across all windows.
  3. Computes `entity_count` (distinct), `word_count`, and `density` (entities per 1,000 words).
  4. `build_graph = entity_count >= self.min_graph_entities`.
  - Returns `{domain, entities, entity_count, word_count, density, build_graph}`.
  - Cost: at most `INGEST_ENTITY_SCAN_MAX_WINDOWS` (8) cheap calls + 1 domain call, all at ingest time (amortized). The scanned entities also enrich the catalog, which improves query-time routing — so the work isn't wasted even for vector-only docs.
- **`ingest(doc_id, text, title="")`** → the Agent 1 workflow (invokes the router graph):
  1. `assess_document(...)` → decides `build_graph`.
  2. **Route:** `build_graph` → `store_graph` (`graph.upsert_document_tree(...)`, + vector only if `STORE_GRAPH_DOCS_IN_VECTOR`); else → `store_vector` (`dis.upload_document(...)`).
  3. `write_catalog` computes the flags **deterministically** from `build_graph` + the dual-store config (`in_graph = build_graph`; `in_vector = not build_graph or dual`) and writes the `CatalogEntry` — domain, distinct entities (capped at 40), `section_tree_ref="neptune"` for graph docs, `meta` with `entity_count`/`density`/`word_count`.
  > Flags are computed in `write_catalog` rather than propagated as node state — a deliberate robustness choice (cross-branch state merges on a conditional edge proved fragile).
- **`ingest_documents(docs)`** — batch helper: ingests a list of loader-style records `{doc_id, text, title}` and returns the `CatalogEntry` list. All docs accumulate in the same shared catalog + stores (multi-document corpus).
  - Returns the `CatalogEntry`.

**Why the gate changed:** the old version counted a curated ≤5-item sample from the
first 2000 chars, so a threshold like "3" reflected the *sample size*, not the
document. The new gate reads the **whole document**, counts **distinct** entities, and
compares to `INGEST_MIN_GRAPH_ENTITIES` (8) — so the number now means what you'd
intuitively expect: "at least 8 distinct entities across the doc to justify a graph."
Vector storage still happens regardless; the gate only decides whether to *also* build
a graph.

---

## 6. `retrieval.py` — the three retrieval modes

All three return a `RetrievalResult` so the CRAG loop treats them uniformly.

### `vector_retrieve(dis, query, collection, top_k=None)`
Calls `dis.retrieve(...)` (defaults to `RETRIEVE_TOP_K=5`), joins passage texts into
`text`. `source="vector"`.

### `graph_retrieve(graph, query, limit=None)`
Calls `graph.keyword_search(...)`, joins hit texts. `source="graph"`. Used as the
primary for relational-sounding queries.

### `vectorless_retrieve(lmaas, graph, query, doc_ids=None, max_sections=4)` — the corrective
This is the PageIndex-style reasoning retriever. Steps:
1. `graph.get_sections(doc_ids)` → pull the **headings only** (tiny/cheap).
2. Build a "table of contents" string and ask the LLM (`_NAV_SYSTEM`, **cheap** model) which `section_ids` are relevant — reasoning over meaning, not keywords. Returns a JSON list.
3. Cap to `max_sections` (4), then `graph.get_section_texts(chosen)` → fetch those sections' full text.
4. Join into `text` with `## heading` markers. `source="vectorless"`.
- If there are no sections, or the navigator picks nothing, returns an **empty** result — which is how the CRAG loop knows to fall back further.
- Key property: it reads only headings to *decide*, then fetches only the chosen bodies — so it's cheap even on large documents, and needs **no embeddings**.
- **Multi-document scoping:** Agent 2 passes `doc_ids` (from `catalog.candidate_doc_ids(query)`) so the navigator sees a **focused** table-of-contents from just the relevant documents, not the whole corpus — important once many docs are ingested.

---

## 7. `crag.py` — Agent 2 (the CRAG query loop)

**Built as a LangGraph `StateGraph`.** The state is `QueryState` (query, primary_source,
primary result, verdict, knowledge, answer, sources, calls, trace). Nodes:
`route → retrieve → evaluate →` *(conditional on the verdict)* `→ correct | ambiguous | incorrect → generate → END`.
The `_route_after_eval` function returns the verdict label, which the conditional edge
maps to the matching branch node. `evaluate`/`refine`/`generate` are module-level pure
functions (below); the graph nodes (`_n_*`) are thin wrappers that update state and
count LLM calls. `CRAGQueryAgent.run()` compiles the graph once and `invoke`s it,
then packs the final state into a `QueryResult` (unchanged public API).

### Module-level prompts
- `_EVAL_SYSTEM` — the retrieval evaluator: returns `{score 0..1, reason}`.
- `_REFINE_SYSTEM` — decompose → filter → recompose (drop irrelevant context).
- `_GEN_SYSTEM` — answer using ONLY the provided knowledge; say what's missing; don't invent.
- `_RELATIONAL_HINTS` — words ("relationship", "depend", "connect", "between", "how does", …) that bias the router toward the **graph** store.

### `evaluate(lmaas, query, result) → Verdict`
- If retrieval is empty → immediate `incorrect` (score 0), **no LLM call spent**.
- Otherwise one **cheap** LLM call scores relevance/sufficiency of the context (first 4000 chars) to the query.
- Score is clamped to [0,1] and bucketed by the two thresholds into correct / ambiguous / incorrect.
- If the evaluator call itself errors → default to **ambiguous** (0.5) so the pipeline still proceeds safely.

### `refine(lmaas, query, text) → (str, called)`
- **Skips** the call entirely when `len(text.split()) < REFINE_MIN_TOKENS` (180) — small context isn't worth cleaning. Returns `(text, False)`.
- Otherwise one **cheap** LLM call strips irrelevant material and returns `(cleaned, True)`. On error, returns the raw text (never fails the query). The `called` flag lets the node count the LLM call accurately.

### `generate(lmaas, query, knowledge) → str`
- One **strong**-model call (`cheap=False`) producing the final answer. This is the single expensive call per query.

### `CRAGQueryAgent`
- `__init__(lmaas, dis, graph, catalog=None, collection=None)` — inject clients + catalog.
- **`_primary_source(query)`** — chooses vector vs graph for the *primary* retrieval.
  It first asks the catalog which stores even hold docs relevant to the query
  (`candidates`, driven by the ingest routing), then applies this precedence:

```
                          query
                            │
                            ▼
        candidates = catalog.candidates(query)      → {vector: bool, graph: bool}
        (which stores hold docs whose title/domain/entities overlap the query;
         reflects Agent 1's routing — entity-dense docs are graph candidates,
         prose docs are vector candidates)
                            │
                            ▼
        ┌─ Is the query RELATIONAL?  (matches _RELATIONAL_HINTS:
        │   "how does", "depend", "connect", "between", "related", "cause"…)
        │        │
        │        └─ yes AND graph is a candidate ───────────────▶  GRAPH
        │                                                          (multi-hop / relationships)
        ▼ (not relational, or graph not a candidate)
        vector is a candidate? ─── yes ───────────────────────────▶  VECTOR
        │ no                                                         (semantic passage lookup)
        ▼
        graph is a candidate?  ─── yes ───────────────────────────▶  GRAPH
        │ no
        ▼
        VECTOR   (default — nothing catalogued yet)
```

  In plain terms: **relational-sounding questions prefer the graph; otherwise the
  store that actually holds the relevant doc wins** (a query about an entity-dense
  doc lands on graph, a query about a prose doc lands on vector). Whatever the
  primary pick, the CRAG evaluator still checks it and can correct via vectorless.
- **`_retrieve(source, query)`** — dispatches to `graph_retrieve` or `vector_retrieve`.
- **`_n_route / _n_retrieve / _n_evaluate / _n_correct / _n_ambiguous / _n_incorrect / _n_generate`** — the graph nodes; each updates `QueryState` and increments `calls`.
- **`run(query) → QueryResult`** — `invoke`s the compiled StateGraph, then builds the result. What flows through the nodes (tracking `calls` and `trace`):
  1. `route` picks the primary source; `retrieve` fetches it. (No LLM call yet.)
  2. `evaluate(...)`. Counts 1 call *unless* retrieval was empty (then 0 — the empty-shortcut).
  3. Branch on verdict:
     - **correct:** `refine` (counts a call only if context ≥ 180 words) → knowledge = refined primary. `sources = [primary]`. *Early exit — cheapest path.*
     - **ambiguous:** `refine` primary (k_in) + `vectorless_retrieve` (k_ex, +1 nav call) → `_fuse(k_in, k_ex)`. `sources = [primary] (+ "vectorless" if it returned anything)`.
     - **incorrect:** `vectorless_retrieve` (+1 call). If it's empty, **fall back to the other internal store** (graph↔vector) — **never the web**. `sources` reflects whichever recovered.
  4. If no knowledge was assembled → **abstain** with a fixed message (no generation call). Otherwise `generate(...)` (+1 strong call).
  - Returns `QueryResult(answer, verdict, sources, llm_calls, trace)`.

### `_fuse(k_in, k_ex)`
Concatenates the two knowledge blocks with a `---` separator (drops empties). This is
the "**fusion, not selection**" step — the generator sees both the primary context
and the vectorless recovery, rather than us picking one.

### The cost story, precisely
| Verdict | LLM calls | Which models |
|---|---|---|
| correct | 1 eval + (0–1 refine) + 1 gen = **2–3** | cheap, cheap, **strong** |
| ambiguous | 1 eval + (0–1 refine) + 1 nav + 1 gen = **3–4** | cheap×N, **strong** |
| incorrect | 1 eval + 1 nav + 1 gen = **3** (+1 if fallback) | cheap×N, **strong** |
| empty retrieval → incorrect | 0 eval + 1 nav + 1 gen = **2** | cheap, **strong** |
The strong model fires **once** (generation); everything else is the cheap tier.

---

## 8. `fakes.py` — offline stand-ins

These duck-type the real clients so `demo.py`/tests run with no credentials.

### Overlap primitives
- `_terms(text)` — words of length ≥ 4 (same idea as the catalog's).
- `_recall(query, context)` — |query terms ∩ context terms| / |query terms|. This single number **drives the CRAG verdict offline**: high overlap → high score → correct; none → incorrect. So verdicts depend on retrieval quality, not randomness.
- `_section(user, marker, end_markers)` — pulls the text after a marker (e.g. `QUESTION:`) out of a prompt, so the fake can "read" what it's being asked.

### `FakeLMaaS`
Inspects the **system prompt** to tell which step is calling, then responds:
- domain → `{domain}` (keyword guess); entity-scan → `{entities}` (capitalized phrases in the window via `_caps_phrases`, with a small stopword filter).
- evaluate → `{score = _recall(question, context), reason}`.
- navigate → `{section_ids}` = headings whose text overlaps the question.
- refine → returns the first ~80 words (pretends to keep the salient part).
- generate → `"[fake answer] Based on the retrieved knowledge: <first 50 words>"`.
- Records every call in `self.calls` (handy for assertions).

### `FakeDIS(coverage_words=60)`
In-memory vector store with **deliberately shallow coverage**: `upload_document`
stores only the first `coverage_words` of each doc. `retrieve` scores those heads by
`_recall` and returns the top-k with score > 0. The shallow index is what makes the
**corrective path demonstrable** — deep-section queries miss vector and get recovered
by vectorless navigation over the full graph tree. (Tests set `coverage_words=12` for
tight control.)

### `FakeGraph`
In-memory `section_id → {doc_id, heading, chunks}`. Implements the same method surface
as the real `GraphClient` (`upsert_document_tree`, `get_sections`, `get_section_texts`,
`keyword_search`) so it's a drop-in.

---

## 9. `demo.py` and `tests/test_hybrid.py`

- **`demo.py`** — wires the fakes (`USE_FAKES=True`), ingests a manual + a policy doc
  through Agent 1, then runs four queries through Agent 2, printing verdict, sources,
  LLM-call count, and the step trace. Flip `USE_FAKES=False` (and fill config) for real.
- **`tests/test_hybrid.py`** — 7 offline tests proving: catalog round-trip; Agent 1
  routes manuals to both stores and prose to vector-only; and the four Agent 2
  outcomes — **correct** (early exit), **incorrect** (recovers via vectorless),
  **ambiguous** (fuses), and **abstain** (nothing internal matches).

---

## 10. End-to-end narration (say this in the demo)

**Ingest:** "A document comes in. Agent 1 always stores it in DIS for vector search,
and — if a cheap LLM classifier judges it entity-dense — also builds a knowledge
graph in Neptune. It records where the doc went in a routing catalog."

**Query:** "A question comes in. Agent 2 checks the catalog to pick the right store,
retrieves, then a cheap **evaluator** scores whether that context can actually answer
the question. Above 0.7 we trust it and answer. Below 0.3 we discard it and correct
by **reasoning over the document's structure** — vectorless retrieval — staying fully
internal, no web. In between, we keep both and fuse them. Only the final answer uses
the expensive model; every check runs on a cheap one."

**Why it's better than plain RAG:** "It doesn't blindly trust one store or one
retrieval — it picks the right tool per document and per query, checks its own work,
and corrects when the evidence is weak."
