"""Central configuration — all secrets/endpoints are REPLACE_ME placeholders.

Every value can be overridden by an environment variable. The OFFLINE demo/tests
don't read any of these (they use fakes); they only matter when you wire the real
LMaaS / DIS / Neptune clients.

NEVER hard-code real secrets here — fill them via env var / secret injection.
"""

from __future__ import annotations

import os

# =============================================================================
# LMaaS  (Azure OpenAI gateway + IDAM client-credentials) — same as pdf-to-kg-agent
# =============================================================================
LMAAS_ENDPOINT: str = os.environ.get("LMAAS_ENDPOINT", "REPLACE_ME")          # https://lmaas-<env>.ailab.gehealthcare.net/
LMAAS_DEPLOYMENT_STRONG: str = os.environ.get("LMAAS_DEPLOYMENT_STRONG", "REPLACE_ME")  # generator (quality)
LMAAS_DEPLOYMENT_CHEAP: str = os.environ.get("LMAAS_DEPLOYMENT_CHEAP", "REPLACE_ME")    # evaluator/nav/refine (mini/nano)
LMAAS_API_VERSION: str = os.environ.get("LMAAS_API_VERSION", "2025-04-01-preview")
LMAAS_TIMEOUT: int = int(os.environ.get("LMAAS_TIMEOUT", "120"))
# TLS verification for the IDAM token calls AND the LMaaS gateway. The official
# sample disables it (verify=False) because the internal cert isn't in the
# default trust store; set LMAAS_VERIFY_SSL=false if you hit SSLCertVerificationError.
LMAAS_VERIFY_SSL: bool = os.environ.get("LMAAS_VERIFY_SSL", "true").lower() == "true"
# GPT-5.x / reasoning deployments require `max_completion_tokens` (not `max_tokens`)
# and reject any non-default `temperature`. Keep true for gpt-5.* deployments; set
# false for legacy gpt-4o-style deployments that use max_tokens + temperature.
LMAAS_REASONING_MODEL: bool = os.environ.get("LMAAS_REASONING_MODEL", "true").lower() == "true"

# IDAM client-credentials (LMaaS token source)
IDAM_TOKEN_ENDPOINT: str = os.environ.get("IDAM_TOKEN_ENDPOINT", "https://idam.gehealthcloud.io/oauth2/token")
IDAM_APP_CLIENT_ID: str = os.environ.get("IDAM_APP_CLIENT_ID", "REPLACE_ME")
IDAM_APP_CLIENT_SECRET: str = os.environ.get("IDAM_APP_CLIENT_SECRET", "REPLACE_ME")
LMAAS_AUDIENCE: str = os.environ.get("LMAAS_AUDIENCE", "REPLACE_ME")

# =============================================================================
# DIS  (managed vector RAG + IDAM token-exchange)
# =============================================================================
# DIS auth is IDAM app-to-app with a TOKEN-EXCHANGE step: get an app token, then
# exchange it for a DIS-scoped token (audience = DIS app id) carrying RBAC roles.
DIS_APP_AUDIENCE: str = os.environ.get("DIS_APP_AUDIENCE", "REPLACE_ME")       # DIS app client id (e.g. ShePtGvVv4kWAbUpe70IpgqaoT4a)
# DIS REST base + service paths. Exact paths live in the "DIS API Collection"
# Bruno export — fill these once you have it.
DIS_BASE_URL: str = os.environ.get("DIS_BASE_URL", "REPLACE_ME")              # https://<dis-host>.ailab.gehealthcare.net
DIS_PATH_CREATE_COLLECTION: str = os.environ.get("DIS_PATH_CREATE_COLLECTION", "REPLACE_ME")  # e.g. /collections
DIS_PATH_UPLOAD: str = os.environ.get("DIS_PATH_UPLOAD", "REPLACE_ME")        # e.g. /collections/{collection}/documents
DIS_PATH_RETRIEVE: str = os.environ.get("DIS_PATH_RETRIEVE", "REPLACE_ME")    # e.g. /collections/{collection}/retrieve
DIS_COLLECTION: str = os.environ.get("DIS_COLLECTION", "hybrid-rag-demo")
DIS_VERIFY_SSL: bool = os.environ.get("DIS_VERIFY_SSL", "true").lower() == "true"

# =============================================================================
# Neptune  (graph + vectorless section-tree)
# =============================================================================
NEPTUNE_ENDPOINT: str = os.environ.get("NEPTUNE_ENDPOINT", "REPLACE_ME")
NEPTUNE_PORT: int = int(os.environ.get("NEPTUNE_PORT", "8182"))
NEPTUNE_USE_IAM: bool = os.environ.get("NEPTUNE_USE_IAM", "true").lower() == "true"
AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")

# =============================================================================
# Routing catalog
# =============================================================================
CATALOG_PATH: str = os.environ.get("CATALOG_PATH", "routing_catalog.json")

# =============================================================================
# Retrieval + CRAG tuning
# =============================================================================
RETRIEVE_TOP_K: int = int(os.environ.get("RETRIEVE_TOP_K", "5"))

# Retrieval-evaluator thresholds (score in [0,1]): the CRAG verdict boundaries.
CRAG_CORRECT_THRESHOLD: float = float(os.environ.get("CRAG_CORRECT_THRESHOLD", "0.7"))
CRAG_INCORRECT_THRESHOLD: float = float(os.environ.get("CRAG_INCORRECT_THRESHOLD", "0.3"))

# Only run the (extra) refinement call when context is bigger than this (tokens ~ words).
REFINE_MIN_TOKENS: int = int(os.environ.get("REFINE_MIN_TOKENS", "180"))

# --- Agent-1 ingest: full-document graph-worthiness gate ---------------------
# Agent 1 scans the WHOLE document (windowed, cheap model), counts the DISTINCT
# entities, and builds a graph only if there are enough to make relationships
# worthwhile. This replaces the old "count a 2000-char sample" heuristic.
INGEST_MIN_GRAPH_ENTITIES: int = int(os.environ.get("INGEST_MIN_GRAPH_ENTITIES", "8"))
# Words per entity-scan window (each window = one cheap LLM extraction call).
INGEST_ENTITY_SCAN_WINDOW_WORDS: int = int(os.environ.get("INGEST_ENTITY_SCAN_WINDOW_WORDS", "400"))
# Cost cap: never make more than this many scan calls per document. If the doc
# has more windows than this, they are sampled EVENLY across it (so we still
# "look at the whole document", just not every window on very long docs).
INGEST_ENTITY_SCAN_MAX_WINDOWS: int = int(os.environ.get("INGEST_ENTITY_SCAN_MAX_WINDOWS", "8"))

# Storage-routing policy. By default an entity-dense document goes to the GRAPH
# ONLY (not also the vector store) — the graph + vectorless retrieval serve it,
# and we avoid duplicating entity-rich content as vectors. Set this true to also
# vectorize graph docs (dual-store) if you want semantic search on them too.
STORE_GRAPH_DOCS_IN_VECTOR: bool = (
    os.environ.get("STORE_GRAPH_DOCS_IN_VECTOR", "false").lower() == "true"
)


def is_configured(value: str) -> bool:
    """True if a setting has been filled in (not left as REPLACE_ME/empty)."""
    return bool(value) and value != "REPLACE_ME"
