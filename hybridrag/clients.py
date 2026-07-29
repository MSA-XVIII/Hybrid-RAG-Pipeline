"""Clients for the three real backends. Each is behind a small interface so the
fakes in ``fakes.py`` can stand in for offline testing.

  * LMaaSClient  — Azure OpenAI gateway, IDAM client-credentials auth (like pdf-to-kg-agent)
  * DISClient    — DIS vector RAG, IDAM TOKEN-EXCHANGE auth + RBAC roles
  * GraphClient  — Neptune Gremlin over HTTPS

Third-party imports (openai/requests/boto3) are LAZY — importing this module
costs nothing, so the offline demo/tests run without those packages installed.
All secrets/endpoints come from config.py, defaulting to REPLACE_ME; a client
raises NotConfigured if you try to use it before filling them in.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from hybridrag import config

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.S)
_TOKEN_SKEW_S = 60


class NotConfigured(RuntimeError):
    """Raised when a real client is used while its config is still REPLACE_ME."""


class BackendError(RuntimeError):
    """Transport/parse failure talking to a backend."""


def _require(*pairs: tuple[str, str]) -> None:
    missing = [name for name, val in pairs if not config.is_configured(val)]
    if missing:
        raise NotConfigured(
            "These settings are still REPLACE_ME: " + ", ".join(missing)
            + ". Fill them in config.py (or env), or run with fakes."
        )


def _lit(value: Any) -> str:
    """Quote a value as a Gremlin/Groovy string literal."""
    s = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def _label(value: str, fallback: str) -> str:
    """Sanitize an LLM-supplied type/relation into a safe Gremlin label."""
    s = re.sub(r"[^A-Za-z0-9_]", "", str(value or ""))
    return s or fallback


def _values(resp: dict) -> list[dict]:
    """Pull the list of result maps from a Gremlin HTTP response."""
    try:
        return resp["result"]["data"] or []
    except (KeyError, TypeError):
        return []


def _parse_json(content: str) -> dict | list:
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = _FENCED_JSON_RE.search(content)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    raise BackendError(f"Could not parse JSON from model output: {content[:300]}")


# =============================================================================
# IDAM token generators
# =============================================================================

class IDAMClientCredentials:
    """LMaaS auth. TWO-leg IDAM flow, matching the official aifabric-lmaas-sample
    (idam_token_generator.py):

        1. client_credentials grant          -> base access token
        2. token_exchange grant (Basic auth) -> LMaaS-scoped token (audience=LMAAS_AUDIENCE)

    The *exchange* token (leg 2) is what the gateway accepts as the bearer token;
    a plain leg-1 client-credentials token is rejected. Cached + auto-refreshed.
    """

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def _access_token(self) -> str:
        """Leg 1: client-credentials grant -> base access token."""
        import uuid  # lazy
        import requests  # lazy

        data = {
            "grant_type": "client_credentials",
            "client_id": config.IDAM_APP_CLIENT_ID,
            "client_secret": config.IDAM_APP_CLIENT_SECRET,
            "scope": str(uuid.uuid4()),
        }
        resp = requests.post(
            config.IDAM_TOKEN_ENDPOINT, data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=config.LMAAS_TIMEOUT, verify=config.LMAAS_VERIFY_SSL,
        )
        if resp.status_code >= 400:
            raise BackendError(f"IDAM access-token {resp.status_code}: {resp.text[:300]}")
        return resp.json()["access_token"]

    def token(self) -> str:
        if self._token and time.monotonic() < self._expires_at:
            return self._token
        with self._lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            _require(
                ("IDAM_APP_CLIENT_ID", config.IDAM_APP_CLIENT_ID),
                ("IDAM_APP_CLIENT_SECRET", config.IDAM_APP_CLIENT_SECRET),
                ("LMAAS_AUDIENCE", config.LMAAS_AUDIENCE),
            )
            import uuid  # lazy
            import requests  # lazy
            from requests.auth import HTTPBasicAuth

            access_token = self._access_token()
            # Leg 2: exchange the base token for an LMaaS-scoped token.
            payload = {
                "grant_type": "token_exchange",
                "token": access_token,
                "audience": config.LMAAS_AUDIENCE,
                "scope": str(uuid.uuid4()),
            }
            resp = requests.post(
                config.IDAM_TOKEN_ENDPOINT,
                auth=HTTPBasicAuth(config.IDAM_APP_CLIENT_ID, config.IDAM_APP_CLIENT_SECRET),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=payload, timeout=config.LMAAS_TIMEOUT, verify=config.LMAAS_VERIFY_SSL,
            )
            if resp.status_code >= 400:
                raise BackendError(f"IDAM token-exchange {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
            self._token = body["access_token"]
            self._expires_at = time.monotonic() + max(0.0, float(body.get("expires_in", 300)) - _TOKEN_SKEW_S)
            return self._token


class IDAMTokenExchange:
    """OAuth2 token-exchange token (DIS): app token -> DIS-scoped token w/ roles."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def _app_token(self) -> str:
        # DIS onboarding issues an app_user token; here we reuse the same IDAM
        # client-credentials grant to obtain the base app token before exchange.
        import requests  # lazy

        data = {
            "grant_type": "client_credentials",
            "client_id": config.IDAM_APP_CLIENT_ID,
            "client_secret": config.IDAM_APP_CLIENT_SECRET,
            "scope": "openid",
        }
        resp = requests.post(
            config.IDAM_TOKEN_ENDPOINT, data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=config.LMAAS_TIMEOUT, verify=config.DIS_VERIFY_SSL,
        )
        if resp.status_code >= 400:
            raise BackendError(f"IDAM app-token {resp.status_code}: {resp.text[:300]}")
        return resp.json()["access_token"]

    def token(self) -> str:
        if self._token and time.monotonic() < self._expires_at:
            return self._token
        with self._lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            _require(
                ("IDAM_APP_CLIENT_ID", config.IDAM_APP_CLIENT_ID),
                ("IDAM_APP_CLIENT_SECRET", config.IDAM_APP_CLIENT_SECRET),
                ("DIS_APP_AUDIENCE", config.DIS_APP_AUDIENCE),
            )
            import requests  # lazy
            from requests.auth import HTTPBasicAuth

            app_token = self._app_token()
            # token-exchange -> DIS-scoped token carrying mapped RBAC roles
            payload = {
                "grant_type": "token_exchange",
                "scope": "openid",
                "token": app_token,
                "audience": config.DIS_APP_AUDIENCE,
            }
            resp = requests.post(
                config.IDAM_TOKEN_ENDPOINT,
                auth=HTTPBasicAuth(config.IDAM_APP_CLIENT_ID, config.IDAM_APP_CLIENT_SECRET),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=payload, timeout=config.LMAAS_TIMEOUT, verify=config.DIS_VERIFY_SSL,
            )
            if resp.status_code >= 400:
                raise BackendError(f"IDAM token-exchange {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
            self._token = body["access_token"]
            self._expires_at = time.monotonic() + max(0.0, float(body.get("expires_in", 300)) - _TOKEN_SKEW_S)
            return self._token


# =============================================================================
# LMaaS  (Azure OpenAI + IDAM client-credentials)
# =============================================================================

class LMaaSClient:
    """Chat client. ``cheap=True`` selects the cheap deployment (evaluator / nav /
    refine); the strong deployment is reserved for final generation."""

    def __init__(self, token_gen: IDAMClientCredentials | None = None) -> None:
        self._token_gen = token_gen or IDAMClientCredentials()
        self._client: Any = None
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    _require(("LMAAS_ENDPOINT", config.LMAAS_ENDPOINT))
                    from openai import AzureOpenAI  # lazy

                    # Honour LMAAS_VERIFY_SSL for the gateway too (not just IDAM):
                    # when False, hand the SDK an httpx client with TLS off.
                    http_client = None
                    if not config.LMAAS_VERIFY_SSL:
                        import httpx  # lazy (bundled with openai)
                        http_client = httpx.Client(verify=False)

                    self._client = AzureOpenAI(
                        azure_endpoint=config.LMAAS_ENDPOINT.rstrip("/") + "/",
                        api_version=config.LMAAS_API_VERSION,
                        azure_ad_token_provider=self._token_gen.token,
                        timeout=config.LMAAS_TIMEOUT,
                        http_client=http_client,
                    )
        return self._client

    def _deployment(self, cheap: bool) -> str:
        dep = config.LMAAS_DEPLOYMENT_CHEAP if cheap else config.LMAAS_DEPLOYMENT_STRONG
        _require(("LMAAS_DEPLOYMENT", dep))
        return dep

    def _sampling(self, max_tokens: int, temperature: float) -> dict:
        """Params vary by model family: gpt-5.x/reasoning models use
        `max_completion_tokens` and reject non-default `temperature`; legacy
        gpt-4o-style models use `max_tokens` + `temperature` (see LMAAS_REASONING_MODEL)."""
        if config.LMAAS_REASONING_MODEL:
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens, "temperature": temperature}

    def complete_text(self, system: str, user: str, *, cheap: bool = False,
                      max_tokens: int = 1024, temperature: float = 0.0) -> str:
        resp = self._get_client().chat.completions.create(
            model=self._deployment(cheap),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self._sampling(max_tokens, temperature),
        )
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise BackendError(f"Unexpected LMaaS response: {str(resp)[:300]}") from exc

    def complete_json(self, system: str, user: str, *, cheap: bool = False,
                     max_tokens: int = 4096) -> dict | list:
        resp = self._get_client().chat.completions.create(
            model=self._deployment(cheap),
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self._sampling(max_tokens, 0.0),
        )
        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise BackendError(f"Unexpected LMaaS response: {str(resp)[:300]}") from exc
        return _parse_json(content)


# =============================================================================
# DIS  (managed vector RAG + IDAM token-exchange)
# =============================================================================

class DISClient:
    """Vector RAG over DIS. Needs the Admin role for ingest and User for query.

    Endpoint PATHS are REPLACE_ME until pulled from the DIS API Collection (Bruno).
    Methods construct the documented request shape and attach the DIS-scoped token.
    """

    def __init__(self, token_gen: IDAMTokenExchange | None = None) -> None:
        self._token_gen = token_gen or IDAMTokenExchange()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token_gen.token()}",
                "Content-Type": "application/json"}

    def _url(self, path: str, **fmt: str) -> str:
        _require(("DIS_BASE_URL", config.DIS_BASE_URL), ("DIS path", path))
        return config.DIS_BASE_URL.rstrip("/") + "/" + path.lstrip("/").format(**fmt)

    def create_collection(self, name: str) -> str:
        import requests  # lazy
        url = self._url(config.DIS_PATH_CREATE_COLLECTION)
        resp = requests.post(url, headers=self._headers(), json={"name": name},
                             timeout=config.LMAAS_TIMEOUT, verify=config.DIS_VERIFY_SSL)
        if resp.status_code >= 400:
            raise BackendError(f"DIS create_collection {resp.status_code}: {resp.text[:300]}")
        return (resp.json() or {}).get("id", name)

    def upload_document(self, collection: str, doc_id: str, text: str,
                        metadata: dict | None = None) -> None:
        import requests  # lazy
        url = self._url(config.DIS_PATH_UPLOAD, collection=collection)
        payload = {"doc_id": doc_id, "text": text, "metadata": metadata or {}}
        resp = requests.post(url, headers=self._headers(), json=payload,
                             timeout=config.LMAAS_TIMEOUT, verify=config.DIS_VERIFY_SSL)
        if resp.status_code >= 400:
            raise BackendError(f"DIS upload {resp.status_code}: {resp.text[:300]}")

    def retrieve(self, query: str, collection: str, top_k: int = 5) -> list[dict]:
        """Return [{doc_id, text, score, ...}] passages for the query."""
        import requests  # lazy
        url = self._url(config.DIS_PATH_RETRIEVE, collection=collection)
        payload = {"query": query, "top_k": top_k}
        resp = requests.post(url, headers=self._headers(), json=payload,
                             timeout=config.LMAAS_TIMEOUT, verify=config.DIS_VERIFY_SSL)
        if resp.status_code >= 400:
            raise BackendError(f"DIS retrieve {resp.status_code}: {resp.text[:300]}")
        body = resp.json() or {}
        # Shape TBD from the Bruno collection; accept a few common shapes.
        return body.get("results") or body.get("passages") or body.get("chunks") or []


# =============================================================================
# Graph  (Neptune Gremlin over HTTPS)
# =============================================================================

class GraphClient:
    """Neptune-backed graph store.

    Exposes the higher-level operations Agent 1 (ingest) and the graph/vectorless
    retrievers need, implemented as Gremlin over the pdf-to-kg-agent schema
    (Document -HAS_SECTION-> Section -HAS_CHUNK-> Chunk). The FakeGraph in
    fakes.py implements the SAME method surface in memory for offline runs.

    The Gremlin below targets that schema and may need tuning to your exact
    property names; it is only exercised against a real Neptune (offline uses
    the fake), so treat these as REPLACE_ME-quality starting points.
    """

    # --- high-level store API (mirrored by FakeGraph) -----------------------

    def upsert_document_tree(self, doc_id: str, title: str, sections: list[dict]) -> None:
        """sections = [{"order": int, "heading": str, "chunks": [str, ...]}]."""
        # Minimal idempotent upsert of Document/Section/Chunk. A full entity
        # graph is built by the pdf-to-kg-agent pipeline; here we persist the
        # structural tree that powers graph + vectorless retrieval.
        g = [f"g.V('doc_{doc_id}').fold().coalesce(unfold(),"
             f"addV('Document').property(id,'doc_{doc_id}')).property('title',{_lit(title)})"]
        for s in sections:
            sid = f"sec_{doc_id}_{s.get('order', 0)}"
            g.append(f"g.V('{sid}').fold().coalesce(unfold(),addV('Section').property(id,'{sid}'))"
                     f".property('heading',{_lit(s.get('heading',''))}).property('docId','doc_{doc_id}')")
            g.append(f"g.V('doc_{doc_id}').as('d').V('{sid}').coalesce(__.inE('HAS_SECTION'),"
                     f"__.addE('HAS_SECTION').from('d'))")
            for i, ch in enumerate(s.get("chunks", []) or []):
                cid = f"{sid}_c{i}"
                g.append(f"g.V('{cid}').fold().coalesce(unfold(),addV('Chunk').property(id,'{cid}'))"
                         f".property('text',{_lit(ch)}).property('docId','doc_{doc_id}')")
                g.append(f"g.V('{sid}').as('s').V('{cid}').coalesce(__.inE('HAS_CHUNK'),"
                         f"__.addE('HAS_CHUNK').from('s'))")
        for stmt in g:
            self.run_gremlin(stmt)

    def upsert_entities(self, doc_id: str, mentions: list[dict]) -> None:
        """Create Entity nodes and Section-MENTIONS->Entity edges.

        ``mentions`` = [{"order": int, "entity": str}]. Entity ids are GLOBAL
        (name-derived), so the same entity across sections/docs merges into one
        node — the shared node that interconnects the graph.
        """
        from hybridrag.common import entity_id
        g: list[str] = []
        for m in mentions:
            eid = entity_id(m["entity"])
            sid = f"sec_{doc_id}_{m.get('order', 0)}"
            g.append(f"g.V('{eid}').fold().coalesce(unfold(),"
                     f"addV('Entity').property(id,'{eid}')).property('name',{_lit(m['entity'])})")
            # add the edge only if this specific section->entity edge is absent
            g.append(f"g.V('{sid}').as('s').V('{eid}').coalesce("
                     f"__.inE('MENTIONS').where(__.outV().hasId('{sid}')),"
                     f"__.addE('MENTIONS').from('s'))")
        for stmt in g:
            self.run_gremlin(stmt)

    def upsert_typed_entities(self, nodes: list[dict], edges: list[dict]) -> None:
        """Persist an LMaaS-built typed entity graph (see KnowledgeGraph.to_store()).

        ``nodes`` = [{id, name, type}] -> vertices labelled with the LLM-assigned type
        ``edges`` = [{source, target, relation, count}] -> typed entity->entity edges

        Idempotent: vertices upsert by id, an edge upserts on (source, relation, target).
        Neptune cannot change a vertex label after creation, so the type is ALSO stored
        as an ``etype`` property — that is what reads/filters use.
        """
        for n in nodes:
            nid, label = n["id"], _label(n.get("type"), "Entity")
            self.run_gremlin(
                f"g.V({_lit(nid)}).fold().coalesce(unfold(),"
                f"addV('{label}').property(id,{_lit(nid)}))"
                f".property('name',{_lit(n.get('name', nid))})"
                f".property('etype',{_lit(n.get('type', 'Entity'))})")
        for e in edges:
            sid, oid, rel = e["source"], e["target"], _label(e.get("relation"), "RELATED_TO")
            self.run_gremlin(
                f"g.V({_lit(sid)}).as('s').V({_lit(oid)}).coalesce("
                f"__.inE('{rel}').where(__.outV().hasId({_lit(sid)})),"
                f"__.addE('{rel}').from('s')).property('count',{int(e.get('count', 1))})")

    def get_entity_graph(self, limit: int = 400) -> tuple[list[dict], list[dict]]:
        """Read the typed entity graph back out of Neptune.

        Returns (nodes, edges) in the shape visualize.entity_kg_html() expects.
        """
        nres = self.run_gremlin(
            f"g.V().has('etype').limit({limit})"
            ".project('id','label','group').by(id).by('name').by('etype')")
        eres = self.run_gremlin(
            f"g.V().has('etype').outE().where(__.inV().has('etype')).limit({limit * 4})"
            ".project('from','to','label').by(__.outV().id()).by(__.inV().id()).by(label)")
        return _values(nres), _values(eres)

    def get_sections(self, doc_ids: list[str] | None = None) -> list[dict]:
        """Headings only (cheap) — for vectorless navigation."""
        filt = ""
        if doc_ids:
            ids = ",".join(f"'doc_{d}'" for d in doc_ids)
            filt = f".has('docId',within({ids}))"
        res = self.run_gremlin(
            f"g.V().hasLabel('Section'){filt}.project('section_id','heading','doc_id')"
            ".by(id).by('heading').by('docId')")
        return _values(res)

    def get_section_texts(self, section_ids: list[str]) -> list[dict]:
        ids = ",".join(f"'{s}'" for s in section_ids)
        res = self.run_gremlin(
            f"g.V({ids}).project('section_id','heading','text')"
            ".by(id).by('heading').by(__.out('HAS_CHUNK').values('text').fold())")
        return _values(res)

    def get_tree(self, doc_ids: list[str] | None = None) -> list[dict]:
        """Full Document -> Section -> Chunk tree (for visualization)."""
        filt = ""
        if doc_ids:
            ids = ",".join(f"'doc_{d}'" for d in doc_ids)
            filt = f".has(id,within({ids}))"
        res = self.run_gremlin(
            f"g.V().hasLabel('Document'){filt}.project('doc_id','title','sections','entities')"
            ".by(id).by(coalesce(values('title'),constant('')))"
            ".by(__.out('HAS_SECTION').project('section_id','order','heading','chunks','entities')"
            ".by(id).by(coalesce(values('order'),constant(0))).by('heading')"
            ".by(__.out('HAS_CHUNK').project('chunk_id','text').by(id).by('text').fold())"
            ".by(__.out('MENTIONS').id().fold()).fold())"
            ".by(__.out('HAS_SECTION').out('MENTIONS').dedup()"
            ".project('entity_id','name').by(id).by('name').fold())")
        return _values(res)

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        """Containment search over Chunk text. Uses TextP.containing on the first
        salient term; for production, back this with Neptune full-text search
        (OpenSearch) rather than a substring scan."""
        terms = [t for t in query.lower().split() if len(t) >= 4]
        if not terms:
            return []
        res = self.run_gremlin(
            f"g.V().hasLabel('Chunk').has('text',TextP.containing({_lit(terms[0])}))"
            f".limit({limit}).project('doc_id','section_id','text').by('docId').by(id).by('text')")
        return _values(res)

    # --- raw Gremlin --------------------------------------------------------

    def run_gremlin(self, gremlin: str) -> dict:
        import requests  # lazy
        _require(("NEPTUNE_ENDPOINT", config.NEPTUNE_ENDPOINT))
        url = f"https://{config.NEPTUNE_ENDPOINT}:{config.NEPTUNE_PORT}/gremlin"
        # Ask Neptune for UNTYPED GraphSON so result.data is plain JSON (list/dict/int),
        # not typed wrappers like {"@type":"g:List","@value":[...]}. Without this the read
        # helpers (_values, get_tree) receive dicts where they expect lists.
        headers = {"Content-Type": "application/json",
                   "Accept": "application/vnd.gremlin-v3.0+json;types=false"}
        data = json.dumps({"gremlin": gremlin})
        if config.NEPTUNE_USE_IAM:
            headers.update(self._sigv4_headers(url, data))
        resp = requests.post(url, data=data, headers=headers, timeout=60,
                             verify=config.DIS_VERIFY_SSL)
        if resp.status_code >= 400:
            raise BackendError(f"Neptune {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _sigv4_headers(self, url: str, body: str) -> dict[str, str]:
        import boto3  # lazy
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        session = boto3.Session()
        creds = session.get_credentials()
        req = AWSRequest(method="POST", url=url, data=body,
                         headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "neptune-db", config.AWS_REGION).add_auth(req)
        return dict(req.headers)
