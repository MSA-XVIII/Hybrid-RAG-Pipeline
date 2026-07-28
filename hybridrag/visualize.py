"""Self-contained HTML visualizations of the knowledge graph and the PageIndex tree.

Two views over the same Document -> Section -> Chunk data (from a graph store's
``get_tree()`` — fake or real Neptune):

  * graph_html(tree)      — force-directed knowledge graph (Document/Section/Chunk
                            nodes + HAS_SECTION / HAS_CHUNK edges), like the earlier
                            patient / service-manual KG viewers.
  * pageindex_html(tree)  — hierarchical top-down tree (Document -> chapters ->
                            sub-sections), the structure the VECTORLESS retriever
                            reasons over. Sub-section nesting is inferred from
                            dotted heading numbers (e.g. 2.4.3 sits under 2.4).

Each returns a complete standalone HTML document (vis-network from a CDN). Render
inline in a notebook via an <iframe srcdoc>, or save to a file with save().
No third-party Python deps.
"""

from __future__ import annotations

import json
import re

_COLORS = {"Document": "#4C78A8", "Section": "#54A24B", "Chunk": "#E45756",
           "Entity": "#F58518"}

_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>html,body{margin:0;height:100%;font-family:Segoe UI,Arial,sans-serif}
#hdr{padding:8px 12px;font-weight:600;background:#f4f6f8;border-bottom:1px solid #e2e6ea}
#net{width:100%;height:calc(100% - 38px)}</style></head>
<body><div id="hdr">__TITLE__</div><div id="net"></div>
<script>(function(){function go(){if(typeof vis==='undefined'){return setTimeout(go,50);}
var nodes=new vis.DataSet(__NODES__);var edges=new vis.DataSet(__EDGES__);
new vis.Network(document.getElementById('net'),{nodes:nodes,edges:edges},__OPTIONS__);}go();})();</script>
</body></html>"""

_GROUPS = ("{Document:{color:'%(Document)s',shape:'box',font:{color:'#fff',bold:true}},"
           "Section:{color:'%(Section)s',shape:'box',font:{color:'#fff'}},"
           "Chunk:{color:'%(Chunk)s',shape:'dot',size:8},"
           "Entity:{color:'%(Entity)s',shape:'dot',size:14,font:{size:13}}}" % _COLORS)

_GRAPH_OPTS = ("{groups:%s,nodes:{font:{size:14}},"
               "edges:{arrows:'to',color:'#9aa5b1',font:{size:10,align:'middle',color:'#66707a'}},"
               "physics:{stabilization:true,barnesHut:{springLength:130}}}" % _GROUPS)

_PAGEINDEX_OPTS = ("{groups:%s,layout:{hierarchical:{enabled:true,direction:'UD',"
                   "sortMethod:'directed',levelSeparation:120,nodeSpacing:190}},physics:false,"
                   "nodes:{shape:'box',font:{size:13}},edges:{arrows:'to',color:'#9aa5b1'}}" % _GROUPS)


def _render(nodes: list[dict], edges: list[dict], title: str, options: str) -> str:
    return (_TEMPLATE.replace("__TITLE__", title)
            .replace("__NODES__", json.dumps(nodes))
            .replace("__EDGES__", json.dumps(edges))
            .replace("__OPTIONS__", options))


# --- knowledge-graph (network) view -----------------------------------------

def _graph_elements(tree: list[dict]) -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []
    seen_ent: set[str] = set()   # entity nodes are shared across docs — add once
    for doc in tree:
        did = f"doc_{doc['doc_id']}"
        nodes.append({"id": did, "label": doc.get("title") or doc["doc_id"], "group": "Document"})
        for ent in doc.get("entities", []):
            eid = ent["entity_id"]
            if eid not in seen_ent:
                seen_ent.add(eid)
                nodes.append({"id": eid, "label": ent.get("name") or eid, "group": "Entity"})
        for s in doc.get("sections", []):
            sid = s["section_id"]
            nodes.append({"id": sid, "label": s.get("heading") or sid,
                          "group": "Section", "title": sid})
            edges.append({"from": did, "to": sid, "label": "HAS_SECTION"})
            for c in s.get("chunks", []):
                snippet = c["text"][:80] + ("…" if len(c["text"]) > 80 else "")
                nodes.append({"id": c["chunk_id"], "label": "chunk", "group": "Chunk", "title": snippet})
                edges.append({"from": sid, "to": c["chunk_id"], "label": "HAS_CHUNK"})
            for eid in s.get("entities", []):   # Section -MENTIONS-> Entity (the interconnections)
                edges.append({"from": sid, "to": eid, "label": "MENTIONS"})
    return nodes, edges


def graph_html(tree: list[dict], title: str = "Hybrid RAG — Knowledge Graph") -> str:
    nodes, edges = _graph_elements(tree)
    return _render(nodes, edges, title, _GRAPH_OPTS)


# --- PageIndex (hierarchical tree) view --------------------------------------

def _leading_number(heading: str) -> str:
    m = re.match(r"^(\d+(?:\.\d+)*)\b", (heading or "").strip())
    return m.group(1) if m else ""


def _pageindex_elements(tree: list[dict]) -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []
    for doc in tree:
        did = f"doc_{doc['doc_id']}"
        nodes.append({"id": did, "label": f"Document\n{doc.get('title') or doc['doc_id']}",
                      "group": "Document", "level": 0})
        # map dotted number -> section id, to infer parent/child nesting
        by_num = {}
        for s in doc.get("sections", []):
            num = _leading_number(s.get("heading", ""))
            if num:
                by_num[num] = s["section_id"]
        for s in doc.get("sections", []):
            sid = s["section_id"]
            num = _leading_number(s.get("heading", ""))
            depth = len(num.split(".")) if num else 1
            parent = did
            if num:
                parts = num.split(".")
                for k in range(len(parts) - 1, 0, -1):
                    cand = ".".join(parts[:k])
                    if cand in by_num:
                        parent = by_num[cand]
                        break
            label = f"{s.get('heading') or sid}\nNode ID: {sid}"
            nodes.append({"id": sid, "label": label, "group": "Section", "level": depth})
            edges.append({"from": parent, "to": sid})
    return nodes, edges


def pageindex_html(tree: list[dict], title: str = "Vectorless — PageIndex Tree") -> str:
    nodes, edges = _pageindex_elements(tree)
    return _render(nodes, edges, title, _PAGEINDEX_OPTS)


# --- LMaaS-built typed knowledge graph (entity triples) ----------------------

# Cycled across whatever entity TYPES the LLM assigns (types are open-ended, so
# colours are allocated on the fly rather than from a fixed label map).
_TYPE_PALETTE = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
                 "#B279A2", "#EECA3B", "#FF9DA6", "#9D755D", "#BAB0AC")


def _pageindex_tree_elements(root: dict) -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []

    def walk(node: dict, node_id: str, level: int, parent_id: str | None) -> None:
        title = (node.get("title") or node_id).strip()
        if level == 0:
            label = f"Document\n{title}"
        else:
            summary = " ".join((node.get("summary") or "").split())
            if len(summary) > 70:
                summary = summary[:67] + "…"
            label = f"{title}\n{summary}" if summary else title
        nodes.append({"id": node_id, "label": label,
                      "group": "Document" if level == 0 else "Section", "level": level})
        if parent_id is not None:
            edges.append({"from": parent_id, "to": node_id})
        for i, child in enumerate(node.get("children") or []):
            walk(child, f"{node_id}.{i}", level + 1, node_id)

    walk(root, "root", 0, None)
    return nodes, edges


def pageindex_tree_html(root: dict, title: str = "LMaaS-built PageIndex Tree") -> str:
    """Hierarchical top-down view of an LLM-built PageIndex tree (nested
    {title, summary, children} from :meth:`PageIndexExtractor.build_one`)."""
    nodes, edges = _pageindex_tree_elements(root)
    return _render(nodes, edges, title, _PAGEINDEX_OPTS)


def entity_kg_html(nodes: list[dict], edges: list[dict],
                   title: str = "LMaaS-built Knowledge Graph") -> str:
    """Force-directed view of a typed entity graph from :meth:`KnowledgeGraph.to_vis`.

    ``nodes`` = [{id, label, group=<type>, title?}]; ``edges`` = [{from, to, label=<relation>}].
    Nodes are coloured by their ``group`` (entity type); edges are labelled by relation.
    """
    types = sorted({n.get("group", "Entity") for n in nodes})
    groups = ",".join(
        f"'{t}':{{color:'{_TYPE_PALETTE[i % len(_TYPE_PALETTE)]}',shape:'dot',"
        f"size:16,font:{{size:13}}}}" for i, t in enumerate(types))
    options = ("{groups:{%s},nodes:{font:{size:14}},"
               "edges:{arrows:'to',color:'#9aa5b1',font:{size:10,align:'middle',color:'#66707a'}},"
               "physics:{stabilization:true,barnesHut:{springLength:150}}}" % groups)
    return _render(nodes, edges, title, options)


# --- output helpers ----------------------------------------------------------

def save(html: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def iframe_srcdoc(html: str, height: int = 540) -> str:
    """Wrap a standalone doc as an isolated <iframe srcdoc> for notebook display."""
    import html as _h
    return (f'<iframe srcdoc="{_h.escape(html, quote=True)}" width="100%" '
            f'height="{height}" style="border:1px solid #ddd;border-radius:6px"></iframe>')
