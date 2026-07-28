"""Load documents from a folder so you can drop in your own files and ingest them.

Supports .md / .txt (read directly) and .pdf (via pdfplumber, lazily imported).
Returns records shaped for IngestAgent.ingest(doc_id, text, title).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

_SUPPORTED = {".md", ".txt", ".pdf"}

# Lines that are pure document furniture (page numbers, the "5000A EN 20020315"
# style footer code). Dropped entirely so they never become sections or chunks.
_BOILERPLATE_RE = re.compile(
    r"^(?:\d{1,4}|\d{3,}[A-Za-z]?\s+EN\s+\d{4,}|\d{3,}[A-Za-z]?|page\s+\d+)$",
    re.I,
)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "doc"


def _is_boilerplate(line: str) -> bool:
    s = line.strip()
    return not s or bool(_BOILERPLATE_RE.match(s))


def _read_pdf(path: Path) -> str:
    """Extract PDF text, marking real headings with a leading ``# `` so the
    ingester's structurer builds a meaningful tree.

    Headings are detected by TYPOGRAPHY, not word patterns: a line whose font is
    clearly larger than the document's body size (or bold and short) is a
    heading. This avoids mistaking numbered procedure steps and ALL-CAPS UI
    labels for headings — the failure mode of a pure text heuristic on real PDFs.
    """
    try:
        import pdfplumber  # lazy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Reading {path.name} needs pdfplumber — `pip install pdfplumber`."
        ) from exc

    # (text, font_size, is_bold) per reconstructed line, across all pages.
    lines: list[tuple[str, float, bool]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                words = page.extract_words(extra_attrs=["size", "fontname"])
            except Exception:  # noqa: BLE001 — fall back to plain text for this page
                words = []
            if not words:
                for ln in (page.extract_text() or "").splitlines():
                    lines.append((ln.strip(), 0.0, False))
                continue
            rows: dict[int, list[dict]] = defaultdict(list)
            for w in words:                       # group words into visual lines by y-position
                rows[round(float(w.get("top", 0)))].append(w)
            for top in sorted(rows):
                ws = sorted(rows[top], key=lambda x: float(x.get("x0", 0)))
                text = " ".join(w["text"] for w in ws).strip()
                size = max((float(w.get("size", 0)) for w in ws), default=0.0)
                bold = any("bold" in str(w.get("fontname", "")).lower() for w in ws)
                lines.append((text, round(size, 1), bold))

    # body size = the most common font size among substantial lines
    body_sizes = [sz for t, sz, _ in lines if sz > 0 and len(t.split()) >= 4]
    body = Counter(body_sizes).most_common(1)[0][0] if body_sizes else 0.0

    out: list[str] = []
    for text, size, bold in lines:
        if _is_boilerplate(text):
            continue
        n = len(text.split())
        is_heading = bool(body) and 0 < n <= 12 and not text.isdigit() and (
            size >= body * 1.15 or (bold and n <= 8 and size >= body * 0.98)
        )
        out.append(f"# {text}" if is_heading else text)
    return "\n".join(out)


def load_documents(docs_dir: str = "docs") -> list[dict]:
    """Return [{doc_id, title, text}] for every supported file in ``docs_dir``."""
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"docs folder not found: {root.resolve()}")
    out: list[dict] = []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in _SUPPORTED:
            continue
        text = _read_pdf(p) if p.suffix.lower() == ".pdf" else p.read_text(encoding="utf-8")
        if not text.strip():
            continue
        out.append({"doc_id": _slug(p.stem),
                    "title": p.stem.replace("_", " ").replace("-", " ").title(),
                    "text": text})
    return out
