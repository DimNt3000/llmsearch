"""SQLite-backed inverted index with BM25 ranking.

Documents are split into overlapping chunks; the inverted index maps
term -> (chunk, term frequency). Search scores chunks with BM25 and
results can be grouped per document for display or kept chunk-level
for retrieval-augmented answering.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have if in into is it its no not
    of on or such that the their then there these they this to was were will with
    you your we our i me my he she his her him they them what which who whom when
    where why how all any both each few more most other some than too very can
    just do does did doing""".split()
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(
    id       INTEGER PRIMARY KEY,
    url      TEXT UNIQUE NOT NULL,
    title    TEXT,
    source   TEXT NOT NULL,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks(
    id     INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    seq    INTEGER NOT NULL,
    text   TEXT NOT NULL,
    length INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS postings(
    term     TEXT NOT NULL,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tf       INTEGER NOT NULL,
    PRIMARY KEY(term, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


def tokenize(text: str) -> list[str]:
    return [
        t
        for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if 2 <= len(t) <= 40 and t not in STOPWORDS
    ]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~size-char chunks, preferring paragraph boundaries."""
    size = max(1, size)
    # The hard-split below advances by (size - overlap); clamp so a
    # misconfigured overlap >= size can never stall the loop.
    overlap = min(max(0, overlap), size // 2)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # Hard-split paragraphs that alone exceed the chunk size.
        while len(para) > size:
            head, para = para[:size], para[max(0, size - overlap):]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(para) + 2 > size and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def add_document(conn: sqlite3.Connection, url: str, title: str, text: str,
                 source: str, cfg: dict) -> int:
    """Index (or re-index) one document. Returns the number of chunks stored."""
    conn.execute("DELETE FROM docs WHERE url = ?", (url,))
    cur = conn.execute(
        "INSERT INTO docs(url, title, source, added_at) VALUES (?, ?, ?, ?)",
        (url, title or url, source, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    doc_id = cur.lastrowid
    chunks = chunk_text(text, cfg["chunk_chars"], cfg["chunk_overlap"])
    for seq, chunk in enumerate(chunks):
        tokens = tokenize(chunk)
        cur = conn.execute(
            "INSERT INTO chunks(doc_id, seq, text, length) VALUES (?, ?, ?, ?)",
            (doc_id, seq, chunk, len(tokens)),
        )
        chunk_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO postings(term, chunk_id, tf) VALUES (?, ?, ?)",
            [(term, chunk_id, tf) for term, tf in Counter(tokens).items()],
        )
    conn.commit()
    return len(chunks)


def search_chunks(conn: sqlite3.Connection, query: str, cfg: dict,
                  limit: int = 50) -> list[dict]:
    """BM25 over chunks. Returns [{chunk_id, doc_id, score, text, url, title}]."""
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []
    row = conn.execute("SELECT COUNT(*), COALESCE(AVG(length), 0) FROM chunks").fetchone()
    n_chunks, avgdl = row[0], row[1] or 1.0
    if not n_chunks:
        return []

    k1, b = cfg["bm25_k1"], cfg["bm25_b"]
    scores: dict[int, float] = {}
    doc_of: dict[int, int] = {}
    for term in terms:
        rows = conn.execute(
            "SELECT p.chunk_id, p.tf, c.length, c.doc_id "
            "FROM postings p JOIN chunks c ON c.id = p.chunk_id WHERE p.term = ?",
            (term,),
        ).fetchall()
        df = len(rows)
        if not df:
            continue
        idf = math.log(1 + (n_chunks - df + 0.5) / (df + 0.5))
        for chunk_id, tf, length, doc_id in rows:
            denom = tf + k1 * (1 - b + b * length / avgdl)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (tf * (k1 + 1)) / denom
            doc_of[chunk_id] = doc_id

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    if not top:
        return []
    placeholders = ",".join("?" * len(top))
    meta = {
        cid: (text, url, title)
        for cid, text, url, title in conn.execute(
            f"SELECT c.id, c.text, d.url, d.title FROM chunks c "
            f"JOIN docs d ON d.id = c.doc_id WHERE c.id IN ({placeholders})",
            [cid for cid, _ in top],
        )
    }
    return [
        {
            "chunk_id": cid,
            "doc_id": doc_of[cid],
            "score": score,
            "text": meta[cid][0],
            "url": meta[cid][1],
            "title": meta[cid][2],
        }
        for cid, score in top
        if cid in meta
    ]


def group_by_doc(chunks: list[dict], k: int) -> list[dict]:
    """Keep the best-scoring chunk per document, top k documents."""
    best: dict[int, dict] = {}
    for c in chunks:
        if c["doc_id"] not in best:
            best[c["doc_id"]] = c
    return list(best.values())[:k]


def top_contexts(chunks: list[dict], k: int, per_doc: int = 3) -> list[dict]:
    """Top k chunks for RAG, with at most per_doc chunks from one document."""
    out: list[dict] = []
    seen: Counter = Counter()
    for c in chunks:
        if seen[c["doc_id"]] < per_doc:
            out.append(c)
            seen[c["doc_id"]] += 1
        if len(out) >= k:
            break
    return out


def rrf_merge(rankings: list[list[dict]], c: int = 60) -> list[dict]:
    """Reciprocal-rank fusion of several chunk rankings (used by --smart)."""
    fused: dict[int, float] = {}
    by_id: dict[int, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            cid = item["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (c + rank + 1)
            by_id.setdefault(cid, item)
    merged = []
    for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        item = dict(by_id[cid])
        item["score"] = score
        merged.append(item)
    return merged


def make_snippet(text: str, query: str, max_chars: int) -> str:
    """Pick the window of the chunk containing the most query terms."""
    terms = set(tokenize(query))
    if not terms or len(text) <= max_chars:
        return " ".join(text[:max_chars].split())
    words = text.split()
    best_start, best_hits = 0, -1
    window = max(10, max_chars // 7)  # rough words-per-window
    step = max(1, window // 2)
    for start in range(0, max(1, len(words) - window + 1), step):
        hits = sum(1 for w in words[start:start + window] if w.strip(".,;:!?()[]\"'").lower() in terms)
        if hits > best_hits:
            best_start, best_hits = start, hits
    snippet = " ".join(words[best_start:best_start + window])
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0]
    prefix = "..." if best_start > 0 else ""
    return f"{prefix}{snippet}..."


def stats(conn: sqlite3.Connection) -> dict:
    docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    terms = conn.execute("SELECT COUNT(DISTINCT term) FROM postings").fetchone()[0]
    by_source = conn.execute(
        "SELECT source, COUNT(*) FROM docs GROUP BY source ORDER BY 2 DESC"
    ).fetchall()
    return {"documents": docs, "chunks": chunks, "unique_terms": terms,
            "by_source": dict(by_source)}
