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
from urllib.parse import unquote, urlparse

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
-- Re-indexing a document cascades from its chunks to their postings; without
-- this index every deleted chunk scanned the whole postings table.
CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);
-- The title field: terms of each document's title and URL name.
CREATE TABLE IF NOT EXISTS title_postings(
    term   TEXT NOT NULL,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    tf     INTEGER NOT NULL,
    PRIMARY KEY(term, doc_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_title_postings_doc ON title_postings(doc_id);
"""

# Stored in PRAGMA user_version. 1: documents have a title field.
_SCHEMA_VERSION = 1


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an index built by an older version up to date, once."""
    if conn.execute("PRAGMA user_version").fetchone()[0] >= _SCHEMA_VERSION:
        return
    conn.execute("BEGIN IMMEDIATE")  # one process migrates; the others wait, then skip
    try:
        if conn.execute("PRAGMA user_version").fetchone()[0] < 1:
            conn.execute("DELETE FROM title_postings")
            for doc_id, title, url in conn.execute("SELECT id, title, url FROM docs").fetchall():
                _index_title(conn, doc_id, title, url)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def title_terms(title: str, url: str) -> Counter:
    """Terms of a document's title field: its title plus the last part of its URL.

    For API documentation the URL often carries the full name that the title
    shortens: "Conv2d" is the title of both torch.nn.Conv2d.html and
    torch.ao.nn.qat.Conv2d.html.
    """
    name = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])
    return Counter(tokenize(f"{title or ''} {name}"))


def _index_title(conn: sqlite3.Connection, doc_id: int, title: str, url: str) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO title_postings(term, doc_id, tf) VALUES (?, ?, ?)",
        [(term, doc_id, tf) for term, tf in title_terms(title, url).items()],
    )


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
    _index_title(conn, doc_id, title or url, url)
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


def _batches(items: list, size: int = 500):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _title_tf(conn: sqlite3.Connection, terms: list[str], b: float) -> dict[str, dict[int, float]]:
    """Length-normalized title-field frequency of each query term, per document."""
    n_docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    total = conn.execute("SELECT COALESCE(SUM(tf), 0) FROM title_postings").fetchone()[0]
    if not n_docs or not total:
        return {}
    rows_by_term = {
        t: conn.execute("SELECT doc_id, tf FROM title_postings WHERE term = ?", (t,)).fetchall()
        for t in terms
    }
    doc_ids = sorted({doc_id for rows in rows_by_term.values() for doc_id, _ in rows})
    lengths: dict[int, int] = {}
    for batch in _batches(doc_ids):
        lengths.update(conn.execute(
            f"SELECT doc_id, SUM(tf) FROM title_postings "
            f"WHERE doc_id IN ({','.join('?' * len(batch))}) GROUP BY doc_id", batch))
    avg_len = total / n_docs
    return {t: {doc_id: tf / (1 - b + b * lengths[doc_id] / avg_len) for doc_id, tf in rows}
            for t, rows in rows_by_term.items() if rows}


def search_chunks(conn: sqlite3.Connection, query: str, cfg: dict,
                  limit: int = 50) -> list[dict]:
    """BM25F over chunks: the chunk's text plus its document's title field.

    Returns [{chunk_id, doc_id, score, text, url, title}]. For each query term,
    the chunk's length-normalized frequency and cfg["title_weight"] times the
    title field's are summed before BM25 saturates them, so a term counts once
    wherever it matches: a title lifts its page, but one title word cannot
    outweigh the rest of the query. A title weight of 0 gives plain BM25. A
    document that matches only in its title contributes its first chunk.
    """
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []
    row = conn.execute("SELECT COUNT(*), COALESCE(AVG(length), 0) FROM chunks").fetchone()
    n_chunks, avgdl = row[0], row[1] or 1.0
    if not n_chunks:
        return []

    k1, b = cfg["bm25_k1"], cfg["bm25_b"]
    weight = float(cfg.get("title_weight", 5.0))
    body: dict[str, dict[int, float]] = {}
    doc_of: dict[int, int] = {}
    for term in terms:
        rows = conn.execute(
            "SELECT p.chunk_id, p.tf, c.length, c.doc_id "
            "FROM postings p JOIN chunks c ON c.id = p.chunk_id WHERE p.term = ?",
            (term,),
        ).fetchall()
        body[term] = {chunk_id: tf / (1 - b + b * length / avgdl)
                      for chunk_id, tf, length, _ in rows}
        doc_of.update((chunk_id, doc_id) for chunk_id, _, _, doc_id in rows)

    title = _title_tf(conn, terms, b) if weight > 0 else {}
    missing = sorted({d for docs in title.values() for d in docs} - set(doc_of.values()))
    for batch in _batches(missing):
        doc_of.update(conn.execute(
            f"SELECT id, doc_id FROM chunks WHERE seq = 0 "
            f"AND doc_id IN ({','.join('?' * len(batch))})", batch))

    scores: dict[int, float] = {}
    for term in terms:
        in_body, in_title = body[term], title.get(term, {})
        df = len(in_body) or len(in_title)  # a term found only in titles
        if not df:
            continue
        idf = math.log(1 + (n_chunks - df + 0.5) / (df + 0.5))
        for chunk_id, doc_id in doc_of.items():
            f = in_body.get(chunk_id, 0.0) + weight * in_title.get(doc_id, 0.0)
            if f:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * f * (k1 + 1) / (f + k1)

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
