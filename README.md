<div align="center">

# llmsearch

**Build your own search engine, then let an LLM reason over it.**

A crawler, an inverted index, and BM25 ranking written from scratch in pure Python,
with Claude layered on top for query expansion, reranking, and cited answers.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Core dependencies](https://img.shields.io/badge/core%20dependencies-0-success)
[![Tests](https://github.com/DimNt3000/llmsearch/actions/workflows/tests.yml/badge.svg)](https://github.com/DimNt3000/llmsearch/actions/workflows/tests.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

</div>

---

Search engines are usually treated as a black box you call. This one is the box:
every layer is implemented rather than imported. A polite web crawler feeds a
SQLite inverted index, BM25 scores the results, and a language model sits on top
to expand queries, rerank hits, and answer questions **strictly from the retrieved
sources, with citations**.

The retrieval core, roughly 1,400 lines across the package, uses **only the Python
standard library**. `pip install` is optional and unlocks just the LLM and live web
search layers.

A [companion mobile app](https://github.com/DimNt3000/llmsearch-mobile) (React
Native, TypeScript) searches, asks, and crawls from a phone through `llmsearch serve`.

## Demo

Ask a question and get a grounded answer, not a guess:

```console
$ llmsearch ask "how does BM25 handle long documents?"

BM25 handles long documents through **document length normalization** [1]. Long
documents naturally contain more term occurrences, so BM25 divides the term
frequency by a factor based on the document's length relative to the average
document length in the collection [1]. The **b parameter** (typically 0.75)
controls how strong this normalization is: setting b = 1 fully normalizes by
length (penalizing long documents the most), while b = 0 ignores length
entirely [1].

Sources
  [1] bm25 - file:///…/samples/bm25.md
```

Every claim carries a citation, and when the index does not cover the question the
model is instructed to say so instead of falling back on its own knowledge.

`--smart` shows why the LLM layer earns its place. The query below shares no exact
term with any indexed document, so plain BM25 finds nothing useful. The model
rewrites it, and reciprocal rank fusion merges the results:

```console
$ llmsearch search "reducing hallucinations" --smart
Also searching: minimizing AI confabulation errors; decreasing LLM false output
generation; mitigating fabricated responses large language models

 1. rag  0.032
    file:///…/samples/rag.md
    # Retrieval-augmented generation (RAG) Retrieval-augmented generation combines
    a search system with a large language model. Instead of asking the model to
    answer from its training data alone, the system first retrieves passages...
```

The same pipeline also runs against the live web, fetching the top results and
answering from their actual content rather than from search snippets:

```console
$ llmsearch web "what is the latest stable Python version" --ask
fetching https://en.wikipedia.org/wiki/History_of_Python...
fetching https://devguide.python.org/versions/...
fetching https://www.python.org/downloads/...

The latest stable Python version is **Python 3.14.7**, available for download
from the official Python.org site [3].
```

## Architecture

```
   SOURCES              INGESTION                RETRIEVAL              REASONING
 ┌───────────┐      ┌─────────────────┐      ┌──────────────┐      ┌───────────────┐
 │  websites │─────▶│ crawler         │─────▶│ SQLite       │─────▶│ query         │
 │  (crawl)  │      │  robots.txt     │      │ inverted     │      │  expansion    │
 ├───────────┤      │  throttling     │      │ index        │      ├───────────────┤
 │  local    │─────▶│  gzip/charset   │      │              │      │ reranking     │
 │  files    │      │  HTML → text    │      │ term →       │      ├───────────────┤
 │  (add)    │      ├─────────────────┤      │  (chunk, tf) │      │ cited answers │
 ├───────────┤      │ chunker         │      │              │      │  (RAG)        │
 │ DuckDuck  │─────▶│  ~1400 chars    │─────▶│ BM25 scoring │─────▶│ summaries     │
 │ Go (web)  │      │  paragraph-wise │      │  k₁ · b · IDF│      │               │
 └───────────┘      └─────────────────┘      └──────────────┘      └───────────────┘
                                                     │                     │
                                                     └─── rank fusion ◀────┘
```

| Module | Responsibility |
|---|---|
| [`crawler.py`](llmsearch/crawler.py) | BFS crawler on `urllib`: robots.txt, throttling, redirects, gzip/deflate, charset detection, HTML to text |
| [`indexer.py`](llmsearch/indexer.py) | Chunking, tokenization, SQLite inverted index, BM25 scoring, reciprocal rank fusion, snippets |
| [`llm.py`](llmsearch/llm.py) | Backend resolution and the four LLM capabilities, each degrading safely on failure |
| [`websearch.py`](llmsearch/websearch.py) | DuckDuckGo results, normalized across two independent backends |
| [`service.py`](llmsearch/service.py) | The search, ask, summarize, web, and crawl pipelines, shared by CLI and server |
| [`server.py`](llmsearch/server.py) | `llmsearch serve`: JSON API, token auth, input validation, background crawl jobs |
| [`config.py`](llmsearch/config.py) | Layered defaults, `config.json`, `.env` loading |
| [`cli.py`](llmsearch/cli.py) | Argument parsing, colored output, error handling, output sanitization |

## Commands

| Command | What it does | Needs an LLM |
|---|---|---|
| `crawl URL [--depth N] [--max-pages N] [--all-domains] [--path-prefix PATH]` | Crawl a site into the index, politely, optionally within one section | no |
| `add PATH...` | Index local `.md` / `.txt` / `.rst` / `.html` files or folders | no |
| `search QUERY` | BM25 search with highlighted snippets | no |
| `search QUERY --smart` | LLM query expansion, merged with reciprocal rank fusion | optional |
| `search QUERY --rerank` | Over-fetch, then let the model reorder by relevance | optional |
| `ask QUESTION` | Retrieval-augmented answer with `[n]` citations | yes |
| `summarize QUERY` | Digest of what the top results collectively say | yes |
| `web QUERY [--ask]` | Live DuckDuckGo search, optionally answered from page content | `--ask` only |
| `serve [--host] [--port] [--new-token]` | JSON API for the mobile app, see below | for LLM endpoints |
| `stats` / `config` | Index statistics, active backend, and every tunable setting | no |

## Quickstart

```bash
git clone https://github.com/DimNt3000/llmsearch.git
cd llmsearch
python run.py add samples          # index the bundled documents
python run.py search "inverted index"
python run.py ask "how does BM25 handle long documents?"
```

Nothing to install for the retrieval half. `pip install anthropic` enables the LLM
layer (or reuse an existing Claude Code login, see below), `pip install ddgs` makes
live web search more robust, and `pip install -e .` gives you a global `llmsearch`
command.

## HTTP API and mobile app

`llmsearch serve` exposes the whole engine as a small JSON API, which the companion
[mobile app](https://github.com/DimNt3000/llmsearch-mobile) uses to search, ask, and
crawl from a phone. It is built on the standard
library's `http.server`, so serving adds no dependencies, and it runs the exact same
pipeline as the CLI through a shared service layer.

```console
$ llmsearch serve
llmsearch server v0.2.0
  Local:    http://127.0.0.1:8765
  Network:  http://192.168.1.42:8765   <- enter this in the app
  Token:    (a random 24-character token)
```

Every endpoint except `/api/health` requires `Authorization: Bearer <token>`. The token
is generated once and stored in the data directory, so a paired phone survives server
restarts; `serve --new-token` rotates it.

| Endpoint | Input | Returns |
|---|---|---|
| `GET /api/health` | | Liveness and version, no token needed |
| `GET /api/stats` | | Index size and the active LLM backend |
| `GET /api/search` | `q`, `smart=1`, `rerank=1` | Ranked results with snippets and full chunk text |
| `POST /api/ask` | `{"question", "smart"}` | Cited answer plus the numbered sources |
| `POST /api/summarize` | `{"query"}` | Digest plus sources |
| `GET /api/web` | `q`, `n` | Live DuckDuckGo results |
| `POST /api/web/ask` | `{"query"}` | Answer written from the fetched result pages |
| `POST /api/crawl` | `{"url", "depth", "max_pages", "all_domains"}` | Starts a background crawl (202) |
| `GET /api/crawl` | | Progress and log of the current or last crawl |
| `DELETE /api/crawl` | | Cancels the running crawl |

Errors share one shape, `{"error": {"code", "message"}}`, with codes a client can act
on: `unauthorized` (401), `bad_request` (400), `no_context` (422), `busy` (409),
`payload_too_large` (413), `llm_unavailable` (503), and `llm_error` / `web_error` (502).

The phone and the computer need to share a network. On Windows, allow Python through
the firewall for private networks the first time the server starts.

## How it works

**Chunking.** Documents are split at paragraph boundaries into roughly 1,400-character
chunks with overlap, because retrieval is sharpest when each indexed unit is about
one thing, and because only a handful of chunks fit in a prompt. Oversized paragraphs
fall back to a hard split whose stride is clamped so a misconfigured overlap cannot
stall it.

**The index.** Chunks are tokenized with a Unicode-aware regex (Greek, accents, and
mixed scripts included), stopwords are dropped, and postings are stored in a
`WITHOUT ROWID` SQLite table keyed on `(term, chunk_id)`, so a lookup is a direct
index seek rather than a scan.

**Ranking.** Classic BM25, computed from the postings list:

```
score(D,Q) = Σ  IDF(t) · ──────tf · (k₁ + 1)───────
            t∈Q          tf + k₁ · (1 − b + b·|D|/avgdl)

IDF(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
```

`k₁` controls term-frequency saturation, `b` controls length normalization, and both
are tunable at runtime through `config set`.

**Rank fusion.** When one query becomes several, the result lists are merged with
reciprocal rank fusion, which combines rankings without needing their scores to be
comparable:

```
RRF(d) = Σ  1 / (60 + rank_r(d))
         r
```

**Grounded answering.** Retrieved chunks (capped per document, so one verbose source
cannot crowd out the rest) are passed to the model with instructions to answer only
from them and to cite each claim. Citations are the point: an answer with sources can
be checked, an answer without them has to be trusted.

## Engineering notes

The parts that took the real work, and the reasoning behind them:

- **Two LLM backends, resolved automatically.** The Anthropic API is used when
  credentials exist; otherwise the tool shells out to the `claude` CLI and reuses an
  existing Claude Code login, so it works with zero setup. If neither is available,
  the LLM features report exactly how to enable one and the search half keeps working.
- **Every LLM call fails open.** Model output is untrusted input. Query expansion and
  reranking parse JSON out of prose or code fences, and on anything unparseable they
  fall back to the original BM25 ordering rather than failing the command.
- **Terminal injection is not possible.** A crawled page controls its own title and
  text. Before anything untrusted reaches stdout, complete ANSI CSI/OSC/Fe sequences
  and stray control bytes are stripped, so a hostile page cannot repaint the terminal
  or forge output. Verified with an adversarial page carrying embedded escapes.
- **The crawler behaves.** robots.txt is honored per origin with RFC 9309 matching
  (`*` and `$` wildcards, the longest rule wins), requests are throttled, responses
  are capped at 3 MB, content types are allow-listed, and gzip and charset are handled
  explicitly instead of hoping the bytes are UTF-8. robots.txt is fetched with the
  crawler's own user agent: bot protection answers Python's default one with a 403,
  which the standard library's parser reads as a ban on the whole site.
- **Boilerplate stays out of the index.** Text inside `<nav>` and `<footer>` is not
  indexed, while the links in them are still followed. On the PyTorch docs a sidebar
  listing the whole API was 80% of every page, so each class name matched every page
  and real results sank. `--path-prefix` keeps a crawl inside one section of a site,
  such as a single version of a documentation set.
- **Encodings are treated as hostile.** `.env` files are sniffed for UTF-16 and UTF-8
  BOMs, which is exactly what a Windows shell writes by default, and a malformed file
  can never take a command down with it.
- **The API assumes a hostile network.** A bearer token compared in constant time
  guards every endpoint, request bodies are size-capped, every parameter is validated
  against bounds, unexpected errors return a generic 500 without internals, and
  concurrent LLM work is capped so a client cannot spawn unbounded model processes.
- **Failure paths are first-class.** Unreachable hosts, empty indexes, stopword-only
  queries, refusals, rate limits, and timeouts each produce a clear message and a
  meaningful exit code, never a traceback. The `claude` CLI retries a failed API call
  ten times by default, which turned an expired login into a three-minute wait; the
  backend caps it at two retries and recognizes auth errors, so an expired login is
  reported in seconds along with the command that fixes it, and a stopped local API
  proxy is detected in about a second without starting the CLI at all.

## Testing

```bash
python tests/test_e2e.py      # 110 checks: the engine and the CLI
python tests/test_server.py   # 50 checks: the HTTP API over real sockets
```

Both harnesses exit non-zero on failure, so they double as CI gates. They make **no
network and no LLM calls**, which keeps them fast and free: model output is faked
in-process, and the crawl tests crawl a throwaway site served from a temporary
directory on localhost. Neither touches your index or settings.

[`tests/test_e2e.py`](tests/test_e2e.py) covers indexing and re-indexing, BM25
properties (rare terms outrank common ones, length normalization behaves), chunking
edge cases including the regression where a misconfigured overlap could stall the
splitter, Unicode handling, backend resolution across every configuration, LLM
fail-open behavior against deliberately malformed model output, CLI-backend failures,
terminal-escape sanitization against a hostile page, `.env` encoding traps, robots.txt
rules and the status codes a robots.txt fetch can return, menus kept out of indexed
text, crawls scoped to a path prefix, exit codes
for every subcommand, concurrent readers, and packaging consistency.

[`tests/test_server.py`](tests/test_server.py) starts the real server on an ephemeral
port and covers token handling, authentication, validation of every input, each error
status, CORS preflight, oversized bodies, internals never leaking into a 500, the full
lifecycle of a background crawl (start, progress, conflict, cancellation, unreachable
host), and 16 concurrent clients. Checks that need something absent from the machine
report as skipped rather than failing.

The same harness runs in CI on Ubuntu and Windows across Python 3.10 through 3.13,
once with nothing installed (which is what proves the zero-dependency claim) and once
with the optional layers present. See
[`.github/workflows/tests.yml`](.github/workflows/tests.yml).

The live paths, crawling real sites and answering with a real model, were exercised
manually end to end.

## Configuration

Everything is tunable, and every setting persists in `config.json`:

```bash
python run.py config set model claude-sonnet-5      # cheaper, faster answers
python run.py config set answer_style "terse, bullet points only"
python run.py config set bm25_b 0.3                 # weaken length normalization
python run.py config show
```

| Key | Default | Meaning |
|---|---|---|
| `model` / `cli_model` | `claude-opus-5` / `opus` | Model for the API and CLI backends |
| `backend` | `auto` | `auto`, `api`, `claude-cli`, or `off` |
| `answer_style` | `concise and factual` | Injected into every prompt |
| `top_k` / `ask_contexts` | `8` / `6` | Results shown, chunks fed to the model |
| `chunk_chars` / `chunk_overlap` | `1400` / `200` | Chunking granularity |
| `bm25_k1` / `bm25_b` | `1.5` / `0.75` | Saturation and length normalization |
| `crawl_delay` / `http_timeout` | `0.5` / `20` | Crawler politeness |
| `cli_timeout` / `cli_max_retries` | `240` / `2` | Time limit and API retries for each `claude` CLI call |
| `snippet_chars`, `web_results`, `user_agent`, `max_answer_tokens` | … | Snippet width, result count, crawler identity, answer budget |
| `serve_host` / `serve_port` | `0.0.0.0` / `8765` | Where `llmsearch serve` listens |

The index lives in `data/index.db`, overridable with `LLMSEARCH_DATA_DIR`. Deleting
that file resets the engine; re-crawling a URL replaces its old version in place.

## Limitations

Lexical retrieval only: there are no embeddings, so vocabulary mismatch is handled by
LLM query expansion rather than semantic similarity. No stemming, and the stopword
list is English. The crawler is single-threaded and does not execute JavaScript.
DuckDuckGo throttles bots, so the `ddgs` package is the reliable path.

## Roadmap

Hybrid retrieval with an embedding index alongside BM25, scheduled re-crawls to keep
the index fresh, streamed answers over the API, and PDF ingestion.

## License

MIT, see [LICENSE](LICENSE).
