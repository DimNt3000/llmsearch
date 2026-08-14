# llmsearch — a tiny personal search engine with LLM capabilities

A customizable command-line search engine in pure Python. You build your own
index by crawling websites and adding local files; BM25 ranks the results; and
Claude adds the smart layer on top: query expansion, reranking, summaries, and
cited question-answering (RAG) over whatever you indexed. It can also search
the live web via DuckDuckGo and answer questions from the pages it finds.

```
crawler / files  ->  inverted index (SQLite)  ->  BM25 ranking  ->  Claude
   [crawl, add]         [data/index.db]          [search]     [ask, summarize,
                                                               --smart, --rerank]
```

## Quickstart

```
cd llm-search-engine
python run.py add samples                      # index the bundled sample docs
python run.py search "inverted index"          # plain BM25 search
python run.py ask "how does BM25 handle long documents?"   # cited answer (LLM)
```

No install step is needed — everything except the LLM layer is stdlib-only.
`pip install anthropic` enables the API backend, `pip install ddgs` makes the
`web` command reliable. Or `pip install -e .` to get a global `llmsearch`
command.

## Commands

| Command | What it does | LLM |
|---|---|---|
| `crawl URL [--depth N] [--max-pages N] [--all-domains]` | Politely crawl a site (robots.txt, throttled) into the index | – |
| `add PATH...` | Index local `.md` / `.txt` / `.rst` / `.html` files or folders | – |
| `search QUERY [--smart] [--rerank] [--json]` | BM25 search; `--smart` adds LLM query expansion + rank fusion, `--rerank` lets the LLM reorder results | optional |
| `ask QUESTION [--smart]` | Retrieval-augmented answer from the index, with `[n]` citations | yes |
| `summarize QUERY` | Digest of the top results for a query | yes |
| `web QUERY [-n N] [--ask]` | Live DuckDuckGo search; `--ask` fetches the top pages and answers from them | `--ask` only |
| `stats` | Index size and active LLM backend | – |
| `config [show\|get KEY\|set KEY VALUE]` | Customize everything below | – |

## LLM backends

The LLM layer picks a backend automatically (`config set backend ...` to pin):

1. **`api`** — the official `anthropic` Python SDK. Credentials come from
   `ANTHROPIC_API_KEY` (environment or `.env` in this folder — see
   `.env.example`), `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile.
   Default model: `claude-opus-5`. Answers stream as they generate, and
   server-side refusal fallbacks are enabled by default (if a safety
   classifier declines a request, the API retries it on a fallback model
   automatically).
2. **`claude-cli`** — if no API credentials are found but
   [Claude Code](https://claude.com/claude-code) is installed, the tool shells
   out to `claude -p`, reusing your existing Claude login. Zero setup.
3. **`off`** — no backend found. Plain `search`, `crawl`, `add`, and `web`
   keep working; only the LLM features are disabled.

## Customization

Everything lives in `config.json` (created on first `config set`):

| Key | Default | Meaning |
|---|---|---|
| `model` | `claude-opus-5` | API model (`claude-sonnet-5`, `claude-haiku-4-5`, ... for cheaper/faster) |
| `cli_model` | `opus` | Model alias for the `claude` CLI backend |
| `backend` | `auto` | `auto` / `api` / `claude-cli` / `off` |
| `answer_style` | `concise and factual` | Injected into every LLM prompt — make it yours |
| `top_k` | `8` | Results shown by `search` |
| `ask_contexts` | `6` | Chunks fed to the LLM by `ask` |
| `chunk_chars` / `chunk_overlap` | `1400` / `200` | Chunking granularity |
| `bm25_k1` / `bm25_b` | `1.5` / `0.75` | BM25 saturation / length normalization |
| `crawl_delay` / `http_timeout` | `0.5` / `20` | Crawler politeness |
| `user_agent`, `web_results`, `snippet_chars`, `max_answer_tokens`, `cli_timeout` | ... | See `config show` |

The index lives in `data/index.db` (override with `LLMSEARCH_DATA_DIR`).
Deleting that file resets the engine. Re-crawling or re-adding a URL/file
replaces its old version in the index.

## How it works

- **Crawler** (`crawler.py`): stdlib `urllib` BFS crawler. Respects
  `robots.txt`, stays on the start domain unless `--all-domains`, throttles
  between requests, caps page size at 3 MB, and extracts text/links with
  `html.parser`.
- **Index** (`indexer.py`): documents are split into ~1400-char chunks at
  paragraph boundaries (with overlap), tokenized (Unicode-aware, stopwords
  dropped), and stored in SQLite as an inverted index
  (`term -> chunk, term-frequency`).
- **Ranking**: classic BM25 over chunks — IDF-weighted, saturation-controlled
  (`k1`), length-normalized (`b`). `search` shows the best chunk per document;
  `ask` feeds the top chunks (max 3 per document) to the model.
- **LLM layer** (`llm.py`): four capabilities built on one `complete()` call —
  query expansion (JSON list of variants, merged with reciprocal rank fusion),
  reranking (JSON list of indices), summaries, and cited RAG answers that are
  instructed to use *only* the retrieved sources.

## Limitations

- Lexical search only — no embeddings, so synonyms are handled by `--smart`
  query expansion rather than vector similarity.
- No stemming; English stopword list only (Greek text still indexes fine,
  token-for-token).
- The crawler runs single-threaded and does not execute JavaScript.
- DuckDuckGo's HTML endpoint may throttle bots — install `ddgs` for the
  reliable path.

## Ideas for extending it

Add an embeddings index next to BM25 for hybrid retrieval; schedule `crawl`
re-runs to keep the index fresh; add a `serve` command exposing search over
HTTP; teach `add` to read PDFs.

## License

MIT — see [LICENSE](LICENSE).
