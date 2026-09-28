"""Configuration, paths, and .env loading.

All paths are anchored to the project root (the folder containing run.py),
so commands work no matter which directory you invoke them from.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULTS: dict = {
    # LLM settings
    "model": "claude-opus-5",       # Anthropic API model id
    "cli_model": "opus",            # model alias for the `claude` CLI fallback
    "backend": "auto",              # auto | api | claude-cli | off
    "max_answer_tokens": 4096,
    "cli_timeout": 240,             # seconds for the claude CLI fallback
    "cli_max_retries": 2,           # API retries per CLI call (the CLI's own 10 take minutes)
    "answer_style": "concise and factual",
    # Retrieval settings
    "top_k": 8,                     # results shown by `search`
    "ask_contexts": 6,              # chunks fed to the LLM by `ask`
    "chunk_chars": 1400,
    "chunk_overlap": 200,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "snippet_chars": 240,
    # Crawler / web settings
    "crawl_delay": 0.5,
    "http_timeout": 20,
    "user_agent": "llmsearch/0.1 (personal search engine)",
    "web_results": 8,
    # Server settings (`llmsearch serve`, used by the mobile app)
    "serve_host": "0.0.0.0",
    "serve_port": 8765,
}


def data_dir() -> Path:
    override = os.environ.get("LLMSEARCH_DATA_DIR")
    d = Path(override) if override else PROJECT_ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "index.db"


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from <project>/.env without overriding real env vars.

    Tolerates the encodings Windows tools actually produce (UTF-8 with BOM,
    UTF-16 as written by PowerShell 5.1's `>` redirect) and never raises:
    a malformed .env must not take every command down.
    """
    if not ENV_PATH.is_file():
        return
    try:
        raw = ENV_PATH.read_bytes()
    except OSError:
        return
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        # NULs appear when BOM-less UTF-16 is decoded as UTF-8; stripping
        # them recovers ASCII KEY=VALUE pairs and keeps os.environ happy.
        line = line.replace("\x00", "").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            try:
                os.environ[key] = value
            except (ValueError, OSError):
                continue


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                cfg.update({k: v for k, v in stored.items() if k in DEFAULTS})
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save_config(cfg: dict) -> None:
    changed = {k: v for k, v in cfg.items() if k in DEFAULTS and v != DEFAULTS[k]}
    CONFIG_PATH.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")


def set_value(cfg: dict, key: str, raw: str) -> dict:
    """Set a config key from a CLI string, coercing to the default's type."""
    if key not in DEFAULTS:
        raise KeyError(f"unknown config key: {key!r} (see `config show`)")
    default = DEFAULTS[key]
    if isinstance(default, bool):
        value: object = raw.lower() in ("1", "true", "yes", "on")
    elif isinstance(default, (int, float)):
        try:
            value = type(default)(raw)
        except ValueError:
            raise ValueError(
                f"invalid value for {key}: {raw!r} "
                f"(expected {type(default).__name__})"
            ) from None
    else:
        value = raw
    cfg[key] = value
    return cfg
