#!/usr/bin/env python3
"""Launcher so the CLI works from any directory:

    python path/to/llm-search-engine/run.py search "query"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llmsearch.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
