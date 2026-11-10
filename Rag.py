"""Backward-compatible script entry point. Prefer running `uv run rag`."""

import sys
from pathlib import Path

# Make the src-layout package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag import main


if __name__ == "__main__":
    main()
