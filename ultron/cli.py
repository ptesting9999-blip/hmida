#!/usr/bin/env python3
"""Launch ULTRON in the terminal.  See `python cli.py --help`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultron.cli import main

if __name__ == "__main__":
    main()
