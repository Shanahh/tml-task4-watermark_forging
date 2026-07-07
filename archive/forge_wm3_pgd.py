from __future__ import annotations

import sys

from forge_pgd import main as _main

if __name__ == "__main__":
    sys.argv += ["--category", "WM_3"]
    _main()
