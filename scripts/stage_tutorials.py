"""Copy the tutorial notebooks into ``docs/tutorials/`` so MkDocs (which can only
see files under ``docs_dir``) can render them. Run before ``mkdocs build``."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks"
DST = ROOT / "docs" / "tutorials"

TUTORIALS = [
    "Tutorial01_Basic Nodes.ipynb",
    "Tutorial02_Bayesian Networks.ipynb",
    "Tutorial03_Sampling.ipynb",
    "Tutorial04_Model fitting.ipynb",
]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in TUTORIALS:
        src = SRC / name
        if not src.exists():
            raise SystemExit(f"missing notebook: {src}")
        shutil.copy2(src, DST / name)
        print(f"staged {name}")


if __name__ == "__main__":
    main()
