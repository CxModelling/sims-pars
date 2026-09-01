"""Copy the notebooks into ``docs/`` so MkDocs (which can only see files under
``docs_dir``) can render them. Run before ``mkdocs build``."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks"

# notebook name -> destination directory under docs/
STAGED = {
    "GettingStarted01_The PCore language.ipynb": "getting-started",
    "GettingStarted02_Sampling and intervention.ipynb": "getting-started",
    "GettingStarted03_Fitting a model.ipynb": "getting-started",
    "Tutorial01_Basic Nodes.ipynb": "tutorials",
    "Tutorial02_Bayesian Networks.ipynb": "tutorials",
    "Tutorial03_Sampling.ipynb": "tutorials",
    "Tutorial04_Model fitting.ipynb": "tutorials",
}


def main() -> None:
    for name, subdir in STAGED.items():
        src = SRC / name
        if not src.exists():
            raise SystemExit(f"missing notebook: {src}")
        dst = ROOT / "docs" / subdir
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / name)
        print(f"staged {name} -> docs/{subdir}/")


if __name__ == "__main__":
    main()
