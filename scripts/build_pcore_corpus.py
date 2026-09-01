"""Regenerate tests/pcore/corpus/*.pcore from every PCore script embedded in the
repo (Python sources and notebooks). Run after adding a model to a test or a
notebook so the compatibility oracle keeps covering real usage.

    python scripts/build_pcore_corpus.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "pcore" / "corpus"

_PCORE = re.compile(r"PCore\s+\w+\s*\{.*?\}", re.S | re.I)


def _sources():
    for p in ROOT.rglob("*.py"):
        if ".venv" in p.parts or "corpus" in p.parts:
            continue
        yield p, p.read_text(encoding="utf-8", errors="ignore")
    for p in sorted(ROOT.glob("notebooks/*.ipynb")):
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for cell in nb.get("cells", []):
            yield p, "".join(cell.get("source", []))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.pcore"):
        old.unlink()

    seen: dict[str, str] = {}
    for path, text in _sources():
        for m in _PCORE.finditer(text):
            script = m.group(0).strip()
            norm = re.sub(r"\s+", " ", script)
            if norm in seen:
                continue
            name = re.search(r"PCore\s+(\w+)", script, re.I).group(1)
            key = f"{len(seen):02d}_{name}"
            seen[norm] = key
            (OUT / f"{key}.pcore").write_text(script + "\n", encoding="utf-8")
            print(f"{key:20} <- {path.relative_to(ROOT)}")
    print(f"\n{len(seen)} unique scripts -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
