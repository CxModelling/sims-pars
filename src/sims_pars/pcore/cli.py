"""``sims-pars check`` — parse PCore files and print diagnostics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sims_pars.pcore.lower import check


def _check_paths(paths: list[str]) -> int:
    total = 0
    for name in paths:
        p = Path(name)
        try:
            src = p.read_text(encoding="utf-8")
        except OSError as e:
            print(f"{name}: cannot read ({e})", file=sys.stderr)
            total += 1
            continue
        diags = check(src)
        errs = [d for d in diags if d.is_error]
        if not diags:
            print(f"{name}: ok")
            continue
        print(f"{name}:")
        for d in diags:
            for line in d.render(src).splitlines():
                print(f"  {line}")
        total += len(errs)
    return 1 if total else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sims-pars", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    chk = sub.add_parser("check", help="parse PCore files and report problems")
    chk.add_argument("files", nargs="+", help="PCore script files")
    args = parser.parse_args(argv)
    if args.cmd == "check":
        return _check_paths(args.files)
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
