#!/usr/bin/env python
"""Fail if conf.py loads a Sphinx extension the [docs] extra does not install.

An extension that is present in the developer's environment but missing from the
dependency list builds perfectly on their machine and dies on a clean runner
with `Could not import extension X`. That is exactly how sphinx_copybutton
reached main: it was added to conf.py during a docs overhaul, it was already
installed here, and nothing connected the two facts until CI refused to build.

Sphinx stops at the *first* extension it cannot import, so a red build tells you
about one missing package and hides any others behind it. This checks all of
them at once.

    python docs/check_requirements.py

Extensions under `sphinx.ext.` ship inside Sphinx and need no separate
dependency, so they are skipped. Everything else must appear in the `docs`
extra of pyproject.toml, compared with '-' and '_' treated alike because
`sphinx-copybutton` on PyPI imports as `sphinx_copybutton`.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def normalise(name: str) -> str:
    """PyPI name or import name -> one comparable form."""
    return name.split(">")[0].split("=")[0].split("[")[0].strip().lower().replace("-", "_")


def conf_extensions() -> list[str]:
    text = (HERE / "conf.py").read_text(encoding="utf-8")
    match = re.search(r"^extensions\s*=\s*(\[[^\]]*\])", text, re.S | re.M)
    if match is None:
        raise SystemExit("could not find an `extensions = [...]` list in conf.py")
    return list(ast.literal_eval(match.group(1)))


def declared() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = data["project"]["optional-dependencies"].get("docs")
    if not extra:
        raise SystemExit("pyproject.toml declares no [docs] extra")
    return {normalise(d) for d in extra}


def main() -> int:
    have = declared()
    missing = []
    for ext in conf_extensions():
        if ext.startswith("sphinx.ext."):
            print(f"  ok      {ext:<24} ships with sphinx")
            continue
        if normalise(ext) in have:
            print(f"  ok      {ext:<24} declared in [docs]")
        else:
            print(f"  MISSING {ext:<24} not in the [docs] extra")
            missing.append(ext)

    if missing:
        print(f"\n{len(missing)} extension(s) would fail on a clean install: "
              f"{', '.join(missing)}")
        print("Add them to [project.optional-dependencies].docs in pyproject.toml.")
        return 1

    print(f"\nall {len(conf_extensions())} extensions are installable from [docs]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
