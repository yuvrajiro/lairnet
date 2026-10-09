#!/usr/bin/env python
"""Fail if two pages quote different numbers for the same computation.

``gallery.rst`` and ``index.rst`` quote an ``explain_model`` summary as a
literal block. ``docs/examples/02_interpreting_a_model.ipynb`` produces one by
running the same code on the same data. They must agree, and there is nothing
in a Sphinx build that would notice if they did not.

    python docs/check_numbers_agree.py

They did not agree once. The notebooks were executed by the ambient ``python3``
kernelspec, which was a different interpreter with numpy 2.1 and scipy 1.13
instead of numpy 1.23 and scipy 1.11. L-BFGS converged to a different point and
the notebook reported 0.8476 where the gallery reported 0.8369 -- two honest
numbers for the same three lines of code, published on adjacent pages.
``make_notebooks.py`` now pins the kernel to its own interpreter; this checks
that the pinning is still working.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "examples" / "02_interpreting_a_model.ipynb"
# The README quotes the same summary and is the GitHub and PyPI front page, so
# it drifts exactly as easily as the pages do.
PAGES = (HERE / "gallery.rst", HERE / "index.md",
         HERE.parent / "README.md")

# Lines that carry a number and appear in both the notebook output and the
# quoted blocks. Compared after collapsing whitespace, because the quoted
# blocks are indented into a literal block and the notebook output is not.
KEYS = (
    "score on the evaluation data:",
    "with anchor",
    "without anchor",
    "the anchor is worth",
    "aggregating all depths is worth",
    "best at depth",
)


def squash(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines()
            if " ".join(line.split())]


def notebook_summary() -> list[str]:
    book = nbformat.read(NOTEBOOK, as_version=4)
    for cell in book.cells:
        for output in cell.get("outputs", []):
            text = str(output.get("text") or "")
            if "score on the evaluation data" in text:
                return squash(text)
    raise SystemExit(f"no explain_model summary found in {NOTEBOOK.name}; "
                     "run python docs/make_notebooks.py")


def main() -> int:
    truth = notebook_summary()
    wanted = {}
    for line in truth:
        for key in KEYS:
            if line.startswith(key):
                wanted[key] = line

    missing = [k for k in KEYS if k not in wanted]
    if missing:
        raise SystemExit(f"the notebook summary no longer contains {missing}; "
                         "update KEYS in this script to match")

    failures = []
    for page in PAGES:
        lines = squash(page.read_text(encoding="utf-8"))
        for key, expected in wanted.items():
            found = [ln for ln in lines if ln.startswith(key)]
            if not found:
                continue          # a page need not quote every line
            for line in found:
                status = "ok " if line == expected else "BAD"
                print(f"{status} {page.name:<12} {line}")
                if line != expected:
                    failures.append((page.name, expected, line))

    if failures:
        print(f"\n{len(failures)} line(s) disagree with "
              f"{NOTEBOOK.name}, which is what the code actually printed:")
        for name, expected, line in failures:
            print(f"    {name}\n      page     {line}\n      notebook {expected}")
        print("\nRe-run python docs/make_gallery.py and paste its summary into "
              "the pages, or re-run python docs/make_notebooks.py -- but first "
              "check both ran under the same interpreter.")
        return 1

    print(f"\nall quoted numbers agree with {NOTEBOOK.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
