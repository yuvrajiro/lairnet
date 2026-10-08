#!/usr/bin/env python
"""Fail if a page names a CSS class that ``_static/custom.css`` never styles.

Sphinx does not warn about this. A ``:class-container: split-hero`` on a grid
that has no ``.split-hero`` rule builds cleanly, produces valid HTML, and
renders as if the class were not there. That is exactly how the landing page
shipped once: seven of its thirteen classes had no rule, so the hero was a
column of plain paragraphs while the source said otherwise.

    python docs/check_styles.py

What this does *not* catch is the other half of the same failure -- a rule that
exists but loses on specificity. The theme ships
``.docutils.container { max-width: unset }``, which silently discarded every
width written as a bare ``.hero-lede``. Catching that needs a browser, because
it is a property of the cascade rather than of the text. Checked by hand at
1440px, 375px, and in dark mode.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSS = HERE / "_static" / "custom.css"

# Classes that come from the theme or from sphinx-design rather than from us.
# Naming one of these is legitimate: it selects behaviour that already exists.
EXTERNAL = {
    "sd-card", "sd-row", "sd-container-fluid", "sd-sphinx-override",
    "docutils", "container", "admonition", "topic", "sidebar",
}

# Every construct that can put a class on an element in a page source.
PATTERNS = (
    re.compile(r"^\s*\.\.\s+container::\s+(.+)$", re.M),
    re.compile(r"^\s*\.\.\s+rst-class::\s+(.+)$", re.M),
    re.compile(r"^\s*:class(?:-card|-container|-item|-body|-title)?:\s+(.+)$",
               re.M),
)


def used_classes() -> dict[str, list[str]]:
    """Map class name -> the pages that name it."""
    found: dict[str, list[str]] = {}
    for page in sorted(HERE.rglob("*.rst")):
        if "_build" in page.parts:
            continue
        text = page.read_text(encoding="utf-8")
        for pattern in PATTERNS:
            for match in pattern.findall(text):
                for name in match.split():
                    if name in EXTERNAL:
                        continue
                    found.setdefault(name, []).append(
                        str(page.relative_to(HERE)))
    return found


def strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def check_syntax(css: str) -> list[str]:
    """Report text sitting outside both a comment and a rule body.

    A browser recovering from a syntax error discards everything up to the next
    balanced brace, which can take a whole rule block with it and leaves no
    trace anywhere. That happened here: an edit closed a comment early, the
    leftover prose invalidated the block underneath it, and the figures lost
    their border and radius while every other rule in the file kept working.
    """
    problems = []
    stripped = strip_comments(css)
    depth = 0
    buffer, line = "", 1
    start = 1
    for char in stripped:
        if char == "\n":
            line += 1
        if char == "{":
            if depth == 0:
                selector = " ".join(buffer.split())
                # A selector is the only thing allowed before an opening brace.
                if selector and not re.fullmatch(
                        r"[-\w\s.#,:()\[\]='\"*>+~@%]+", selector):
                    problems.append(f"line {start}: not a selector: {selector!r}")
                buffer, start = "", line
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                problems.append(f"line {line}: unbalanced closing brace")
                depth = 0
            buffer, start = "", line
        elif depth == 0:
            if not buffer.strip():
                start = line
            buffer += char
    if depth != 0:
        problems.append(f"{depth} unclosed brace(s) at end of file")
    if buffer.strip():
        problems.append(f"line {start}: trailing text outside any rule: "
                        f"{' '.join(buffer.split())[:70]!r}")
    if css.count("/*") != css.count("*/"):
        problems.append(f"unbalanced comment markers: {css.count('/*')} /* "
                        f"against {css.count('*/')} */")
    return problems


def styled_classes() -> set[str]:
    # Strip comments first so a class named only inside a comment does not
    # count as styled -- the comments here quote selectors on purpose.
    return set(re.findall(r"\.([A-Za-z][\w-]*)",
                          strip_comments(CSS.read_text(encoding="utf-8"))))


def main() -> int:
    if not CSS.exists():
        print(f"missing {CSS}")
        return 1

    syntax = check_syntax(CSS.read_text(encoding="utf-8"))
    if syntax:
        print(f"{CSS.name} does not parse cleanly:")
        for problem in syntax:
            print(f"    {problem}")
        print("\nA browser drops everything up to the next balanced brace, so "
              "rules near this will silently stop applying.")
        return 1

    used = used_classes()
    styled = styled_classes()
    missing = {name: pages for name, pages in used.items() if name not in styled}

    for name in sorted(used):
        mark = "  " if name in styled else "!!"
        pages = ", ".join(sorted(set(used[name])))
        print(f"{mark} .{name:<22} {pages}")

    if missing:
        print(f"\n{len(missing)} class(es) named in a page with no rule in "
              f"{CSS.relative_to(HERE.parent)}:")
        for name, pages in sorted(missing.items()):
            print(f"    .{name} -- used by {', '.join(sorted(set(pages)))}")
        print("\nEither add a rule or drop the class. A class with no rule "
              "renders as if it were not written.")
        return 1

    print(f"\nall {len(used)} classes styled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
