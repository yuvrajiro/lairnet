"""Refuse to run two benchmarks at once.

Timings taken while another benchmark is running are not measurements. The
first run of ``bench_anchor.py`` was contaminated exactly this way -- a second
process was started when the first appeared to have stalled, and the same
configuration came out 45% apart between runs. Nothing in either output
distinguished the clean run from the dirty one.

    with single_instance("anchor"):
        ...
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def single_instance(name: str):
    """Hold an exclusive lock for the duration, or exit with an explanation."""
    lock = Path(tempfile.gettempdir()) / f"lairnet-bench-{name}.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = lock.read_text(encoding="utf-8").strip()
        except OSError:
            holder = "unknown"
        print(f"ABORT: another '{name}' benchmark is running (pid {holder}).\n"
              f"       Timings taken alongside it would not be measurements.\n"
              f"       If that process is dead, delete {lock}",
              file=sys.stderr)
        raise SystemExit(2)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            lock.unlink()
        except OSError:  # pragma: no cover
            pass
