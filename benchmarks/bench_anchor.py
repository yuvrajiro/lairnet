#!/usr/bin/env python
"""Which anchor fitter reaches a given solution quality soonest?

The obvious benchmark -- give every optimiser the same iteration budget and
time it -- does not answer that question. Implementations converge at different
rates, so one that stops early with a worse solution looks fast. A first pass
here showed exactly that pathology: at n=1000 the scipy path reached a lower
training loss than sklearn, and at n=5000 a higher one, purely because 500
iterations means different things to different L-BFGS implementations.

So this measures time to a common objective value instead.

  1. Every backend is run under the selected profile, recording
     (elapsed, objective) at each function evaluation. The profile fixes the
     tolerance and the iteration cap; see ``PROFILES`` below for which one is
     strict evidence and which one is a sanity check.
  2. The target is the best objective any backend reached, relaxed by a
     tolerance, so it is a value all of them can be asked for.
  3. Each backend is scored by the wall-clock at which its trajectory first
     crosses that target.

A backend that never reaches the target is reported as such rather than given a
number, because "did not get there" is the result in that case.

    python benchmarks/bench_anchor.py --profile quick
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lairnet._anchor import AnchorNet  # noqa: E402
from _guard import single_instance  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 0
HIDDEN = 100
L2 = 1e-4

#: Three regimes, because one set of settings cannot serve two purposes.
#:
#: ``strict`` is what the backend threshold in ``_anchor.py`` rests on. It runs
#: every backend to a far tighter tolerance than the package ever asks for, and
#: takes roughly forty minutes. That is defensible for a decision made once and
#: indefensible as something anyone re-runs.
#:
#: ``package`` uses the package's own defaults, so it measures the regime users
#: actually get rather than a precision they never request.
#:
#: ``quick`` is small enough to run before a release. It is a sanity check, not
#: evidence for a threshold, and the output says so.
PROFILES = {
    "quick":   dict(shapes=[(500, 10), (2000, 20)], repeats=2,
                    maxiter=5000, tol=1e-8),
    "package": dict(shapes=[(1000, 10), (5000, 30)], repeats=3,
                    maxiter=5000, tol=1e-8),
    "strict":  dict(shapes=[(1000, 10), (5000, 30), (20000, 50)], repeats=3,
                    maxiter=2000, tol=1e-10),
}

# Set from the selected profile at start-up. Module-level because this is a
# script and ``run()`` reads them directly; if this ever becomes importable
# library machinery they should become explicit arguments instead.
MAXITER = 5000
TOL = 1e-8
#: Relaxation applied to the target, once the target level is chosen.
TARGET_SLACK = 1.01
#: Independent restarts per backend. This is a non-convex problem and the
#: backends do not share an initialisation -- sklearn seeds itself -- so a
#: single run measures the basin it happened to land in as much as the
#: optimiser. Repeats turn that into a distribution instead of a claim.
REPEATS = 3


@dataclass
class Trace:
    """Objective against wall-clock for one backend on one problem."""

    name: str
    times: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    total_time: float = 0.0
    final_loss: float = float("nan")
    note: str = ""

    def time_to(self, target: float) -> Optional[float]:
        for t, l in zip(self.times, self.losses):
            if l <= target:
                return t
        return None

    @property
    def best(self) -> float:
        return min(self.losses) if self.losses else float("inf")


def make_data(n: int, d: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (2.0 * np.tanh(X @ w) + 0.5 * X[:, 0] * X[:, 1]
         + 0.3 * rng.standard_normal(n))
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


# --------------------------------------------------------------- backends
# Every backend is timed through AnchorNet, the class the package ships, using
# its `callback` hook. An earlier version reimplemented the torch path here.
# That copy was correct and the shipped one had a bug in how it read the final
# loss, so the benchmark was measuring the wrong implementation and would never
# have found it. Measuring one implementation says nothing about another.


def run(X, y, backend: str, seed: int = SEED, device=None) -> Trace:
    name = backend if device is None else f"{backend}-{device}"
    tr = Trace(name)
    t0 = time.perf_counter()

    def record(loss):
        tr.times.append(time.perf_counter() - t0)
        tr.losses.append(float(loss))

    model = AnchorNet(n_hidden=HIDDEN, activation="tanh", alpha=L2,
                      max_iter=MAXITER, tol=TOL, backend=backend,
                      device=device, random_state=seed,
                      callback=record).fit(X, y)
    tr.total_time = time.perf_counter() - t0
    tr.final_loss = float(model.loss_)
    if not tr.losses:
        # sklearn's L-BFGS exposes no per-iteration hook, so there is no
        # trajectory to record -- only the end point.
        tr.times = [tr.total_time]
        tr.losses = [tr.final_loss]
        tr.note = "final point only (no per-iteration hook)"
    return tr


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="quick",
                    help="which regime to measure (default: quick)")
    ap.add_argument("--json", metavar="PATH", help="write raw traces")
    args = ap.parse_args()

    global MAXITER, TOL, REPEATS
    profile = PROFILES[args.profile]
    shapes = profile["shapes"]
    REPEATS = profile["repeats"]
    MAXITER = profile["maxiter"]
    TOL = profile["tol"]
    print(f"profile: {args.profile}  shapes={shapes}  repeats={REPEATS}  "
          f"maxiter={MAXITER}  tol={TOL:g}")
    if args.profile == "quick":
        print("  a sanity check, not evidence for a backend threshold -- "
              "use --profile strict for that")

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    out: Dict[str, list] = {}
    for n, d in shapes:
        rng = np.random.default_rng(SEED)
        X, y = make_data(n, d, rng)
        runners = [("sklearn", lambda s: run(X, y, "sklearn", s)),
                   ("numpy", lambda s: run(X, y, "numpy", s))]
        try:
            import numba  # noqa: F401
            runners.append(("numba", lambda s: run(X, y, "numba", s)))
        except ImportError:
            pass
        try:
            import torch  # noqa: F401
            runners.append(("torch-cpu",
                            lambda s: run(X, y, "torch", s, "cpu")))
            if has_cuda:
                runners.append(("torch-cuda",
                                lambda s: run(X, y, "torch", s, "cuda")))
        except ImportError:
            pass

        # Repeats with different starts. One run measures the basin the
        # optimiser happened to land in as much as the optimiser itself.
        repeated = {name: [fn(SEED + r) for r in range(REPEATS)]
                    for name, fn in runners}
        traces: List[Trace] = [rs[0] for rs in repeated.values()]

        # The target is the WORST of the per-backend best losses, relaxed
        # slightly: the best level every backend demonstrably reaches. Using
        # the single best instead makes the target unreachable for everyone
        # else and reports "not reached" across the board, which is what the
        # first version of this benchmark did.
        target = max(max(r.best for r in runs)
                     for runs in repeated.values()) * TARGET_SLACK
        print(f"\nn={n}  d={d}  hidden={HIDDEN}")
        print(f"  target objective {target:.6g} "
              f"(best reached, relaxed by {TARGET_SLACK:g}x)")
        print(f"  {'backend':<12} {'to target':>10} {'speedup':>8} "
              f"{'total':>8} {'final obj':>11}")
        base = None
        for name, runs in repeated.items():
            hits = [r.time_to(target) for r in runs]
            got = [h for h in hits if h is not None]
            if not got:
                print(f"  {name:<12} {'not reached':>10} in any of "
                      f"{len(hits)} runs")
                continue
            med = float(np.median(got))
            lo, hi = min(got), max(got)
            if name == "sklearn":
                base = med
            sp = f"{base / med:5.2f}x" if base else "  --"
            miss = "" if len(got) == len(hits) else                 f"  ({len(hits) - len(got)} of {len(hits)} missed)"
            print(f"  {name:<12} {med:8.3f}s {sp:>8}  spread "
                  f"[{lo:.2f}, {hi:.2f}]"
                  f"  final {np.median([r.final_loss for r in runs]):.6g}{miss}")
        out[f"{n}x{d}"] = [
            {"name": t.name, "times": t.times, "losses": t.losses,
             "total": t.total_time, "final": t.final_loss} for t in traces]

    if args.json:
        Path(args.json).write_text(json.dumps(out), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def main() -> int:
    with single_instance("anchor"):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
