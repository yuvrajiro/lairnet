# Benchmark results

Measurements, then the decision they support, kept apart on purpose. A number
and a policy are different kinds of claim and mixing them is how a default ends
up justified by a measurement that never supported it.

Reproduce with:

```bash
python benchmarks/bench_anchor.py            # four shapes
python benchmarks/bench_anchor.py --quick    # two
```

---

## How the anchor benchmark is run

The anchor is a single-hidden-layer network fitted by L-BFGS. Five backends
for that fit are compared: `sklearn`, `numpy`, `numba`, `torch-cpu` and
`torch-cuda`.

`numba` and both `torch` rows are **optional and comparative**. Neither is a
required dependency of the package, and neither is part of the default. They
appear here because a speed claim needs the alternatives measured, not because
installing them is expected.

**What reproduces which rows.** A plain `pip install lairnet` gives the
`sklearn` and `numpy` rows, which are the only ones the default path uses. The
`numba` and `torch` rows need those packages present; they were measured in the
development environment, and whether they are installable through an extra is a
packaging decision recorded in `pyproject.toml` rather than assumed here. If
the extra does not exist, these rows are not reproducible from a released
install, and that is the honest reading of the table.

**Time to a common objective, not time to a fixed iteration count.** Different
L-BFGS implementations converge at different rates, so a fixed budget times a
different amount of progress in each and a backend that stops early with a
worse solution looks fast. The first version of this benchmark did exactly
that, and produced the tell that gave it away: the scipy path reached a *lower*
training loss than sklearn at `n=1000` and a *higher* one at `n=5000`.

The target is the **worst** of the per-backend best losses, relaxed by 1%: the
best level every backend demonstrably reaches. Using the single best instead
makes the target unreachable for everyone else, which the second version of
this benchmark also did before it was corrected.

**Three repeats per backend, different seeds.** The problem is non-convex and
the backends do not share an initialisation — sklearn seeds itself — so a
single run measures which basin it landed in as much as which optimiser ran.
The spread is reported and is part of the result.

**The benchmark times the code the package ships.** Every backend is run
through `AnchorNet`, using its `callback` hook to record the objective at each
evaluation. An earlier version reimplemented the torch path inside the
benchmark. That copy was correct and the shipped one had a bug in how it read
its final loss, so the benchmark was measuring an implementation the package
does not use and could never have found the defect. The numbers below were
taken before that refactor, from a benchmark whose scipy paths already called
the package's own objective; the torch rows were measured from the parallel
implementation, which computed the same objective by the same optimiser and
differed only in the reporting bug, so the timings stand. The refactor removes
the class of error rather than correcting a number.

**One benchmark at a time.** `benchmarks/_guard.py` takes an exclusive lock and
refuses to start beside another. This exists because two concurrent runs of an
earlier version produced timings 45% apart for the same configuration, with
nothing in either output distinguishing the clean run from the contaminated
one. The lock narrows that failure. It does not remove it: it says nothing
about a test suite, an editor indexing, or anything else on the machine. These
timings are taken under a load controlled as far as it can be seen, not under
no load.

---

## 1. Measurements

Three profiles, and only one of them sets policy.

| profile | shapes | repeats | maxiter | tol | role |
|---|---|---|---|---|---|
| `quick` | 500, 2000 | 2 | 5000 | 1e-8 | pre-release sanity check |
| `package` | 1000, 5000 | 3 | 5000 | 1e-8 | **sets the default** |
| `strict` | 1000, 5000, 20000 | 3 | 2000 | 1e-10 | numerical stress context |

`package` runs at the package's own defaults, so it measures the regime users
are in. `strict` runs to a far tighter tolerance than the package ever asks
for; its numbers are kept as stress-test context and **do not set policy**.

They were not always separated, and that mattered. The threshold was first set
from `strict`, which is a regime the package never runs in. When the profiles
were split -- a usability change, not an audit -- the ordering came out
different, which is how the mismatch surfaced. That was a **policy-evidence
bug, not a correctness bug**: nothing computed a wrong answer, but a default was
justified by a measurement that did not describe the situation it governed.

### Package profile (sets the default)

**n = 1000, d = 10**

| backend | median | vs sklearn | spread |
|---|---|---|---|
| sklearn | 7.17 s | 1.00x | [6.91, 7.20] |
| numpy | **5.01 s** | 1.43x | [4.89, 5.14] |
| numba | 8.64 s | 0.83x | [7.83, 10.11] |
| torch-cpu | 6.13 s | 1.17x | [5.88, 7.29] |
| torch-cuda | 16.57 s | 0.43x | [14.94, 17.47] |

**n = 5000, d = 30**

| backend | median | vs sklearn | spread |
|---|---|---|---|
| sklearn | 208.09 s | 1.00x | [204.25, 226.71] |
| numpy | 178.75 s | 1.16x | [148.67, 212.42] |
| numba | 93.15 s | 2.23x | [82.13, 107.30] |
| torch-cpu | **29.52 s** | 7.05x | [25.13, 33.49] |
| torch-cuda | 36.39 s | 5.72x | [29.49, 44.40] |

numba is 0.58x of numpy at 1000 and 1.92x of it at 5000, with spreads separated
in both directions. torch on CPU is 7.05x sklearn and 6.06x numpy, separated
from everything.

### Strict profile (stress context, not policy)

`hidden=100`, `alpha=1e-4`, tanh, `maxiter=2000`, `tol=1e-10`, three repeats.
Times are the median wall-clock to reach the common target; spread is the range
over repeats.

### n = 1000, d = 10

| backend | median | vs sklearn | spread | final objective |
|---|---|---|---|---|
| sklearn | 5.55 s | 1.00x | [5.45, 5.64] | 2.06e-05 |
| numpy | **4.72 s** | 1.17x | [4.60, 5.74] | 2.35e-05 |
| numba | 8.56 s | 0.65x | [8.45, 9.50] | 2.32e-05 |
| torch-cpu | 5.77 s | 0.96x | [5.20, 6.91] | 2.61e-05 |
| torch-cuda | 15.90 s | 0.35x | [15.46, 16.35] | 2.58e-05 |

numpy is fastest by median, but its spread `[4.60, 5.74]` overlaps sklearn's
`[5.45, 5.64]`. At this size the advantage is at the edge of the noise.

numba loses to plain numpy: the JIT does not repay its overhead on a problem
this small. torch-cuda is three times slower than torch on CPU, because
transfers and kernel launches dominate an optimisation this size.

### n = 5000, d = 30

| backend | median | vs sklearn | spread | final objective |
|---|---|---|---|---|
| sklearn | 82.03 s | 1.00x | [81.76, 85.22] | 2.73e-03 |
| numpy | 83.81 s | 0.98x | [71.82, 88.86] | 3.90e-03 |
| numba | 49.32 s | 1.66x | [43.29, 49.98] | 3.89e-03 |
| torch-cpu | **19.68 s** | 4.17x | [18.52, 21.39] | 3.87e-03 |
| torch-cuda | 25.38 s | 3.23x | [20.52, 28.91] | 3.74e-03 |

The ordering at `n=1000` does not survive. numpy's 1.17x advantage there becomes
0.98x here -- no advantage at all -- and its spread is the widest of the five.

numba reverses: 0.65x of numpy at `n=1000`, 1.70x of it here, with spreads that
do not overlap in either direction.

torch on CPU is 4.17x sklearn and 4.26x numpy, and its spread `[18.52, 21.39]`
is separated from every other backend. It is unambiguously the fastest fitter
measured at this size.

torch on CUDA is *slower than torch on CPU* at both sizes. Its spread here,
`[20.52, 28.91]`, overlaps CPU's, so it is not distinguishable from it even
before accounting for the median being worse. A problem of this shape does not
have enough arithmetic per transfer to pay for the device.

---

## 2. Decision rule

A backend becomes the default only if it clears **both**:

1. a median improvement large enough to matter in practice, and
2. a spread clearly separated from the incumbent's.

Where a backend clears only the first, the default goes to the lowest
dependency path instead, and the documentation calls it a conservative default
rather than a fast one.

This rule exists because the first numbers taken here supported a claim of a
1.49x speed-up that repeated clean measurement reduced to 1.17x with
overlapping spreads.

---

## 3. Package default

`backend="auto"` resolves to **numpy** below 5000 samples and **numba** at or
above it, when numba is importable. torch is never chosen automatically.

Applying the rule of section 2 to the **package profile**:

| size | numba vs numpy | spreads | clears both bars? | chosen |
|---|---|---|---|---|
| n = 1000 | 0.58x — loses | separated | no | numpy |
| n = 5000 | 1.92x — wins | separated | yes | numba |

**At `n=1000` numba is clearly worse** than numpy, so the default is numpy.
That is also the lowest-dependency path: scipy is already required, and it
avoids making sklearn's `MLPRegressor` the fitter.

**At `n=5000` torch-cpu is fastest by a distance and is still not chosen.** It
is 7.05x sklearn and 6.06x numpy with a separated spread, so it clears both
bars comfortably. The decision against it is a dependency one, not a
measurement one: torch is a multi-gigabyte install, and a default that silently
prefers it when present would make the same code behave differently in two
environments. numba is second, is a light dependency, and clears both bars
against numpy on its own account, so it takes the slot.

**5000 is the smallest package-profile size measured to clear the rule, not the
crossover.** The `quick` profile shows numba already ahead of numpy at n=2000
with separated spreads, so the true crossing is very likely lower. That is
follow-up evidence and not the release threshold, because `quick` is a
sanity check and this file says in section 1 that it does not set policy.
Using it here would be exactly the misuse the warning was written to prevent.

Tightening the threshold needs a package-profile run at n=2000. Until someone
does that, `auto` declines numba across a range where it probably helps, which
costs some speed and no correctness.

**If you have torch and your problem is large, set `backend="torch"`.** That is
documented, it is 4x faster at `n=5000`, and it is your decision rather than
one the package makes on your behalf. Do not reach for `device="cuda"` on a
problem this size: it was slower than the CPU in every regime measured.

---

## 4. Wording

The documentation does not describe any backend as "fastest" outside the
regime where it was measured to be. Where a default is chosen on dependency
weight or implementation simplicity, the documentation says so.
