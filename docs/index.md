# LAIR-Net

```{raw} html
<div class="ln-hero">
  <span class="ln-eyebrow">v0.1.0 &middot; randomized networks for tabular regression &middot; 2026</span>
  <h2 class="ln-title">A randomized network you can steer,<br><em>and then see inside.</em></h2>
  <p class="ln-lead">
    A shallow network fitted to your target supplies an anchor. A stack of fixed
    random layers runs with every layer pulled part-way back towards it, and a
    ridge readout is fitted at every depth. Nothing backpropagates through the
    stack, so it trains in seconds and it tells you what it did.
  </p>
  <div class="ln-actions">
    <a class="ln-btn ln-btn-solid" href="quickstart.html">Get started</a>
    <a class="ln-btn ln-btn-ghost" href="#how-it-works">How it works</a>
    <span class="ln-install">$ pip install lairnet</span>
  </div>
</div>

<div class="ln-cards">
  <div class="ln-card">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
         stroke-linejoin="round" aria-hidden="true">
      <path d="M3 20h18"/><path d="M6 20V10"/><path d="M11 20V4"/>
      <path d="M16 20v-7"/><path d="M21 20v-3"/>
    </svg>
    <h3>Which features it uses</h3>
    <p>Permutation importance on held-out data, with any bar whose error crosses
       zero greyed out rather than ranked.</p>
  </div>
  <div class="ln-card">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
         stroke-linejoin="round" aria-hidden="true">
      <path d="M3 17l5-6 4 3 5-7 4 4"/><path d="M3 21h18"/>
    </svg>
    <h3>Whether depth is helping</h3>
    <p>The score from aggregating the first <em>k</em> depths, for every
       <em>k</em>, in one forward pass instead of twenty refits.</p>
  </div>
  <div class="ln-card">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
         stroke-linejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/>
      <path d="M12 2v2"/><path d="M12 20v2"/><path d="M2 12h2"/><path d="M20 12h2"/>
    </svg>
    <h3>What the anchor is worth</h3>
    <p>Refit at <code>align=0</code> and compare. A flat curve means the
       target-aware component is doing nothing on your data.</p>
  </div>
</div>

<h2 id="how-it-works">How it works</h2>

<div class="ln-steps">
  <div class="ln-step">
    <span class="ln-step-num">01</span>
    <h3>Fit the anchor</h3>
    <p>One shallow network is trained on your target. It is the only iterative
       optimisation in the model.</p>
    <span class="ln-step-arrow">&rarr;</span>
  </div>
  <div class="ln-step">
    <span class="ln-step-num">02</span>
    <h3>Run the stack</h3>
    <p>Twenty fixed random layers, each under a leaky transition so the state
       changes gradually.</p>
    <span class="ln-step-arrow">&rarr;</span>
  </div>
  <div class="ln-step">
    <span class="ln-step-num">03</span>
    <h3>Pull towards it</h3>
    <p>Every layer is mixed part-way back to the anchor, which is what stops
       perturbations compounding with depth.</p>
    <span class="ln-step-arrow">&rarr;</span>
  </div>
  <div class="ln-step">
    <span class="ln-step-num">04</span>
    <h3>Read out and aggregate</h3>
    <p>A closed-form ridge readout at every depth, combined by a median across
       the stack.</p>
  </div>
</div>
```

## Quick start

```python
from lairnet import LAIRNetRegressor

model = LAIRNetRegressor().fit(X_train, y_train)
model.predict(X_test)

# and then, unlike most randomized networks:
model.anchor_converged_      # did the anchor fit converge, per the optimiser?
model.layer_conditions_      # how well conditioned is each readout?
model.layer_predictions(X)   # what did every depth predict?
```

One call reports the lot:

```python
from lairnet.explain import explain_model

print(explain_model(model, X_train, y_train, X_test, y_test).summary())
```

```text
score on the evaluation data: 0.8369

features the model uses
  feature 3                +0.6931 +/- 0.0397
  feature 0                +0.4900 +/- 0.0644
  ...

depth
  best at depth 12 of 12
  aggregating all depths is worth +0.1069 over one
  depths correlate 0.942 on average, so aggregation recovers less variance
  than independent errors would give

anchor
  with anchor    0.8369
  without anchor 0.7472
  the anchor is worth +0.0897 here
```

Called without held-out data it still works, and says so: every score above the
note is then measured on the training data.

```{raw} html
<figure class="ln-figure">
  <img src="_static/gallery/depth_path.png"
       alt="Score against depth, aggregated and per layer, on Friedman-1.">
  <figcaption>
    Score against depth: aggregating the first <em>k</em> depths, against the
    score from depth <em>k</em> alone. The gap between the two lines is what
    aggregation buys, and a curve that flattens early means the remaining
    layers are not paying for themselves.
  </figcaption>
</figure>

<div class="ln-cite">
  Goswami, R., Bhambu, A. and Karmakar, B. <em>LAIR-Net: Leaky
  Alignment-Impulse Residual Networks for Tabular Regression.</em>
  arXiv:2610.11538, 2026.
</div>
```

```bibtex
@misc{goswami2026lairnet,
  title  = {LAIR-Net: Leaky Alignment-Impulse Residual Networks for Tabular Regression},
  author = {Rahul Goswami and Aryan Bhambu and Bittu Karmakar},
  year   = {2026},
  eprint = {2610.11538},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url    = {https://arxiv.org/abs/2610.11538}
}
```

```{toctree}
:caption: Guide
:hidden:

installation
quickstart
usage
gallery
```

```{toctree}
:caption: Reference
:hidden:

api
development
```
