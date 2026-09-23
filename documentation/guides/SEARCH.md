# The Search

This guide explains how `ProgressiveLayerSearch` turns a seed assignment into a tuned one.

---

## 1. The Shape of the Problem

Llama-2-7B has 32 layers, and the paper notes that its softmax distribution *"varies
significantly across layers"* (§3.4). A single window applied to all 32 layers therefore
costs more accuracy than it needs to; Figure 7 shows per-layer tuning recovering most of it.

The published numbers for Llama-2-7B:

| Configuration | Perplexity |
|---|---|
| Baseline (unapproximated) | 5.75 |
| VLP, single window for all layers | 6.21 |
| VLP, per-layer tuned (Figure 7) | **5.98** |

So per-layer tuning is worth **0.23 perplexity** across 32 layers, or about **0.0072 per
layer** if the gain were spread evenly. Figure 7 shows it is not evenly spread — some layers
move the curve and most do not.

That is the target the search is aiming at, and it is small. Small enough that every
candidate has to be scored with a full perplexity evaluation rather than a cheaper proxy —
which is why the cost of the search is the dominant design constraint throughout the rest of
this guide.

---

## 2. Progressive, Not Joint

The paper's Figure 7 caption says *"Tuning is done progressively across layers."* The search
implements exactly that:

- Layers are visited in order.
- A layer that has been decided is **frozen** at its chosen window.
- Layers not yet reached sit at their **seed** window.
- Only one layer's window varies at a time.

```python
for layer in self.layers:
    outcome = self._run_layer(layer, frozen, incumbent, by_layer[layer])
    frozen = frozen.with_override(layer, self.site, outcome.chosen)
    incumbent = min(incumbent, outcome.best_ppl)
```

This is greedy. It does not explore interactions between layers, and it cannot revisit an
earlier decision.

---

## 3. Seeding

Each layer's starting window comes from `HistogramSeeder`, which reads the measured exponent
distribution for that layer:

1. Load `layer_<n>/seq_len_<m>.pt` histograms, summed across files.
2. `TopKWindow.place()` finds the narrowest band holding most of the mass — threshold 0.92
   for softmax, tuned until the band is 4–5 bins wide.
3. `AutoCluster.cluster()` decides `max` or `min` anchoring.
4. `SoftmaxWindow.from_profile()` builds the window.
5. The anchor is clamped to the searchable range `[−7, 7]`.

Every failure path degrades to a fallback window **and records the reason**:

```python
except (ValueError, RuntimeError, IndexError, TypeError) as e:
    return self._fallback(layer, site, f"{type(e).__name__}: {e}")
```

The degradation ends up in the artifact, not in a swallowed exception. At startup:

```
seeded 27/32 layers from histograms, 5 from fallback
```

> **Only attention is seeded per layer.** `seed_assignment` assigns the *same* `FfnWindow`
> to every layer. The search tunes softmax; SiLU stays uniform.

---

## 4. Candidate Generation

`CandidateGrid.candidates_for(seed)` builds the candidate set:

```python
pairs = {(anchor, "max") for anchor in self.max_anchors}
pairs |= {(anchor, "min") for anchor in self.min_anchors}
pairs.add((seed.anchor, seed.anchor_side))
combos = {(anchor, side, dim) for (anchor, side) in pairs for dim in dims}
```

With the defaults — `max_anchors = 0..7`, `min_anchors = −7..0` — that is **16 anchor
placements**: eight max-anchored, eight min-anchored, with the seed guaranteed present.

**The candidate count is not fixed at 16.** It is 16 × however many `exp_dim` values are
being swept:

| Flags | Candidates per layer |
|---|---|
| default | 16 |
| `--exp_dims 8,10,12` | 48 |
| `--radius 1` | only anchors within 1 of the seed, plus the seed |

Candidates are ordered by distance from the seed, so the most promising are tried first and
early stopping costs the least.

---

## 5. Evaluation

Each candidate is applied to the **live model** — the weights are never reloaded:

```python
t0 = time.perf_counter()
apply_assignment(self.host, assignment)
apply_ms = (time.perf_counter() - t0) * 1000.0
```

`apply_assignment` calls `reset_lut` on each patched site, which rebuilds the lookup table
in place. This is the central engineering property of the framework: a 13 GB model loads
once and is re-parameterised thousands of times.

For scale, measured on Delta with Llama-2-7B: a model load is **38.1 s**, a single
evaluation is **207.4 s**.

---

## 6. Keeping or Rejecting a Candidate

After each evaluation the search decides whether the candidate replaces the window it is
currently holding. There are two acceptance rules, and they answer different questions.

**Unpaired (default).** The candidate wins if it beats the incumbent by more than
`--noise_floor`:

```python
return (incumbent - score) > self.budget.noise_floor, None
```

Here `--noise_floor` has to carry the *entire* significance test, because this rule has no
notion of how uncertain either score is. A floor measured this way — a bootstrap CI width on
one candidate's absolute perplexity — comes out large (order 1) relative to the ~0.007
per-layer effect the paper reports (§1), because it's dominated by which batches of text got
drawn, not by which window is better.

**Paired (`--paired`).** Every candidate in a layer is scored on the same batches as the
incumbent, so the comparison can bootstrap the batch-by-batch *differences* instead of one
candidate's absolute score. The shared "some text is harder" noise cancels out of a
difference, leaving an interval tight enough to resolve the real effect size:

```python
ci = paired_ppl_ci(incumbent_losses, candidate_losses, ...)
return ci[0] > self.budget.noise_floor, ci
```

The candidate wins only if the *entire* confidence interval on the difference clears the
floor — the improvement has to be statistically significant, not just numerically lower.
Under `--paired`, `--noise_floor` is on the difference scale (order 0.01, not order 1) and is
an optional *extra* margin on top of the significance test the paired interval already
performs — `--noise_floor 0` is a legitimate value here, not a degenerate one, because the
interval itself is doing the real work.

**The two floors are not interchangeable, and this is enforced.** `--noise_summary` points at
a `_summary.yaml` written by `--mode noise`, which measures the *unpaired* CI — the wrong
scale for `--paired`. Passing both together is a hard error:

```
--paired needs --noise_floor, not --noise_summary '...'. --noise_summary/--noise_metric read
the *_summary.yaml written by --mode noise, which measures an UNPAIRED bootstrap CI on
absolute perplexity (order 1) ... --paired compares per-layer PPL DIFFERENCES instead
(order 0.01) ...
```

Under `--paired`, pass `--noise_floor` explicitly.

**`--mode search` requires a floor one way or another.** There is no default — omit both
flags and the run exits before loading the model:

```
--mode search needs a noise floor: pass --noise_floor, ...
```

Passing `--noise_floor 0` under the *unpaired* rule makes it "accept any improvement at
all" — the loosest setting available, and generally not what you want, since it can't tell a
real improvement from noise. The same value under `--paired` means something different and
safer: see above.

---

## 7. Stopping

Three budgets can end a layer or a run:

| Budget | Flag | Effect |
|---|---|---|
| Patience | `--patience` (default 3) | Stop this layer after N consecutive rejections |
| Per-layer evals | `--max_evals_per_layer` | Hard cap per layer |
| Total evals | `--max_evals` | Hard cap per run |
| Wall clock | `--time_budget_s` | Finalise at the seed and **still write artifacts** |

The time budget matters on Slurm. Set it below the job wall clock — if the job is SIGKILLed
instead, no artifacts are written at all and the run is a total loss.

> When patience runs out, the recorded stop reason is the enum value `NOISE_FLOOR`. The name
> is misleading — it fires on N consecutive rejections whatever caused them. Read it as
> "patience exhausted."

---

## 8. Cost

For a 32-layer run at defaults, the exhaustive count is fixed and computable: 15 candidates
evaluated per layer (16 minus the seed, which is never re-measured), plus one seed
evaluation and one final evaluation.

```
32 x 15 + 2 = 482 evaluations
```

With `--patience 3` the count is **data-dependent** — it depends on how early three
consecutive candidates fail in each layer — so it cannot be stated as a constant. Read the
actual figure from `n_evals` in `search_trace_summary.yaml` after a run.

Reload savings versus a model reload per configuration are modest on machine time alone —
about **1.01×**, because a load is 38 s against a 207 s evaluation. The real saving is
structural: the manual method is 32 separate Slurm submissions, each with its own queue
wait and its own human edit between runs. That cost does not appear in the ratio.

---

That completes the series. For how the result reaches the runtime, see
[Runtime Integration](RUNTIME_INTEGRATION.md).
