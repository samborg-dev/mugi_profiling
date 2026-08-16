# Window search — first GPU results, 2026-08-15

Llama-2-7b (NousResearch mirror, identical weights, ungated), allenai/c4
validation, n_samples=8, batch_size=2 → 4 batches, 32768 tokens/eval.
2×H200 on NCSA Delta. Code commit in logs/COMMIT.txt.

## Runs

| Job | Mode | Result |
|---|---|---|
| 21174619 | noise | cancelled — submitted before the HF fix landed |
| 21174668 | noise, repeats=10 | baseline 3.896045, uniform 4.731204, **spread 0** |
| 21175365 | noise | cancelled — redundant with 21175526 |
| 21175483 | search, layers 0-3 | 4.731204 → 4.537334, 4/4 layers, 20 evals |
| 21175526 | noise, repeats=2 | adds bootstrap CI on the PPL estimate |

## Key numbers

- model load 39-48s; eval 26.1s median; apply 4.5ms
- ten repeats of one assignment were **bit-identical** — the evaluation is
  deterministic, so run-to-run spread is structurally zero, not merely small
- layer 2's anchor sweep is smooth and unimodal (6.31 / 4.57 / 4.70 / 5.43 /
  5.44 for anchors 0-4), so the search is finding real structure

## Caveats

- baseline 3.896 is NOT comparable to the paper's 5.75; n_samples=8 is a
  smoke-test slice, not the paper's evaluation set
- NOISE_FLOOR=0.001 was chosen to exercise the accept path, not measured
- the savings line printed in logs/search_21175483.out says 3.6x and is wrong:
  it pairs 32 loads with this run's 20 evals. The correct value for a 4-layer
  search is in search_trace_summary.yaml (saved_s 144.2, speedup 1.25).
  Fixed in code after this run.
- histogram seeding did not run — profile/ is empty, so all 32 layers used the
  fallback seed, which reproduces the manual starting condition
