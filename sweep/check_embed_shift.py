"""Synthetic check for embed_shift.py, since runs/ and the embeddings it needs are gitignored and never published.
Builds fake rows and embeddings for one run (4 prompts x 2 doses, 40 random controls, 3 neurons), with one neuron's
embeddings shifted the same way on every prompt (should score highest on econs), one shifted a different way per
prompt (should score near the control median: large edist, unremarkable econs), and one left at the control noise
level (should score near the control median on both).

    python sweep/check_embed_shift.py
"""
import json

import numpy as np

from embed_shift import embed_shift_scores

rng = np.random.default_rng(0)
PROMPTS, DOSES, DIM = ["who", "snake", "orange", "free"], [0.8, -0.7], 32

rows, vecs = [], []


def add(kind, neuron, prompt, dose, vec):
    rows.append({"kind": kind, "neuron": neuron, "prompt": prompt, "dose": dose, "rel_push": abs(dose),
                 "layer": 22, "ref_neuron": neuron if kind == "neuron" else None, "text": ""})
    vecs.append(vec)


base = rng.normal(size=DIM)
# 40 random-direction controls: noise around the shared base point, same for every prompt/dose
for c in range(40):
    drift = rng.normal(scale=0.05, size=DIM)
    for p in PROMPTS:
        for d in DOSES:
            add("random", c, p, d, base + drift + rng.normal(scale=0.05, size=DIM))

consistent_dir = rng.normal(size=DIM)  # neuron 1: same displacement direction on every prompt
for p in PROMPTS:
    for d in DOSES:
        add("neuron", 1, p, d, base + 0.6 * consistent_dir + rng.normal(scale=0.05, size=DIM))

for p in PROMPTS:  # neuron 2: a different, large displacement per prompt (degenerate-looking, inconsistent)
    inconsistent_dir = rng.normal(size=DIM)
    for d in DOSES:
        add("neuron", 2, p, d, base + 0.6 * inconsistent_dir + rng.normal(scale=0.05, size=DIM))

for p in PROMPTS:  # neuron 3: control-level noise, no real shift
    for d in DOSES:
        add("neuron", 3, p, d, base + rng.normal(scale=0.05, size=DIM))

emb = np.array(vecs, dtype=np.float32)
scores = embed_shift_scores(rows, emb)

econs = {u: s["econs"] for u, s in scores.items()}
null = sorted(v for (k, _), v in econs.items() if k == "random")
p95 = np.percentile(null, 95)
n1, n2, n3 = econs[("neuron", 1)], econs[("neuron", 2)], econs[("neuron", 3)]
print(f"control econs: p50={np.percentile(null, 50):.3f} p95={p95:.3f} max={max(null):.3f}")
print(f"neuron 1 (consistent shift):   econs={n1:.3f}  {'ABOVE' if n1 > p95 else 'not above'} control p95")
print(f"neuron 2 (inconsistent shift): econs={n2:.3f}  {'ABOVE' if n2 > p95 else 'not above'} control p95")
print(f"neuron 3 (no real shift):      econs={n3:.3f}  {'ABOVE' if n3 > p95 else 'not above'} control p95")

ok = n1 > p95 and n2 <= p95 and n3 <= p95
print("PASS" if ok else "FAIL")
if not ok:
    raise SystemExit(1)
