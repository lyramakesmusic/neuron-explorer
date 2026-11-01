"""Embedding-space steering score per unit, measured against the random-direction controls of the same run — the
lexical-free counterpart to shift.py. Where shift.py counts words, this measures how far a unit's rollouts sit from
the controls' rollouts in sentence-embedding space, on the same prompt and dose, and whether that displacement
points the same way across prompts. It is meant to settle whether a layer with a low word-shift score (research_log.md,
"the mid-layer dip survives an all-words metric") is actually inert, or only inert in vocabulary.

Needs <run>.emb.npy (build_embeddings.py's normalized bge-small embeddings), one row per line of <run>.jsonl, in the
same order.

For each unit and (prompt, dose) cell: edist = 1 - cosine(rollout embedding, that cell's control centroid),
leave-one-out when the unit is itself a control. econs = mean pairwise cosine similarity between the unit's
per-prompt mean displacement (rollout embedding minus its cell's control centroid), over the prompts it ran on;
None for a unit run on a single prompt (every "free" row, and any run with one prompt per unit).

edist alone rewards any large change, including degenerate rollouts (loops, empty <think>), which random directions
also produce. econs is the discriminating number: a unit that moves every prompt the same way scores high, a unit
that moves each prompt differently (or breaks differently) does not.

    from embed_shift import embed_shift_scores; scores = embed_shift_scores(rows, emb)   # {(kind, neuron): {...}}
"""
import argparse
import collections
import json

import numpy as np


def _unit_norm(v):
    n = np.linalg.norm(v)
    return v / n if n else None


def embed_shift_scores(rows, emb, max_rel=1.6):
    """rows and emb must be the same length and order (emb's row i is rows[i]'s embedding)."""
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms == 0, 1, norms)

    cell_idx = collections.defaultdict(list)  # (prompt, dose) -> row indices of random controls in that cell
    unit_idx = collections.defaultdict(list)  # (kind, neuron) -> row indices
    for i, r in enumerate(rows):
        if r["kind"] == "baseline" or r["rel_push"] > max_rel:
            continue
        unit_idx[(r["kind"], r["neuron"])].append(i)
        if r["kind"] == "random":
            cell_idx[(r["prompt"], r["dose"])].append(i)
    cell_sum = {c: emb[idx].sum(axis=0) for c, idx in cell_idx.items()}
    cell_n = {c: len(idx) for c, idx in cell_idx.items()}

    def centroid(cell, exclude_i, is_control):
        if cell not in cell_sum:
            return None
        s, n = cell_sum[cell], cell_n[cell]
        if is_control:
            s, n = s - emb[exclude_i], n - 1
        return _unit_norm(s / n) if n > 0 else None

    out = {}
    for u, idx in unit_idx.items():
        is_control = u[0] == "random"
        dists, by_prompt = [], collections.defaultdict(list)
        for i in idx:
            r = rows[i]
            c = centroid((r["prompt"], r["dose"]), i, is_control)
            if c is None:
                continue
            dists.append(1.0 - float(np.dot(emb[i], c)))
            by_prompt[r["prompt"]].append(emb[i] - c)
        prompt_vecs = [np.mean(vs, axis=0) for vs in by_prompt.values()]
        cons = []
        for a in range(len(prompt_vecs)):
            for b in range(a + 1, len(prompt_vecs)):
                va, vb = _unit_norm(prompt_vecs[a]), _unit_norm(prompt_vecs[b])
                if va is not None and vb is not None:
                    cons.append(float(np.dot(va, vb)))
        out[u] = {
            "edist": round(float(np.mean(dists)), 4) if dists else None,
            "econs": round(float(np.mean(cons)), 4) if cons else None,
        }

    for key in ("edist", "econs"):
        null = np.sort([v[key] for k, v in out.items() if k[0] == "random" and v[key] is not None])
        for v in out.values():
            v[f"null_pct_{key}"] = (round(float(np.searchsorted(null, v[key]) / len(null)), 4)
                                     if v[key] is not None and len(null) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--emb", default=None, help="defaults to <jsonl without .jsonl>.emb.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rel", type=float, default=1.6)
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--find", default="", help="comma-separated neuron ids to report ranks for")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl)]
    emb = np.load(a.emb or a.jsonl.replace(".jsonl", ".emb.npy"))
    scores = embed_shift_scores(rows, emb, a.max_rel)

    by_neuron = {}
    for r in rows:
        by_neuron.setdefault((r["kind"], r["neuron"]), r)
    records = [
        {"kind": k, "neuron": n, "layer": by_neuron[(k, n)]["layer"],
         "ref_neuron": by_neuron[(k, n)].get("ref_neuron"), **s}
        for (k, n), s in scores.items()
    ]
    records.sort(key=lambda r: (r["econs"] is None, -(r["econs"] or 0)))
    with open(a.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    null = np.array([r["econs"] for r in records if r["kind"] == "random" and r["econs"] is not None])
    neurons = [r for r in records if r["kind"] == "neuron" and r["econs"] is not None]
    if len(null) and neurons:
        p95 = np.percentile(null, 95)
        above = sum(r["econs"] > p95 for r in neurons)
        print(f"{len(neurons)} neurons, {len(null)} random controls; random econs p50/p95/max = "
              f"{np.percentile(null, 50):.2f} / {p95:.2f} / {null.max():.2f}")
        print(f"neurons above the control p95: {above} ({above / len(neurons):.1%})")
    for i, r in enumerate(records[: a.show]):
        print(f"{i + 1:4d}  {r['kind'][:3]} {r['neuron']:5d}  edist {r['edist']}  econs {r['econs']}")
    for n in [int(x) for x in a.find.split(",") if x]:
        rec = next((r for r in neurons if r["neuron"] == n), None)
        if rec:
            rank = neurons.index(rec) + 1
            print(f"neuron {n}: rank {rank}/{len(neurons)} among neurons, edist {rec['edist']}, econs {rec['econs']}, "
                  f"null_pct_econs {rec['null_pct_econs']}")


if __name__ == "__main__":
    main()
