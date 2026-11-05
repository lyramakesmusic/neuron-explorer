"""Joins sweep/steer_lens.py's per-layer files into the explorer's steer index.

    uv run python sweep/steer_lens.py                  # stats/steer_lens_<model>_L{l}.npz, every layer
    uv run python explorer/build_steer_index.py        # -> explorer/cache[/models/<model>]/steer_index.npz

Writes up, down [L, N, k] int32 token ids and up_kl, down_kl [L, N] float32. Layers without a file get -1 / NaN
(missing layers are listed), so a partial run still loads.
"""
import argparse
import os

import numpy as np
from transformers import AutoConfig

from paths import CACHE, MODEL, NAME, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default=os.path.join(ROOT, "stats", f"steer_lens_{NAME}"))
a = ap.parse_args()

cfg = AutoConfig.from_pretrained(MODEL).get_text_config()
L, N = cfg.num_hidden_layers, cfg.intermediate_size
out, missing = None, []
for l in range(L):
    path = f"{a.prefix}_L{l}.npz"
    if not os.path.exists(path):
        missing.append(l)
        continue
    z = np.load(path)
    if out is None:
        k = z["up_tok"].shape[1]
        out = {"up": np.full((L, N, k), -1, np.int32), "down": np.full((L, N, k), -1, np.int32),
               "up_kl": np.full((L, N), np.nan, np.float32), "down_kl": np.full((L, N), np.nan, np.float32)}
    out["up"][l], out["down"][l] = z["up_tok"], z["down_tok"]
    out["up_kl"][l], out["down_kl"][l] = z["up_kl"], z["down_kl"]
if out is None:
    raise SystemExit(f"no {a.prefix}_L*.npz files; run sweep/steer_lens.py first")
dst = os.path.join(CACHE, "steer_index.npz")
np.savez(dst, **out)
print("->", dst, f"(missing layers: {missing})" if missing else "")
