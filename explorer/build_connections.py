"""Direct-path connections for every neuron in every layer, for the explorer's details tab.

For layers i < j, P[B, A] = (W_gate_j[B] . v_A) * (W_up_j[B] . v_A) with v_A = (1 + w_j) * W_down_i[:, A], w_j the
pre-MLP norm weight of layer j. Neuron A's downstream links in layer j are column A of P, |P| z-scored over B; neuron
B's upstream links in layer i are row B of P, |P| z-scored over A. Each neuron keeps its k strongest links per side
across all layers, with sign(P).

    uv run python explorer/build_connections.py            # CUDA, about 5 min on a 4090

Writes explorer/cache/connections.npz: {down,up}_layer [L, N, k] int8, {down,up}_n int16, {down,up}_z float16,
{down,up}_sign int8.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from paths import CACHE, MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=MODEL)
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--out", default=os.path.join(CACHE, "connections.npz"))
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "mps"

root = a.model if os.path.isdir(a.model) else snapshot_download(a.model, allow_patterns=["*.safetensors", "*.json"])
index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
cfg = json.load(open(os.path.join(root, "config.json")))["text_config"]
L, N, K = cfg["num_hidden_layers"], cfg["intermediate_size"], a.k


def tensor(name):
    with safe_open(os.path.join(root, index[name]), "pt") as f:
        return f.get_tensor(name).to(dev, torch.bfloat16)


pre = "model.language_model.layers.{}."
down = [tensor(pre.format(l) + "mlp.down_proj.weight") for l in range(L)]
gate = [tensor(pre.format(l) + "mlp.gate_proj.weight") for l in range(L)]
up = [tensor(pre.format(l) + "mlp.up_proj.weight") for l in range(L)]
norm = [(1.0 + tensor(pre.format(l) + "post_attention_layernorm.weight").float()).to(torch.bfloat16) for l in range(L)]

# running top-k per neuron and side: score, source layer, source neuron, sign
best = {s: (torch.full((L, N, K), -1e9, device=dev), torch.zeros((L, N, K), dtype=torch.int16, device=dev),
            torch.zeros((L, N, K), dtype=torch.int16, device=dev), torch.zeros((L, N, K), dtype=torch.int8, device=dev))
        for s in ("down", "up")}


def merge(side, l, z, idx, other_layer, sign):
    bz, bl, bn, bs = best[side]
    cz = torch.cat([bz[l], z], 1)
    cl = torch.cat([bl[l], torch.full_like(idx, other_layer, dtype=torch.int16)], 1)
    cn = torch.cat([bn[l], idx.to(torch.int16)], 1)
    cs = torch.cat([bs[l], sign], 1)
    top = cz.topk(K, dim=1)
    bz[l], bl[l], bn[l], bs[l] = top.values, cl.gather(1, top.indices), cn.gather(1, top.indices), cs.gather(1, top.indices)


t0 = time.time()
with torch.inference_mode():
    for j in range(1, L):
        for i in range(j):
            V = down[i] * norm[j][:, None]  # [hidden, N_i]
            P = (gate[j] @ V).float() * (up[j] @ V).float()  # [N_j (B), N_i (A)]
            M = P.abs()
            # downstream of A (layer i): column A over B
            zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-8)
            top = zc.topk(K, dim=0)  # [K, N_i]
            sgn = torch.where(P.gather(0, top.indices) > 0, 1, -1).to(torch.int8)
            merge("down", i, top.values.T, top.indices.T, j, sgn.T)
            # upstream of B (layer j): row B over A
            zr = (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-8)
            top = zr.topk(K, dim=1)  # [N_j, K]
            sgn = torch.where(P.gather(1, top.indices) > 0, 1, -1).to(torch.int8)
            merge("up", j, top.values, top.indices, i, sgn)
            del P, M, zc, zr
        print(f"L{j} done ({time.time() - t0:.0f}s)", flush=True)

out = {}
for side, (bz, bl, bn, bs) in best.items():
    out[f"{side}_z"] = bz.clamp(min=-60000).cpu().numpy().astype(np.float16)
    out[f"{side}_layer"] = bl.cpu().numpy().astype(np.int8)
    out[f"{side}_n"] = bn.cpu().numpy()
    out[f"{side}_sign"] = bs.cpu().numpy()
np.savez_compressed(a.out, **out, k=K)
print("->", a.out, f"{os.path.getsize(a.out) / 1e6:.0f} MB")
