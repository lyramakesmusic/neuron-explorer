"""R- and J++-lens readouts for every neuron in every layer, for the hover and the neuron panel.

Each neuron's down column (both signs) is carried from its block to block 31 by the R (or J++) map of
bubskee/qwen3.5-4b-r-jpp-lenses, then read through the final RMSNorm (1 + weight) and LM head in FP32, top k tokens,
with the release's token filter. Layer 31 is the lenses' target, so its row is the same filtered FP32 readout with no
transport. Needs the release in explorer/cache/rjpp/.

    uv run python explorer/build_rjpp_index.py                # R and J++, MPS or CUDA, a few minutes each

Writes explorer/cache/r_index.npz and jpp_index.npz: up, down [L, N, k] int32 token ids.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--lenses", nargs="+", default=["r", "jpp"])
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "mps")

cfg = AutoConfig.from_pretrained(a.model).get_text_config()
model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, dtype=torch.bfloat16).to(dev).eval()
lm = model.model
head = model.lm_head.weight.detach().float()
gain, eps = 1 + lm.norm.weight.detach().float(), lm.norm.eps
excl = torch.zeros(head.shape[0], dtype=torch.bool, device=dev)
excl[torch.tensor(json.load(open(os.path.join(CACHE, "rjpp", "excluded_token_ids.json"))), device=dev)] = True
L, N = len(lm.layers), lm.layers[0].mlp.down_proj.in_features
for lens in a.lenses:
    M_all = torch.load(os.path.join(CACHE, "rjpp", "lenses", f"{lens}.pt"), map_location="cpu", weights_only=True, mmap=True)["parameters"]["jacobians"]
    up, down = np.zeros((L, N, a.k), np.int32), np.zeros((L, N, a.k), np.int32)
    with torch.inference_mode():
        for li in range(L):
            W = lm.layers[li].mlp.down_proj.weight
            M = M_all[li].to(dev) if li < L - 1 else None
            for s in range(0, N, a.chunk):
                Z = W[:, s:s + a.chunk].T.float()
                if M is not None:
                    Z = Z @ M.T
                Z = Z * torch.rsqrt(Z.square().mean(-1, keepdim=True) + eps) * gain
                raw = Z @ head.T
                up[li, s:s + a.chunk] = raw.masked_fill(excl, float("-inf")).topk(a.k).indices.cpu().numpy()
                down[li, s:s + a.chunk] = (-raw).masked_fill(excl, float("-inf")).topk(a.k).indices.cpu().numpy()
            print(f"{lens} layer {li + 1}/{L}", flush=True)
    out = os.path.join(CACHE, f"{lens}_index.npz")
    np.savez(out, up=up, down=down)
    print(f"saved {out}")
