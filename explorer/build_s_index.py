"""S-lens readout of every neuron's read side, for the neuron panel and the agent.

A neuron at layer L applies its gate row a and up row b to the residual entering its MLP (after the pre-MLP norm, so
each row is scaled by that norm's 1 + w). The + pole reads along a^ + b^ (gate and up agree), the - pole along
a^ - b^. Each pole, with the layer's mean read direction removed, is pulled back through the fitted input map S_L
(sweep/fit_input_lens.py) to the token embeddings and scored against every token: the input tokens whose presence in
the context pushes the residual along that read. Top k tokens per pole, with the R/J++ release's token filter.

    uv run python explorer/build_s_index.py            # MPS or CUDA, a few minutes

Reads explorer/cache/slens/S.pt; writes explorer/cache/s_index.npz: up, down [L, N, k] int32 token ids.
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
ap.add_argument("--maps", default=os.path.join(CACHE, "slens", "S.pt"))
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--raw", action="store_true", help="keep the layer's mean read direction")
ap.add_argument("--out", default=os.path.join(CACHE, "s_index.npz"))
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "mps")

cfg = AutoConfig.from_pretrained(a.model).get_text_config()
model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, dtype=torch.bfloat16).to(dev).eval()
lm = model.model
E = lm.embed_tokens.weight.detach().float()
excl = torch.zeros(E.shape[0], dtype=torch.bool, device=dev)
excl[torch.tensor(json.load(open(os.path.join(CACHE, "rjpp", "excluded_token_ids.json"))), device=dev)] = True
S = torch.load(a.maps, map_location="cpu", weights_only=True)["maps"]

L, N = len(lm.layers), lm.layers[0].mlp.down_proj.in_features
up, down = np.zeros((L, N, a.k), np.int32), np.zeros((L, N, a.k), np.int32)
with torch.inference_mode():
    for li, layer in enumerate(lm.layers):
        g = 1 + layer.post_attention_layernorm.weight.float()
        A = layer.mlp.gate_proj.weight.float() * g
        B = layer.mlp.up_proj.weight.float() * g
        A, B = A / A.norm(dim=-1, keepdim=True), B / B.norm(dim=-1, keepdim=True)
        SL = S[li].to(dev)
        for sign, arr in ((1, up), (-1, down)):
            Rd = A + sign * B
            if not a.raw:
                Rd = Rd - Rd.mean(0)
            for s in range(0, N, a.chunk):
                scores = (Rd[s:s + a.chunk] @ SL) @ E.T
                arr[li, s:s + a.chunk] = scores.masked_fill(excl, float("-inf")).topk(a.k).indices.cpu().numpy()
        print(f"layer {li + 1}/{L}", flush=True)
np.savez(a.out, up=up, down=down)
print(f"saved {a.out}")
