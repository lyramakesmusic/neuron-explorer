"""Input-side lens maps for Qwen3.5-4B: how the token embeddings move the residual each MLP reads.

    S_L = mean over prompts of [ mean over kept source positions p of  sum_{kept p' >= p} d h_L[p'] / d e[p] ]

e is the token embedding (embed_tokens output), h_L the residual entering layer L's MLP (the input of its
post_attention_layernorm), L = 0..31. Kept positions are 16 .. seq_len-2, as in the R/J++ fit. With --mode relp the
derivative is taken through the R-lens RelP rules of safety-research/jpp_lens (d7efc0d): residual RMSNorm scale
detached, SiLU's sigmoid detached (stock SiLU forward values, as in the R/J++ release), gate*up product half rule;
attention and GatedDeltaNet keep their ordinary derivatives. --mode grad takes plain derivatives (the K map).

One forward-mode pass per batch of input directions: row b carries tangent e_(start+b) on the embedding at every kept
position, and each layer's tangent summed over kept positions is column (start+b) of every S_L at once.

    python fit_input_lens.py PROMPTS.json OUT.pt --mode relp --jpp-src /path/to/jpp_lens/src [--rows 32] [--shard i/n]
    python fit_input_lens.py PROMPTS.json --check --mode relp --jpp-src ...   # JVP/VJP and finite-difference checks

Saves {"maps": {L: [2560 target, 2560 embedding] fp32}, "num_prompts", "positions", "mode", ...}; a running
checkpoint (OUT.partial) lets a stopped run resume.
"""
import argparse
import contextlib
import json
import os
import sys
import time

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ap = argparse.ArgumentParser()
ap.add_argument("prompts")
ap.add_argument("out", nargs="?")
ap.add_argument("--mode", choices=["relp", "grad"], default="relp")
ap.add_argument("--jpp-src", help="src/ of safety-research/jpp_lens at d7efc0d (needed for --mode relp)")
ap.add_argument("--prompts-n", type=int, default=64)
ap.add_argument("--rows", type=int, default=32)
ap.add_argument("--skip", type=int, default=16)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--check", action="store_true")
ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
ap.add_argument("--shard", default="0/1", help="i/n: fit prompts i, i+n, i+2n, ... (merge shards weighted by num_prompts)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
a = ap.parse_args()
dev = torch.device(a.device)
if dev.type == "mps":  # MPS linear has no forward-mode derivative; the matmul form does
    _linear = F.linear
    F.linear = lambda x, w, b=None: (x @ w.T + b if b is not None else x @ w.T) if fwAD._current_level >= 0 else _linear(x, w, b)


class ValueExactSiLU(torch.autograd.Function):
    """Stock SiLU forward values with the identity-rule derivative sigmoid(z), in both autodiff directions."""

    @staticmethod
    def forward(z):
        return F.silu(z)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (z,) = inputs
        ctx.save_for_backward(z)
        ctx.save_for_forward(z)

    @staticmethod
    def backward(ctx, g):
        (z,) = ctx.saved_tensors
        return g * torch.sigmoid(z)

    @staticmethod
    def jvp(ctx, zt):
        (z,) = ctx.saved_tensors
        return zt * torch.sigmoid(z)


@contextlib.contextmanager
def rules(model):
    if a.mode == "grad":
        yield None
        return
    sys.path.insert(0, a.jpp_src)
    import workspace_lens.lrp.lrp as lrp
    saved = dict(lrp._IDENTITY_RULE_ACTIVATIONS)
    lrp._IDENTITY_RULE_ACTIVATIONS["SiLU"] = ValueExactSiLU.apply
    lrp._IDENTITY_RULE_ACTIVATIONS["SiLUActivation"] = ValueExactSiLU.apply
    try:
        with lrp.apply_lrp_rules_for_mode(model, "rlens") as report:
            print(f"RelP rules: {len(report.ln_rule)} RMSNorms, {len(report.mlp_rule)} MLPs", flush=True)
            yield report
    finally:
        lrp._IDENTITY_RULE_ACTIVATIONS.clear()
        lrp._IDENTITY_RULE_ACTIVATIONS.update(saved)


cfg = AutoConfig.from_pretrained(MODEL, revision=REV).get_text_config()
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, config=cfg, dtype=getattr(torch, a.dtype),
                                             attn_implementation="eager").to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
lm = model.model
L, D = len(lm.layers), cfg.hidden_size
_si, _sn = map(int, a.shard.split("/"))
prompts = json.load(open(a.prompts))[:a.prompts_n][_si::_sn]

state = {"tangent": None, "seen": {}}


def emb_hook(mod, inp, out):
    t = state["tangent"]
    if t is None:
        return out
    if t == "grad":  # reverse-mode check: the embedding output becomes the leaf
        out = out.detach().requires_grad_(True)
        state["emb"] = out
        return out
    return fwAD.make_dual(out, t)


def mid_hook(li):
    def f(mod, inp):
        state["seen"][li] = inp[0]
    return f


lm.embed_tokens.register_forward_hook(emb_hook)
for li, layer in enumerate(lm.layers):
    layer.post_attention_layernorm.register_forward_pre_hook(mid_hook(li))


def encode(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=a.max_len).input_ids.to(dev)
    return ids, torch.arange(a.skip, ids.shape[1] - 1, device=dev)


def jvp_columns(ids, kept, V):
    """V: [B, D] input directions. Returns {L: [B, D]} = sum over kept targets of the tangent of h_L, with V placed at
    every kept source position."""
    B = V.shape[0]
    t = torch.zeros(B, ids.shape[1], D, dtype=model.dtype, device=dev)
    t[:, kept] = V[:, None].to(model.dtype)
    state["tangent"], state["seen"] = t, {}
    with torch.no_grad(), fwAD.dual_level():
        lm(input_ids=ids.expand(B, -1), use_cache=False)
        out = {li: fwAD.unpack_dual(x).tangent[:, kept].float().sum(1) for li, x in state["seen"].items()}
    state["tangent"], state["seen"] = None, {}
    return out


def check(ids, kept):
    """relp and grad: u . (S v) from forward mode against (u^T S) . v from reverse mode through the same rules;
    grad only: S v against a central finite difference."""
    torch.manual_seed(0)
    P = kept.numel()
    v = torch.randn(1, D, device=dev)
    fw = jvp_columns(ids, kept, v)
    for li in (0, 8, 16, 24, 31):
        u = torch.randn(D, device=dev)
        state["tangent"], state["seen"] = "grad", {}
        with torch.enable_grad():
            lm(input_ids=ids, use_cache=False)
            target = (state["seen"][li][0, kept].float() @ u).sum()
            (g,) = torch.autograd.grad(target, state["emb"])
        state["tangent"], state["seen"] = None, {}
        rev = float((g[0, kept].float() * v).sum())
        fwd = float(fw[li][0] @ u)
        print(f"L{li:2d}  forward u.(Sv) {fwd / P:+.5f}   reverse (u^T S).v {rev / P:+.5f}   rel {abs(fwd - rev) / max(abs(rev), 1e-9):.4f}",
              flush=True)
    if a.mode == "grad":
        emb = lm.embed_tokens(ids).float()
        eps = 0.05 * float(emb[0, kept].norm(dim=-1).mean())
        vv = v / v.norm() * eps
        res = {}
        for sgn in (1, -1):
            def add(mod, inp, out, s=sgn):
                o = out.clone(); o[:, kept] += (s * vv).to(o.dtype); return o
            h = lm.embed_tokens.register_forward_hook(add)
            state["seen"] = {}
            with torch.no_grad():
                lm(input_ids=ids, use_cache=False)
            h.remove()
            res[sgn] = {li: x[0, kept].float().sum(0) for li, x in state["seen"].items()}
        for li in (0, 8, 16, 24, 31):
            fd = (res[1][li] - res[-1][li]) / 2
            est = fw[li][0] * (eps / float(v.norm()))
            print(f"L{li:2d}  cosine(Sv, finite difference) {float(F.cosine_similarity(est, fd, dim=0)):.4f}  "
                  f"rel err {float((est - fd).norm() / fd.norm()):.3f}", flush=True)


with rules(model):
    if a.check:
        ids, kept = encode(prompts[0])
        t0 = time.time(); jvp_columns(ids, kept, torch.eye(D, device=dev)[:a.rows])
        if dev.type == "cuda": torch.cuda.synchronize()
        print(f"one forward-mode pass, {a.rows} directions: {time.time() - t0:.1f}s", flush=True)
        check(ids, kept)
        sys.exit()
    ck = a.out + ".partial"
    st = torch.load(ck, weights_only=True) if os.path.exists(ck) else \
        {"sum": {li: torch.zeros(D, D) for li in range(L)}, "done": 0, "positions": 0}
    if st["done"]:
        print(f"resuming after {st['done']} prompts", flush=True)
    eye = torch.eye(D, device=dev)
    t0 = time.time()
    for i, text in enumerate(prompts):
        if i < st["done"]:
            continue
        t1 = time.time()
        ids, kept = encode(text)
        P = kept.numel()
        for s in range(0, D, a.rows):
            cols = jvp_columns(ids, kept, eye[s:s + a.rows])
            for li, c in cols.items():
                st["sum"][li][:, s:s + c.shape[0]] += (c / P).T.cpu()
        st["done"], st["positions"] = i + 1, st["positions"] + P
        torch.save(st, ck + ".tmp"); os.replace(ck + ".tmp", ck)
        print(f"prompt {i + 1}/{len(prompts)}: {time.time() - t1:.1f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    torch.save({"maps": {li: st["sum"][li] / st["done"] for li in range(L)}, "num_prompts": st["done"],
                "positions": st["positions"], "mode": a.mode, "dtype": a.dtype, "shard": a.shard, "model_revision": REV, "skip": a.skip, "max_len": a.max_len,
                "estimator": "mean over prompts and kept sources p of sum over kept p' >= p of d h_L[p'] / d e[p]; "
                             "h_L = input of layer L's post_attention_layernorm"}, a.out)
    print(f"saved {a.out}", flush=True)
