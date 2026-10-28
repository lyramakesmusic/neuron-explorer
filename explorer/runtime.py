"""Live Qwen3.5 model for the explorer: steered generation (add or multiply one neuron, per batch row), per-token
neuron activations on arbitrary text, and logit-lens tokens for a neuron's down column.

The hook sits on the input of a layer's down_proj, where the neuron activations silu(gate) * up live, so add,
multiply, and reading are all exact. Add doses are in units of the median residual norm leaving the layer, the same
unit as the sweep's `--dose-unit rel`.
"""
import json
import os
import sqlite3
import threading

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, l2norm

MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-4B")
NO_MODEL = bool(os.environ.get("EXPLORER_NO_MODEL"))  # cache-only: never download or load weights
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _locked(f):
    """Runs a method entirely under the model lock, so its GPU temporaries are created and freed while no other
    thread is using the GPU (MPS fails when two threads' work interleaves)."""
    def g(self, *a, **k):
        with self.lock:
            return f(self, *a, **k)
    g.__doc__, g.__name__ = f.__doc__, f.__name__
    return g


class Runtime:
    def __init__(self, stats_dir):
        self.stats_dir = stats_dir
        os.makedirs(CACHE, exist_ok=True)
        self._db = sqlite3.connect(os.path.join(CACHE, "connections.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS con (key TEXT PRIMARY KEY, val TEXT)")
        self._db_lock = threading.Lock()
        self._lens_idx = None
        self._con_idx = None
        self.lock = threading.RLock()  # re-entrant: a locked method can call another
        self.model = self.tok = None
        self.status = "idle"
        self.resid_p50 = {}
        self._push = None  # [(layer, neuron, mode, per-row values tensor)], any number of neurons at once (neuron None = all)
        self._heads = None  # [(layer, head, scale)]: attention heads scaled (0 = ablated) during a read; head None = all
        self._read = None  # (layer, neuron) while reading
        self._seen = {}
        self._con_cache = {}
        self._capture_all = None
        self._capture_from = 0
        self._capture_full = None
        self._capture_heads = None  # while set: each layer's query/key side and each head's write size, per token
        self.heads = None  # the last text's captures: {"T", "q", "writes"}, see full_activations(heads=True)
        self._lens_cache = {}

    # ---------- loading ----------
    def ensure_loaded(self):
        if self.model is not None:
            return
        if NO_MODEL:
            raise RuntimeError("live model disabled (EXPLORER_NO_MODEL)")
        with self.lock:
            if self.model is not None:
                return
            self.status = "loading"
            dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(MODEL)
            # straight onto the device: loading on cpu and then moving needs two copies of the weights at once
            if dev == "mps":  # one tensor at a time: transformers' threaded loading can deadlock copying to mps
                os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
            m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
            m.model.visual = None  # the vision tower is never used; frees its weights
            if dev == "mps":
                torch.mps.empty_cache()
            self.dev, self.model, self.lm = dev, m, m.model.language_model
            for i, layer in enumerate(self.lm.layers):
                layer.mlp.down_proj.register_forward_pre_hook(self._hook(i))
                layer.register_forward_hook(self._resid_hook(i))
                # attention output projection: its input is the heads side by side (full attention: o_proj, 16 x 256
                # after the output gate; linear attention: out_proj, 32 value heads x 128 after the gated norm)
                if hasattr(layer, "self_attn"):
                    a = layer.self_attn
                    a.o_proj.register_forward_pre_hook(self._head_hook(i, a.o_proj.in_features // a.head_dim))
                else:
                    a = layer.linear_attn
                    a.out_proj.register_forward_pre_hook(self._head_hook(i, a.num_v_heads))
                a.register_forward_pre_hook(self._qk_hook(i), with_kwargs=True)
                if hasattr(layer, "linear_attn"):
                    a.norm.register_forward_pre_hook(self._gnorm_hook(i, a.num_v_heads))
                layer.post_attention_layernorm.register_forward_hook(self._mlp_in_hook(i))
            self.status = "ready"

    def _hook(self, i):
        def hook(mod, inp):
            h = inp[0]
            if self._capture_full is not None:
                self._capture_full[i] = h[0].float().cpu()  # [tokens, inter]
            if self._capture_all is not None:
                a = h[:, self._capture_from:].float().flatten(0, 1)  # [tokens, inter]
                mx, amx = a.max(0)
                mn, amn = a.min(0)
                self._capture_all[i] = (mx.cpu(), amx.cpu(), mn.cpu(), amn.cpu())
            if self._read is not None and self._read[0] == i:
                n = self._read[1]
                self._seen["act"] = h[0, :, n].float().cpu()
            mine = [p for p in self._push if p[0] == i] if self._push else []
            if mine:
                h = h.clone()
                for _, n, mode, vals in mine:
                    if n is None:  # the whole layer's MLP (multiply only)
                        h = h * vals[:, None, None].to(h.dtype)
                    elif mode == "mul":
                        h[:, :, n] = h[:, :, n] * vals[:, None].to(h.dtype)
                    else:
                        h[:, :, n] = h[:, :, n] + vals[:, None].to(h.dtype)
                return (h,)
        return hook

    def _mlp_in_hook(self, i):
        def hook(mod, inp, out):
            if self._capture_heads is not None:  # the MLP's input after the norm, and the residual's rms per token
                r = inp[0][0].float()
                self._capture_heads["x"][i] = out[0].half()
                self._capture_heads["rms"][i] = (r.pow(2).mean(-1) + mod.eps).sqrt()
        return hook

    def _gnorm_hook(self, i, n_heads):
        def hook(mod, args):
            if self._capture_heads is not None:  # linear attention: each head's output = scale * (sum of A_ij v_j)
                o, z = args[0].float(), args[1].float()
                sc = mod.weight.float() * F.silu(z) * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + mod.variance_epsilon)
                self._capture_heads["s"][i] = sc.view(-1, n_heads, sc.shape[-1]).transpose(0, 1).half()  # [H, T, D]
        return hook

    def _head_hook(self, i, n_heads):
        def hook(mod, inp):
            if self._capture_heads is not None:  # each head's write into the residual: |W_O[:, head] . out_head|
                x = inp[0][0]
                xv = x.view(x.shape[0], n_heads, -1)
                W = mod.weight.view(mod.weight.shape[0], n_heads, -1)
                self._capture_heads["writes"][i] = torch.einsum("thd,ohd->tho", xv, W).float().norm(dim=-1).cpu()  # [T, H]
                self._capture_heads["y"][i] = xv.transpose(0, 1).half()  # [H, T, D]: each head's output
            mine = [x for x in self._heads if x[0] == i] if self._heads else []
            if not mine:
                return None
            h = inp[0].clone()
            v = h.view(*h.shape[:-1], n_heads, h.shape[-1] // n_heads)
            for _, head, scale in mine:
                if head is None:
                    v.mul_(scale)
                else:
                    v[..., head, :] *= scale
            return (h,)
        return hook

    def _qk_hook(self, i):
        """While capturing heads: what decides each head's attention pattern. Full attention: queries and keys after
        norm and rotary (softmax pattern). Linear attention (Gated DeltaNet): queries and keys after the short conv and
        l2 norm, plus each token's write strength (beta) and the per-step log decay; head_row() turns these into the
        exact effective attention row, the weight each earlier token's value gets in a query's output."""
        def hook(mod, args, kwargs):
            if self._capture_heads is None:
                return None
            x = kwargs.get("hidden_states", args[0] if args else None)
            with torch.inference_mode():
                T = x.shape[1]
                if hasattr(mod, "q_norm"):
                    cos, sin = kwargs["position_embeddings"]
                    hd = mod.head_dim
                    q, gate = torch.chunk(mod.q_proj(x).view(1, T, -1, hd * 2), 2, dim=-1)
                    q = mod.q_norm(q).transpose(1, 2)
                    k = mod.k_norm(mod.k_proj(x).view(1, T, -1, hd)).transpose(1, 2)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    v = mod.v_proj(x).view(T, -1, hd).transpose(0, 1)
                    self._capture_heads["q"][i] = {"kind": "full", "q": q[0].half(), "k": k[0].half(), "scale": mod.scaling,
                                                   "v": v.half(), "s": gate[0].float().sigmoid().transpose(0, 1).half()}
                else:
                    qkv = mod.in_proj_qkv(x).transpose(1, 2)
                    w = mod.conv1d.weight
                    qkv = F.silu(F.conv1d(F.pad(qkv, (w.shape[-1] - 1, 0)), w, groups=mod.conv_dim)).transpose(1, 2)[0]
                    q, k, v = torch.split(qkv, [mod.key_dim, mod.key_dim, mod.value_dim], -1)
                    q = l2norm(q.float().view(T, -1, mod.head_k_dim)) / mod.head_k_dim ** 0.5
                    k = l2norm(k.float().view(T, -1, mod.head_k_dim))
                    beta = mod.in_proj_b(x)[0].float().sigmoid()  # [T, Hv]
                    logd = -mod.A_log.float().exp() * F.softplus(mod.in_proj_a(x)[0].float() + mod.dt_bias)  # [T, Hv]
                    self._capture_heads["q"][i] = {"kind": "linear", "q": q.transpose(0, 1).half(), "k": k.transpose(0, 1).half(),
                                                   "beta": beta.T.contiguous(), "logd": logd.T.contiguous(),
                                                   "rep": mod.num_v_heads // mod.num_k_heads,
                                                   "v": v.view(T, -1, mod.head_v_dim).transpose(0, 1).half()}
            return None
        return hook

    def head_row(self, layer, head, pos):
        with self.lock:
            return self._head_row(layer, head, pos)

    def _head_row(self, layer, head, pos):
        """Attention row of one head for query token `pos` of the last captured text: the weight of every token
        0..pos in that head's output there. Full attention: the softmax pattern. Linear attention: the exact effective
        weight (signed; queries carried back through each later token's decay and delta-rule write), scaled so its
        absolute values sum to 1."""
        c = self.heads["q"][layer]
        with torch.inference_mode():
            if c["kind"] == "full":
                kv = head // (c["q"].shape[0] // c["k"].shape[0])
                s = (c["k"][kv, :pos + 1].float() @ c["q"][head, pos].float()) * c["scale"]
                return s.softmax(-1).cpu().tolist()
            kh = head // c["rep"]
            q, k = c["q"][kh].float(), c["k"][kh].float()
            beta, d = c["beta"][head], c["logd"][head].exp()
            u, row = q[pos].clone(), torch.zeros(pos + 1, device=q.device)
            for j in range(pos, -1, -1):
                dot = u @ k[j]
                row[j] = beta[j] * dot
                u = d[j] * (u - beta[j] * dot * k[j])
            return (row / row.abs().sum().clamp(min=1e-12)).cpu().tolist()

    def head_pattern(self, layer, head):
        with self.lock:
            return self._head_pattern(layer, head)

    def _head_pattern(self, layer, head):
        """The whole attention pattern of one head on the last captured text: [T, T], row = query token, column = the
        earlier token it reads (zero above the diagonal). Linear layers give the exact effective weights, each row
        scaled so its absolute values sum to 1, as in head_row()."""
        c = self.heads["q"][layer]
        with torch.inference_mode():
            if c["kind"] == "full":
                kv = head // (c["q"].shape[0] // c["k"].shape[0])
                s = (c["q"][head].float() @ c["k"][kv].float().T) * c["scale"]
                T = s.shape[0]
                s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device), 1), float("-inf"))
                return s.softmax(-1).cpu().numpy()
            kh = head // c["rep"]
            q, k = c["q"][kh].float(), c["k"][kh].float()
            beta, d = c["beta"][head], c["logd"][head].exp()
            T = q.shape[0]
            A = torch.zeros(T, T, device=q.device)
            U = torch.zeros(T, q.shape[1], device=q.device)  # each query carried back through the later writes
            for j in range(T - 1, -1, -1):
                U[j] = q[j]
                dot = U @ k[j]
                A[:, j] = beta[j] * dot
                U = d[j] * (U - (beta[j] * dot)[:, None] * k[j][None])
            A = A / A.abs().sum(-1, keepdim=True).clamp(min=1e-12)
            return A.cpu().numpy()

    def raw_row(self, layer, head, pos, caps=None):
        """(Called with the model lock held.) The attention row of one head at token `pos`, unscaled: softmax weights (full attention) or the exact
        effective weights (linear attention), so that the head's output there is scale * sum_j row[j] * v_j."""
        c = (caps or self.heads)["q"][layer]
        with torch.inference_mode():
            if c["kind"] == "full":
                kv = head // (c["q"].shape[0] // c["k"].shape[0])
                return ((c["k"][kv, :pos + 1].float() @ c["q"][head, pos].float()) * c["scale"]).softmax(-1)
            kh = head // c["rep"]
            q, k = c["q"][kh].float(), c["k"][kh].float()
            beta, d = c["beta"][head], c["logd"][head].exp()
            u, row = q[pos].clone(), torch.zeros(pos + 1, device=q.device)
            for j in range(pos, -1, -1):
                dot = u @ k[j]
                row[j] = beta[j] * dot
                u = d[j] * (u - beta[j] * dot * k[j])
            return row

    def attn_drive(self, layer, n, pos, caps=None, top=6):
        with self.lock:
            return self._attn_drive(layer, n, pos, caps, top)

    def _attn_drive(self, layer, n, pos, caps=None, top=6):
        """How attention drives neuron n of `layer` at token `pos` of a captured text (full_activations(heads=True)).
        The activation silu(gate) * up is linearised in the residual at this token, through the pre-MLP norm with its
        scale held at this token's value. Returns act (the activation), total (the linearised drive of the whole
        residual), heads ([(layer, head, contribution)] for every head at this layer or earlier: its output read by
        the linearisation) and src ({(layer, head): [contribution per token 0..pos]}, for the `top` heads by size:
        attention weight times what that token's value carries)."""
        c = caps or self.heads
        me = self.lm.layers[layer]
        with torch.inference_mode():
            x, rms = c["x"][layer][pos].float(), c["rms"][layer][pos]
            wg, wu = me.mlp.gate_proj.weight[n].float(), me.mlp.up_proj.weight[n].float()
            g, u = wg @ x, wu @ x
            sg = torch.sigmoid(g)
            J = ((sg * (1 + g * (1 - sg))) * u * wg + g * sg * wu) * (1 + me.post_attention_layernorm.weight.float()) / rms
            out, T = [], {}
            for l in range(layer + 1):
                lay = self.lm.layers[l]
                W = (lay.self_attn.o_proj if hasattr(lay, "self_attn") else lay.linear_attn.out_proj).weight.float()
                y = c["y"][l][:, pos].float()  # [H, D]
                t = (W.T @ J).view(y.shape)
                T[l] = t
                out += [(l, h, float(v)) for h, v in enumerate((t * y).sum(-1).tolist())]
            src = {}
            for l, h, _ in sorted(out, key=lambda r: -abs(r[2]))[:top]:
                q = c["q"][l]
                if q["kind"] == "full":
                    v = q["v"][h // (q["q"].shape[0] // q["k"].shape[0]), :pos + 1].float()
                    sc = q["s"][h, pos].float()
                else:
                    v, sc = q["v"][h, :pos + 1].float(), c["s"][l][h, pos].float()
                src[(l, h)] = (self.raw_row(l, h, pos, c) * (v @ (sc * T[l][h]))).tolist()
            return {"act": float(g * sg * u), "total": float((sg * (1 + g * (1 - sg)) * g + g * sg) * u),
                    "heads": out, "src": src}

    def layer_patterns(self, layer, caps=None):
        with self.lock:
            return self._layer_patterns(layer, caps)

    def _layer_patterns(self, layer, caps=None):
        """head_pattern() for every head of a layer at once: [H, T, T], from `caps` (a self.heads capture) or the
        last captured text."""
        c = (caps or self.heads)["q"][layer]
        with torch.inference_mode():
            if c["kind"] == "full":
                k = c["k"].float().repeat_interleave(c["q"].shape[0] // c["k"].shape[0], 0)
                s = (c["q"].float() @ k.transpose(1, 2)) * c["scale"]
                T = s.shape[-1]
                s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device), 1), float("-inf"))
                return s.softmax(-1).cpu().numpy()
            q = c["q"].float().repeat_interleave(c["rep"], 0)
            k = c["k"].float().repeat_interleave(c["rep"], 0)
            beta, d = c["beta"].float(), c["logd"].float().exp()  # [H, T]
            H, T, D = q.shape
            A = torch.zeros(H, T, T, device=q.device)
            U = torch.zeros(H, T, D, device=q.device)
            for j in range(T - 1, -1, -1):
                U[:, j] = q[:, j]
                dot = (U @ k[:, j, :, None])[..., 0]  # [H, T]
                A[:, :, j] = beta[:, j, None] * dot
                U = d[:, j, None, None] * (U - (beta[:, j, None] * dot)[..., None] * k[:, j, None, :])
            A = A / A.abs().sum(-1, keepdim=True).clamp(min=1e-12)
            return A.cpu().numpy()

    @_locked
    def head_links(self, layer, head, k=40):
        """Weight-level reads and writes of one attention head. reads: earlier neurons whose down column lands in the
        head's value read space, as the share of the column's length the head picks up (after the pre-attention norm
        weight); writes: neurons at this layer or later whose gate and up rows read the head's output space, the same
        share for each row, multiplied. Shares make large-norm neurons that line up with every head (hubs) sink. Each
        list is the top k by that share, z-scored over all candidate neurons. q and k: earlier neurons whose down
        column lands in the head's query or key read space, the same way as reads."""
        key = f"{layer}:{head}:{k}:qk"
        with self._db_lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS headcon (key TEXT PRIMARY KEY, val TEXT)")
            row = self._db.execute("SELECT val FROM headcon WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            lay = self.lm.layers[layer]
            g = (1.0 + lay.input_layernorm.weight.float())
            if hasattr(lay, "self_attn"):
                a = lay.self_attn; hd = a.head_dim
                nq, nkv = a.o_proj.in_features // hd, a.k_proj.out_features // hd
                kv = head // (nq // nkv)
                WV = a.v_proj.weight[kv * hd:(kv + 1) * hd].float() * g
                WO = a.o_proj.weight[:, head * hd:(head + 1) * hd].float()
                WQ = a.q_proj.weight.view(nq, 2 * hd, -1)[head, :hd].float() * g
                WK = a.k_proj.weight[kv * hd:(kv + 1) * hd].float() * g
            else:
                a = lay.linear_attn; hd = a.head_v_dim
                WV = a.in_proj_qkv.weight[2 * a.key_dim + head * hd: 2 * a.key_dim + (head + 1) * hd].float() * g
                WO = a.out_proj.weight[:, head * hd:(head + 1) * hd].float() * a.norm.weight.float()
                kh, kd = head // (a.num_v_heads // a.num_k_heads), a.head_k_dim
                WQ = a.in_proj_qkv.weight[kh * kd:(kh + 1) * kd].float() * g
                WK = a.in_proj_qkv.weight[a.key_dim + kh * kd:a.key_dim + (kh + 1) * kd].float() * g
            # orthonormal bases of the read and write spaces, so a share is a plain projection length
            QV = torch.linalg.qr(WV.T.cpu())[0].to(WV.device)  # [hidden, hd]
            QO = torch.linalg.qr(WO.cpu())[0].to(WO.device)
            def top(scores):
                s = torch.cat(scores)
                z = (s - s.mean()) / (s.std() + 1e-8)
                t = z.topk(k)
                N = scores[0].numel()
                return t.values.tolist(), t.indices.tolist(), N
            QQ = torch.linalg.qr(WQ.T.cpu())[0].to(WQ.device)
            QK = torch.linalg.qr(WK.T.cpu())[0].to(WK.device)
            reads, rq, rk, wr = [], [], [], []
            for l in range(layer):
                D = self.lm.layers[l].mlp.down_proj.weight.float()  # [hidden, N]
                dn = D.norm(dim=0).clamp(min=1e-8)
                reads.append(((QV.T @ D).norm(dim=0) / dn).cpu())
                rq.append(((QQ.T @ D).norm(dim=0) / dn).cpu())
                rk.append(((QK.T @ D).norm(dim=0) / dn).cpu())
            for l in range(layer, len(self.lm.layers)):
                L2 = self.lm.layers[l]
                w = 1.0 + L2.post_attention_layernorm.weight.float()
                G, U = L2.mlp.gate_proj.weight.float() * w, L2.mlp.up_proj.weight.float() * w
                share = lambda R: (R @ QO).norm(dim=1) / R.norm(dim=1).clamp(min=1e-8)
                wr.append((share(G) * share(U)).cpu())
            out = {}
            for side, scores, l0 in (("reads", reads, 0), ("q", rq, 0), ("k", rk, 0), ("writes", wr, layer)):
                if not scores:
                    out[side] = []
                    continue
                zs, idx, N = top(scores)
                out[side] = [{"layer": l0 + i // N, "n": i % N, "z": round(z, 2)} for z, i in zip(zs, idx)]
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO headcon VALUES (?, ?)", (key, json.dumps(out)))
            self._db.commit()
        return out

    @_locked
    def _head_bases(self):
        """Orthonormal bases of every head's read space (value rows after the pre-attention norm weight) and write
        space (output columns, after the gated norm weight on linear layers), stacked: (QV [hidden, sum hd],
        QO [hidden, sum hd], [(layer, head, start, end)])."""
        if getattr(self, "_hb", None) is not None:
            return self._hb
        self.ensure_loaded()
        qv, qo, spans, at = [], [], [], 0
        with self.lock, torch.inference_mode():
            for l, lay in enumerate(self.lm.layers):
                g = 1.0 + lay.input_layernorm.weight.float()
                if hasattr(lay, "self_attn"):
                    a = lay.self_attn; hd = a.head_dim
                    nq, nkv = a.o_proj.in_features // hd, a.k_proj.out_features // hd
                    heads = [(a.v_proj.weight[(h // (nq // nkv)) * hd:(h // (nq // nkv) + 1) * hd].float() * g,
                              a.o_proj.weight[:, h * hd:(h + 1) * hd].float()) for h in range(nq)]
                else:
                    a = lay.linear_attn; hd = a.head_v_dim; o = 2 * a.key_dim
                    heads = [(a.in_proj_qkv.weight[o + h * hd:o + (h + 1) * hd].float() * g,
                              a.out_proj.weight[:, h * hd:(h + 1) * hd].float() * a.norm.weight.float())
                             for h in range(a.num_v_heads)]
                for h, (WV, WO) in enumerate(heads):
                    qv.append(torch.linalg.qr(WV.T.cpu())[0]); qo.append(torch.linalg.qr(WO.cpu())[0])
                    spans.append((l, h, at, at + hd)); at += hd
        self._hb = (torch.cat(qv, 1).to(self.dev), torch.cat(qo, 1).to(self.dev), spans)
        return self._hb

    @_locked
    def neuron_heads(self, layer, n):
        """Which attention heads carry a neuron's output on, and which feed it, from weights. Later heads: the share
        of the neuron's down column inside each head's read space. Heads at this layer or earlier: the share of its
        gate row inside the head's write space times the same for its up row (after the pre-MLP norm weight). Each
        side is z-scored over its heads; returns {layer: [z per head]}."""
        QV, QO, spans = self._head_bases()
        with self.lock, torch.inference_mode():
            me = self.lm.layers[layer]
            d = me.mlp.down_proj.weight[:, n].float()
            w = 1.0 + me.post_attention_layernorm.weight.float()
            gr, ur = me.mlp.gate_proj.weight[n].float() * w, me.mlp.up_proj.weight[n].float() * w
            pr = (QV.T @ d).cpu()  # projections of the down column on every head's read basis
            pg, pu = (QO.T @ gr).cpu(), (QO.T @ ur).cpu()
        dn, gn, un, hid = float(d.norm()), float(gr.norm()), float(ur.norm()), d.shape[0]
        reads, writes = [], []
        for l, h, a, b in spans:
            base = ((b - a) / hid) ** 0.5  # a random direction's expected share of a (b - a)-dim space
            if l > layer:
                reads.append((l, h, float(pr[a:b].norm()) / dn / base))
            else:
                writes.append((l, h, float(pg[a:b].norm()) / gn * float(pu[a:b].norm()) / un / base ** 2))
        out = {}
        for side in (reads, writes):
            if len(side) < 2:
                continue
            v = np.array([x[2] for x in side])
            z = (v - v.mean()) / (v.std() + 1e-12)
            for (l, h, _), zz in zip(side, z):
                out.setdefault(l, {})[h] = round(float(zz), 2)
        return {l: [hs[h] for h in sorted(hs)] for l, hs in out.items()}

    @_locked
    def _qk_bases(self):
        """Orthonormal bases of every attention head's query and key read spaces (input rows, after the pre-attention
        norm weight), stacked as float16 [hidden, sum dims] with spans [(layer, heads, start, end)]; `heads` lists the
        output heads that share the space (grouped keys in full attention, shared q/k heads in linear attention)."""
        if getattr(self, "_qkb", None) is not None:
            return self._qkb
        self.ensure_loaded()
        out = {}
        with self.lock, torch.inference_mode():
            for role in ("q", "k"):
                cols, spans, at = [], [], 0
                for l, lay in enumerate(self.lm.layers):
                    g = 1.0 + lay.input_layernorm.weight.float()
                    if hasattr(lay, "self_attn"):
                        a = lay.self_attn; hd = a.head_dim
                        nq, nkv = a.o_proj.in_features // hd, a.k_proj.out_features // hd
                        if role == "q":
                            Wq = a.q_proj.weight.view(nq, 2 * hd, -1)[:, :hd]
                            units = [(Wq[h], [h]) for h in range(nq)]
                        else:
                            rep = nq // nkv
                            units = [(a.k_proj.weight[kv * hd:(kv + 1) * hd], list(range(kv * rep, (kv + 1) * rep))) for kv in range(nkv)]
                    else:
                        a = lay.linear_attn; hd = a.head_k_dim; rep = a.num_v_heads // a.num_k_heads
                        o = 0 if role == "q" else a.key_dim
                        units = [(a.in_proj_qkv.weight[o + kh * hd:o + (kh + 1) * hd], list(range(kh * rep, (kh + 1) * rep)))
                                 for kh in range(a.num_k_heads)]
                    for W, hs in units:
                        cols.append(torch.linalg.qr((W.float() * g).T.cpu())[0].half())
                        spans.append((l, hs, at, at + W.shape[0])); at += W.shape[0]
                out[role] = (torch.cat(cols, 1).to(self.dev), spans)
        self._qkb = out
        return out

    @_locked
    def neuron_head_roles(self, layer, n, k=8):
        """The attention heads a neuron meets, by role, from weights. q / k / v: later heads whose query, key or value
        read space holds a large share of the neuron's down column (it shapes what they look for from its token, makes
        its token findable, or is what they carry on). o: heads at this layer or earlier whose output lands in its
        gate and up rows (neuron_heads). Each role z-scored over its heads; the top k each, {layer, head, z}."""
        qk = self._qk_bases()
        zs = self.neuron_heads(layer, n)
        with self.lock, torch.inference_mode():
            d = self.lm.layers[layer].mlp.down_proj.weight[:, n].float()
            dn = float(d.norm())
            proj = {r: (B.T @ d.half()).float().cpu() for r, (B, _) in qk.items()}
        roles = {}
        for r, (_, spans) in qk.items():  # shares relative to a random direction's, as in neuron_heads
            side = [(l, hs, float(proj[r][a:b].norm()) / dn / ((b - a) / d.shape[0]) ** 0.5) for l, hs, a, b in spans if l > layer]
            if len(side) < 2:
                roles[r] = []
                continue
            v = np.array([x[2] for x in side]); z = (v - v.mean()) / (v.std() + 1e-12)
            rows = sorted(((zz, l, hs) for (l, hs, _), zz in zip(side, z)), key=lambda x: -x[0])[:k]
            roles[r] = [{"layer": l, "head": hs[0], "heads": hs, "z": round(float(zz), 1)} for zz, l, hs in rows]
        flat = [(z, int(l), h) for l, row in zs.items() for h, z in enumerate(row)]
        for r, keep in (("v", lambda l: l > layer), ("o", lambda l: l <= layer)):
            rows = []  # heads sharing one value space (grouped in full attention) score alike: one entry each
            for z, l, h in sorted((x for x in flat if keep(x[1])), reverse=True):
                if rows and rows[-1]["layer"] == l and rows[-1]["z"] == round(z, 1):
                    rows[-1]["heads"].append(h)
                elif len(rows) < k:
                    rows.append({"layer": l, "head": h, "heads": [h], "z": round(z, 1)})
                else:
                    break
            for x in rows:
                x["heads"].sort(); x["head"] = x["heads"][0]
            roles[r] = rows
        return roles

    def n_heads(self, layer):
        L = self.lm.layers[layer]
        return L.self_attn.o_proj.in_features // L.self_attn.head_dim if hasattr(L, "self_attn") else L.linear_attn.num_v_heads

    def _resid_hook(self, i):
        def hook(mod, inp, out):
            if self._seen.get("want_resid") == i:
                h = out[0] if isinstance(out, tuple) else out
                self._seen["resid"] = h[0, 1:].float().norm(dim=-1).cpu()
        return hook

    def layer_resid_p50(self, layer):
        if layer in self.resid_p50:
            return self.resid_p50[layer]
        path = os.path.join(self.stats_dir, f"q35-4b_L{layer}.npz")
        if os.path.exists(path):
            v = float(np.load(path)["resid_norm_q10_50_90"][1])
        else:
            ids = self.tok(REFERENCE_TEXT, return_tensors="pt").input_ids.to(self.dev)
            self._seen["want_resid"] = layer
            with torch.inference_mode():
                self.lm(input_ids=ids, use_cache=False)
            self._seen.pop("want_resid")
            v = float(self._seen.pop("resid").median())
        self.resid_p50[layer] = v
        return v

    def down_col(self, layer, n):
        return self.lm.layers[layer].mlp.down_proj.weight[:, n].float()

    # ---------- steering ----------
    def chat_ids(self, prompt):
        return self.messages_ids([{"role": "user", "content": prompt}])

    def messages_ids(self, messages):
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)

    def steer_stream(self, layer, n, rows_messages, dose, mode, max_new=200, cancel=None):
        """Greedy-decodes three conversations in one batch: row 0 pushed up, row 1 unsteered, row 2 pushed down.
        rows_messages holds each row's own message list (the rows can differ in length; they are left-padded).
        Yields (row, text_delta), then (None, None)."""
        yield from self.steer_multi([(layer, n, dose, mode)], rows_messages, max_new, cancel)

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None, temperature=0.0, repetition_penalty=1.0, seed=0, min_p=0.0):
        """steer_stream for any number of neurons at once: pushes = [(layer, n, dose, mode)], all applied together.
        Each is reversed in row 2 the same way a single neuron is (add: -dose; mul: x -dose). Greedy by default; with
        temperature > 0 each row samples from its own generator seeded with `seed` (so a run repeats exactly), keeping
        only tokens with at least min_p times the top token's probability.
        repetition_penalty > 1 divides positive logits (multiplies negative ones) of every token already in the row's
        prompt or output, as in transformers."""
        self.ensure_loaded()
        with self.lock:
            if cancel is not None and cancel.is_set():
                yield None, None
                return
            try:
                push = [self._push_entry(layer, n, dose, mode, (1, 0, -1)) for layer, n, dose, mode in pushes]
                seqs = [self.messages_ids(m) for m in rows_messages]
                T = max(map(len, seqs))
                pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=self.dev)
                mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=self.dev)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                eos = self.tok.convert_tokens_to_ids("<|im_end|>")
                done, out_ids, sent = [False] * 3, [[] for _ in range(3)], [""] * 3
                gens = [torch.Generator().manual_seed(int(seed)) for _ in range(3)]
                seen = None
                if repetition_penalty != 1.0:
                    seen = torch.zeros(3, self.model.lm_head.out_features, dtype=torch.bool, device=self.dev)
                    for r, q in enumerate(seqs):
                        seen[r, torch.tensor(q, device=self.dev)] = True
                self._push = push
                with torch.inference_mode():
                    out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                                     logits_to_keep=1)
                    for _ in range(max_new):
                        if cancel is not None and cancel.is_set():
                            break
                        logits = out.logits[:, -1].float()
                        if seen is not None:
                            pen = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
                            logits = torch.where(seen, pen, logits)
                        if temperature > 0:
                            probs = torch.softmax(logits / temperature, -1)
                            if min_p > 0:
                                probs = torch.where(probs >= min_p * probs.max(-1, keepdim=True).values, probs, 0.0)
                                probs = probs / probs.sum(-1, keepdim=True)
                            probs = probs.cpu()
                            nxt = torch.cat([torch.multinomial(probs[r], 1, generator=gens[r]) for r in range(3)]).to(self.dev)
                        else:
                            nxt = logits.argmax(-1)
                        if seen is not None:
                            seen[torch.arange(3, device=self.dev), nxt] = True
                        for r in range(3):
                            if done[r]:
                                continue
                            t = int(nxt[r])
                            if t == eos:
                                done[r] = True
                                continue
                            out_ids[r].append(t)
                            text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                            if not text.endswith("\ufffd"):
                                yield r, text[len(sent[r]):]
                                sent[r] = text
                        if all(done):
                            break
                        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
                        pos = pos[:, -1:] + 1
                        out = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                                         past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            finally:
                self._push = None
        yield None, None

    def token_units(self, ids):
        """Groups token ids into displayable units: byte-level pieces of one multi-byte character (emoji, CJK split
        across tokens) join the token that completes it. Returns [(text, [token indices])]."""
        units, pending, prev = [], [], ""
        for i in range(len(ids)):
            full = self.tok.decode(ids[: i + 1])
            pending.append(i)
            if full.endswith("\ufffd") and i < len(ids) - 1:
                continue
            units.append((full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[j] for j in pending]), pending))
            prev, pending = full, []
        # keep emoji sequences whole: joiners, variation selectors and skin tones attach to the unit before them
        glue = {"\u200d", "\ufe0f", "\ufe0e", "\u20e3"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
        merged = []
        for text, idx in units:
            if merged and text and (text[0] in glue or merged[-1][0].endswith("\u200d")):
                merged[-1] = (merged[-1][0] + text, merged[-1][1] + idx)
            else:
                merged.append((text, idx))
        return merged

    # ---------- reading ----------
    def _push_entry(self, layer, n, dose, mode, rows):
        """One neuron's push for the down_proj hook, per batch row: rows holds +1 (pushed), 0 (left alone) or -1
        (pushed the other way). add doses are in units of the layer's median residual norm along the neuron's down
        column; mul doses multiply the activation (left alone = x1, the other way = x -dose)."""
        if n is None and mode != "mul":
            raise ValueError("a whole-layer MLP push must be mode 'mul' (e.g. dose 0 to ablate)")
        if mode == "mul":
            vals = [dose if r > 0 else 1.0 if r == 0 else -dose for r in rows]
        else:
            unit = self.layer_resid_p50(layer) / float(self.down_col(layer, n).norm())
            vals = [r * dose * unit for r in rows]
        return (layer, n, mode, torch.tensor(vals, device=self.dev))

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        """Per-token activation of neuron n at `layer` on `text` (as a user turn when chat=True), optionally with other
        neurons pushed at the same time: pushes = [(layer, n, dose, mode)], the same units as steering (n None = the
        layer's whole MLP, mul only), and attention heads scaled: heads = [(layer, head, scale)], head None = all heads,
        scale 0 = ablated. Only earlier layers can move the read (a same-layer MLP push lands after the capture; a
        same-layer head change lands before it)."""
        self.ensure_loaded()
        with self.lock:
            if chat:
                ids = self.chat_ids(text)
            else:
                ids = self.tok(text).input_ids
            self._read = (layer, n)
            self._push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
            self._heads = list(heads) or None if heads else None
            try:
                with torch.inference_mode():
                    self.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
            finally:
                self._read = None
                self._push = None
                self._heads = None
            act = self._seen.pop("act").tolist()
        units = []
        for text, idx in self.token_units(ids):
            vals = [act[i] for i in idx]
            units.append({"text": text, "act": max(vals, key=abs), "pieces": vals})
        return {"units": units}

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        """Log-probabilities of the next token after `text` (as a user turn when chat=True, so the next token opens the
        reply) followed by `prefix`, as a [vocab] tensor. With pushes / heads (same units as read()) it also returns
        the same distribution under that intervention: (base, intervened or None)."""
        self.ensure_loaded()
        ids = self.chat_ids(text) if chat else self.tok(text).input_ids
        if prefix:
            ids = ids + self.tok(prefix, add_special_tokens=False).input_ids

        def run(push, hd):
            with self.lock:
                self._push, self._heads = push, hd
                try:
                    with torch.inference_mode():
                        out = self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)
                finally:
                    self._push = self._heads = None
                return torch.log_softmax(out.logits[0, -1].float(), -1).cpu()

        base = run(None, None)
        if not pushes and not heads:
            return base, None
        push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
        return base, run(push, list(heads) if heads else None)

    def con_index(self):
        """Precomputed connections (build_connections.py), loaded once."""
        if self._con_idx is None:
            path = os.path.join(CACHE, "connections.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._con_idx = {key: z[key] for key in z.files}
                self._con_idx["k"] = int(z["k"])
            else:
                self._con_idx = False
        return self._con_idx or None

    # ---------- logit lens ----------
    def lens_index(self):
        if self._lens_idx is None:
            path = os.path.join(CACHE, "lens_index.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._lens_idx = {"up": z["up"], "down": z["down"]}
        return self._lens_idx

    def steer_index(self):
        if getattr(self, "_steer_idx", None) is None:
            path = os.path.join(CACHE, "steer_index.npz")
            self._steer_idx = dict(np.load(path)) if os.path.exists(path) else False
        return self._steer_idx or None

    @_locked
    def lens(self, layer, n, k=16):
        idx = self.lens_index()
        if idx is not None and k <= idx["up"].shape[-1]:
            v = self.vocab_strings()
            out = {"up": [v[t] for t in idx["up"][layer, n, :k]], "down": [v[t] for t in idx["down"][layer, n, :k]]}
            st = self.steer_index()
            if st is not None:
                out.update(steer_up=[v[t] for t in st["up"][layer, n, :k]], steer_down=[v[t] for t in st["down"][layer, n, :k]],
                           kl_up=round(float(st["up_kl"][layer, n]), 3), kl_down=round(float(st["down_kl"][layer, n]), 3))
            return out
        self.ensure_loaded()
        if (layer, n, k) in self._lens_cache:
            return self._lens_cache[(layer, n, k)]
        with self.lock, torch.inference_mode():
            d = self.down_col(layer, n).to(self.model.dtype)[None, None]
            out = {}
            for sign, name in ((1, "up"), (-1, "down")):
                logits = self.model.lm_head(self.lm.norm(sign * d))[0, 0].float()
                top = logits.topk(k).indices.tolist()
                out[name] = [self.tok.decode([t]) for t in top]
        self._lens_cache[(layer, n, k)] = out
        return out

    # ---------- connections ----------
    @_locked
    def connections(self, layer, n, k=24, lens_k=5):
        """Direct-path weights between neuron (layer, n) and neurons in other layers, ignoring everything between.
        Downstream B in layer j > layer: g = W_gate_j[B] . ((1 + w_j) * d_n), u = W_up_j[B] . (...), where d_n is this
        neuron's down column and w_j the pre-MLP norm weight. Upstream A in layer i < layer: the same with A's down
        column against this neuron's gate and up rows. Score = |g * u| z-scored within each layer; sign = sign(g * u).
        Each returned neuron carries its own logit-lens tokens."""
        key = (layer, n, k, lens_k)
        if key in self._con_cache:
            return self._con_cache[key]
        ci = self.con_index()
        if ci is not None and k <= ci["k"]:
            out = {}
            for side in ("down", "up"):
                z = ci[f"{side}_z"][layer, n]
                out[side] = [{"layer": int(ci[f"{side}_layer"][layer, n, i]), "n": int(ci[f"{side}_n"][layer, n, i]),
                              "z": round(float(z[i]), 2), "sign": int(ci[f"{side}_sign"][layer, n, i])}
                             for i in range(k) if z[i] > -1e4]
            idx = self.steer_index() or self.lens_index()  # linked neurons show their steer lens when there is one
            if lens_k and idx is not None:
                v = self.vocab_strings()
                lg = self.lens_index()
                for c in out["down"] + out["up"]:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
                    if lg is not None and lg is not idx:  # the logit lens too, shown faintly under the steer lens
                        c["logit"] = [v[t] for t in lg["up"][c["layer"], c["n"], :lens_k]]
                        c["logit_neg"] = [v[t] for t in lg["down"][c["layer"], c["n"], :lens_k]]
            return out
        dkey = f"{layer}:{n}:{k}:{lens_k}"
        with self._db_lock:
            row = self._db.execute("SELECT val FROM con WHERE key = ?", (dkey,)).fetchone()
        if row:
            out = json.loads(row[0])
            self._con_cache[key] = out
            return out
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            dt = self.model.dtype
            out = {"down": [], "up": []}
            d = self.lm.layers[layer].mlp.down_proj.weight[:, n]
            me = self.lm.layers[layer]
            scale_me = (1.0 + me.post_attention_layernorm.weight.float()).to(dt)
            r_g = me.mlp.gate_proj.weight[n] * scale_me
            r_u = me.mlp.up_proj.weight[n] * scale_me
            for j, Lyr in enumerate(self.lm.layers):
                if j == layer:
                    continue
                if j > layer:
                    v = d * (1.0 + Lyr.post_attention_layernorm.weight.float()).to(dt)
                    g, u = (Lyr.mlp.gate_proj.weight @ v).float(), (Lyr.mlp.up_proj.weight @ v).float()
                    side = "down"
                else:
                    W = Lyr.mlp.down_proj.weight
                    g, u = (r_g @ W).float(), (r_u @ W).float()
                    side = "up"
                prod = g * u
                mag = prod.abs()
                z = (mag - mag.mean()) / (mag.std() + 1e-8)
                top = z.topk(min(k, z.numel()))
                out[side] += [{"layer": j, "n": int(i), "z": round(float(sc), 2), "sign": 1 if prod[i] > 0 else -1}
                              for sc, i in zip(top.values.tolist(), top.indices.tolist())]
            for side in out:
                out[side] = sorted(out[side], key=lambda c: -c["z"])[:k]
            every = out["down"] + out["up"]
            idx = self.steer_index() or self.lens_index()
            if every and lens_k and idx is not None and lens_k <= idx["up"].shape[-1]:
                v = self.vocab_strings()
                for c in every:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
            elif every and lens_k:
                D = torch.stack([self.lm.layers[c["layer"]].mlp.down_proj.weight[:, c["n"]] for c in every])
                logits = self.model.lm_head(self.lm.norm(D[:, None]))[:, 0].float()
                ids, ids_neg = logits.topk(lens_k).indices.tolist(), (-logits).topk(lens_k).indices.tolist()
                for c, row, row_neg in zip(every, ids, ids_neg):
                    c["lens"] = [self.tok.decode([t]) for t in row]
                    c["lens_neg"] = [self.tok.decode([t]) for t in row_neg]
        if len(self._con_cache) > 2000:
            self._con_cache.clear()
        self._con_cache[key] = out
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO con VALUES (?, ?)", (dkey, json.dumps(out)))
            self._db.commit()
        return out

    # ---------- logit-lens index over every neuron ----------
    def build_lens_index(self, path, k=12, chunk=1024):
        """Top-k tokens each neuron's down column pushes up and down, for every layer. Saved as int32 arrays
        up[L, N, k], down[L, N, k]."""
        self.ensure_loaded()
        L, N = len(self.lm.layers), self.lm.layers[0].mlp.down_proj.in_features
        up = np.zeros((L, N, k), np.int32)
        down = np.zeros((L, N, k), np.int32)
        for li, layer in enumerate(self.lm.layers):
            self.status = f"indexing lens {li}/{L}"
            with self.lock, torch.inference_mode():
                W = layer.mlp.down_proj.weight  # [hidden, inter]
                for s in range(0, N, chunk):
                    D = W[:, s:s + chunk].T[:, None]  # [c, 1, hidden]
                    logits = self.model.lm_head(self.lm.norm(D))[:, 0]
                    up[li, s:s + chunk] = logits.topk(k).indices.cpu().numpy()
                    down[li, s:s + chunk] = (-logits).topk(k).indices.cpu().numpy()
        np.savez(path, up=up, down=down)
        self.status = "ready"
        return path

    def vocab_strings(self):
        if getattr(self, "_vocab", None) is None:
            path = os.path.join(CACHE, "vocab.json")
            if os.path.exists(path):
                self._vocab = json.load(open(path))
            else:
                if self.tok is None:
                    self.tok = AutoTokenizer.from_pretrained(MODEL)
                self._vocab = [self.tok.decode([i]) for i in range(len(self.tok))]
                json.dump(self._vocab, open(path, "w"))
        return self._vocab

    # ---------- which neurons fire on a text ----------
    def all_activations(self, ids, skip_first=False):
        """Per layer: (max, argmax, min, argmin) over tokens for every neuron, on token ids [T] or [B, T]. With
        skip_first, position 0 (the attention sink) is left out and argmax/argmin count from position 1."""
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            self._capture_all, self._capture_from = {}, int(skip_first)
            try:
                x = torch.tensor(ids, device=self.dev)
                self.lm(input_ids=x if x.dim() == 2 else x[None], use_cache=False)
                return self._capture_all
            finally:
                self._capture_all, self._capture_from = None, 0

    def build_act_scale(self, path, tokens=10000, seq=512, seed=0):
        """Each neuron's largest positive and most negative activation over web text (positions >= 1)."""
        from datasets import load_dataset
        self.ensure_loaded()
        L, N = len(self.lm.layers), self.lm.layers[0].mlp.down_proj.in_features
        pos, neg = np.zeros((L, N), np.float32), np.zeros((L, N), np.float32)
        buf, seen = [], 0
        for d in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed):
            buf += self.tok(d["text"], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            while len(buf) >= seq and seen < tokens:
                chunk, buf = buf[:seq], buf[seq:]
                self.status = f"measuring neuron ranges {seen // 1000}k/{tokens // 1000}k tokens"
                acts = self.all_activations(chunk, skip_first=True)
                for l, (mx, _, mn, _) in acts.items():
                    pos[l] = np.maximum(pos[l], mx.numpy()); neg[l] = np.minimum(neg[l], mn.numpy())
                seen += seq
            if seen >= tokens:
                break
        np.savez(path, pos=pos, neg=neg, tokens=seen)
        self.status = "ready"
        return path

    def full_activations(self, ids, heads=False):
        """Every neuron's activation at every token: float32 array [layers, tokens, inter]. With heads, the same pass
        also keeps what head_row() needs and every head's write size per token (self.heads)."""
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            self._capture_full = {}
            if heads:
                self._capture_heads = {"q": {}, "writes": {}, "y": {}, "s": {}, "x": {}, "rms": {}}
            try:
                self.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
                if heads:
                    self.heads = {"T": len(ids), "ids": list(ids), **self._capture_heads}
                return torch.stack([self._capture_full[l] for l in sorted(self._capture_full)]).numpy()
            finally:
                self._capture_full = None
                self._capture_heads = None
