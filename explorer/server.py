"""Neuron explorer server: sweep data, names, embeddings, and the live model.

    uv run python explorer/server.py            # http://localhost:8791
"""
import asyncio
import builtins
import glob
import json
import os
import sys
import threading
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sweep"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank import compute_units  # noqa: E402
from runtime import CACHE, NO_MODEL, Runtime  # noqa: E402
from update import UpdateError, apply_update, check as update_check  # noqa: E402

RUNS, STATS = os.path.join(ROOT, "runs"), os.path.join(ROOT, "stats")
# the agent's neuronary.org tools (agent_mcp.py) read the person's token from here when set
NEURONARY_ENV = {k: os.environ[k] for k in ("NEURONARY_TOKEN", "NEURONARY_API") if os.environ.get(k)}
NAMES = os.path.join(ROOT, "explorer", "names.json")
PORT = int(os.environ.get("PORT", 8791))
_cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B").get_text_config()
ARCH = {"n_layers": _cfg.num_hidden_layers, "hidden": _cfg.hidden_size, "inter": _cfg.intermediate_size,
        "layer_types": list(_cfg.layer_types), "attn_heads": _cfg.num_attention_heads,
        "kv_heads": _cfg.num_key_value_heads, "linear_k_heads": _cfg.linear_num_key_heads,
        "linear_v_heads": _cfg.linear_num_value_heads}

app = FastAPI()
BOOT_ID = uuid.uuid4().hex  # changes on /api/update/restart so the page can tell when new code is live


# only this machine's own pages and tools: a page on another site (or a DNS name rebound to 127.0.0.1) can't reach
# the API, which runs the model and a Claude agent with a shell. Links from elsewhere still open pages (a top-level GET
# navigation, whose response the linking site can't read).
LOCAL_HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}", f"[::1]:{PORT}"}


@app.middleware("http")
async def local_only(request, call_next):
    from urllib.parse import urlparse
    origin = request.headers.get("origin")
    if (request.headers.get("host") not in LOCAL_HOSTS
            or (origin and urlparse(origin).netloc not in LOCAL_HOSTS)
            or (request.headers.get("sec-fetch-site") in ("cross-site", "same-site")
                and not (request.method == "GET" and request.headers.get("sec-fetch-mode") == "navigate"))):
        return PlainTextResponse("forbidden", status_code=403)
    return await call_next(request)


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp

rt = Runtime(STATS)
_cache = {}


def layer_runs():
    """layer -> path of its sweep run: main_rel preferred, then the largest (calibration runs excluded)."""
    best = {}
    for meta_path in glob.glob(os.path.join(RUNS, "*.meta.json")):
        meta = json.load(open(meta_path))
        if meta.get("design_name", "main") == "calib":
            continue
        jsonl = meta_path.replace(".meta.json", ".jsonl")
        if not os.path.exists(jsonl):
            continue
        key = (meta.get("design_name") == "main_rel", os.path.getsize(jsonl))
        if key > best.get(meta["layer"], ("", (False, 0)))[1]:
            best[meta["layer"]] = (jsonl, key)
    return {layer: path for layer, (path, _) in best.items()}


_load_lock = threading.Lock()


def load_layer(layer):
    path = layer_runs().get(layer)
    if path is None:
        return None
    units_path = path.replace(".jsonl", ".units.jsonl")
    key = (path, os.path.getmtime(path), os.path.getmtime(units_path) if os.path.exists(units_path) else 0)
    if _cache.get(layer, {}).get("key") == key:
        return _cache[layer]
    with _load_lock:
        if _cache.get(layer, {}).get("key") == key:
            return _cache[layer]
        if not (os.path.exists(units_path) and os.path.getmtime(units_path) >= os.path.getmtime(path)):
            full = [json.loads(l) for l in open(path)]  # ranking needs token ids; done once, then dropped
            with open(units_path + ".tmp", "w") as f:
                f.writelines(json.dumps(u) + "\n" for u in compute_units(full))
            del full
            os.replace(units_path + ".tmp", units_path)
            key = (path, os.path.getmtime(path), os.path.getmtime(units_path))
        units = [json.loads(l) for l in open(units_path)]
        # keep only what the explorer shows; token ids, logprobs and alternatives stay on disk
        slim = ("kind", "neuron", "prompt", "dose", "rel_push", "text", "finish", "n_gen")
        rows = []
        for l in open(path):
            r = json.loads(l)
            rows.append({k: r[k] for k in slim})
        by_unit = {}
        for i, r in enumerate(rows):
            by_unit.setdefault((r["kind"], r["neuron"]), []).append(i)
        _cache[layer] = {"key": key, "path": path, "rows": rows, "units": units, "by_unit": by_unit,
                         "meta": json.load(open(path.replace(".jsonl", ".meta.json")))}
        return _cache[layer]


_ranked = {}


def ranked_words(layer):
    """{neuron: top words} for the neurons of `layer` with sweep rollouts, read from the run's ranking file only."""
    path = layer_runs().get(layer)
    units_path = path.replace(".jsonl", ".units.jsonl") if path else None
    if not units_path or not os.path.exists(units_path):
        return {}
    key = (units_path, os.path.getmtime(units_path))
    if _ranked.get(layer, (None,))[0] != key:
        words = {}
        for line in open(units_path):
            u = json.loads(line)
            if u["kind"] == "neuron":
                words[u["neuron"]] = [w for w, _, _ in u["words"][:4]]
        _ranked[layer] = (key, words)
    return _ranked[layer][1]


def swept(layer):
    return ranked_words(layer).keys()


def names():
    return json.load(open(NAMES)) if os.path.exists(NAMES) else {}


@app.get("/api/index")
def index():
    runs = layer_runs()
    return {"model": "Qwen/Qwen3.5-4B", "arch": ARCH,
            "layers": {str(l): os.path.basename(p) for l, p in sorted(runs.items())}, "status": rt.status}


@app.get("/api/units")
def units(layer: int):
    d = load_layer(layer)
    if d is None:
        return []
    nm = names()
    return [{"kind": u["kind"], "n": u["neuron"], "z": u.get("mass", u["z"]), "null_pct": u["null_pct"],
             "words": [w for w, _, _ in u["words"][:5]], "name": nm.get(f"L{layer}/N{u['neuron']}", "")}
            for u in d["units"]]


@app.get("/api/unit")
def unit(layer: int, n: int, kind: str = "neuron"):
    d = load_layer(layer)
    out = {"layer": layer, "n": n, "kind": kind, "name": names().get(f"L{layer}/N{n}", "")}
    if d is None:
        return out
    rec = next((u for u in d["units"] if u["neuron"] == n and u["kind"] == kind), None)
    rows = [d["rows"][i] for i in d["by_unit"].get((kind, n), [])]
    base = [r for r in d["rows"] if r["kind"] == "baseline"]
    keep = ("prompt", "dose", "rel_push", "text", "finish", "n_gen")
    neurons = [u for u in d["units"] if u["kind"] == "neuron"]
    out.update({
        "run": os.path.basename(d["path"]), "prompts": d["meta"]["prompts"],
        "rollouts": [{k: r[k] for k in keep} for r in rows],
        "baselines": [{k: r[k] for k in keep} for r in base],
        "stats": rec, "rank": next((i + 1 for i, u in enumerate(neurons) if u["neuron"] == n), None) if rec else None,
        "n_ranked": len(neurons),
    })
    return out


@app.post("/api/name")
async def set_name(req: Request):
    body = await req.json()
    nm = names()
    key = f"L{int(body['layer'])}/N{int(body['n'])}"
    if body.get("name", "").strip():
        nm[key] = body["name"].strip()
    else:
        nm.pop(key, None)
    with open(NAMES, "w") as f:
        json.dump(nm, f, indent=1, sort_keys=True)
    return {"ok": True}


@app.get("/api/embeddings")
def embeddings(layer: int):
    d = load_layer(layer)
    if d is None:
        return {}
    path = d["path"].replace(".jsonl", ".umap.json")
    return FileResponse(path) if os.path.exists(path) else JSONResponse({})


@app.get("/api/rollout")
def rollout(layer: int, i: int):
    r = load_layer(layer)["rows"][i]
    return {k: r[k] for k in ("kind", "neuron", "prompt", "dose", "rel_push", "text")}


@app.get("/api/connections")
async def connections(layer: int, n: int, k: int = 24, lens: int = 5):
    out = await asyncio.to_thread(rt.connections, layer, n, k, lens)
    words = await asyncio.to_thread(lambda: {l: ranked_words(l) for l in {c["layer"] for c in out["down"] + out["up"]}})
    nm = names()
    out = {side: [dict(c) for c in out[side]] for side in ("down", "up")}
    for c in out["down"] + out["up"]:
        c["name"] = nm.get(f"L{c['layer']}/N{c['n']}", "")
        c["words"] = words[c["layer"]].get(c["n"], [])
    return out


@app.get("/api/hover")
async def hover(layer: int, n: int):
    con = await asyncio.to_thread(rt.connections, layer, n, 16, 0)
    lens = await asyncio.to_thread(rt.lens, layer, n, 8)
    return {"lens": lens["up"], "lens_neg": lens["down"], "steer": lens.get("steer_up"), "steer_neg": lens.get("steer_down"),
            "con": con}


@app.get("/api/lens")
async def lens(layer: int, n: int):
    return await asyncio.to_thread(rt.lens, layer, n, 12)


@app.post("/api/reads")
async def reads(req: Request):
    b = await req.json()
    return await asyncio.to_thread(rt.read, int(b["layer"]), int(b["n"]), b["text"], b.get("chat", True))


@app.post("/api/steer")
async def steer(req: Request):
    b = await req.json()
    loop, q, cancel = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

    def work():
        try:
            # each row replays its own earlier replies; the user turns are shared
            rows = []
            for r in range(3):
                msgs = []
                for turn in b.get("history", []):
                    msgs += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["replies"][r]}]
                rows.append(msgs + [{"role": "user", "content": b["prompt"]}])
            for row, delta in rt.steer_stream(int(b["layer"]), int(b["n"]), rows, float(b["dose"]),
                                              b.get("mode", "add"), int(b.get("max_new", 300)), cancel):
                loop.call_soon_threadsafe(q.put_nowait, {"row": row, "t": delta})
        except Exception as e:  # surfaced in the UI next to the chat
            loop.call_soon_threadsafe(q.put_nowait, {"row": None, "error": repr(e)})

    threading.Thread(target=work, daemon=True).start()

    async def events():
        try:
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
                if ev["row"] is None:
                    break
                if await req.is_disconnected():
                    cancel.set()
                    break
        finally:
            cancel.set()

    return StreamingResponse(events(), media_type="text/event-stream")


LENS_INDEX = os.path.join(ROOT, "explorer", "cache", "lens_index.npz")
_lens = {}


def lens_index():
    if "up" not in _lens and os.path.exists(LENS_INDEX):
        z = np.load(LENS_INDEX)
        _lens["up"], _lens["down"] = z["up"], z["down"]
    return _lens if "up" in _lens else None


@app.get("/api/search")
async def search(q: str, limit: int = 300):
    """Neurons whose logit lens (top 12 up or down) contains a token matching q (case-insensitive substring)."""
    idx = lens_index()
    q = q.strip().lower()
    if idx is None or len(q) < 2:
        return {"ready": idx is not None, "results": []}
    vocab = await asyncio.to_thread(rt.vocab_strings)
    match = np.array([i for i, t in enumerate(vocab) if q in t.lower()], dtype=np.int32)
    if not len(match):
        return {"ready": True, "results": [], "tokens": 0}
    out = []
    st = rt.steer_index()
    sources = [("+", idx["up"], "lens"), ("-", idx["down"], "lens")]
    if st is not None:
        sources += [("+", st["up"], "steer"), ("-", st["down"], "steer")]
    for sign, arr, src in sources:
        hit = np.isin(arr, match)  # [L, N, k]
        count = hit.sum(-1)
        best = np.where(hit, np.arange(arr.shape[-1]), 99).min(-1)
        for l, n in zip(*np.nonzero(count)):
            out.append((int(best[l, n]), -int(count[l, n]), int(l), int(n), sign, src))
    # one row per neuron and sign, keeping its best match across the two lenses
    seen, uniq = set(), []
    for row in sorted(out):
        if (row[2], row[3], row[4]) not in seen:
            seen.add((row[2], row[3], row[4])); uniq.append(row)
    out = uniq
    nm, res = names(), []
    sw = {l: swept(l) for l in {row[2] for row in out[:limit]}}
    for rank, _, l, n, sign, src in out[:limit]:
        arr = (idx if src == "lens" else st)["up" if sign == "+" else "down"]
        res.append({"layer": l, "n": n, "sign": sign, "rank": rank, "src": src, "name": nm.get(f"L{l}/N{n}", ""),
                    "swept": n in sw[l], "lens": [vocab[t] for t in arr[l, n, :6]]})
    return {"ready": True, "results": res, "tokens": int(len(match)), "total": len(out)}


ACT_SCALE = os.path.join(ROOT, "explorer", "cache", "act_scale.npz")
MAX_INFER_TOKENS = 1024  # [32, T, 9216] float32 activations plus scores: about 2.4 GB at this length


@app.get("/api/fires")
async def fires(text: str, limit: int = 300, chat: bool = False):
    """Neurons that fire hardest on `text` relative to their typical range over web text (activation / typical max,
    and activation / typical min for the negative side)."""
    if not os.path.exists(ACT_SCALE):
        return {"ready": False, "results": []}
    sc = np.load(ACT_SCALE)
    ids = await asyncio.to_thread(rt.chat_ids if chat else (lambda t: rt.tok(t).input_ids), text)
    acts = await asyncio.to_thread(rt.all_activations, ids)
    toks = [rt.tok.decode([t]) for t in ids]
    cand = []
    for l, (mx, amx, mn, amn) in acts.items():
        p = mx.numpy() / np.maximum(sc["pos"][l], 1e-3)
        q = mn.numpy() / np.minimum(sc["neg"][l], -1e-3)
        for arr, idx, sign, raw in ((p, amx.numpy(), "+", mx.numpy()), (q, amn.numpy(), "-", mn.numpy())):
            top = np.argpartition(-arr, limit)[:limit]
            cand += [(float(arr[n]), l, int(n), sign, int(idx[n]), float(raw[n])) for n in top]
    cand.sort(reverse=True)
    idx_lens, nm = lens_index(), names()
    vocab = await asyncio.to_thread(rt.vocab_strings)
    res = []
    for score, l, n, sign, t, raw in cand[:limit]:
        lens = [vocab[k] for k in (idx_lens["up"] if sign == "+" else idx_lens["down"])[l, n, :6]] if idx_lens else []
        res.append({"layer": l, "n": n, "sign": sign, "score": round(score, 2), "act": round(raw, 2), "token": toks[t],
                    "pos": t, "name": nm.get(f"L{l}/N{n}", ""), "lens": lens})
    return {"ready": True, "results": res, "tokens": toks}


_infer = {}


async def score_text(text, chat=False, heads=False):
    """Every neuron's activation at every token of `text` ([L, T, N]) and its signed score: the positive side as a
    multiple of the neuron's typical max, the negative side of its typical min (act_scale.npz). With heads, the
    attention heads' patterns and write sizes are kept too (rt.heads)."""
    ids = rt.chat_ids(text) if chat else rt.tok(text).input_ids
    ids = ids[:MAX_INFER_TOKENS]
    acts = await asyncio.to_thread(rt.full_activations, ids, heads)  # [L, T, N]
    sc = np.load(ACT_SCALE)
    # a side the neuron barely reaches in calibration is scaled by at least a tenth of its other side
    pos = np.maximum(sc["pos"], np.maximum(0.1 * -sc["neg"], 1e-3))[:, None, :]
    neg = np.minimum(sc["neg"], np.minimum(-0.1 * sc["pos"], -1e-3))[:, None, :]
    score = np.where(acts >= 0, acts / pos, -(acts / neg)).astype(np.float32)
    if score.shape[1] > 1:
        score[:, 0, :] = 0  # position 0 is the attention-sink token, outside the range the scale was measured on
    return ids, acts, score


@app.post("/api/infer")
async def infer(req: Request):
    """Runs text through the model and keeps every neuron's activation at every token for /api/infer/view."""
    b = await req.json()
    if not os.path.exists(ACT_SCALE) or rt.tok is None:
        return {"ready": False}
    ids, acts, score = await score_text(b["text"], b.get("chat", False), heads=True)
    units = rt.token_units(ids)
    unit_of = {i: k for k, (_, idx) in enumerate(units) for i in idx}
    _infer.clear()
    _infer.update(ids=ids, toks=[units[unit_of[i]][0] for i in range(len(ids))], acts=acts, score=score, units=units)
    # one chip per unit; clicking it selects the unit's last token (the one that completes the character)
    return {"ready": True, "tokens": [u[0] for u in units], "pos": [u[1][-1] for u in units]}


@app.get("/api/infer/view")
async def infer_view(pos: int = -1, sign: str = "both", lo: int = 0, hi: int = 99, min: float = 1.0, q: str = "",
                     limit: int = 300):
    """Neurons passing the filters for the last inferred text. pos = token index, or -1 for the strongest token.
    Returns the list plus every neuron's signed score at that position (base64 float16, layer-major) for the map."""
    if "score" not in _infer:
        return {"results": []}
    import base64
    S, A, toks = _infer["score"], _infer["acts"], _infer["toks"]  # [L, T, N]
    if pos >= S.shape[1]:
        return {"results": [], "total": 0}
    if pos >= 0:
        best, at = S[:, pos, :], np.full(S[:, 0, :].shape, pos)
    else:
        at = S.argmax(1) if sign == "+" else S.argmin(1) if sign == "-" else np.abs(S).argmax(1)  # [L, N]
        best = np.take_along_axis(S, at[:, None, :], 1)[:, 0, :]
    view = best.copy()
    keep = np.abs(best) >= min
    if sign == "+": keep &= best > 0
    if sign == "-": keep &= best < 0
    lay = np.arange(best.shape[0])[:, None]
    keep &= (lay >= lo) & (lay <= hi)
    idx = lens_index()
    vocab = await asyncio.to_thread(rt.vocab_strings)
    if q.strip() and idx is not None:
        ql = q.strip().lower()
        match = np.array([i for i, t in enumerate(vocab) if ql in t.lower()], dtype=np.int32)
        lens_hit = np.where(best[..., None] >= 0, np.isin(idx["up"][:, :, :6], match), np.isin(idx["down"][:, :, :6], match)).any(-1)
        keep &= lens_hit
    view[~keep] = 0
    ls, ns = np.nonzero(keep)
    order = np.argsort(-np.abs(best[ls, ns]))[:limit]
    nm, res = names(), []
    for o in order:
        l, n = int(ls[o]), int(ns[o]); sg = "+" if best[l, n] > 0 else "-"
        lens = [vocab[k] for k in (idx["up"] if sg == "+" else idx["down"])[l, n, :6]] if idx is not None else []
        t = int(at[l, n])
        res.append({"layer": l, "n": n, "sign": sg, "score": round(float(abs(best[l, n])), 2),
                    "act": round(float(A[l, t, n]), 2), "token": toks[t], "pos": t,
                    "name": nm.get(f"L{l}/N{n}", ""), "lens": lens})
    blob = base64.b64encode(view.astype(np.float16).tobytes()).decode()
    return {"results": res, "total": int(keep.sum()), "map": blob, "shape": list(view.shape)}


def _heads_for_infer():
    """The head captures of the last inferred text (re-captured if another text ran since)."""
    ids = _infer.get("ids")
    if ids is None:
        return None
    if not rt.heads or rt.heads.get("ids") != list(ids):
        rt.full_activations(ids, True)
    return rt.heads


@app.get("/api/infer/heads")
def infer_heads(pos: int = -1):
    """Every attention head's write size at token `pos` of the last inferred text, per layer, as a share of the
    largest write any head of that layer makes at any token past the first (position 0 is the attention sink)."""
    h = _heads_for_infer() if pos >= 0 else None
    if not h or pos >= h["T"]:
        return {"layers": None}
    out = []
    for l in sorted(h["writes"]):
        W = h["writes"][l]  # [T, H]
        top = float(W[1:].max()) if W.shape[0] > 1 else float(W.max())
        out.append([round(float(x) / max(top, 1e-8), 3) for x in W[pos]])
    return {"layers": out}


@app.get("/api/head/row")
def head_row(layer: int, head: int, pos: int):
    """One head's attention row at token `pos` of the last inferred text, summed per display unit (the chips in the
    run tab); units after pos are null. Full attention: softmax weights. Linear attention: signed effective weights,
    absolute values summing to 1."""
    h = _heads_for_infer() if pos >= 0 else None
    if not h or pos >= h["T"] or layer not in h["q"]:
        return {"units": None}
    row = rt.head_row(layer, head, pos)
    return {"units": [round(sum(row[i] for i in idx), 4) if idx[0] <= pos else None for _, idx in _infer["units"]]}


def _lens_base():
    """The lens used for pooled token summaries (steer when built, else logit), the vocab strings, and each token's
    base rate per layer: its share among all neurons' top-8 lens tokens of that layer."""
    st = rt.steer_index()
    lens = st if st is not None else rt.lens_index()
    v = rt.vocab_strings()
    key_src = "steer" if st is not None else "logit"
    if _idf.get("src") != key_src:
        L, N = lens["up"].shape[:2]
        df = np.stack([np.bincount(np.concatenate([lens["up"][l, :, :8].ravel(), lens["down"][l, :, :8].ravel()]),
                                   minlength=len(v)) for l in range(L)])
        _idf.update(src=key_src, base=((df + 1) / (2 * N)).astype(np.float32))  # [layer, token]
    return lens, v, _idf["base"], key_src


@app.get("/api/neuron/heads")
async def neuron_heads(layer: int, n: int):
    """The attention heads linked to a neuron by weights: z per head, per layer (runtime.neuron_heads), and the top
    heads per role, q / k / v / o (runtime.neuron_head_roles)."""
    return {"layers": await asyncio.to_thread(rt.neuron_heads, layer, n),
            "roles": await asyncio.to_thread(rt.neuron_head_roles, layer, n)}


HEAD_SAMPLE = "Alice called Bob because she wanted to borrow his car. He said yes, and she drove it to the lake."


@app.post("/api/head/pattern")
async def head_pattern(req: Request):
    """One head's attention pattern on {text} (the run text, else a sample sentence), per display unit: rows are
    queries (a unit's last token), columns sum the weight over each earlier unit's tokens."""
    b = await req.json()
    layer, head = int(b["layer"]), int(b["head"])
    text = b.get("text") or HEAD_SAMPLE
    ids = (rt.chat_ids(text) if b.get("chat") else rt.tok(text).input_ids)[:MAX_INFER_TOKENS] if rt.tok else None
    if ids is None:
        return {"error": "model loading"}
    if not rt.heads or rt.heads.get("ids") != ids:
        await asyncio.to_thread(rt.full_activations, ids, True)
    A = await asyncio.to_thread(rt.head_pattern, layer, head)
    units = rt.token_units(ids)
    cols = np.stack([A[:, idx].sum(1) for _, idx in units], 1)  # [T, U]
    rows = [[round(float(x), 4) for x in cols[idx[-1], :u + 1]] for u, (_, idx) in enumerate(units)]
    return {"text": text, "tokens": [t for t, _ in units], "rows": rows,
            "full": ARCH["layer_types"][layer] == "full_attention"}


HEAD_TEXTS = [
    "Alice called Bob because she wanted to borrow his car. He said yes, and she drove it to the lake.",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n    return out + a + b",
    "The committee, however, rejected the proposal; members said it was expensive, rushed, and unclear about who would pay.",
    "Ingredients: 2 cups flour, 1 tsp salt, 3 eggs, and a pinch of sugar. Mix the flour and salt, then add the eggs.",
    "我昨天去了北京，但是天气很冷，所以我们在酒店里待了一整天。",
    "\"Where are you going?\" asked the old man. \"To the market,\" said Mara, \"to sell the goat my brother raised.\"",
    "- Monday: 9:00 standup, 14:30 review with Sam\n- Tuesday: dentist at 11\n- Friday: release 2.4.1, then retro",
    "The mitochondria produce ATP through oxidative phosphorylation, which requires oxygen; without it, cells fall back on glycolysis.",
]
_hprof = {}


def _head_profiles():
    """Where every head looks over HEAD_TEXTS, per display unit, for queries from the 4th unit on:
    dist: mean share of |weight| on self / the previous unit / units 2-5 back / further back / the first unit;
    neg: mean share of negative weight; src: per unit string, the |weight| it receives against a uniform pattern's;
    pairs: the strongest single reads more than one unit back (past the first unit)."""
    path = os.path.join(CACHE, "head_profiles.json")
    key = json.dumps(HEAD_TEXTS)
    if _hprof.get("key") == key:
        return _hprof["data"]
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d.get("key") == key:
            _hprof.update(key=key, data=d)
            return d
    acc = {}
    for t, text in enumerate(HEAD_TEXTS):
        ids = rt.tok(text).input_ids
        rt.full_activations(ids, True)
        caps = rt.heads
        units = rt.token_units(ids)
        toks = [s for s, _ in units]
        U = len(units)
        last = [idx[-1] for _, idx in units]
        for layer in sorted(caps["q"]):
            A = rt.layer_patterns(layer, caps)  # [H, T, T]
            A = np.stack([A[:, :, idx].sum(-1) for _, idx in units], -1)[:, last]  # [H, U, U]
            H = A.shape[0]
            tot = np.abs(A).sum(-1, keepdims=True).clip(1e-12)
            R = (A / tot)[:, 3:]  # rows 3.. [H, U-3, U]
            Ra = np.abs(R)
            qi = np.arange(3, U)
            j = np.arange(U)
            back = qi[:, None] - j[None]  # distance per (query, key)
            dist = np.stack([(Ra * (back == 0)).sum(-1), (Ra * (back == 1)).sum(-1),
                             (Ra * ((back >= 2) & (back <= 5) & (j[None] > 0))).sum(-1),
                             (Ra * ((back > 5) & (j[None] > 0))).sum(-1), Ra[..., 0]], -1).sum(1)  # [H, 5]
            neg = np.clip(-R, 0, None).sum(-1).sum(1)
            mask = (back >= 1) & (j[None] > 0)  # earlier units past the first
            recv = (Ra * mask).sum(1)  # [H, U]
            expv = (mask / (qi[:, None] + 1)).sum(0)  # [U]
            for h in range(H):
                a = acc.setdefault(f"{layer}/{h}", {"dist": np.zeros(5), "neg": 0.0, "n": 0, "src": {}, "pairs": []})
                a["dist"] += dist[h]; a["neg"] += float(neg[h]); a["n"] += len(qi)
                for u in range(1, U - 1):
                    s = a["src"].setdefault(toks[u].strip() or toks[u], [0.0, 0.0, 0])
                    s[0] += float(recv[h, u]); s[1] += float(expv[u]); s[2] += 1
                far = np.where((back > 1) & (j[None] > 0), R[h], 0)
                for r, c in zip(*np.unravel_index(np.argsort(-np.abs(far), None)[:6], far.shape)):
                    a["pairs"].append([round(float(far[r, c]), 3), t, int(qi[r]), int(c)])
    out = {}
    for k, a in acc.items():
        src = sorted(((s, r / e, r - e, n) for s, (r, e, n) in a["src"].items() if n >= 2 and r > e),
                     key=lambda x: -x[2])[:10]
        seen, pairs = set(), []
        for w, t, qu, ku in sorted(a["pairs"], key=lambda p: -abs(p[0])):
            if (t, qu, ku) in seen:
                continue
            seen.add((t, qu, ku))
            pairs.append([w, t, qu, ku])
            if len(pairs) == 8:
                break
        out[k] = {"dist": [round(float(x) / a["n"], 3) for x in a["dist"]], "neg": round(a["neg"] / a["n"], 3),
                  "src": [[s, round(l, 1)] for s, l, _, _ in src], "pairs": pairs}
    d = {"key": key, "texts": HEAD_TEXTS,
         "tokens": [[s for s, _ in rt.token_units(rt.tok(x).input_ids)] for x in HEAD_TEXTS], "heads": out}
    with open(path, "w") as f:
        json.dump(d, f)
    _hprof.update(key=key, data=d)
    return d


@app.get("/api/head/profile")
async def head_profile(layer: int, head: int):
    """One head's profile over the reference texts (_head_profiles), with each pair's query and key unit text and
    the text it comes from. The first call computes every head's profile."""
    if rt.tok is None:
        return {"error": "model loading"}
    d = await asyncio.to_thread(_head_profiles)
    p = d["heads"].get(f"{layer}/{head}")
    if p is None:
        return {"error": "no such head"}
    pairs = [{"w": w, "text": d["texts"][t], "q": qu, "k": ku, "qt": d["tokens"][t][qu], "kt": d["tokens"][t][ku]}
             for w, t, qu, ku in p["pairs"]]
    return {"layer": layer, "head": head, "dist": p["dist"], "neg": p["neg"], "src": p["src"], "pairs": pairs}


@app.get("/api/head")
def head_info(layer: int, head: int, top: int = 24):
    """What one attention head reads and writes, from weights: its linked neurons (runtime.head_links) and their
    positive-pole lens tokens pooled the way the web summary pools a cone (weighted by z, ranked by lift over each
    layer's base rate)."""
    links = rt.head_links(layer, head)
    lens, v, base, src = _lens_base()

    def pool(nodes):
        acc = {}
        zmax = max([c["z"] for c in nodes] + [1e-8])
        for c in nodes:
            for t in lens["up"][c["layer"], c["n"], :8]:
                s = v[t]
                key = s.strip().lower()
                if not key or "\ufffd" in key:
                    continue
                a = acc.setdefault(key, {"n": 0, "w": 0.0, "base": 0.0, "forms": {}})
                a["n"] += 1
                a["w"] += c["z"] / zmax
                a["base"] += float(base[c["layer"], t])
                a["forms"][s.strip()] = a["forms"].get(s.strip(), 0) + 1
        for a in acc.values():
            lift = (a["n"] / max(1, len(nodes))) / (a["base"] / a["n"])
            a["score"] = a["w"] * np.log(lift) if lift > 1 else 0.0
        out = sorted((a for a in acc.values() if a["score"] > 0), key=lambda a: -a["score"])[:top]
        return [{"tok": max(a["forms"], key=a["forms"].get), "n": a["n"], "w": round(float(a["score"]), 2)} for a in out]

    for side in ("reads", "writes", "q", "k"):
        for c in links[side]:
            c["lens"] = [v[t] for t in lens["up"][c["layer"], c["n"], :5]]
    return {"layer": layer, "head": head, "full": ARCH["layer_types"][layer] == "full_attention", "src": src, "reads": links["reads"], "writes": links["writes"], "q": links["q"], "k": links["k"],
            "read_toks": pool(links["reads"]), "write_toks": pool(links["writes"]),
            "q_toks": pool(links["q"]), "k_toks": pool(links["k"])}


@app.get("/api/infer/paths")
def infer_paths(pos: int, top: int = 300, min: float = 1.0):
    """The circuit active at one token of the last inferred text: among the `top` neurons with the largest |score| there
    (at least `min`), every precomputed direct link whose two ends both fired. Same binary format as /api/web, with z
    signed by the push the writer's actual activation gives the reader (link sign x sign of the writer's score), so
    yellow / magenta read as "this firing pushed that one up / down"; hop is 1."""
    if "score" not in _infer or pos >= _infer["score"].shape[1]:
        return Response(b"", media_type="application/octet-stream", headers={"X-Web-Total": "0"})
    S = _infer["score"][:, pos, :]  # [L, N]
    flat = np.abs(S).ravel()
    cand = np.argpartition(-flat, min_k := builtins.min(top, flat.size - 1))[:min_k]
    cand = cand[flat[cand] >= min]
    N = S.shape[1]
    active = {(int(i // N), int(i % N)) for i in cand}
    ci = rt.con_index()
    rows = []
    for l, n in active:
        for i in range(ci["k"]):
            z = float(ci["down_z"][l, n, i])
            if z < -1e4:
                break
            t = (int(ci["down_layer"][l, n, i]), int(ci["down_n"][l, n, i]))
            if t in active:
                push = int(ci["down_sign"][l, n, i]) * (1 if S[l, n] > 0 else -1)
                rows.append((l, n, *t, round(z * 10) * push, 1))
    out = np.array(rows, np.int16).reshape(-1, 6)
    return Response(out.tobytes(), media_type="application/octet-stream",
                    headers={"X-Web-Total": str(len(out)), "X-Active": str(len(active))})


@app.get("/api/layout")
def layout(kind: str = "writes"):
    """Seriated grid layout: cell index (row * side + col) for every neuron, base64 int16, layer-major."""
    import base64
    path = os.path.join(ROOT, "explorer", "cache", f"layout_{kind}.npz")
    if not os.path.exists(path):
        return {"available": False}
    cell = np.load(path)["cell"].astype(np.int16)
    return {"available": True, "shape": list(cell.shape), "cell": base64.b64encode(cell.tobytes()).decode()}


def cone_links(layer, n, k=2, zmin=15, cap=20000, first=None, hops=None):
    """Connection cone of one neuron: everything it writes into, then what those write into, out to the last layer;
    and everything that writes into it, back to layer 0. Each side only moves one way. The first hop takes all of the
    neuron's top `first` links (all when None), later hops each node's top k with z >= zmin, up to `hops` hops; each side keeps at most cap / 2 links.
    Rows: src layer, src neuron, dst layer, dst neuron, z * sign, hop (positive downstream, negative upstream)."""
    ci = rt.con_index()
    out = []
    for side, step in (("down", 1), ("up", -1)):
        zs, ls, ns, sg = ci[f"{side}_z"], ci[f"{side}_layer"], ci[f"{side}_n"], ci[f"{side}_sign"]
        front, seen, hop, start = [(layer, n)], {(layer, n)}, 1, len(out)
        while front and len(out) - start < cap // 2 and (not hops or hop <= hops):
            nxt = []
            for a, b in front:
                for i in range((first or ci["k"]) if hop == 1 else k):
                    z = float(zs[a, b, i])
                    if z < -1e4 or (hop > 1 and z < zmin):
                        break
                    t = (int(ls[a, b, i]), int(ns[a, b, i]))
                    src, dst = ((a, b), t) if side == "down" else (t, (a, b))
                    out.append((*src, *dst, z * int(sg[a, b, i]), hop * step))
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
            front, hop = nxt, hop + 1
        del out[start + cap // 2:]
    return out


@app.get("/api/web")
def web(layer: int, n: int, first: int = 5, k: int = 2, zmin: float = 15, cap: int = 20000, hops: int = 0):
    """The cone for the map. Binary int16, 6 per link: src layer, src neuron, dst layer, dst neuron,
    round(10 * z) * sign, hop."""
    rows = np.array([(*r[:4], round(r[4] * 10), r[5]) for r in cone_links(layer, n, k, zmin, cap, first, hops or None)], np.int16).reshape(-1, 6)
    return Response(rows.tobytes(), media_type="application/octet-stream", headers={"X-Web-Total": str(len(rows))})


_idf = {}


@app.get("/api/web/summary")
def web_summary(layer: int, n: int, first: int = 5, k: int = 2, top: int = 30, src: str = "steer", hops: int = 0):
    """What the neuron's web says, in tokens. Each neuron in the cone is read on the pole the chain drives (signs
    multiplied along the path from the open neuron), and its steer (or logit) lens tokens are pooled, merged across
    case and leading space, and weighted by link z / max z times 0.8 per hop past the first, times log lift: the share
    of web neurons carrying the token over its share among all neurons of the same layers, so tokens a layer's neurons
    commonly carry (`(`, ` a` in early layers) sink and tokens the web carries no more often than chance drop out.
    Downstream gives
    `toward` (the driven pole) and `away` (the opposite pole); upstream gives `fed_by` (the pole that excites it) and
    `held_by` (the pole that inhibits it)."""
    if src == "steer":
        lens, v, base, key_src = _lens_base()
    else:
        lens, v = rt.lens_index(), rt.vocab_strings()
        key_src = "logit"
        if _idf.get("src") != key_src:
            L, N = lens["up"].shape[:2]
            df = np.stack([np.bincount(np.concatenate([lens["up"][l, :, :8].ravel(), lens["down"][l, :, :8].ravel()]),
                                       minlength=len(v)) for l in range(L)])
            _idf.update(src=key_src, base=((df + 1) / (2 * N)).astype(np.float32))
        base = _idf["base"]
    sign, zmax = {(layer, n): 1}, 1.0
    links = cone_links(layer, n, k, 15, 20000, first, hops or None)
    for sl, sn, dl, dn, z, hop in links:
        zmax = max(zmax, abs(z))
    reach = {}
    for sl, sn, dl, dn, z, hop in links:
        near, far = ((sl, sn), (dl, dn)) if hop > 0 else ((dl, dn), (sl, sn))
        if far in sign or near not in sign:
            continue
        sign[far] = sign[near] * (1 if z > 0 else -1)
        reach[far] = (hop, abs(z))

    def pool(nodes, flip):
        acc = {}
        for (l, m), (hop, z) in nodes:
            w = z / zmax * 0.8 ** (abs(hop) - 1)
            pole = "up" if sign[(l, m)] * flip > 0 else "down"
            for t in lens[pole][l, m, :8]:
                s = v[t]
                key = s.strip().lower()
                if not key or "\ufffd" in key:
                    continue
                a = acc.setdefault(key, {"tok": s.strip(), "n": 0, "w": 0.0, "forms": {}})
                a["n"] += 1
                a["w"] += w
                a["base"] = a.get("base", 0.0) + float(base[l, t])
                a["forms"][s.strip()] = a["forms"].get(s.strip(), 0) + 1
        for a in acc.values():
            lift = (a["n"] / max(1, len(nodes))) / (a["base"] / a["n"])
            a["score"] = a["w"] * np.log(lift) if lift > 1 else 0.0
        out = sorted((a for a in acc.values() if a["score"] > 0), key=lambda a: -a["score"])[:top]
        return [{"tok": max(a["forms"], key=a["forms"].get), "n": a["n"], "w": round(float(a["score"]), 2)} for a in out]

    down = [(key, val) for key, val in reach.items() if val[0] > 0]
    up = [(key, val) for key, val in reach.items() if val[0] < 0]
    return {"src": key_src, "links": len(links),
            "down": {"neurons": len(down), "toward": pool(down, 1), "away": pool(down, -1)},
            "up": {"neurons": len(up), "fed_by": pool(up, 1), "held_by": pool(up, -1)}}


# ---------- Claude: autointerp and chat ----------
# Each chat is a `claude -p` session resumed turn by turn. Its tools come from agent_mcp.py, which reads through the
# endpoints below and sends map actions back through /api/agent/ui into the chat's event stream.
AGENT_PROMPT = os.path.join(ROOT, "explorer", "agent_prompt.md")
AGENT_DIR = os.path.join(ROOT, "explorer", "cache", "agent")  # claude's cwd, so no project settings or CLAUDE.md apply
_chats = {}
_act = {}


def act_range(layer, n):
    if "pos" not in _act and os.path.exists(ACT_SCALE):
        z = np.load(ACT_SCALE)
        _act.update(pos=z["pos"], neg=z["neg"], tokens=int(z["tokens"]))
    if "pos" not in _act:
        return None
    return {"act_max": round(float(_act["pos"][layer, n]), 3), "act_min": round(float(_act["neg"][layer, n]), 3),
            "over_tokens": _act["tokens"], "source": "FineWeb-Edu"}


@app.get("/api/agent/overview")
def agent_overview():
    runs = layer_runs()
    return {"model": "Qwen/Qwen3.5-4B", "arch": ARCH, "layers_with_sweeps": sorted(runs),
            "named_neurons": len(names()), "live_model": not NO_MODEL}


@app.get("/api/agent/names")
def agent_names():
    return names()


@app.get("/api/agent/neuron")
async def agent_neuron(layer: int, n: int):
    lens = await asyncio.to_thread(rt.lens, layer, n, 12)
    con = await connections(layer, n, 24, 5)
    u = await asyncio.to_thread(unit, layer, n)
    out = {"neuron": f"L{layer} N{n}", "name": u.get("name", ""),
           "lens_up": lens["up"], "lens_down": lens["down"],
           "steer_up": lens.get("steer_up"), "steer_down": lens.get("steer_down"),
           "kl_up": lens.get("kl_up"), "kl_down": lens.get("kl_down"),
           "activation": act_range(layer, n),
           "sweep": {"rank": u.get("rank"), "n_ranked": u.get("n_ranked"), "stats": u.get("stats"),
                     "n_rollouts": len(u.get("rollouts", []))} if "run" in u else None}
    slim = lambda c: {"neuron": f"L{c['layer']} N{c['n']}", "z": c["z"], "sign": c["sign"], "steer_up": c.get("lens"),
                      "steer_down": c.get("lens_neg"), "words": c.get("words") or None, "name": c.get("name") or None}
    out["writes_into"] = [slim(c) for c in con["down"]]
    out["reads_from"] = [slim(c) for c in con["up"]]
    return out


@app.get("/api/agent/cone")
def agent_cone(layer: int, n: int, k: int = 2, limit: int = 120):
    links = cone_links(layer, n, k)
    best = {}
    for sl, sn, dl, dn, z, hop in links:
        key = (dl, dn) if hop > 0 else (sl, sn)
        if key not in best or abs(z) > abs(best[key][0]):
            best[key] = (z, hop if key not in best else best[key][1])
    v = rt.vocab_strings()
    idx = rt.steer_index() or rt.lens_index()
    nm = names()
    nodes = sorted(best.items(), key=lambda kv: (abs(kv[1][1]), -abs(kv[1][0])))[:limit]
    per_hop = {}
    for _, (_, hop) in best.items():
        per_hop[hop] = per_hop.get(hop, 0) + 1
    return {"links": len(links), "neurons": len(best), "per_hop": dict(sorted(per_hop.items())),
            "note": "weight-only direct-path links chained; paths past hop 1 are leads, not circuits",
            "nodes": [{"neuron": f"L{l} N{m}", "hop": hop, "z": round(abs(z), 1), "sign": 1 if z > 0 else -1,
                       "steer_up": [v[t] for t in idx["up"][l, m, :5]], "name": nm.get(f"L{l}/N{m}") or None}
                      for (l, m), (z, hop) in nodes]}


def _live_off():
    if NO_MODEL:
        return {"error": "the live model is off (server started with EXPLORER_NO_MODEL=1); suggest the test instead"}
    return None


@app.post("/api/agent/steer")
async def agent_steer(req: Request):
    """Steered rollouts for Claude: one prompt, pushed up / unsteered / pushed down; whole texts at once. Greedy unless
    `temperature` is set; `repetition_penalty`, `seed` and `min_p` as in runtime.steer_multi."""
    b = await req.json()
    if off := _live_off():
        return off
    texts = ["", "", ""]
    # one neuron (layer, n) or several (neurons: [{layer, n, dose?, mode?}]), each defaulting to the call's dose/mode
    dose, mode = float(b.get("dose", 0.8)), b.get("mode", "add")
    items = b.get("neurons") or [{"layer": b["layer"], "n": b["n"]}]
    pushes = [(int(x["layer"]), int(x["n"]), float(x.get("dose", dose)), x.get("mode", mode)) for x in items]

    samp = {"temperature": float(b.get("temperature") or 0), "repetition_penalty": float(b.get("repetition_penalty") or 1.0),
            "seed": int(b.get("seed") or 0), "min_p": float(b.get("min_p") or 0)}

    def work():
        for row, delta in rt.steer_multi(pushes, [[{"role": "user", "content": b["prompt"]}]] * 3, int(b.get("max_new", 150)), **samp):
            if row is not None:
                texts[row] += delta
    await asyncio.to_thread(work)
    return {"prompt": b["prompt"], "steered": [{"neuron": f"L{l} N{n}", "dose": d, "mode": m} for l, n, d, m in pushes],
            "sampling": samp if samp["temperature"] or samp["repetition_penalty"] != 1.0 else "greedy",
            "up": texts[0], "baseline": texts[1], "down": texts[2]}


@app.post("/api/agent/activations")
async def agent_activations(req: Request):
    """The neuron's activation on each token of a text, for Claude; compact [token, activation] pairs. With `neurons`
    ([{layer, n, dose?, mode?}]) those are pushed during the same pass (steering units)."""
    b = await req.json()
    if off := _live_off():
        return off
    # optional: other neurons pushed during the same pass, to test whether they drive this one
    # neurons: {layer, n, dose?, mode?}, or {layer, all: true, dose, mode: "mul"} for the whole layer's MLP;
    # heads: {layer, head (int or "all"), dose (a multiplier; 0 = ablate)}. Items with "head" in `neurons` count as heads.
    items, heads = b.get("neurons") or [], list(b.get("heads") or [])
    heads += [x for x in items if "head" in x]
    pushes = [(int(p["layer"]), None if p.get("all") else int(p["n"]), float(p.get("dose", 0 if p.get("all") else 0.8)),
               "mul" if p.get("all") else p.get("mode", "add")) for p in items if "head" not in p]
    hs = [(int(h["layer"]), None if h["head"] in ("all", None) else int(h["head"]), float(h.get("dose", 0))) for h in heads]
    for l, hd, _ in hs:
        if hd is not None and not 0 <= hd < rt.n_heads(l):
            return {"error": f"layer {l} has {rt.n_heads(l)} heads ({ARCH['layer_types'][l]})"}
    r = await asyncio.to_thread(rt.read, int(b["layer"]), int(b["n"]), b["text"], b.get("chat", False), pushes, hs)
    units = [[u["text"], round(u["act"], 3)] for u in r["units"]]
    top = sorted(units, key=lambda u: -abs(u[1]))[:10]
    return {"tokens": units, "strongest": top, "act_range": act_range(int(b["layer"]), int(b["n"])),
            "pushed": [{"neuron": f"L{l} " + ("MLP (all)" if n is None else f"N{n}"), "dose": d, "mode": m} for l, n, d, m in pushes],
            "heads": [{"layer": l, "head": "all" if hd is None else hd, "scale": sc} for l, hd, sc in hs]}


def _interventions(b):
    """pushes and heads from a request body, as activations takes them; (pushes, heads, error)."""
    items, heads = b.get("neurons") or [], list(b.get("heads") or [])
    heads += [x for x in items if "head" in x]
    pushes = [(int(p["layer"]), None if p.get("all") else int(p["n"]), float(p.get("dose", 0 if p.get("all") else 0.8)),
               "mul" if p.get("all") else p.get("mode", "add")) for p in items if "head" not in p]
    hs = [(int(h["layer"]), None if h["head"] in ("all", None) else int(h["head"]), float(h.get("dose", 0))) for h in heads]
    for l, hd, _ in hs:
        if hd is not None and not 0 <= hd < rt.n_heads(l):
            return pushes, hs, f"layer {l} has {rt.n_heads(l)} heads ({ARCH['layer_types'][l]})"
    return pushes, hs, None


def _capture_text(text, chat):
    """Runs a text with head capture: (ids, activations [L, T, N], captures, display units)."""
    ids = (rt.chat_ids(text) if chat else rt.tok(text).input_ids)[:MAX_INFER_TOKENS]
    acts = rt.full_activations(ids, True)
    return ids, acts, rt.heads, rt.token_units(ids)


def _find_unit(units, token):
    """The first display unit containing `token` (exact, then case- and space-insensitive); None if absent."""
    if not token:
        return None
    for match in (lambda u: token in u, lambda u: token.strip().lower() in u.lower()):
        for k, (u, _) in enumerate(units):
            if match(u):
                return k
    return None


@app.post("/api/agent/attention")
async def agent_attention(req: Request):
    """For Claude: one head's attention on a text. Body {layer, head, text, token?, chat?}. At the query token (the
    first containing `token`, else the last) the weight on each earlier token (summed per display unit; linear
    attention is signed, absolute values summing to 1), and the strongest single reads anywhere in the text."""
    b = await req.json()
    if off := _live_off():
        return off
    layer, head = int(b["layer"]), int(b["head"])
    if not 0 <= head < rt.n_heads(layer):
        return {"error": f"layer {layer} has {rt.n_heads(layer)} heads ({ARCH['layer_types'][layer]})"}
    ids, _, caps, units = await asyncio.to_thread(_capture_text, b["text"], b.get("chat", False))
    toks = [u for u, _ in units]
    qu = int(b["at"]) % len(units) if b.get("at") is not None else _find_unit(units, b.get("token"))
    qu = len(units) - 1 if qu is None else qu
    A = await asyncio.to_thread(rt.layer_patterns, layer, caps)
    A = A[head]
    U = np.stack([A[:, idx].sum(1) for _, idx in units], 1)[[idx[-1] for _, idx in units]]  # [U, U]
    row = [[toks[j], round(float(U[qu, j]), 3)] for j in range(qu + 1)]
    pairs = [(float(U[i, j]), i, j) for i in range(len(units)) for j in range(1, i)]
    pairs.sort(key=lambda x: -abs(x[0]))
    return {"layer": layer, "head": head, "kind": ARCH["layer_types"][layer], "tokens": toks,
            "query": {"index": qu, "token": toks[qu]}, "row": row,
            "top_sources": sorted(row, key=lambda r: -abs(r[1]))[:8],
            "first_token_weight": round(float(np.abs(U[1:, 0]).mean()), 3) if len(units) > 1 else None,
            "strongest_reads": [{"query": toks[i], "source": toks[j], "weight": round(w, 3), "back": i - j}
                                for w, i, j in pairs[:12]]}


@app.post("/api/agent/attn_drive")
async def agent_attn_drive(req: Request):
    """For Claude: which attention heads drive a neuron at one token of a text, and which earlier tokens they read
    from. Body {layer, n, text, token?, chat?, top?}. The token is the first containing `token`, else where the neuron
    fires hardest. Contributions are the neuron's activation linearised at that token (runtime.attn_drive)."""
    b = await req.json()
    if off := _live_off():
        return off
    layer, n, top = int(b["layer"]), int(b["n"]), int(b.get("top", 8))
    ids, acts, caps, units = await asyncio.to_thread(_capture_text, b["text"], b.get("chat", False))
    toks = [u for u, _ in units]
    unit_of = {i: k for k, (_, idx) in enumerate(units) for i in idx}
    k = int(b["at"]) % len(units) if b.get("at") is not None else _find_unit(units, b.get("token"))
    if k is None:
        pos = int(np.abs(acts[layer, 1:, n]).argmax()) + 1
        k = unit_of[pos]
    pos = units[k][1][-1]
    d = await asyncio.to_thread(rt.attn_drive, layer, n, pos, caps, top)
    attn = sum(c for _, _, c in d["heads"])
    heads = []
    for l, h, c in sorted(d["heads"], key=lambda r: -abs(r[2]))[:max(top, 12)]:
        row = {"head": f"L{l} h{h}", "contribution": round(c, 4), "share": round(c / d["total"], 3) if d["total"] else None}
        if (l, h) in d["src"]:
            src = d["src"][(l, h)]
            per = {}
            for i, v in enumerate(src):
                per[unit_of[i]] = per.get(unit_of[i], 0.0) + v
            row["from"] = [[toks[u], round(v, 4)] for u, v in sorted(per.items(), key=lambda x: -abs(x[1]))[:5]]
        heads.append(row)
    return {"neuron": f"L{layer} N{n}", "token": toks[k], "index": k, "act": round(d["act"], 4),
            "act_range": act_range(layer, n), "linearised_total": round(d["total"], 4),
            "attention_sum": round(attn, 4), "rest": round(d["total"] - attn, 4), "heads": heads}


@app.post("/api/agent/logprobs")
async def agent_logprobs(req: Request):
    """For Claude: the next token's distribution after a text. Body {text, chat?, prefix?, k?, targets?: [str],
    neurons?, heads?} (neurons / heads as in /api/agent/activations). Without an intervention: {top, targets}. With
    one: baseline and intervened {top, targets}, kl (nats, baseline to intervened) and the tokens that gained and lost
    the most probability over the whole vocabulary."""
    b = await req.json()
    if off := _live_off():
        return off
    pushes, hs, err = _interventions(b)
    if err:
        return {"error": err}
    k = builtins.min(int(b.get("k", 15)), 100)
    base, alt = await asyncio.to_thread(rt.next_logprobs, b["text"], b.get("chat", True), b.get("prefix", ""), pushes, hs)
    dec = lambda i: rt.tok.decode([int(i)])
    targets = []
    for t in b.get("targets") or []:
        tid = rt.tok(t, add_special_tokens=False).input_ids
        if tid:
            targets.append((t, tid[0], len(tid) > 1))

    def view(lp):
        v, i = lp.topk(k)
        return {"top": [[dec(t), round(float(x), 3)] for x, t in zip(v, i)],
                "targets": [{"text": t, "token": dec(tid), "logprob": round(float(lp[tid]), 3),
                             **({"first_token_only": True} if multi else {})} for t, tid, multi in targets]}
    if alt is None:
        return view(base)
    d = alt.exp() - base.exp()  # ranked by probability gained or lost, so the far tail's large log ratios stay out
    up, down = d.topk(12), (-d).topk(12)
    row = lambda i: [dec(i), round(float(base[i]), 2), round(float(alt[i]), 2)]
    return {"baseline": view(base), "intervened": view(alt),
            "kl": round(float((base.exp() * (base - alt)).sum()), 4),
            "rose": [row(i) for i in up.indices], "fell": [row(i) for i in down.indices],
            "rows_are": "[token, baseline logprob, intervened logprob]",
            "pushed": [{"neuron": f"L{l} " + ("MLP (all)" if n is None else f"N{n}"), "dose": dd, "mode": m} for l, n, dd, m in pushes],
            "heads": [{"layer": l, "head": "all" if hd is None else hd, "scale": sc} for l, hd, sc in hs]}


@app.post("/api/agent/compare")
async def agent_compare(req: Request):
    """For Claude: many neurons read across several texts, one forward pass per text. Body {neurons: [{layer, n}],
    texts: [text | {label, text}], chat?}. Per neuron and text: the strongest positive and negative activation with
    its token, and the peak score (activation over the neuron's usual extreme; position 0, the sink, left out)."""
    b = await req.json()
    if off := _live_off():
        return off
    neurons = [(int(x["layer"]), int(x["n"])) for x in b.get("neurons") or []][:300]
    texts = [(t.get("label") or f"t{i + 1}", t["text"]) if isinstance(t, dict) else (f"t{i + 1}", t)
             for i, t in enumerate((b.get("texts") or [])[:24])]
    if not neurons or not texts:
        return {"error": "give neurons [{layer, n}] and texts [text or {label, text}]"}
    nm = names()
    rows = {nk: {"neuron": f"L{nk[0]} N{nk[1]}", "name": nm.get(f"L{nk[0]}/N{nk[1]}") or None,
                 "act_range": act_range(*nk), "by_text": {}} for nk in neurons}
    for label, text in texts:
        ids, acts, score = await score_text(text, b.get("chat", False))
        toks = [rt.tok.decode([i]) for i in ids]
        lo = 1 if len(ids) > 1 else 0
        for l, m in neurons:
            a, sc = acts[l, lo:, m], score[l, lo:, m]
            hi_i, lo_i, pk = int(a.argmax()), int(a.argmin()), int(np.abs(sc).argmax())
            rows[(l, m)]["by_text"][label] = {"max": [round(float(a[hi_i]), 2), toks[hi_i + lo]],
                                             "min": [round(float(a[lo_i]), 2), toks[lo_i + lo]],
                                             "peak_score": round(float(sc[pk]), 2)}
    return {"texts": [{"label": lb, "chars": len(t)} for lb, t in texts], "neurons": list(rows.values())}


@app.post("/api/agent/fires")
async def agent_fires(req: Request):
    """For Claude: run text through the model and list the neurons firing hardest at one token (the first whose text
    contains `token`, else the token where any neuron fires hardest; `at` picks a token by index into `tokens`,
    negative from the end). Leaves the person's paste box alone."""
    b = await req.json()
    if off := _live_off():
        return off
    ids, acts, score = await score_text(b["text"], b.get("chat", False))
    toks = [rt.tok.decode([i]) for i in ids]
    want = b.get("token")
    pos = int(b["at"]) % len(toks) if b.get("at") is not None else next((i for i, t in enumerate(toks) if want and want in t), None)
    if pos is None:
        pos = int(np.abs(score).max(axis=(0, 2)).argmax())
    S = score[:, pos, :]
    if b.get("layers"):
        lo, hi = b["layers"]
        S = S.copy(); S[:lo] = 0; S[hi + 1:] = 0
    flat = np.argsort(-np.abs(S).ravel())[: int(b.get("limit", 30))]
    st = rt.steer_index() or rt.lens_index()
    v, nm, N = rt.vocab_strings(), names(), S.shape[1]
    res = []
    for i in flat:
        l, n = int(i // N), int(i % N)
        sg = "+" if S[l, n] > 0 else "-"
        res.append({"neuron": f"L{l} N{n}", "pole": sg, "score": round(float(abs(S[l, n])), 2),
                    "act": round(float(acts[l, pos, n]), 3), "name": nm.get(f"L{l}/N{n}") or None,
                    "steer": [v[t] for t in st["up" if sg == "+" else "down"][l, n, :6]]})
    return {"tokens": toks, "at": pos, "token": toks[pos],
            "note": "score = activation / the neuron's typical extreme on that side; 1 = as strong as its usual max",
            "neurons": res}


@app.post("/api/agent/ui")
async def agent_ui(req: Request):
    b = await req.json()
    c = _chats.get(b.pop("chat", ""))
    if c is None:
        return {"ok": False, "error": "no such chat"}
    _emit(c, {"t": "ui", **b})
    return {"ok": True, "note": "shown in the explorer"}


UPLOADS = os.path.join(AGENT_DIR, "uploads")
FILE_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]


@app.post("/api/agent/upload")
async def agent_upload(req: Request):
    """A file for the chat, sent as base64 JSON {name, type, data}; kept under explorer/cache/agent/uploads."""
    import base64
    import uuid
    b = await req.json()
    raw = base64.b64decode(b["data"])
    if len(raw) > 25 << 20:
        return JSONResponse({"error": "file over 25 MB"}, status_code=413)
    os.makedirs(UPLOADS, exist_ok=True)
    fid = uuid.uuid4().hex[:12]
    name = os.path.basename(b.get("name") or "file")
    with open(os.path.join(UPLOADS, f"{fid}-{name}"), "wb") as f:
        f.write(raw)
    with open(os.path.join(UPLOADS, f"{fid}.json"), "w") as f:
        json.dump({"name": name, "type": b.get("type") or "", "size": len(raw)}, f)
    return {"id": fid, "name": name, "size": len(raw)}


def attachment_block(fid):
    """An uploaded file as a message content block: images and PDFs as themselves, anything else as text."""
    import base64
    fid = os.path.basename(str(fid))
    meta_path = os.path.join(UPLOADS, f"{fid}.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path))
    raw = open(os.path.join(UPLOADS, f"{fid}-{meta['name']}"), "rb").read()
    kind = meta["type"]
    if kind in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        return {"type": "image", "source": {"type": "base64", "media_type": kind, "data": base64.b64encode(raw).decode()}}
    if kind == "application/pdf" or meta["name"].lower().endswith(".pdf"):
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                               "data": base64.b64encode(raw).decode()}, "title": meta["name"]}
    text = raw.decode("utf-8", errors="replace")
    cut = 200_000
    more = f"\n[... {len(text) - cut} more characters cut]" if len(text) > cut else ""
    return {"type": "text", "text": f'<file name="{meta["name"]}">\n{text[:cut]}{more}\n</file>'}


AGENT_METHOD = os.path.join(ROOT, "explorer", "agent_methodology.md")
AGENT_LOCAL = os.path.join(ROOT, "explorer", "agent_prompt.local.md")  # this machine's own additions, untracked


def agent_prompt_file():
    """agent_prompt.md, then agent_methodology.md and agent_prompt.local.md when present, with {ROOT} filled in as
    this checkout's path, written beside the chat's working files."""
    text = open(AGENT_PROMPT).read()
    for extra in (AGENT_METHOD, AGENT_LOCAL):
        if os.path.exists(extra):
            text = text.rstrip() + "\n\n" + open(extra).read()
    text = text.replace("{ROOT}", ROOT)
    out = os.path.join(AGENT_DIR, "system_prompt.md")
    if not os.path.exists(out) or open(out).read() != text:
        if os.path.exists(out):
            os.chmod(out, 0o644)
        with open(out, "w") as f:
            f.write(text)
        os.chmod(out, 0o444)  # generated: edits belong in the source files above
    return out


async def _stop_turn(proc):
    import signal
    if proc is None or proc.returncode is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            await asyncio.wait_for(proc.wait(), 5)
            return
        except asyncio.TimeoutError:
            pass


def _session_file(sid):
    hits = glob.glob(os.path.join(os.path.expanduser("~/.claude/projects"), "*", f"{os.path.basename(sid)}.jsonl"))
    return hits[0] if hits else None


def _user_text(m):
    """The text of a message the person sent (None for tool results and CLI-injected notes)."""
    if m.get("type") != "user" or m.get("isMeta"):
        return None
    c = (m.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list) and not any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
        # the first text block is what they typed; attached text files follow as blocks of their own
        return next((x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text"), "")
    return None


def _tool_results(m):
    """The tool_result events in a session's user message, outputs cut to 1500 characters."""
    out = []
    for blk in (m.get("message") or {}).get("content") or []:
        if isinstance(blk, dict) and blk.get("type") == "tool_result":
            o = blk.get("content")
            if isinstance(o, list):
                o = "\n".join(x.get("text", "") for x in o if isinstance(x, dict))
            o = str(o or "")
            out.append({"t": "tool_result", "id": blk.get("tool_use_id"), "error": bool(blk.get("is_error")),
                        "output": o[:1500] + (f" … ({len(o)} chars)" if len(o) > 1500 else "")})
    return out


def _refusal(m):
    """A reply the API's safety classifier stopped: {id, text} naming the classifier, else None."""
    msg = m.get("message") if isinstance(m.get("message"), dict) else {}
    if m.get("type") != "assistant" or msg.get("stop_reason") != "refusal":
        return None
    cat = (msg.get("stop_details") or {}).get("category")
    return {"id": msg.get("id"), "text": f"stopped by the {cat} safety classifier" if cat else "stopped by a safety classifier"}


def _shown_text(t):
    """What the person typed, without the note about what they had open."""
    import re
    return re.sub(r"^\[The person has [^\]]*\]\n\n", "", t)


AGENT_SESSIONS = os.path.join(os.path.expanduser("~/.claude/projects"), AGENT_DIR.replace("/", "-").replace(".", "-"))
_chat_list = {}  # path -> (mtime, summary)


@app.get("/api/agent/chats")
def agent_chats(limit: int = 60):
    """Past chats, newest first: {chat, t (unix time of the last change), title (the first message), turns}."""
    out = []
    for path in glob.glob(os.path.join(AGENT_SESSIONS, "*.jsonl")):
        mt = os.path.getmtime(path)
        hit = _chat_list.get(path)
        if not hit or hit[0] != mt:
            title, turns = None, 0
            with open(path) as f:
                for line in f:
                    try:
                        t = _user_text(json.loads(line))
                    except ValueError:
                        continue
                    if t is not None:
                        turns += 1
                        title = title or _shown_text(t).strip()[:120]
            hit = _chat_list[path] = (mt, {"chat": os.path.basename(path)[:-6], "t": int(mt), "title": title, "turns": turns, "agent": "claude"})
        if hit[1]["turns"]:
            out.append(hit[1])
    for cid, m in _meta().items():
        path = os.path.join(CODEX_LOGS, f"{os.path.basename(cid)}.jsonl")
        if m.get("agent") != "codex" or not os.path.exists(path):
            continue
        users = [e for e in map(json.loads, open(path)) if e.get("t") == "user"]
        if users:
            out.append({"chat": cid, "t": int(os.path.getmtime(path)), "title": users[0]["text"].strip()[:120], "turns": len(users), "agent": "codex"})
    return sorted(out, key=lambda c: -c["t"])[:limit]


@app.get("/api/agent/history")
def agent_history(chat: str):
    """A past chat as the events a live turn streams (text, tool, tool_result), with a user event per message."""
    if _meta().get(chat, {}).get("agent") == "codex":
        path = os.path.join(CODEX_LOGS, f"{os.path.basename(chat)}.jsonl")
        ev = [e for e in map(json.loads, open(path)) if e.get("t") in ("user", "text", "tool", "tool_result")] if os.path.exists(path) else []
        return {"chat": chat, "agent": "codex", "events": ev}
    path = _session_file(chat)
    if not path:
        return {"events": []}
    ev, seen_refusals = [], set()
    with open(path) as f:
        for line in f:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("isSidechain") or m.get("isMeta"):
                continue
            t = _user_text(m)
            if t is not None:
                ev.append({"t": "user", "text": _shown_text(t)})
            elif m.get("type") == "assistant":
                if (r := _refusal(m)) and r["id"] not in seen_refusals:
                    seen_refusals.add(r["id"])
                    ev.append({"t": "error", "error": r["text"]})
                for blk in (m.get("message") or {}).get("content", []):
                    if blk.get("type") == "text" and blk.get("text"):
                        ev.append({"t": "text", "text": blk["text"]})
                    elif blk.get("type") == "tool_use":
                        ev.append({"t": "tool", "id": blk.get("id"), "input": blk.get("input"),
                                   "name": blk["name"].removeprefix("mcp__explorer__")})
            elif m.get("type") == "user":
                ev += _tool_results(m)
    return {"chat": chat, "events": ev}


@app.post("/api/agent/rewind")
async def agent_rewind(req: Request):
    """Cut a chat's claude session back to just before the last message the person sent that ends with `text`
    (everything after it goes too: the reply, a refusal, CLI notes). The cut session is backed up beside it first.
    No match (already cut) is not an error."""
    import shutil
    import time
    b = await req.json()
    sid, text = b["chat"], (b.get("text") or "").strip()
    meta = _meta().get(sid, {})
    if meta.get("agent") == "codex":  # drop the last turn from the thread and the replay log
        c = _chat(sid)
        turns = meta.get("turns") or []
        if turns:
            try:
                await _ensure_proc(c)
                await _rpc(c, "thread/revert", {"threadId": c["thread"], "beforeTurnId": turns[-1]})
            finally:
                await _end_chat(c)
            _meta_set(sid, turns=turns[:-1])
        path = os.path.join(CODEX_LOGS, f"{os.path.basename(sid)}.jsonl")
        if os.path.exists(path):
            ev = [json.loads(x) for x in open(path)]
            last = max((i for i, e in enumerate(ev) if e.get("t") == "user"), default=None)
            if last is not None:
                with open(path, "w") as f:
                    f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in ev[:last])
        return {"ok": True, "cut": bool(turns)}
    if sid in _chats:  # a live process holds the conversation in memory; the next message resumes from the cut file
        await _end_chat(_chats[sid])
    path = _session_file(sid)
    if not path:
        return {"ok": True, "cut": False}
    lines = open(path).read().splitlines(True)
    at = None
    for i in range(len(lines) - 1, -1, -1):
        try:
            t = _user_text(json.loads(lines[i]))
        except ValueError:
            continue
        if t is not None and (not text or t.strip() == text or t.strip().endswith("\n\n" + text) or t.split("\n\n", 1)[-1].strip() == text):
            at = i
            break
    if at is None:
        return {"ok": True, "cut": False}
    shutil.copy(path, f"{path}.bak-{int(time.time())}")
    keep = lines[:at]
    if not any(_user_text(json.loads(x)) is not None for x in keep if x.strip()):
        os.remove(path)  # nothing the person said is left: the next turn starts the session afresh
    else:
        with open(path, "w") as f:
            f.writelines(keep)
    return {"ok": True, "cut": True}


# One claude process per chat, its input left open: messages written mid-turn reach Claude between tool calls,
# interrupt stops the reply without ending the process (its background commands keep running and report back in a
# turn of their own), and every event goes to the chat's listeners. Events since the last turn ended are kept, so a
# browser that reconnects mid-turn sees the whole turn.
CHAT_IDLE = 30 * 60  # seconds without listeners or turns before a chat's input is closed (it exits once its work is done)


def _claude_cmd(cid):
    mcp = {"mcpServers": {"explorer": {"command": sys.executable, "args": [os.path.join(ROOT, "explorer", "agent_mcp.py")],
                                       "env": {"EXPLORER_URL": f"http://127.0.0.1:{PORT}", "EXPLORER_CHAT": cid, **NEURONARY_ENV}}}}
    cmd = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--system-prompt-file", agent_prompt_file(), "--mcp-config", json.dumps(mcp), "--strict-mcp-config",
           # file tools and a shell, so it can run scripts against this server (e.g. score a file it shouldn't read)
           "--tools", *FILE_TOOLS, "--allowedTools", "mcp__explorer", *FILE_TOOLS,
           "--add-dir", os.path.expanduser("~"), "--setting-sources", "project"]
    if os.environ.get("EXPLORER_CLAUDE_MODEL"):
        cmd += ["--model", os.environ["EXPLORER_CLAUDE_MODEL"]]
    # resume once the session exists on disk (any earlier turn creates it)
    return cmd + (["--resume", cid] if _session_file(cid) else ["--session-id", cid])


CHAT_META = os.path.join(AGENT_DIR, "chats.json")  # per chat: agent ("claude" / "codex"), codex thread id and turn ids
CODEX_LOGS = os.path.join(AGENT_DIR, "codex_logs")  # per codex chat: its events, for past-chat replay


def _meta():
    try:
        return json.load(open(CHAT_META))
    except (OSError, ValueError):
        return {}


def _meta_set(cid, **kw):
    m = _meta()
    m.setdefault(cid, {}).update(kw)
    os.makedirs(AGENT_DIR, exist_ok=True)
    with open(CHAT_META, "w") as f:
        json.dump(m, f)


def _chat(cid, agent=None):
    c = _chats.get(cid)
    # a chat that hasn't run yet (an event stream can open before its first message) takes the agent it's sent with
    if c is not None and agent in ("claude", "codex") and c["proc"] is None and not _meta().get(cid, {}).get("agent") \
            and not _session_file(cid) and c["agent"] != agent:
        del _chats[cid]
        c2 = _chat(cid, agent)
        c2["listeners"] = c["listeners"]
        return c2
    if cid not in _chats:
        known = _meta().get(cid, {}).get("agent")
        kind = known or (agent if agent in ("claude", "codex") else "claude")
        if not known and kind == "codex":
            _meta_set(cid, agent="codex")
        _chats[cid] = {"session": cid, "agent": kind, "proc": None, "listeners": set(), "turn": [], "busy": False, "last": 0.0}
    return _chats[cid]


def _codex_log(cid, ev):
    os.makedirs(CODEX_LOGS, exist_ok=True)
    with open(os.path.join(CODEX_LOGS, f"{os.path.basename(cid)}.jsonl"), "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _emit(c, ev):
    import time
    c["last"] = time.time()
    if ev.get("t") != "ui":
        c["turn"].append(ev)
        if c["agent"] == "codex" and ev.get("t") in ("text", "tool", "tool_result", "error", "done"):
            _codex_log(c["session"], ev)
    for q in list(c["listeners"]):
        q.put_nowait(ev)


async def _pump(c, proc):
    """Turns the process's stream-json output into chat events until it exits."""
    async for line in proc.stdout:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("type") == "stream_event":
            e = m["event"]
            if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta":
                c["busy"] = True
                _emit(c, {"t": "text", "text": e["delta"]["text"]})
        elif m.get("type") == "assistant":
            for blk in m["message"].get("content", []):
                if blk.get("type") == "tool_use":
                    c["busy"] = True
                    _emit(c, {"t": "tool", "id": blk.get("id"), "input": blk.get("input"),
                              "name": blk["name"].removeprefix("mcp__explorer__")})
            if (r := _refusal(m)) and r["id"] not in c.setdefault("refused", set()):
                c["refused"].add(r["id"])
                _emit(c, {"t": "error", "error": r["text"]})
        elif m.get("type") == "user":
            for ev in _tool_results(m):
                _emit(c, ev)
        elif m.get("type") == "result":
            if m.get("is_error") and m.get("subtype") != "error_during_execution":
                _emit(c, {"t": "error", "error": str(m.get("result"))[:600]})
            c["busy"] = False
            _emit(c, {"t": "done", "cost": m.get("total_cost_usd")})
            c["turn"] = []
    err = (await proc.stderr.read()).decode(errors="replace")
    code = await proc.wait()
    if c.get("proc") is proc:
        c["proc"] = None
        if c["busy"] or code not in (0, None):
            _emit(c, {"t": "error", "error": err.strip()[-600:] or f"claude exited {code}"})
        if c["busy"]:
            c["busy"] = False
            _emit(c, {"t": "done"})
        c["turn"] = []


CODEX_HOME = os.path.join(AGENT_DIR, "codex-home")


def _codex_home():
    """Codex's own home for the explorer agent: the person's login, their model and effort, and nothing else (no
    plugins, apps, memories or global instructions), so its tools are the explorer's and its shell."""
    import tomllib
    os.makedirs(CODEX_HOME, exist_ok=True)
    auth, src = os.path.join(CODEX_HOME, "auth.json"), os.path.expanduser("~/.codex/auth.json")
    if not os.path.lexists(auth) and os.path.exists(src):
        os.symlink(src, auth)
    try:
        user = tomllib.load(open(os.path.expanduser("~/.codex/config.toml"), "rb"))
    except (OSError, ValueError):
        user = {}
    lines = [f'{k} = {json.dumps(user[k])}' for k in ("model", "model_reasoning_effort") if isinstance(user.get(k), str)]
    lines += ["", "[features]", "apps = false", "memories = false"]
    with open(os.path.join(CODEX_HOME, "config.toml"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return CODEX_HOME


CODEX_NOTE = """

## This session

You are running as Codex. Your shell and file editing are Codex's own: where these instructions name Read, Glob,
Grep, Write, Edit or Bash, use your shell and apply_patch. Start anything that runs longer than about a minute
detached (`nohup {ROOT}/.venv/bin/python script.py > script.log 2>&1 &`), say where the log is, and end your reply;
read the log in a later turn."""


async def _rpc(c, method, params, timeout=120):
    rpc = c["rpc"]
    rpc["next"] += 1
    fut = asyncio.get_running_loop().create_future()
    rpc["pending"][rpc["next"]] = fut
    await _write(c["proc"], {"jsonrpc": "2.0", "id": rpc["next"], "method": method, "params": params})
    r = await asyncio.wait_for(fut, timeout)
    if "error" in r:
        raise RuntimeError(f"codex {method}: {r['error'].get('message', r['error'])}")
    return r.get("result") or {}


def _codex_text(item):
    """A tool item's input and output as the events show them: (name, input, output, failed)."""
    k = item.get("type")
    if k == "mcpToolCall":
        res = item.get("result") or {}
        out = "\n".join(x.get("text", "") for x in res.get("content") or [] if isinstance(x, dict)) or str(item.get("error") or "")
        return item.get("tool", "tool"), item.get("arguments") or {}, out, item.get("status") == "failed"
    if k == "commandExecution":
        return "Bash", {"command": item.get("command")}, item.get("aggregatedOutput") or "", item.get("status") in ("failed", "declined")
    if k == "fileChange":
        ch = item.get("changes") or []
        return "Edit", {"file_path": ", ".join(str(x.get("path", "")) for x in ch if isinstance(x, dict))}, item.get("status", ""), item.get("status") == "failed"
    if k == "webSearch":
        return "WebSearch", {"query": item.get("query")}, "", False
    return None


async def _codex_pump(c, proc):
    rpc = c["rpc"]
    async for line in proc.stdout:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if "id" in m and ("result" in m or "error" in m) and "method" not in m:
            fut = rpc["pending"].pop(m["id"], None)
            if fut and not fut.done():
                fut.set_result(m)
            continue
        meth, prm = m.get("method"), m.get("params") or {}
        if "id" in m and meth:  # a request from codex (approvals): the thread runs without them; decline anything else
            await _write(proc, {"jsonrpc": "2.0", "id": m["id"], "result": {"decision": "decline"}})
            continue
        if meth == "turn/started":
            c["turn_id"], c["busy"] = prm["turn"]["id"], True
        elif meth == "item/agentMessage/delta":
            c["busy"] = True
            if c.get("msg_item") != prm.get("itemId"):
                if c.get("msg_item"):
                    _emit(c, {"t": "text", "text": "\n\n"})
                c["msg_item"] = prm.get("itemId")
            _emit(c, {"t": "text", "text": prm.get("delta", "")})
        elif meth in ("item/started", "item/completed"):
            t = _codex_text(prm.get("item") or {})
            if t:
                name, inp, out, failed = t
                iid = prm["item"].get("id")
                if meth == "item/started":
                    c["msg_item"] = None
                    _emit(c, {"t": "tool", "id": iid, "name": name, "input": inp})
                else:
                    _emit(c, {"t": "tool_result", "id": iid, "error": failed,
                              "output": out[:1500] + (f" … ({len(out)} chars)" if len(out) > 1500 else "")})
        elif meth == "turn/completed":
            turn = prm.get("turn") or {}
            if turn.get("status") == "failed" and turn.get("error"):
                _emit(c, {"t": "error", "error": str(turn["error"].get("message", turn["error"]))[:600]})
            c["busy"], c["turn_id"], c["msg_item"] = False, None, None
            _emit(c, {"t": "done"})
            c["turn"] = []
        elif meth == "error":
            _emit(c, {"t": "error", "error": str(prm.get("error", {}).get("message", prm))[:600]})
    await proc.wait()
    if c.get("proc") is proc:
        c["proc"] = None
        if c["busy"]:
            c["busy"] = False
            _emit(c, {"t": "error", "error": "codex exited"})
            _emit(c, {"t": "done"})
        c["turn"] = []


async def _ensure_codex(c):
    proc = c.get("proc")
    if proc is not None and proc.returncode is None:
        return proc
    os.makedirs(AGENT_DIR, exist_ok=True)
    env = {**os.environ, "CODEX_HOME": _codex_home()}
    proc = c["proc"] = await asyncio.create_subprocess_exec("codex", "app-server", cwd=AGENT_DIR, env=env, stdin=asyncio.subprocess.PIPE,
                                                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                                                            limit=1 << 24, start_new_session=True)
    c["rpc"] = {"next": 0, "pending": {}}
    c["pump"] = asyncio.create_task(_codex_pump(c, proc))
    await _rpc(c, "initialize", {"clientInfo": {"name": "neuron-explorer", "version": "0.1"}})
    mcp = {"explorer": {"command": sys.executable, "args": [os.path.join(ROOT, "explorer", "agent_mcp.py")],
                        "env": {"EXPLORER_URL": f"http://127.0.0.1:{PORT}", "EXPLORER_CHAT": c["session"], **NEURONARY_ENV}}}
    opts = {"cwd": AGENT_DIR, "approvalPolicy": "never", "sandbox": "danger-full-access", "config": {"mcp_servers": mcp},
            "developerInstructions": open(agent_prompt_file()).read() + CODEX_NOTE.replace("{ROOT}", ROOT)}
    thread = _meta().get(c["session"], {}).get("thread")
    r = await _rpc(c, "thread/resume", {"threadId": thread, **opts}) if thread else await _rpc(c, "thread/start", opts)
    c["thread"] = r["thread"]["id"]
    if not thread:
        _meta_set(c["session"], thread=c["thread"])
    return proc


def _agent_files_mtime():
    """When the agent's tools or instructions last changed (agent_mcp.py and the prompt files)."""
    paths = [os.path.join(ROOT, "explorer", "agent_mcp.py"), AGENT_PROMPT, AGENT_METHOD, AGENT_LOCAL]
    return max((os.path.getmtime(x) for x in paths if os.path.exists(x)), default=0)


async def _ensure_proc(c):
    import time
    # a chat between turns picks up changed tools or instructions: its process restarts, resuming the conversation
    proc = c.get("proc")
    if proc is not None and proc.returncode is None and not c["busy"] and c.get("spawned", 0) < _agent_files_mtime():
        await _end_chat(c)
    if c.get("proc") is None or c["proc"].returncode is not None:
        c["spawned"] = time.time()
    if c["agent"] == "codex":
        return await _ensure_codex(c)
    proc = c.get("proc")
    if proc is not None and proc.returncode is None:
        return proc
    os.makedirs(AGENT_DIR, exist_ok=True)
    # its own process group, so ending a chat also ends what it started (MCP server, shell scripts)
    proc = c["proc"] = await asyncio.create_subprocess_exec(*_claude_cmd(c["session"]), cwd=AGENT_DIR, stdin=asyncio.subprocess.PIPE,
                                                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                            limit=1 << 24, start_new_session=True)
    c["pump"] = asyncio.create_task(_pump(c, proc))
    return proc


async def _write(proc, obj):
    proc.stdin.write((json.dumps(obj) + "\n").encode())
    await proc.stdin.drain()


@app.post("/api/agent/send")
async def agent_send(req: Request):
    """A message into a chat: {chat, message, layer?, n? or head?, attachments?: [upload ids]}. Starts (or resumes) the
    chat's claude process if needed. Mid-turn, the message reaches Claude between tool calls. Events arrive on
    /api/agent/events."""
    import time
    b = await req.json()
    c = _chat(b["chat"], b.get("agent"))
    msg = b["message"]
    shown = msg
    if b.get("layer") is not None and b.get("head") is not None:
        msg = f"[The person has attention head L{int(b['layer'])} h{int(b['head'])} open in the explorer.]\n\n{msg}"
    elif b.get("layer") is not None:
        msg = f"[The person has L{int(b['layer'])} N{int(b['n'])} open in the explorer.]\n\n{msg}"
    if c["agent"] == "codex":
        try:
            await _ensure_proc(c)
        except Exception as e:
            return {"error": f"codex: {e}"}
        _codex_log(c["session"], {"t": "user", "text": shown})
        items = [{"type": "text", "text": msg, "text_elements": []}] + [x for a in b.get("attachments") or [] if (x := _codex_input(a))]
        c["busy"], c["last"] = True, time.time()
        _emit(c, {"t": "sent"})
        if c.get("turn_id"):  # mid-turn: the message joins the running turn
            try:
                await _rpc(c, "turn/steer", {"threadId": c["thread"], "expectedTurnId": c["turn_id"], "input": items})
                return {"ok": True}
            except Exception:
                pass  # the turn ended meanwhile: start a new one
        r = await _rpc(c, "turn/start", {"threadId": c["thread"], "input": items})
        c["turn_id"] = r["turn"]["id"]
        _meta_set(c["session"], turns=_meta().get(c["session"], {}).get("turns", []) + [c["turn_id"]])
        return {"ok": True}
    content = [{"type": "text", "text": msg}] + [blk for a in b.get("attachments") or [] if (blk := attachment_block(a))]
    proc = await _ensure_proc(c)
    c["busy"], c["last"] = True, time.time()
    _emit(c, {"t": "sent"})
    await _write(proc, {"type": "user", "message": {"role": "user", "content": content}})
    return {"ok": True}


def _codex_input(fid):
    """An uploaded file as codex input: images by path, anything else as text (as attachment_block gives it)."""
    fid = os.path.basename(str(fid))
    mp = os.path.join(UPLOADS, f"{fid}.json")
    if not os.path.exists(mp):
        return None
    meta = json.load(open(mp))
    if meta["type"].startswith("image/"):
        return {"type": "localImage", "path": os.path.join(UPLOADS, f"{fid}-{meta['name']}")}
    blk = attachment_block(fid)
    return {"type": "text", "text": blk["text"], "text_elements": []} if blk and blk.get("type") == "text" else None


@app.post("/api/agent/interrupt")
async def agent_interrupt(req: Request):
    """Stops the chat's current reply; the process and its background commands keep running."""
    import uuid
    b = await req.json()
    c = _chats.get(b.get("chat", ""))
    proc = c and c.get("proc")
    if not proc or proc.returncode is not None or not c["busy"]:
        return {"ok": True, "running": False}
    if c["agent"] == "codex":
        if not c.get("turn_id"):
            return {"ok": True, "running": False}
        await _rpc(c, "turn/interrupt", {"threadId": c["thread"], "turnId": c["turn_id"]})
        return {"ok": True, "running": True}
    await _write(proc, {"type": "control_request", "request_id": str(uuid.uuid4()), "request": {"subtype": "interrupt"}})
    return {"ok": True, "running": True}


@app.get("/api/agent/events")
async def agent_events(chat: str):
    """The chat's events as NDJSON, starting with {t: "state", busy} and the current turn's events so far: text, tool,
    tool_result, ui, error, done (a turn ended), sent (a message went in)."""
    c = _chat(chat)
    q = asyncio.Queue()
    c["listeners"].add(q)

    async def gen():
        try:
            yield json.dumps({"t": "state", "busy": c["busy"], "running": bool(c.get("proc") and c["proc"].returncode is None)}) + "\n"
            for ev in list(c["turn"]):
                yield json.dumps(ev) + "\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), 20)
                except asyncio.TimeoutError:
                    yield json.dumps({"t": "ping"}) + "\n"
                    continue
                yield json.dumps(ev) + "\n"
        finally:
            c["listeners"].discard(q)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


async def _end_chat(c):
    """Ends a chat's process now (and its background commands)."""
    proc = c.get("proc")
    c["proc"] = None
    await _stop_turn(proc)


async def _reap_chats():
    """Closes the input of chats idle past CHAT_IDLE with no one listening; claude exits once its own work is done."""
    import time
    while True:
        await asyncio.sleep(60)
        for c in list(_chats.values()):
            proc = c.get("proc")
            if proc and proc.returncode is None and not c["busy"] and not c["listeners"] and time.time() - c["last"] > CHAT_IDLE:
                try:
                    proc.stdin.close()
                except Exception:
                    pass


@app.on_event("startup")
async def _start_reaper():
    asyncio.create_task(_reap_chats())


@app.get("/api/update/check")
def api_update_check():
    """Check the author's main branch without touching the working tree."""
    try:
        return update_check()
    except UpdateError as e:
        return {"ok": False, "code": e.code, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/update/apply")
def api_update_apply():
    """Fast-forward to upstream main and reconcile dependencies.  Data directories are gitignored and untouched."""
    try:
        return apply_update()
    except UpdateError as e:
        return {"ok": False, "code": e.code, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/update/status")
def api_update_status():
    """The boot id changes only when the process is re-exec'd (or manually restarted)."""
    return {"ok": True, "boot": BOOT_ID, "pid": os.getpid()}


@app.post("/api/update/restart")
def api_update_restart():
    """Re-exec this server after the response has been sent, so new code and static files load."""
    import time
    def go():
        time.sleep(0.8)
        env = {**os.environ, "PORT": str(PORT)}
        os.execve(sys.executable, [sys.executable, os.path.join(ROOT, "explorer", "server.py")], env)
    threading.Thread(target=go, daemon=True).start()
    return {"ok": True, "boot": BOOT_ID}


@app.get("/api/status")
def status():
    return {"status": rt.status}


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "explorer", "static"), html=True), name="static")

@app.on_event("startup")
def preload():
    def go():
        if NO_MODEL:
            rt.status = "no model"
            rt.vocab_strings()
            return
        rt.ensure_loaded()
        if not os.path.exists(LENS_INDEX):
            os.makedirs(os.path.dirname(LENS_INDEX), exist_ok=True)
            rt.build_lens_index(LENS_INDEX)
        rt.vocab_strings()
        if not os.path.exists(ACT_SCALE):
            rt.build_act_scale(ACT_SCALE)
    threading.Thread(target=go, daemon=True).start()


if __name__ == "__main__":
    # chat event streams stay open indefinitely; on shutdown, close them after a few seconds instead of waiting
    uvicorn.run(app, host="127.0.0.1", port=PORT, timeout_graceful_shutdown=3)
