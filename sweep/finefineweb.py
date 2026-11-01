"""What every neuron reads on web text: top-activating contexts and a per-domain profile over FineFineWeb.

Samples a domain-balanced corpus from m-a-p/FineFineWeb (66 domains, a few random shards each, documents truncated to
--max-len tokens), runs it through the model one document per row, and records each neuron's activation (the input to
down_proj, silu(gate) * up) at every position except the first. Keeps, per layer, neuron and sign, the --k strongest
positions, and per domain the token count, max, min, and summed positive and negative activation.

    uv run python sweep/finefineweb.py --out ffw/q35-4b                      # sample + sweep
    uv run python sweep/finefineweb.py --out ffw/q35-4b --tokens-per-domain 2000 --domains 2   # smoke test

Writes, under --out:
    corpus.npz    tokens [T] uint32 (documents back to back), doc_start, doc_len [D] int64, doc_domain [D] int16,
                  domains [66] str
    urls.json     each document's source url
    acts.npz      top_val, bot_val [L, k, N] float16 (top: largest activations, bot: most negative), top_pos, bot_pos
                  [L, k, N] int32 (indices into tokens), dom_count [66] int64, dom_max, dom_min, dom_pos, dom_neg
                  [66, L, N] float32, sum, sumsq [L, N] float64, count
A checkpoint (state.pt) is written as it goes; rerunning the same command resumes.
"""
import argparse
import json
import os
import random
import time
import urllib.request

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

REPO = "m-a-p/FineFineWeb"


def hf_get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{rng - 1}"} if rng else {})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def sample_corpus(tok, a):
    """A few random shards per domain; from each, the head of the file (a range request), parsed line by line."""
    files = json.loads(hf_get(f"https://huggingface.co/api/datasets/{REPO}"))["siblings"]
    by_dom = {}
    for f in files:
        p = f["rfilename"]
        if p.endswith(".jsonl") and "/" in p:
            by_dom.setdefault(p.split("/")[0], []).append(p)
    domains = sorted(by_dom)[: a.domains] if a.domains else sorted(by_dom)
    rnd = random.Random(a.seed)
    toks, starts, lens, doms, urls = [], [], [], [], []
    total = 0
    for di, d in enumerate(domains):
        shards = rnd.sample(by_dom[d], min(a.shards, len(by_dom[d])))
        per_shard, got = a.tokens_per_domain // len(shards) + 1, 0
        for sh in shards:
            raw = hf_get(f"https://huggingface.co/datasets/{REPO}/resolve/main/{sh}", rng=a.shard_bytes)
            lines = raw.split(b"\n")[:-1]  # the last line may be cut by the range
            took = 0
            for line in lines:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ids = tok(rec["text"], add_special_tokens=False).input_ids[: a.max_len]
                if len(ids) < a.min_len:
                    continue
                starts.append(total); lens.append(len(ids)); doms.append(di); urls.append(rec.get("url", ""))
                toks.append(np.asarray(ids, np.uint32)); total += len(ids); took += len(ids)
                if took >= per_shard:
                    break
            got += took
        print(f"{d}: {got} tokens from {len(shards)} shards", flush=True)
    return {"tokens": np.concatenate(toks), "doc_start": np.asarray(starts, np.int64), "doc_len": np.asarray(lens, np.int64),
            "doc_domain": np.asarray(doms, np.int16), "domains": np.asarray(domains)}, urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-per-domain", type=int, default=300_000)
    ap.add_argument("--domains", type=int, default=0, help="first n domains only (0: all)")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--shard-bytes", type=int, default=4_000_000)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-len", type=int, default=32)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
    tok = AutoTokenizer.from_pretrained(a.model)

    cpath = os.path.join(a.out, "corpus.npz")
    if not os.path.exists(cpath):
        corpus, urls = sample_corpus(tok, a)
        json.dump(urls, open(os.path.join(a.out, "urls.json"), "w"))
        np.savez(cpath, **corpus)
    c = dict(np.load(cpath))
    D, ND = len(c["doc_len"]), len(c["domains"])
    print(f"corpus: {D} documents, {int(c['doc_len'].sum())} tokens, {ND} domains", flush=True)

    # batches of documents of similar length (less padding), in a fixed shuffled order
    order = np.argsort(c["doc_len"], kind="stable")
    batches, cur, longest = [], [], 0
    for i in order:
        longest_next = max(longest, int(c["doc_len"][i]))
        if cur and longest_next * (len(cur) + 1) > a.batch_tokens:
            batches.append(cur); cur, longest_next = [], int(c["doc_len"][i])
        cur.append(int(i)); longest = longest_next
    batches.append(cur)
    random.Random(a.seed).shuffle(batches)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    lm = model.model.language_model
    L, N, K = len(lm.layers), lm.layers[0].mlp.down_proj.in_features, a.k
    spath = os.path.join(a.out, "state.pt")
    if os.path.exists(spath):
        st = torch.load(spath, map_location=dev, weights_only=True)
        print(f"resuming at batch {st['done']}/{len(batches)}", flush=True)
    else:
        z = lambda *s, dt=torch.float32, v=0.0: torch.full(s, v, dtype=dt, device=dev)
        f64 = torch.float64 if dev.type == "cuda" else torch.float32  # MPS has no float64
        st = {"done": 0, "top_val": z(L, K, N, v=-1e9), "bot_val": z(L, K, N, v=-1e9),
              "top_pos": z(L, K, N, dt=torch.int64, v=-1), "bot_pos": z(L, K, N, dt=torch.int64, v=-1),
              "dom_count": z(ND, dt=torch.int64), "dom_max": z(ND, L, N, v=-1e9), "dom_min": z(ND, L, N, v=1e9),
              "dom_pos": z(ND, L, N), "dom_neg": z(ND, L, N), "sum": z(L, N, dt=f64),
              "sumsq": z(L, N, dt=f64), "count": z(1, dt=torch.int64)}

    cur_batch = {}

    def hook(li):
        def f(mod, inp):
            h = inp[0]
            sel, gpos, dom = cur_batch["sel"], cur_batch["gpos"], cur_batch["dom"]
            x = h[sel].float()  # [T, N], valid positions only
            for sign, vk, pk in ((1, "top_val", "top_pos"), (-1, "bot_val", "bot_pos")):
                v, ix = (sign * x).topk(min(K, x.shape[0]), dim=0)
                allv = torch.cat([st[vk][li], v]); allp = torch.cat([st[pk][li], gpos[ix]])
                keep = allv.topk(K, dim=0).indices
                st[vk][li] = allv.gather(0, keep); st[pk][li] = allp.gather(0, keep)
            idx = dom[:, None].expand(-1, N)
            st["dom_max"][:, li].scatter_reduce_(0, idx, x, "amax")
            st["dom_min"][:, li].scatter_reduce_(0, idx, x, "amin")
            st["dom_pos"][:, li].index_add_(0, dom, x.clamp(min=0))
            st["dom_neg"][:, li].index_add_(0, dom, x.clamp(max=0))
            st["sum"][li] += x.sum(0).to(st["sum"].dtype); st["sumsq"][li] += x.square().sum(0).to(st["sum"].dtype)
        return f

    for li, layer in enumerate(lm.layers):
        layer.mlp.down_proj.register_forward_pre_hook(hook(li))
    tokens = torch.from_numpy(c["tokens"].astype(np.int64))
    t0, last_save = time.time(), time.time()
    done_tok = 0
    with torch.inference_mode():
        for bi in range(st["done"], len(batches)):
            docs = batches[bi]
            S = int(max(c["doc_len"][i] for i in docs))
            ids = torch.full((len(docs), S), tok.pad_token_id or 0, dtype=torch.int64)
            sel = torch.zeros((len(docs), S), dtype=torch.bool)
            gpos = torch.zeros((len(docs), S), dtype=torch.int64)
            dom = torch.zeros((len(docs), S), dtype=torch.int64)
            for r, i in enumerate(docs):
                s, n = int(c["doc_start"][i]), int(c["doc_len"][i])
                ids[r, :n] = tokens[s:s + n]; sel[r, 1:n] = True
                gpos[r, :n] = torch.arange(s, s + n); dom[r, :n] = int(c["doc_domain"][i])
            sel = sel.to(dev)
            cur_batch.update(sel=sel, gpos=gpos.to(dev)[sel], dom=dom.to(dev)[sel])
            # right padding: every real token precedes the pads in its row, so causal layers never see them
            lm(input_ids=ids.to(dev), use_cache=False)
            n_valid = int(sel.sum())
            st["count"] += n_valid
            st["dom_count"].index_add_(0, cur_batch["dom"], torch.ones_like(cur_batch["dom"]))
            st["done"] = bi + 1
            done_tok += n_valid
            if bi % 20 == 0 or bi == len(batches) - 1:
                el = time.time() - t0
                print(f"batch {bi + 1}/{len(batches)}  {done_tok / el:,.0f} tok/s  elapsed {el / 60:.1f} min", flush=True)
            if time.time() - last_save > 300:
                torch.save(st, spath + ".tmp"); os.replace(spath + ".tmp", spath); last_save = time.time()

    out = {k: v.cpu().numpy() for k, v in st.items() if k != "done"}
    out["bot_val"] = -out["bot_val"]
    for k in ("top_val", "bot_val"):
        out[k] = out[k].astype(np.float16)
    for k in ("top_pos", "bot_pos"):
        out[k] = out[k].astype(np.int32)
    out["count"] = int(out["count"][0])
    np.savez(os.path.join(a.out, "acts.npz"), **out)
    torch.save(st, spath)
    print(f"saved {a.out}/acts.npz", flush=True)


if __name__ == "__main__":
    main()
