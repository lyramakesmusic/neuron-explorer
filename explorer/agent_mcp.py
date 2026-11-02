"""Stdio MCP server that gives the explorer's Claude chat its tools: read any neuron data through the explorer's HTTP API,
run the live model (steered rollouts, activations) when it is loaded, and drive the map (select, highlight) in the browser that started the chat. Standard library only; started by
`claude -p --mcp-config ...` from server.py with EXPLORER_URL and EXPLORER_CHAT set.

The neuronary_* tools talk to neuronary.org (search, download, publish recipes). Reads need nothing; writes need a token
from neuronary.org -> account menu -> API & agents, in NEURONARY_TOKEN or ~/.neuronary/token.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

URL = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
CHAT = os.environ.get("EXPLORER_CHAT", "")
NEURONARY = os.environ.get("NEURONARY_API", "https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api")
MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-4B")

LN = {"layer": {"type": "integer", "description": "layer, 0-31"}, "n": {"type": "integer", "description": "neuron, 0-9215"}}
HEAD = {"layer": {"type": "integer", "description": "layer, 0-31"}, "head": {"type": "integer", "description": "head: 0-15 on full-attention layers, 0-31 on linear-attention layers"}}
TOOLS = [
    {"name": "overview", "description": "Model architecture and which layers have sweep runs (rollouts and rankings).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "neuron", "description": "Everything cached for one neuron: name, logit lens, R and J++ lens (`rjpp`), S lens of "
     "its read side (`s_lens`), and steer lens (with KL) for both poles, activation range, sweep rank if any, and its 24 strongest direct-path links each way (writes_into, "
     "reads_from) with each linked neuron's steer lens, top words and name.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "exemplars", "description": "Where a neuron fires hardest on web text (about 20M tokens of FineFineWeb, "
     "66 domains): for each sign, its strongest contexts (one per document) with the peak token in [[ ]], the peak "
     "activation, the domain, and the other tokens in the window where it is at least half the peak (from a live "
     "trace); plus the domains it concentrates in (lift: its mean activation per token in that domain over the mean "
     "across all domains) and its activation range, mean and std over the whole corpus.",
     "inputSchema": {"type": "object", "properties": {**LN, "k": {"type": "integer", "default": 8}}, "required": ["layer", "n"]}},
    {"name": "rollouts", "description": "Sweep rollouts for a neuron: steered text per prompt and dose beside the "
     "unsteered baselines, distinctive words and stats. Empty when the layer has no sweep run.",
     "inputSchema": {"type": "object", "properties": {**LN, "max_chars": {"type": "integer", "default": 500}},
                     "required": ["layer", "n"]}},
    {"name": "cone", "description": "The neuron's connection cone: links chained forward (downstream) and backward "
     "(upstream), first hop all links, later hops each node's top k. Returns the neurons reached with hop, best z and "
     "steer lens. Multi-hop paths are weight-only guesses: leads to check, not circuits.",
     "inputSchema": {"type": "object", "properties": {**LN, "k": {"type": "integer", "default": 2},
                                                       "limit": {"type": "integer", "default": 120}},
                     "required": ["layer", "n"]}},
    {"name": "web_summary", "description": "The token summary of the neuron's web, the same one the person sees in the "
     "left panel: steer-lens tokens pooled over its cone, each connected neuron read on the pole the chain drives, ranked "
     "by lift over that layer's base rate. `toward` / `away`: what its downstream neurons push the model toward and away "
     "from when this neuron fires (indirect effect). `fed_by` / `held_by`: what its upstream writers say when they "
     "excite / inhibit it (its inputs). Each token has n (neurons carrying it) and a score.",
     "inputSchema": {"type": "object", "properties": {**LN, "first": {"type": "integer", "default": 5},
                                                       "k": {"type": "integer", "default": 2}},
                     "required": ["layer", "n"]}},
    {"name": "search_lens", "description": "Neurons whose logit, R or steer lens (top 12, either pole) contains a token "
     "matching the query (case-insensitive substring). Best matches first.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 40}},
                     "required": ["query"]}},
    {"name": "top_units", "description": "A layer's neurons ranked by sweep word-cluster mass against random-direction "
     "controls, with top words and names. Empty when the layer has no sweep run.",
     "inputSchema": {"type": "object", "properties": {"layer": LN["layer"], "limit": {"type": "integer", "default": 30}},
                     "required": ["layer"]}},
    {"name": "set_name", "description": "Name a neuron in the explorer (the name field in its panel, shown in lists and on "
     "the map), or clear its name with an empty string. Only when the person asks, or after they agree to a name you "
     "suggested.",
     "inputSchema": {"type": "object", "properties": {**LN, "name": {"type": "string"}}, "required": ["layer", "n", "name"]}},
    {"name": "names", "description": "Every neuron the person has named so far, as {\"L14/N3262\": name}.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "steer", "description": "Live: generate from a chat prompt three ways in one batch, pushed up, unsteered, "
     "and pushed down (greedy). Steer one neuron (layer, n) or several at once (neurons: [{layer, n, dose?, mode?}]; all "
     "are applied together, each reversed in the down row; a negative per-neuron dose pushes that one the other way). "
     "dose in units of the layer's median residual norm for mode 'add' (0.8 is the steer-lens push; 2-3 is strong), or "
     "a multiplier for mode 'mul'. Greedy by default; temperature, min_p, repetition_penalty and seed match a recipe's published "
     "sampling (the result says which was used). Needs the live model; errors if it is off.",
     "inputSchema": {"type": "object", "properties": {**LN, "prompt": {"type": "string"},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer", "n"]}},
                                                       "dose": {"type": "number", "default": 0.8},
                                                       "mode": {"type": "string", "enum": ["add", "mul"], "default": "add"},
                                                       "max_new": {"type": "integer", "default": 150},
                                                       "temperature": {"type": "number", "default": 0,
                                                                       "description": "0 = greedy; e.g. 0.7 to match a recipe's sampling"},
                                                       "repetition_penalty": {"type": "number", "default": 1.0,
                                                                              "description": "e.g. 1.1; >1 discourages repeats (loops)"},
                                                       "min_p": {"type": "number", "default": 0,
                                                                 "description": "with temperature: drop tokens below min_p x the top token's probability, e.g. 0.05"},
                                                       "seed": {"type": "integer", "default": 0,
                                                                "description": "each row samples with this seed, so a run repeats exactly"}},
                     "required": ["prompt"]}},
    {"name": "activations", "description": "Live: the neuron's raw activation on every token of a text (chat=true wraps "
     "it as a user turn). Compare against act_range from `neuron`. Optional `neurons` [{layer, n, dose?, mode?}] are "
     "pushed during the same pass (steering units), to test causally whether they drive this neuron: compare with the "
     "same call without them. {layer, all: true, dose: 0} zeroes that layer's whole MLP. Optional `heads` [{layer, "
     "head, dose}] scale attention heads (dose 0 = ablate; head 'all' = every head; full-attention layers have 16 heads, "
     "linear-attention layers 32 value heads; layer types are in `overview`). Only earlier layers can move it. Needs "
     "the live model; errors if it is off.",
     "inputSchema": {"type": "object", "properties": {**LN, "text": {"type": "string"},
                                                       "chat": {"type": "boolean", "default": False},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "all": {"type": "boolean"}, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer"]}},
                                                       "heads": {"type": "array", "items": {"type": "object", "properties": {
                                                           "layer": {"type": "integer"},
                                                           "head": {"type": ["integer", "string"]},
                                                           "dose": {"type": "number", "default": 0}},
                                                           "required": ["layer", "head"]}}},
                     "required": ["layer", "n", "text"]}},
    {"name": "compare", "description": "Live: read many neurons across several texts in one call (one forward pass per "
     "text). Per neuron and text: its strongest positive and negative activation with the token, and peak_score (the "
     "activation over the neuron's usual extreme; 1 = as strong as its usual max). For screening candidates against "
     "target texts and controls; `activations` gives one neuron's full per-token trace and takes pushes for causal "
     "tests. Up to 300 neurons and 24 texts.",
     "inputSchema": {"type": "object", "properties": {
         "neurons": {"type": "array", "items": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
         "texts": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {
             "label": {"type": "string"}, "text": {"type": "string"}}, "required": ["text"]}]}},
         "chat": {"type": "boolean", "default": False}},
                     "required": ["neurons", "texts"]}},
    {"name": "logprobs", "description": "Live: the next token's log-probabilities after a text (chat=true by default: "
     "the text is a user turn and the next token opens the reply; `prefix` continues from the start of a reply or any "
     "text). Returns the top k tokens and the logprob of each `targets` string's first token. With `neurons` and/or "
     "`heads` (same as `activations`: pushes, whole-layer zeroing, head scaling) it runs with and without them and "
     "returns both, the KL from baseline to intervened, and the tokens that gained and lost the most probability "
     "across the whole vocabulary: a direct read of what an intervention does to the output.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "chat": {"type": "boolean", "default": True},
                                                       "prefix": {"type": "string"}, "k": {"type": "integer", "default": 15},
                                                       "targets": {"type": "array", "items": {"type": "string"}},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "all": {"type": "boolean"}, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer"]}},
                                                       "heads": {"type": "array", "items": {"type": "object", "properties": {
                                                           "layer": {"type": "integer"},
                                                           "head": {"type": ["integer", "string"]},
                                                           "dose": {"type": "number", "default": 0}},
                                                           "required": ["layer", "head"]}}},
                     "required": ["text"]}},
    {"name": "top_firing", "description": "Live: run text through the model and list the neurons firing hardest at one "
     "token (the first token containing `token`; omit it for the token with the strongest firing). Each neuron has its "
     "pole, score (activation over its usual extreme; 1 = as strong as its usual max), raw activation and steer lens. "
     "`layers` [lo, hi] restricts the layers. Doesn't change the person's screen; use show_text to show them.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "at": {"type": "integer", "description": "token index into the returned tokens (negative counts from the end); overrides token"}, "token": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 30},
                                                       "layers": {"type": "array", "items": {"type": "integer"}},
                                                       "chat": {"type": "boolean", "default": False}},
                     "required": ["text"]}},
    {"name": "camera", "description": "Move the person's 3D view. mode 'home': the whole model. 'layer': look at one "
     "layer's neuron sheet face-on (give layer). 'neuron': fly close to one neuron (give layer and n; closeness 0.1 = very "
     "close, 0.5 = wide).",
     "inputSchema": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["home", "layer", "neuron"]},
                                                       **LN, "closeness": {"type": "number", "default": 0.22}},
                     "required": ["mode"]}},
    {"name": "annotate_neurons", "description": "Pin short text labels to neurons on the 3D map (they follow the camera). "
     "Replaces the previous labels; an empty list clears them. Keep each label a few words.",
     "inputSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {
         "type": "object", "properties": {**LN, "text": {"type": "string"}}, "required": ["layer", "n", "text"]}}},
                     "required": ["items"]}},
    {"name": "show_text", "description": "Put text in the explorer's paste box and run it through the live model, so the "
     "person sees its tokens; pin the first token containing `token` to show the neurons firing most there and the "
     "direct links among them on the map. Needs the live model.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "show_search", "description": "Run a lens search in the explorer's left panel, so the person sees the "
     "matching neurons listed and can click them.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "open_tab", "description": "Switch the open neuron's panel to a tab: examples (sweep rollouts), chat (the "
     "person's own steered chat) or details (lenses and connections).",
     "inputSchema": {"type": "object", "properties": {"tab": {"type": "string", "enum": ["examples", "chat", "details"]}},
                     "required": ["tab"]}},
    {"name": "select_neuron", "description": "Open a neuron in the explorer: its panel, and its connection web on the map.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "head", "description": "One attention head from weights, by role. q / k / v: the earlier neurons whose "
     "output feeds its query (what it looks for from the current token), its key (what makes an earlier token "
     "findable) and its value (what it carries); o: the later neurons reading its output. Each role has its top neurons "
     "(z, logit lens, name) and their steer-lens tokens pooled. `profile`: where it looks over 8 reference texts: dist is "
     "the mean share of weight on self / previous token / 2-5 back / further / first token, neg the share of negative "
     "weight (linear-attention heads are signed), src the tokens it reads most against chance, pairs its strongest "
     "single reads (query <- source) with the text they come from.",
     "inputSchema": {"type": "object", "properties": HEAD, "required": ["layer", "head"]}},
    {"name": "neuron_heads", "description": "The attention heads a neuron meets, from weights, by role: q / k / v are "
     "later heads whose query, key or value reads its output (it shapes what they look for from its token, makes its "
     "token findable, or is what they carry on); o are heads at its layer or earlier whose output it reads. z per head "
     "(share of the neuron's vector in that head's space, against a random direction, z-scored over heads); grouped "
     "heads share one key or value space.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "attention", "description": "Live: one head's attention on a text. At the query token (first containing "
     "`token`, else the last) the weight on every earlier token, and the strongest single reads anywhere in the text. "
     "Full attention: softmax weights. Linear attention (Gated DeltaNet): the exact effective weights, signed, absolute "
     "values summing to 1. first_token_weight is the mean weight on token 0 (the attention sink).",
     "inputSchema": {"type": "object", "properties": {**HEAD, "text": {"type": "string"}, "at": {"type": "integer", "description": "token index into the returned tokens (negative counts from the end); overrides token"}, "token": {"type": "string"},
                                                       "chat": {"type": "boolean", "default": False}},
                     "required": ["layer", "head", "text"]}},
    {"name": "attn_drive", "description": "Live: which attention heads drive a neuron at one token of a text (first "
     "containing `token`, else where it fires hardest), and which earlier tokens each top head read that from. The "
     "activation silu(gate)*up is linearised at that token; each head's contribution is its output there read by that "
     "linearisation, `share` its fraction of linearised_total (the whole residual's drive, about 2x the activation), "
     "attention_sum all heads together and rest everything else (MLPs, embedding). `from` splits a head's contribution "
     "exactly over source tokens.",
     "inputSchema": {"type": "object", "properties": {**LN, "text": {"type": "string"}, "at": {"type": "integer", "description": "token index into the returned tokens (negative counts from the end); overrides token"}, "token": {"type": "string"},
                                                       "chat": {"type": "boolean", "default": False},
                                                       "top": {"type": "integer", "default": 8}},
                     "required": ["layer", "n", "text"]}},
    {"name": "select_head", "description": "Open an attention head in the explorer: its q / k / v / o panel, and its "
     "links on the map.",
     "inputSchema": {"type": "object", "properties": HEAD, "required": ["layer", "head"]}},
    {"name": "neuronary_search", "description": "Search neuronary.org, the public dictionary of neuron steering recipes. "
     "Free text matches names, descriptions, tags and authors; neuron ids like 'L14 N8949' or 'L14N8949' match recipes "
     "that use that neuron. Use it to see whether a neuron already has known recipes, or before publishing.",
     "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}, "tag": {"type": "string"},
                                                       "author": {"type": "string"},
                                                       "sort": {"type": "string", "enum": ["top", "new", "hot"]},
                                                       "limit": {"type": "integer", "default": 20}}}},
    {"name": "neuronary_get", "description": "Download one neuronary recipe: neurons and doses (as `steer_neurons`, ready "
     "for the `steer` tool's `neurons`: same dose units), recommended sampling, examples, notes.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "neuronary_publish", "description": "Publish a recipe to neuronary.org under the person's account. Only when "
     "the person asks or agrees. Doses are in the same units as `steer` (median residual norm, mode add). Examples must be "
     "real `steer` outputs, verbatim: the prompt, the pushed-up text as output and the unsteered text as base_output. Put "
     "a needed repetition penalty or temperature in sampling. Returns the recipe's URL.",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "short, e.g. 'Evil Qwen'"},
         "description": {"type": "string", "description": "what it does, one or two sentences"},
         "neurons": {"type": "array", "items": {"type": "object", "properties": {**LN, "dose": {"type": "number"}},
                                                "required": ["layer", "n", "dose"]}},
         "tags": {"type": "array", "items": {"type": "string"}},
         "examples": {"type": "array", "items": {"type": "object", "properties": {
             "prompt": {"type": "string"}, "output": {"type": "string"}, "base_output": {"type": "string"}},
             "required": ["prompt", "output"]}},
         "sampling": {"type": "object", "description": "e.g. {repetition_penalty: 1.1, temperature: 0.7}"},
         "mode": {"type": "string", "enum": ["add", "mul"], "default": "add"},
         "notes": {"type": "string"}, "source": {"type": "string"}},
                     "required": ["name", "description", "neurons", "tags", "examples"]}},
    {"name": "neuronary_update", "description": "Edit one of the person's own neuronary recipes: add examples (examples replaces the whole "
     "list), change the neurons or doses (neurons replaces them all; update the examples to match). Only when the person asks.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "name": {"type": "string"},
                                                       "description": {"type": "string"},
                                                       "neurons": {"type": "array", "description": "replaces the recipe's neurons and doses (same units as `steer`)",
                                                                   "items": {"type": "object", "properties": {**LN, "dose": {"type": "number"}},
                                                                             "required": ["layer", "n", "dose"]}},
                                                       "tags": {"type": "array", "items": {"type": "string"}},
                                                       "examples": {"type": "array", "items": {"type": "object"}},
                                                       "sampling": {"type": "object"}, "notes": {"type": "string"}},
                     "required": ["id"]}},
    {"name": "neuronary_comment", "description": "Comment on a neuronary recipe as the person. Only when they ask.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "body": {"type": "string"},
                                                       "parent_id": {"type": "integer"}}, "required": ["id", "body"]}},
    {"name": "neuronary_reproduce", "description": "Report your run of someone else's neuronary recipe as the person: whether "
     "it reproduced (yes / partial / no), the steered output verbatim from `steer`, the prompt, and the setup. Shown on the "
     "recipe and counted; one report per recipe per person, sending again replaces it. Not for the person's own recipes "
     "(add an example with neuronary_update instead). Only when the person asks or agrees.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "output": {"type": "string", "description": "the steered output, verbatim"},
         "prompt": {"type": "string"}, "result": {"type": "string", "enum": ["yes", "partial", "no"], "default": "yes"},
         "setup": {"type": "string", "description": "default: bf16 transformers (neuron explorer), greedy"},
         "note": {"type": "string", "description": "dose changes, caveats, what it took to reproduce"}},
                     "required": ["id", "output"]}},
    {"name": "neuronary_vote", "description": "Vote on a neuronary recipe as the person (1, -1, or 0 to clear). Only when "
     "they ask.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"},
                                                       "value": {"type": "integer", "enum": [-1, 0, 1]}},
                     "required": ["id", "value"]}},
    {"name": "highlight_neurons", "description": "Mark a set of neurons on the 3D map (replaces the previous marks; an "
     "empty list clears them). sign '-' draws a neuron in the negative-pole colour.",
     "inputSchema": {"type": "object", "properties": {"neurons": {"type": "array", "items": {
         "type": "object", "properties": {**LN, "sign": {"type": "string", "enum": ["+", "-"]}},
         "required": ["layer", "n"]}}}, "required": ["neurons"]}},
]


def get(path, **q):
    with urllib.request.urlopen(f"{URL}{path}?{urllib.parse.urlencode(q)}", timeout=60) as r:
        return json.load(r)


def post(path, body, timeout=60):
    req = urllib.request.Request(f"{URL}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def neuronary_token():
    t = os.environ.get("NEURONARY_TOKEN", "").strip()
    if not t:
        try:
            t = open(os.path.expanduser("~/.neuronary/token"), encoding="utf-8").read().strip()
        except OSError:
            t = ""
    return t


def neuronary(method, path, body=None, params=None):
    q = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    url = NEURONARY + path + ("?" + urllib.parse.urlencode(q) if q else "")
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode())
    req.add_header("Content-Type", "application/json")
    tok = neuronary_token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    elif method != "GET":
        raise ValueError("no neuronary token: the person can create one at neuronary.org (account menu -> API & agents) "
                         "and put it in NEURONARY_TOKEN or ~/.neuronary/token")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e).get("error")
        except Exception:
            msg = str(e)
        raise ValueError(f"neuronary {e.code}: {msg}") from None


# ---------- neuronary atlas: a neuron's readouts and interpretations ----------
LN_ = {"layer": {"type": "integer"}, "n": {"type": "integer"}}

ATLAS_TOOLS = [
    {"name": "neuronary_neuron", "description": "What neuronary.org has on one neuron of this model: everyone's "
     "interpretations (text, confidence, who), its readouts, and the recipes that use it.",
     "inputSchema": {"type": "object", "properties": LN_, "required": ["layer", "n"]}},
    {"name": "neuronary_readouts", "description": "Re-send this neuron's readouts to neuronary.org: the explorer's own "
     "readout panels (logit, steer, R, J++ and S lenses, activation range, FineFineWeb contexts with per-token heat, net, "
     "heads), gathered automatically. Usually unnecessary: readouts go up by themselves the first time a neuron is "
     "interpreted or published. Use it when the explorer has gained a readout neuronary doesn't have yet (a new lens): "
     "extra_fields is only for such readouts the automatic gather doesn't cover, as neuronary blocks, never for prose, "
     "arguments or your own tests. only: send just these block names.",
     "inputSchema": {"type": "object", "properties": {
         **LN_, "extra_fields": {"type": "array", "items": {"type": "object"}},
         "only": {"type": "array", "items": {"type": "string"}, "description": "send just these block names"}},
                     "required": ["layer", "n"]}},
    {"name": "neuronary_interpret", "description": "Post (or replace) the person's interpretation of one neuron on "
     "neuronary.org, like a reproduction: text covers both poles, e.g. '(+) ascribes a measured attribute (of density, has "
     "a thickness); (−) copula predication (flow is turbulent)'; confidence 0-1 is how sure, honestly (under 0.3 shows as "
     "questionable). The neuron's readouts are attached automatically if neuronary doesn't have them yet. Only when the "
     "person asks or agrees; show the text first.",
     "inputSchema": {"type": "object", "properties": {
         **LN_, "text": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
         "method": {"type": "string", "description": "how you got it, e.g. 'lens + 12 steered rollouts at ±0.8'"}},
                     "required": ["layer", "n", "text", "confidence"]}},
]


def _toks(xs, top=12):
    """lens lists come as strings, [tok, value] pairs or {tok|token, ...} dicts; neuronary takes tok or [tok, value]"""
    out = []
    for x in (xs or [])[:top]:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, (list, tuple)) and x:
            out.append([str(x[0]), round(float(x[1]), 3)] if len(x) > 1 and isinstance(x[1], (int, float)) else str(x[0]))
        elif isinstance(x, dict):
            t = x.get("tok", x.get("token", x.get("t")))
            v = x.get("w", x.get("n", x.get("z", x.get("p"))))
            if t is not None:
                out.append([str(t), round(float(v), 3)] if isinstance(v, (int, float)) else str(t))
    return out


def _head_label(h):
    if isinstance(h, str):
        return h
    if isinstance(h, dict):
        lab = h.get("label") or f"L{h.get('layer')} h{h.get('head', h.get('heads'))}"
        v = h.get("z", h.get("score", h.get("w")))
        return [lab, round(float(v), 2)] if isinstance(v, (int, float)) else lab
    if isinstance(h, (list, tuple)) and h:
        return [str(h[0]), round(float(h[1]), 2)] if len(h) > 1 and isinstance(h[1], (int, float)) else str(h[0])
    return str(h)


def _readouts(layer, n, trace=True):
    fields = []
    try:
        d = get("/api/agent/neuron", layer=layer, n=n)
        if d.get("lens_up") or d.get("lens_down"):
            fields.append({"type": "tokens", "name": "logit lens", "rows": {"+": _toks(d.get("lens_up")), "-": _toks(d.get("lens_down"))}})
        if d.get("steer_up") or d.get("steer_down"):
            fields.append({"type": "tokens", "name": "steer lens", "rows": {"+": _toks(d.get("steer_up")), "-": _toks(d.get("steer_down"))}})
        rj = d.get("rjpp") or {}
        for name, key in (("R lens", "r"), ("J++ lens", "jpp")):
            if rj.get(f"{key}_up"):
                fields.append({"type": "tokens", "name": name, "rows": {"+": _toks(rj[f"{key}_up"]), "-": _toks(rj[f"{key}_down"])}})
        sl = d.get("s_lens") or {}
        if sl.get("s_up"):
            fields.append({"type": "tokens", "name": "S lens", "rows": {"+": _toks(sl["s_up"]), "-": _toks(sl["s_down"])}})
        act = d.get("activation")
        if act:
            items = [["max", act.get("act_max")], ["min", act.get("act_min")]]
            items += [[k, act[k]] for k in ("mean", "std") if act.get(k) is not None]
            fields.append({"type": "stats", "name": "activation range", "items": items + [["over tokens", act.get("over_tokens")], ["source", act.get("source")]]})
    except Exception:
        pass
    try:
        w = get("/api/web/summary", layer=layer, n=n)
        up, down = w.get("up", {}), w.get("down", {})
        groups = [g for g in [
            {"label": "fed by", "sign": "+", "note": f"{up.get('neurons', 0)} upstream neurons", "items": [[x["tok"], x.get("n")] for x in up.get("fed_by", [])[:24]]},
            {"label": "held back by", "sign": "-", "items": [[x["tok"], x.get("n")] for x in up.get("held_by", [])[:24]]},
            {"label": "feeds → pushes toward", "sign": "+", "note": f"{down.get('neurons', 0)} downstream neurons", "items": [[x["tok"], x.get("n")] for x in down.get("toward", [])[:24]]},
            {"label": "feeds → pushes away from", "sign": "-", "items": [[x["tok"], x.get("n")] for x in down.get("away", [])[:24]]},
        ] if g["items"]]
        if groups:
            fields.append({"type": "units", "name": "net", "note": f"{w.get('links', 0)} links · {w.get('src', 'steer')} lens, ranked by lift over each layer", "groups": groups})
    except Exception:
        pass
    try:
        h = get("/api/neuron/heads", layer=layer, n=n).get("roles") or {}
        rows = {k: [_head_label(x) for x in (h.get(k) or [])[:8]] for k in ("q", "k", "v", "o")}
        if any(rows.values()):
            fields.append({"type": "heads", "name": "heads", "rows": rows})
    except Exception:
        pass
    try:
        fields += _contexts(layer, n, trace)
    except Exception:
        pass
    return fields


def _context_text(r):
    """A FineFineWeb window as neuronary context text: the peak as ⟦tok⟧, other tokens of the same sign at a third of
    the peak or more as ⟦tok|heat⟧ (heat = activation / peak, from the live trace)."""
    out, peak = [], r["act"]
    for i, t in enumerate(r["tokens"]):
        t = t.replace("\n", "⏎").replace("⟦", "[").replace("⟧", "]").replace("|", "¦")
        h = 1.0 if i == r["peak"] else (r["trace"][i] / peak if r.get("trace") and peak else 0)
        if i == r["peak"]:
            out.append(f"⟦{t}⟧")
        elif h >= 0.33:
            out.append(f"⟦{t}|{min(h, 1):.2f}⟧")
        else:
            out.append(t)
    return "".join(out)


def _contexts(layer, n, trace=True):
    """The neuron's strongest FineFineWeb contexts per sign (8 and 5, one per document) and the domains each pole
    concentrates in, as one neuronary contexts block; with trace, each window's other strong tokens marked by heat."""
    e = get("/api/exemplars", layer=layer, n=n, k=8, trace="true" if trace else "false")
    if not e or not e.get("+"):
        return []
    rows = {s: [{"text": _context_text(r), "tag": r["domain"].replace("_", " "), "value": round(r["act"], 3)} for r in e[s][:k]]
            for s, k in (("+", 8), ("-", 5))}
    tags = {s: [[d["domain"].replace("_", " "), d["lift"]] for d in e.get(f"domains{s}", []) if d["lift"] >= 1.25][:6] for s in "+-"}
    note = f"FineFineWeb, {e['tokens'] / 1e6:.0f}M tokens; tags: mean activation per token in the domain over the mean"
    return [{"type": "contexts", "name": "activations", "note": note, "tags": tags, "rows": rows}]


def _neuron(a):
    return neuronary("GET", f"/neurons/L{int(a['layer'])}N{int(a['n'])}", params={"model": MODEL})  


def ensure_readouts(layer, n, trace=True):
    """Send the neuron's readouts unless neuronary already has some; returns the block names sent (or [])."""
    try:
        have = neuronary("GET", f"/neurons/L{layer}N{n}", params={"model": MODEL}).get("readouts") or []
        if have:
            return []
        return _push_readouts({"layer": layer, "n": n}, trace).get("blocks", [])
    except Exception:
        return []


def _push_readouts(a, trace=True):
    layer, n = int(a["layer"]), int(a["n"])
    fields = _readouts(layer, n, trace) + list(a.get("extra_fields") or [])
    if a.get("only"):
        keep = {x.lower() for x in a["only"]}
        fields = [f for f in fields if str(f.get("name", "")).lower() in keep]
    if not fields:
        return {"written": 0, "note": "no readouts to send"}
    r = neuronary("POST", "/readouts", {"model": MODEL, "layer": layer, "neuron": n, "source": "neuron explorer", "fields": fields})
    r["blocks"] = [f.get("name") for f in fields]
    r["url"] = f"https://neuronary.org/n/L{layer}N{n}"
    return r


def _interpret(a):
    layer, n = int(a["layer"]), int(a["n"])
    body = {"model": MODEL, "layer": layer, "neuron": n, "text": a["text"], "confidence": float(a["confidence"]),
            "source": "neuron explorer"}
    if a.get("method"):
        body["method"] = a["method"]
    r = neuronary("POST", "/interpretations", body)
    r["readouts_sent"] = ensure_readouts(layer, n)
    r["url"] = f"https://neuronary.org/n/L{layer}N{n}"
    return r


ATLAS_HANDLERS = {"neuronary_neuron": _neuron, "neuronary_readouts": _push_readouts, "neuronary_interpret": _interpret}

TOOLS += ATLAS_TOOLS


def to_explorer(ns):
    return [{"layer": x["layer"], "n": x["neuron"], "dose": x["dose"]} for x in ns]


def call(name, a):
    if name in ATLAS_HANDLERS:
        return ATLAS_HANDLERS[name](a)
    if name == "neuronary_search":
        r = neuronary("GET", "/recipes", params={"q": a.get("q"), "tag": a.get("tag"), "author": a.get("author"),
                                                 "sort": a.get("sort"), "limit": a.get("limit", 20)})
        for x in r["recipes"]:
            x["steer_neurons"] = to_explorer(x.pop("neurons"))
        return r
    if name == "neuronary_get":
        r = neuronary("GET", f"/recipes/{a['id']}")
        r["steer_neurons"] = to_explorer(r.pop("neurons"))
        return r
    if name == "neuronary_publish":
        body = {k: a[k] for k in ("name", "description", "tags", "examples", "sampling", "notes", "source") if a.get(k) is not None}
        body["model"] = MODEL
        body["neurons"] = [{"layer": int(x["layer"]), "neuron": int(x["n"]), "dose": float(x["dose"])} for x in a["neurons"]]
        if a.get("mode") == "mul":
            body["method"] = {"op": "mul"}
        r = neuronary("POST", "/recipes", body)
        # the recipe's neurons carry their readouts up with it (peaks only: a live trace per neuron is too slow here)
        r["readouts_sent"] = {f"L{x['layer']}N{x['neuron']}": ensure_readouts(x["layer"], x["neuron"], trace=False)
                              for x in body["neurons"]}
        return r
    if name == "neuronary_update":
        body = {k: v for k, v in a.items() if k not in ("id", "neurons")}
        if a.get("neurons"):
            body["neurons"] = [{"layer": int(x["layer"]), "neuron": int(x["n"]), "dose": float(x["dose"])} for x in a["neurons"]]
        return neuronary("PATCH", f"/recipes/{a['id']}", body)
    if name == "neuronary_comment":
        return neuronary("POST", f"/recipes/{a['id']}/comments", {"body": a["body"], "parent_id": a.get("parent_id")})
    if name == "neuronary_reproduce":
        body = {"output": a["output"], "result": a.get("result", "yes"),
                "setup": a.get("setup") or "bf16 transformers (neuron explorer), greedy"}
        body.update({k: a[k] for k in ("prompt", "note") if a.get(k)})
        return neuronary("POST", f"/recipes/{a['id']}/reproductions", body)
    if name == "neuronary_vote":
        return neuronary("POST", f"/recipes/{a['id']}/vote", {"value": a["value"]})

    if name == "overview":
        return get("/api/agent/overview")
    if name == "neuron":
        return get("/api/agent/neuron", layer=a["layer"], n=a["n"])
    if name == "exemplars":
        r = get("/api/exemplars", layer=a["layer"], n=a["n"], k=a.get("k", 8), trace="true")
        if not r:
            return {"error": "no FineFineWeb sweep in explorer/cache/ffw"}
        def ctx(x):
            toks, p = x["tokens"], x["peak"]
            text = "".join(toks[:p]) + "[[" + toks[p] + "]]" + "".join(toks[p + 1:])
            hot = [[t, v] for i, (t, v) in enumerate(zip(toks, x.get("trace") or [])) if i != p and v * x["act"] > 0 and abs(v) >= 0.5 * abs(x["act"])]
            return {"text": text, "act": x["act"], "domain": x["domain"], "also": hot}
        dom = lambda lst: [f"{d['domain']} {d['lift']}" for d in lst[:6]]
        return {"range": r["range"], "positive": [ctx(x) for x in r["+"]], "negative": [ctx(x) for x in r["-"]],
                "domains_positive": dom(r["domains+"]), "domains_negative": dom(r["domains-"])}
    if name == "rollouts":
        u = get("/api/unit", layer=a["layer"], n=a["n"])
        cut = int(a.get("max_chars", 500))
        for key in ("rollouts", "baselines"):
            for r in u.get(key, []):
                r["text"] = r["text"][:cut]
        return u
    if name == "cone":
        return get("/api/agent/cone", layer=a["layer"], n=a["n"], k=a.get("k", 2), limit=a.get("limit", 120))
    if name == "web_summary":
        r = get("/api/web/summary", layer=a["layer"], n=a["n"], first=a.get("first", 5), k=a.get("k", 2), top=20)
        slim = lambda lst: [f"{t['tok']} ({t['n']})" for t in lst]
        return {"lens": r["src"], "links": r["links"],
                "downstream_neurons": r["down"]["neurons"], "toward": slim(r["down"]["toward"]), "away": slim(r["down"]["away"]),
                "upstream_neurons": r["up"]["neurons"], "fed_by": slim(r["up"]["fed_by"]), "held_by": slim(r["up"]["held_by"])}
    if name == "search_lens":
        r = get("/api/search", q=a["query"], limit=a.get("limit", 40))
        return {"tokens_matched": r.get("tokens"), "total": r.get("total"), "results": r.get("results", [])}
    if name == "top_units":
        us = sorted((u for u in get("/api/units", layer=a["layer"]) if u["kind"] == "neuron"), key=lambda u: -u["z"])
        return us[: int(a.get("limit", 30))]
    if name == "set_name":
        post("/api/name", {"layer": a["layer"], "n": a["n"], "name": a["name"]})
        post("/api/agent/ui", {"chat": CHAT, "action": "rename", "layer": a["layer"], "n": a["n"], "name": a["name"]})
        return {"ok": True, "neuron": f"L{a['layer']} N{a['n']}", "name": a["name"].strip() or None}
    if name == "names":
        return get("/api/agent/names")
    if name == "head":
        r = get("/api/head", layer=a["layer"], head=a["head"])
        slim = lambda lst: [{"neuron": f"L{c['layer']} N{c['n']}", "z": c["z"], "lens": c.get("lens", [])} for c in lst[:12]]
        toks = lambda lst: [f"{t['tok']} ({t['n']})" for t in lst[:16]]
        out = {"head": f"L{a['layer']} h{a['head']}", "kind": "full attention" if r["full"] else "linear attention",
               "q": {"neurons": slim(r["q"]), "tokens": toks(r["q_toks"])},
               "k": {"neurons": slim(r["k"]), "tokens": toks(r["k_toks"])},
               "v": {"neurons": slim(r["reads"]), "tokens": toks(r["read_toks"])},
               "o": {"neurons": slim(r["writes"]), "tokens": toks(r["write_toks"])}}
        try:
            p = get("/api/head/profile", layer=a["layer"], head=a["head"])
            if "dist" in p:
                out["profile"] = {"dist": dict(zip(["self", "prev", "2-5 back", "further", "first"], p["dist"])),
                                  "neg": p["neg"], "src": p["src"],
                                  "pairs": [{"query": x["qt"], "source": x["kt"], "weight": x["w"], "text": x["text"][:160]}
                                            for x in p["pairs"]]}
        except Exception:
            pass
        return out
    if name == "neuron_heads":
        r = get("/api/neuron/heads", layer=a["layer"], n=a["n"])
        return {role: [{"head": f"L{x['layer']} h{x['heads'][0]}" + (f"-{x['heads'][-1]}" if len(x["heads"]) > 1 else ""), "z": x["z"]}
                       for x in rows] for role, rows in r["roles"].items()}
    if name == "attention":
        return post("/api/agent/attention", a, timeout=900)
    if name == "attn_drive":
        return post("/api/agent/attn_drive", a, timeout=900)
    if name == "select_head":
        return post("/api/agent/ui", {"chat": CHAT, "action": "head", "layer": a["layer"], "head": a["head"]})
    if name == "steer":
        return post("/api/agent/steer", a, timeout=900)  # the first call may wait for the model to load
    if name == "top_firing":
        return post("/api/agent/fires", a, timeout=900)
    if name == "logprobs":
        return post("/api/agent/logprobs", a, timeout=900)
    if name == "compare":
        return post("/api/agent/compare", a, timeout=900)
    if name == "activations":
        return post("/api/agent/activations", a, timeout=900)
    ui = {"camera": "camera", "annotate_neurons": "annotate", "show_text": "text", "show_search": "search", "open_tab": "tab"}
    if name in ui:
        return post("/api/agent/ui", {"chat": CHAT, "action": ui[name], **a})
    if name == "select_neuron":
        return post("/api/agent/ui", {"chat": CHAT, "action": "select", "layer": a["layer"], "n": a["n"]})
    if name == "highlight_neurons":
        return post("/api/agent/ui", {"chat": CHAT, "action": "highlight", "neurons": a["neurons"]})
    raise ValueError(f"unknown tool {name}")


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def serve(tools=None, handler=None, name="explorer", instructions=None):
    """the stdio JSON-RPC loop; explorer/mcp.py reuses it with a smaller tool list and an in-process backend"""
    tools, handler = tools or TOOLS, handler or call
    for line in sys.stdin:
        if not line.strip():
            continue
        m = json.loads(line)
        method, id_ = m.get("method"), m.get("id")
        if id_ is None:
            continue  # notifications
        if method == "initialize":
            info = {"protocolVersion": m["params"].get("protocolVersion", "2025-06-18"),
                    "capabilities": {"tools": {}}, "serverInfo": {"name": name, "version": "0.1"}}
            if instructions:
                info["instructions"] = instructions
            reply(id_, info)
        elif method == "tools/list":
            reply(id_, {"tools": tools})
        elif method == "tools/call":
            p = m["params"]
            try:
                out = handler(p["name"], p.get("arguments") or {})
                reply(id_, {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]})
            except Exception as e:  # tool errors go back to the model, not the protocol
                reply(id_, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        elif method == "ping":
            reply(id_, {})
        else:
            reply(id_, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    serve()
