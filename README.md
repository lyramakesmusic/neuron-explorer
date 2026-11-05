# neuron explorer

Steering-first neuron explorer for the Qwen3.5 family. Every MLP neuron in a layer is pushed up and down on a few
prompts, every rollout is kept, and the explorer lays the results out over a 3D view of the model.

This is a fork of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), which targets
Qwen3.5-4B. The fork runs the live model on a 6 GB GPU and on Windows. It uses **Qwen3.5-2B** by default, can switch to
0.8B or 4B, and keeps a separate cache for each model. See [Changes in this fork](#changes-in-this-fork).

![neuron explorer](explorer.png)

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

`EXPLORER_MODEL` picks the model: `Qwen/Qwen3.5-2B` (default), `Qwen/Qwen3.5-0.8B` or `Qwen/Qwen3.5-4B`. It accepts a
Hub id or a local folder, and weights in `models/<name>/` are used when present. Approximate bf16 sizes: 0.8B about
1.7 GB, 2B about 4.5 GB, 4B about 9.3 GB. The vision tower is kept in RAM, so on a 6 GB card the 2B uses about 3.6 GB
of VRAM at peak.

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below. Loads the model
locally (MPS, CUDA or CPU) for live steered chat, lens, connections and live inference over your own text.

**Caches are per model.** The Qwen3.5-4B caches ship in `explorer/cache/`, and the Qwen3.5-2B caches ship in
`explorer/cache/models/qwen3.5-2b/`. Any other model gets its own `explorer/cache/models/<name>/`. On first start the
server builds the logit-lens index and activation ranges. The steer lens, connections and map layout come from the
build scripts:

    uv run python sweep/steer_lens.py                 # stats/steer_lens_<model>_L{l}.npz, every layer
    uv run python explorer/build_steer_index.py       # -> steer_index.npz
    uv run python explorer/build_connections.py       # -> connections.npz
    uv run python explorer/build_layout.py --kind writes

On an RTX 3050 6 GB laptop GPU the full 2B build takes about two hours, most of it the steer lens (about 4 min per
layer). The R, J++ and S lens rows exist only for the 4B, because their maps were fitted on that model. Neuron names
are kept per model: `explorer/names.json` for the 4B and `explorer/names.<model>.json` for the others.

### Windows

    winget install astral-sh.uv
    uv sync
    .\run_windows.ps1                  # Qwen3.5-2B; also: .\run_windows.ps1 0.8B, .\run_windows.ps1 4B -NoModel
    .\build_cache_windows.ps1          # builds the caches above, skipping finished steps

On Windows and Linux, `pyproject.toml` installs the CUDA 13.0 build of torch, plus `flash-linear-attention` and
`triton-windows` for the linear-attention layers. These kernels make decode about 3.5× faster than the PyTorch
fallback. The first run compiles them, which takes about a minute. `run_windows.ps1` sets `PYTHONUTF8=1`, because the
vocab and prompts hold non-Latin text and Windows Python otherwise reads files as cp1250.

Behind a VPN that inspects TLS (for example Cloudflare WARP), Python HTTPS uses the Windows certificate store through
`truststore`, and the `hf_xet` downloader is turned off because it stalls there. If a Hub download still hangs, fetch
the weights into `models/<name>/` with `curl`.

The R and J++ lens rows read the maps of [bubskee/qwen3.5-4b-r-jpp-lenses](https://huggingface.co/bubskee/qwen3.5-4b-r-jpp-lenses)
(Apache-2.0):

    hf download bubskee/qwen3.5-4b-r-jpp-lenses --local-dir explorer/cache/rjpp

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and R lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer, R and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, a name, and where it fires on web text: its strongest FineFineWeb contexts per sign, each token shaded
  by the neuron's activation, and the domains it concentrates in.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **language**: the interface follows the browser language on first visit; the menu at the right of the header
  switches between English and Simplified Chinese (the choice is kept in this browser). Translations live in
  `explorer/static/i18n.js`.
- **Claude** (fourth neuron-panel tab, or the right-edge handle): a chat running the `claude` CLI (installed and
  logged in) with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine. On Windows, put
  the CLI on `PATH` (the native installer puts it in `%USERPROFILE%\.local\bin`). A running chat is stopped with
  `taskkill /T`. It uses the `claude` login and counts against that account's usage.

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_<model>_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron (plain transformers, no vLLM needed) |
| `explorer/build_steer_index.py` | `steer_index.npz` in the model's cache folder: the steer-lens layers joined into the explorer's index |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `layout_writes.npz` in the model's cache folder (sheets that aren't square, e.g. 6144 neurons, leave the last row partly empty) |
| `explorer/build_connections.py` | `connections.npz` in the model's cache folder: each neuron's 24 strongest direct-path links up and down |
| `sweep/fit_input_lens.py` | `S.pt`: the fitted input map S_L (token embeddings to the residual each MLP reads, layers 0–31), RelP rules of safety-research/jpp_lens; shards over prompts, merged by `num_prompts`; copy to `explorer/cache/slens/S.pt` |
| `explorer/build_s_index.py` | `explorer/cache/s_index.npz`: each neuron's top S-lens (read side) tokens, both poles |
| `explorer/build_rjpp_index.py` | `explorer/cache/r_index.npz`, `jpp_index.npz`: each neuron's top R- and J++-lens tokens, both poles (needs the R/J++ release in `explorer/cache/rjpp/`) |
| `sweep/finefineweb.py` | `<out>/corpus.npz`, `urls.json`, `acts.npz`: a domain-balanced FineFineWeb sample (about 20M tokens over 66 domains) and every neuron's top contexts per sign and per-domain activation statistics; copy the three files to `explorer/cache/ffw/` |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes. Both describe Qwen3.5-4B, so the neuron
ids in them name different neurons in the smaller models.

## Changes in this fork

- **Model choice and per-model caches.** The new `explorer/paths.py` reads `EXPLORER_MODEL` (default Qwen3.5-2B), finds
  local weights in `models/`, and points every cache read and write at the model's own folder. Each model also gets
  its own neuron names. The explorer's caches for Qwen3.5-2B ship in `explorer/cache/models/qwen3.5-2b/`.
- **Small GPUs.** The vision tower loads to RAM, not VRAM.
- **Steer lens without vLLM.** `sweep/steer_lens.py` defaults to the chosen model and all of its layers.
  `explorer/build_steer_index.py` is new, and `build_layout.py` handles sheets that aren't square.
- **Windows.**
  - CUDA torch on Windows, with Triton and flash-linear-attention kernels.
  - `run_windows.ps1` and `build_cache_windows.ps1`.
  - `truststore`, with `hf_xet` turned off, for TLS-inspecting VPNs.
  - The Claude chat finds `claude.exe`, stops chats with `taskkill`, and finds its session folder.
- **Agent prompt.** It fills in the running model's name and size, and says that the reference neurons in the
  methodology come from Qwen3.5-4B.

## License

MIT, see [LICENSE](LICENSE). The original work is copyright (c) 2026 lyramakesmusic
([lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)), and the LICENSE file keeps that
notice unchanged. The changes in this fork are released under the same MIT license. The R/J++ lens maps
([bubskee/qwen3.5-4b-r-jpp-lenses](https://huggingface.co/bubskee/qwen3.5-4b-r-jpp-lenses)) are Apache-2.0. The Qwen3.5
weights are not part of this repository and carry their own license.
