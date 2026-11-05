"""Which model the explorer runs and where its per-model data lives.

EXPLORER_MODEL picks the model (default Qwen/Qwen3.5-2B, which fits a 6 GB GPU in bf16). The shipped caches in
explorer/cache/ belong to Qwen3.5-4B; every other model gets its own folder, explorer/cache/models/<name>/, built on
first use (lens index, act scale) or by the build scripts (steer index, connections, layout). Files that don't depend
on the model's weights (the tokenizer's vocab.json, shared by all Qwen3.5 sizes) are copied from the 4B cache.
"""
import os
import re
import shutil

try:  # the OS certificate store for HTTPS (Hugging Face downloads behind TLS-inspecting VPNs such as Cloudflare WARP)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
# hf_xet (Rust) has its own TLS stack that ignores truststore and stalls behind such VPNs; plain HTTPS works
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-2B")  # a Hub id or a local folder (models/Qwen3.5-2B)
_base = re.split(r"[\\/]", MODEL.rstrip("/\\"))[-1]
if os.path.isdir(MODEL) or not os.path.isdir(os.path.join(ROOT, "models", _base)):
    pass
else:  # weights fetched into models/<name> (run_windows.ps1 downloads there) win over the Hub cache
    MODEL = os.path.join(ROOT, "models", _base)
NAME = _base.lower()                                      # "qwen3.5-2b"
MODEL_ID = f"Qwen/{_base}"                                # the Hub id, for display and neuronary.org
BASE_CACHE = os.path.join(HERE, "cache")
IS_4B = NAME == "qwen3.5-4b"
CACHE = BASE_CACHE if IS_4B else os.path.join(BASE_CACHE, "models", NAME)
STATS_PREFIX = NAME.replace("qwen3.5-", "q35-")          # stats/q35-2b_L{l}.npz, as the sweep names them

os.makedirs(CACHE, exist_ok=True)
for _shared in ("vocab.json",):
    _src, _dst = os.path.join(BASE_CACHE, _shared), os.path.join(CACHE, _shared)
    if not IS_4B and os.path.exists(_src) and not os.path.exists(_dst):
        shutil.copyfile(_src, _dst)
