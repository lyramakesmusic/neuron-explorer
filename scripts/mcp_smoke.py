"""Drive explorer/mcp.py over stdio like an MCP client: initialize, list tools, call a few. Cache-only by default.

    EXPLORER_NO_MODEL=1 uv run python scripts/mcp_smoke.py
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = subprocess.Popen([sys.executable, os.path.join(ROOT, "explorer", "mcp.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env={**os.environ, "EXPLORER_NO_MODEL": os.environ.get("EXPLORER_NO_MODEL", "1")})
n = 0


def rpc(method, params=None):
    global n
    n += 1
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": n, "method": method, "params": params or {}}) + "\n")
    p.stdin.flush()
    return json.loads(p.stdout.readline())


def tool(name, **a):
    t = time.time()
    r = rpc("tools/call", {"name": name, "arguments": a})["result"]
    text = r["content"][0]["text"]
    print(f"{'ERR ' if r.get('isError') else 'ok  '}{name:16s} {time.time() - t:5.1f}s  {text[:160]}")
    return r


init = rpc("initialize", {"protocolVersion": "2025-06-18"})["result"]
print("server:", init["serverInfo"], "| instructions:", bool(init.get("instructions")))
tools = [t["name"] for t in rpc("tools/list")["result"]["tools"]]
print(len(tools), "tools:", ", ".join(tools))
assert not {"camera", "highlight_neurons", "select_neuron"} & set(tools), "UI tools leaked"
tool("overview")
tool("neuron", layer=17, n=6555)
tool("search_lens", query="average", limit=5)
tool("cone", layer=17, n=6555, limit=10)
tool("web_summary", layer=17, n=6555)
tool("neuron_heads", layer=17, n=6555)
tool("exemplars", layer=17, n=6555, k=3)
tool("steer", layer=17, n=6555, prompt="hi", dose=0.8)
tool("camera", mode="home")
tool("neuronary_search", q="jesus", limit=2)
p.stdin.close()
p.wait(timeout=30)
