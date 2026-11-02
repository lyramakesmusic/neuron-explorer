"""The neuron explorer as a plain MCP server: every neuron tool the explorer's Claude chat has (lenses, cones, webs,
heads, steering, activations, logprobs, FineFineWeb exemplars, sweep rollouts, neuronary), with no browser, no UI and
no HTTP server. The explorer's FastAPI app runs in this process and the tools call it directly.

    claude mcp add neuron-explorer -- uv run --directory /path/to/neuron-explorer python explorer/mcp.py

EXPLORER_NO_MODEL=1 serves the cached data only (lenses, cones, webs, heads, search) without downloading or loading
Qwen3.5-4B; the live tools then say so. Otherwise the model loads on first use (MPS, CUDA or CPU), as in the explorer.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# stdout carries the protocol: anything the explorer prints (model loading, warnings) goes to stderr instead
PROTOCOL = sys.stdout
sys.stdout = sys.stderr

import agent_mcp  # noqa: E402  (tool definitions, tool logic and the stdio loop)
from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402  (the explorer's app: endpoints, caches, runtime)

# tools that move the person's map, list or panels: meaningless without the browser
UI_TOOLS = {"camera", "annotate_neurons", "show_text", "show_search", "open_tab", "select_neuron", "select_head",
            "highlight_neurons"}

client = TestClient(server.app, base_url=f"http://127.0.0.1:{server.PORT}")  # the app only answers local Host headers
client.__enter__()  # runs the app's startup hooks (model preload unless EXPLORER_NO_MODEL)


def get(path, **q):
    r = client.get(path, params=q)
    r.raise_for_status()
    return r.json()


def post(path, body, timeout=60):
    r = client.post(path, json=body)
    r.raise_for_status()
    return r.json()


agent_mcp.get, agent_mcp.post = get, post


def reply(id_, result=None, error=None):
    import json
    msg = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    PROTOCOL.write(json.dumps(msg) + "\n")
    PROTOCOL.flush()


agent_mcp.reply = reply


def handle(name, a):
    if name in UI_TOOLS:
        raise ValueError(f"{name} drives the explorer's browser view, which this server doesn't have")
    if name == "set_name":  # keep the name, skip the browser update
        post("/api/name", {"layer": a["layer"], "n": a["n"], "name": a["name"]})
        return {"ok": True}
    return agent_mcp.call(name, a)


TOOLS = [t for t in agent_mcp.TOOLS if t["name"] not in UI_TOOLS]

INSTRUCTIONS = (
    "Neuron explorer for Qwen/Qwen3.5-4B (32 layers x 9216 MLP neurons). Neurons are (layer, n). Start with `neuron` "
    "(lenses for both poles: steer, R / J++, S, logit; links each way), `exemplars` (where it fires on web text) and "
    "`steer` (pushed up / unsteered / pushed down rollouts) before you name what a neuron does: lenses suggest, steering "
    "and exemplars show. Doses are in median-residual units: one neuron is usually coherent around 0.5-1.0. "
    "A negative push is not the opposite behavior. neuronary_* tools read and publish to neuronary.org."
)

if __name__ == "__main__":
    agent_mcp.serve(tools=TOOLS, handler=handle, name="neuron-explorer", instructions=INSTRUCTIONS)
