"""Update this checkout from the author's main branch.

The explorer runs from a git checkout in the normal case, so updates use git:
fetch `lyramakesmusic/neuron-explorer` main into a private ref, then fast-forward
only when the working tree is clean and the histories allow it.  Data directories
(`runs/`, `stats/`, `captures/`, caches, names) are gitignored and are never
touched.

The server exposes the same two operations through /api/update/check and
/api/update/apply; applying returns `restart_required` and the page can call
/api/update/restart to re-exec the server.
"""

import os
import shutil
import subprocess

ROOT = os.path.abspath(os.environ.get("EXPLORER_ROOT") or os.path.join(os.path.dirname(__file__), ".."))
UPSTREAM = os.environ.get("EXPLORER_UPSTREAM", "https://github.com/lyramakesmusic/neuron-explorer.git")
BRANCH = os.environ.get("EXPLORER_UPSTREAM_BRANCH", "main")
REF = "refs/remotes/neuron-upstream/" + BRANCH


class UpdateError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _run(args, timeout=180, check=True):
    try:
        p = subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise UpdateError("update.git_missing", "git is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        raise UpdateError("update.timeout", "git timed out after %s seconds" % timeout)
    if check and p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-4:]
        raise UpdateError("update.git_failed", " / ".join(tail) or "git exited %s" % p.returncode)
    return p


def _sha(rev):
    return _run(["rev-parse", rev]).stdout.strip()


def _is_ancestor(a, b):
    return _run(["merge-base", "--is-ancestor", a, b], check=False).returncode == 0


def _dirty():
    out = _run(["status", "--porcelain=v1", "--untracked-files=no"], check=False).stdout
    return [line[3:] for line in out.splitlines() if line.strip()][:20]


def _commit_count(current, latest):
    out = _run(["rev-list", "--count", "%s..%s" % (current, latest)]).stdout.strip()
    return int(out) if out.isdigit() else 0


def _repo():
    if not os.path.isdir(os.path.join(ROOT, ".git")) and not os.path.isfile(os.path.join(ROOT, ".git")):
        raise UpdateError("update.not_git",
                          "this explorer is not a git checkout; update it the way it was installed")
    top = _run(["rev-parse", "--show-toplevel"]).stdout.strip()
    if os.path.realpath(top) != os.path.realpath(ROOT):
        raise UpdateError("update.not_git", "the git checkout is at %s, not %s" % (top, ROOT))


def check(fetch=True):
    """Return a JSON-ready status.  Never modifies the working tree."""
    _repo()
    if fetch:
        try:
            _run(["fetch", "--quiet", "--no-tags", UPSTREAM, "+refs/heads/%s:%s" % (BRANCH, REF)], timeout=60)
        except UpdateError as e:
            if e.code == "update.timeout":
                raise
            raise UpdateError("update.fetch_failed", str(e))
    current = _sha("HEAD")
    try:
        latest = _sha(REF)
    except UpdateError:
        raise UpdateError("update.fetch_failed", "upstream %s/%s was not fetched" % (UPSTREAM, BRANCH))
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"], check=False).stdout.strip() or "HEAD"
    dirty = _dirty()
    if current == latest:
        return {"ok": True, "available": False, "current": current, "latest": latest,
                "branch": branch, "dirty": dirty, "ahead": False, "fast_forward": False, "count": 0}
    fast_forward = _is_ancestor("HEAD", REF)
    ahead = _is_ancestor(REF, "HEAD")
    return {
        "ok": True,
        "available": not ahead,
        "current": current,
        "latest": latest,
        "branch": branch,
        "dirty": dirty,
        "ahead": ahead,
        "fast_forward": fast_forward,
        "count": _commit_count(current, latest),
    }


def _sync_deps():
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        return False, "uv is not installed; install dependencies with `uv sync` before restarting"
    lock = os.path.join(ROOT, "uv.lock")
    args = [uv, "sync", "--project", ROOT, "--no-progress"]
    if os.path.isfile(lock):
        args.append("--frozen")
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return False, "uv sync timed out after 900 seconds"
    except OSError as e:
        return False, "could not run uv: %s" % e
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-4:]
        return False, "uv sync failed: " + (" / ".join(tail) or "exit %s" % p.returncode)
    return True, ""


def apply_update():
    """Fast-forward to upstream and reconcile dependencies.  Refuses to touch a dirty tree."""
    info = check()
    if not info["available"]:
        return {"ok": True, "up_to_date": True, "commit": info["current"], "restart_required": False}
    if info["dirty"]:
        raise UpdateError("update.dirty",
                          "the working tree has changes; commit or stash them before updating: " +
                          ", ".join(info["dirty"][:5]))
    if not info["fast_forward"]:
        raise UpdateError("update.diverged",
                          "local commits diverged from %s/%s; merge or rebase manually" % (UPSTREAM, BRANCH))
    p = _run(["merge", "--ff-only", REF], timeout=180, check=False)
    if p.returncode != 0:
        raise UpdateError("update.apply_failed", (p.stderr or p.stdout).strip()[-800:])
    commit = _sha("HEAD")
    deps_ok, deps_error = _sync_deps()
    if not deps_ok:
        # Keep the old code runnable: restore the pre-update commit and report the dependency failure.
        _run(["reset", "--hard", info["current"]], check=False)
        raise UpdateError("update.deps_failed",
                          deps_error or "dependency sync failed; the update was rolled back")
    return {"ok": True, "up_to_date": False, "commit": commit, "previous": info["current"],
            "restart_required": True, "deps_ok": True, "deps_error": ""}
