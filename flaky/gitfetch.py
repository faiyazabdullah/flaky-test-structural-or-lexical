"""Minimal git plumbing for 05: fetch one commit and read files either side.

``git fetch --depth=2 --filter=blob:none <sha>`` pulls the commit and its
parent with no file contents; ``git show <sha>:path`` then fetches just the
blobs actually needed.  For a repo the size of hadoop this is a few hundred KB
instead of a full clone, and everything lands in ``cache/git/`` so a re-run or a
resumed run costs nothing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

FETCH_TIMEOUT = 600
SHOW_TIMEOUT = 120


class GitError(RuntimeError):
    pass


def _run(args: list[str], cwd: Path | None = None, timeout: int = 120,
         binary: bool = False):
    p = subprocess.run(args, cwd=str(cwd) if cwd else None, capture_output=True,
                       timeout=timeout)
    if p.returncode != 0:
        raise GitError(f"{' '.join(args[:4])}...: {p.stderr.decode('utf-8', 'replace')[:300]}")
    return p.stdout if binary else p.stdout.decode("utf-8", "replace")


def repo_dir(cache: Path, owner_repo: str) -> Path:
    return Path(cache) / "git" / owner_repo.replace("/", "__")


def ensure_repo(cache: Path, owner_repo: str) -> Path:
    d = repo_dir(cache, owner_repo)
    if not (d / ".git").exists() and not (d / "HEAD").exists():
        d.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q", "."], cwd=d)
        _run(["git", "remote", "add", "origin", f"https://github.com/{owner_repo}"], cwd=d)
        _run(["git", "config", "protocol.version", "2"], cwd=d)
    return d


def fetch_commit(cache: Path, owner_repo: str, sha: str) -> Path:
    """Fetch ``sha`` and its parent, blobs excluded.  Idempotent."""
    d = ensure_repo(cache, owner_repo)
    try:  # already present from an earlier run?
        _run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=d, timeout=30)
        _run(["git", "cat-file", "-e", f"{sha}^^{{commit}}"], cwd=d, timeout=30)
        return d
    except (GitError, subprocess.TimeoutExpired):
        pass
    _run(["git", "fetch", "-q", "--depth=2", "--filter=blob:none", "origin", sha],
         cwd=d, timeout=FETCH_TIMEOUT)
    return d


def changed_files(repo: Path, sha: str) -> list[tuple[str, str]]:
    """``[(status, path), ...]`` for ``sha`` against its first parent."""
    out = _run(["git", "diff", "--name-status", "-M", f"{sha}^", sha], cwd=repo,
               timeout=SHOW_TIMEOUT)
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0][:1], parts[-1]))
    return rows


def file_at(repo: Path, rev: str, path: str) -> str | None:
    try:
        return _run(["git", "show", f"{rev}:{path}"], cwd=repo, timeout=SHOW_TIMEOUT,
                    binary=True).decode("utf-8", "replace")
    except (GitError, subprocess.TimeoutExpired):
        return None


def commit_meta(repo: Path, sha: str) -> dict:
    out = _run(["git", "show", "-s", "--format=%H%n%an%n%aI%n%s", sha], cwd=repo,
               timeout=SHOW_TIMEOUT)
    lines = out.splitlines()
    return {"sha": lines[0] if lines else sha,
            "author": lines[1] if len(lines) > 1 else "",
            "date": lines[2] if len(lines) > 2 else "",
            "subject": lines[3] if len(lines) > 3 else ""}
