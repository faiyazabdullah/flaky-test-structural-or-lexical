"""Shared plumbing: paths, seeding, logging, JSON IO, resumable checkpoints.

Every stage writes a JSON result file to ``out/`` and a log file to ``logs/``.
Long loops append to a JSONL checkpoint so a killed run resumes where it
stopped rather than starting over.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# run context
# --------------------------------------------------------------------------


@dataclass
class Ctx:
    """Where a run reads and writes.

    ``smoke`` runs are fully isolated: separate ``data/``, ``out/`` and
    ``ckpt/`` trees, so a smoke run can never contaminate real results.
    """

    smoke: bool = False
    seed: int = 0
    root: Path = ROOT
    suffix: str = field(init=False)

    def __post_init__(self) -> None:
        self.suffix = "_smoke" if self.smoke else ""
        for d in (self.data, self.out, self.ckpt, self.logs, self.cache):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def data(self) -> Path:
        return self.root / f"data{self.suffix}"

    @property
    def out(self) -> Path:
        return self.root / f"out{self.suffix}"

    @property
    def ckpt(self) -> Path:
        return self.root / f"ckpt{self.suffix}"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def cache(self) -> Path:
        # Network caches (git objects, patches, HF models) are shared between
        # smoke and full runs on purpose -- they are inputs, not results.
        return self.root / "cache"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"


def add_common_args(parser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run on the reduced smoke subset, writing to data_smoke/ and out_smoke/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore existing checkpoints and recompute from scratch",
    )


def ctx_from_args(args) -> Ctx:
    return Ctx(smoke=bool(getattr(args, "smoke", False)), seed=int(getattr(args, "seed", 0)))


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed every RNG we might touch.  Called at the top of every stage."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # cuBLAS reductions stay fast
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

_LOG_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(stage: str, ctx: Ctx | None = None) -> logging.Logger:
    """Logger that writes to stdout and to ``logs/<stage>.log`` (appending)."""
    logger = logging.getLogger(stage)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(_LOG_FMT, datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logs_dir = (ctx.logs if ctx is not None else ROOT / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke." if (ctx is not None and ctx.smoke) else ""
    fh = logging.FileHandler(logs_dir / f"{tag}{stage}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    logger.addHandler(fh)

    logger.info("=" * 72)
    logger.info("stage %s starting (pid %d)", stage, os.getpid())
    return logger


class Timer:
    """``with Timer(log, 'thing'):`` -> logs wall time on exit."""

    def __init__(self, logger: logging.Logger, what: str):
        self.logger = logger
        self.what = what

    def __enter__(self):
        self.t0 = time.monotonic()
        self.logger.info("%s ...", self.what)
        return self

    def __exit__(self, *exc):
        dt = time.monotonic() - self.t0
        if exc[0] is None:
            self.logger.info("%s done in %s", self.what, fmt_dur(dt))
        else:
            self.logger.error("%s FAILED after %s", self.what, fmt_dur(dt))
        return False


def fmt_dur(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# --------------------------------------------------------------------------
# progress
# --------------------------------------------------------------------------


def progress(iterable: Iterable, desc: str, total: int | None = None, **kw) -> Iterator:
    """tqdm wrapper that degrades gracefully and never spams a log file."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        return iter(iterable)
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        **kw,
    )


# --------------------------------------------------------------------------
# JSON IO
# --------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Make numpy scalars/arrays and Paths JSON-serialisable."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if v == v and abs(v) != float("inf") else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def save_json(path: Path, obj: Any) -> Path:
    """Atomic write, so a kill mid-write cannot leave a truncated result file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_jsonable)
        f.write("\n")
    os.replace(tmp, path)
    return path


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------


def digest(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class JsonlCheckpoint:
    """Append-only keyed store for resumable loops.

    Each line is ``{"k": <key>, "v": <value>}``.  Re-running skips keys already
    present.  Appends are flushed and fsynced so a hard kill loses at most the
    item in flight.
    """

    def __init__(self, path: Path, enabled: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._done: dict[str, Any] = {}
        self._fh = None
        if enabled and self.path.exists():
            self._load()

    def _load(self) -> None:
        bad = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._done[rec["k"]] = rec["v"]
                except Exception:
                    bad += 1  # truncated tail from a hard kill; ignore
        self._bad_lines = bad

    def __contains__(self, key: str) -> bool:
        return self.enabled and key in self._done

    def __len__(self) -> int:
        return len(self._done)

    def get(self, key: str, default: Any = None) -> Any:
        return self._done.get(key, default)

    def items(self):
        return self._done.items()

    def values(self):
        return self._done.values()

    def put(self, key: str, value: Any) -> None:
        self._done[key] = value
        if not self.enabled:
            return
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps({"k": key, "v": value}, default=_jsonable) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def reset(self) -> None:
        self.close()
        self._done.clear()
        if self.path.exists():
            self.path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def resumable(
    ctx: Ctx,
    name: str,
    items: Sequence[Any],
    key_fn: Callable[[Any], str],
    work_fn: Callable[[Any], Any],
    desc: str,
    logger: logging.Logger | None = None,
    force: bool = False,
) -> list[Any]:
    """Run ``work_fn`` over ``items``, skipping anything already checkpointed."""
    cp = JsonlCheckpoint(ctx.ckpt / f"{name}.jsonl")
    if force:
        cp.reset()
    todo = [it for it in items if key_fn(it) not in cp]
    if logger is not None and len(todo) < len(items):
        logger.info("%s: resuming, %d/%d already done", name, len(items) - len(todo), len(items))
    with cp:
        for it in progress(todo, desc=desc, total=len(todo)):
            cp.put(key_fn(it), work_fn(it))
    return [cp.get(key_fn(it)) for it in items]


class StageMarker:
    """``ckpt/<stage>.done`` records the inputs a completed stage saw."""

    def __init__(self, ctx: Ctx, stage: str):
        self.path = ctx.ckpt / f"{stage}.done"

    def matches(self, sig: str) -> bool:
        if not self.path.exists():
            return False
        try:
            return load_json(self.path).get("sig") == sig
        except Exception:
            return False

    def write(self, sig: str, info: dict | None = None) -> None:
        save_json(self.path, {"sig": sig, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(info or {})})

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------


def gpu_report() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda not available"}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "total_mem_gb": round(props.total_memory / 2**30, 2),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "torch": torch.__version__,
    }


def peak_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except ImportError:
        pass
    return 0.0


def report_acceptance(logger, checks: dict, smoke: bool = False) -> bool:
    """Log each acceptance check and return whether the stage should pass.

    In smoke mode a failure is reported but not fatal: the smoke run exists to
    validate the pipeline end to end on a few hundred rows, and the
    sample-size-dependent criteria (matched-subset medians, fold counts, pair
    power) cannot be met there.  It never suppresses the log line.
    """
    bools = {k: v for k, v in checks.items() if isinstance(v, bool)}
    failed = [k for k, v in bools.items() if not v]
    for k, v in bools.items():
        logger.info("acceptance %-42s %s", k, "PASS" if v else "FAIL")
    if not failed:
        return True
    if smoke:
        logger.warning("acceptance failures in SMOKE mode (not fatal, sample is too small "
                       "to satisfy size-dependent criteria): %s", ", ".join(failed))
        return True
    logger.error("acceptance FAILED: %s", ", ".join(failed))
    return False
