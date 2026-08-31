"""06 -- hidden-state extraction.

Turing (sm_75) constraints, violating any of which produces silent garbage or a
crash:

* ``torch_dtype=torch.float16``.  **Not** bfloat16 -- sm_75 has no bf16 support.
* ``attn_implementation`` is ``sdpa`` or ``eager``.  FlashAttention-2 needs Ampere.
* no 4-bit quantisation in Phase 1 -- NF4 perturbs the representations being
  measured.

Output is ``out/emb_{model}_{variant}_{pool}.npy``, ``[n, L+1, d]``, fp16, with
row order matching ``data/prepped.csv``.  Written through a ``.npy`` memmap and
a batch-level checkpoint, so a killed extraction resumes at the next batch
rather than restarting.
"""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import JsonlCheckpoint, progress

FP16_MAX = 65504.0


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    kind: str            # "encoder" | "decoder"
    max_length: int
    batch_size: int
    pools: tuple[str, ...]


MODELS = {
    "codebert": ModelSpec("codebert", "microsoft/codebert-base", "encoder", 512, 16, ("mean",)),
    "qwen1_5b": ModelSpec("qwen1_5b", "Qwen/Qwen2.5-Coder-1.5B", "decoder", 1024, 4,
                          ("mean", "last")),
    # Stretch target only, and only after the 1.5B pipeline is validated end to end.
    "qwen3b": ModelSpec("qwen3b", "Qwen/Qwen2.5-Coder-3B", "decoder", 1024, 1, ("mean", "last")),
}


def assert_turing_safe(model) -> None:
    """Fail loudly rather than silently producing garbage on sm_75."""
    import torch

    dtypes = {p.dtype for p in model.parameters()}
    assert torch.bfloat16 not in dtypes, (
        "bfloat16 parameters found; Turing (sm_75) has no bf16 support")
    assert dtypes <= {torch.float16, torch.float32}, f"unexpected parameter dtypes: {dtypes}"
    impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    assert impl in (None, "sdpa", "eager"), f"attn_implementation={impl!r} needs Ampere+"


def load_model(spec: ModelSpec, cache_dir: Path | None = None, device: str = "cuda"):
    import torch
    from transformers import AutoModel, AutoTokenizer

    kw = {"cache_dir": str(cache_dir)} if cache_dir else {}
    tok = AutoTokenizer.from_pretrained(spec.hf_id, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # A decoder's final-position state must be a real token, not padding.
    tok.padding_side = "left" if spec.kind == "decoder" else "right"

    model = AutoModel.from_pretrained(
        spec.hf_id,
        torch_dtype=torch.float16,       # fp16 only -- see module docstring
        attn_implementation="sdpa",
        **kw,
    )
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    assert_turing_safe(model)
    return tok, model


def token_lengths(tok, texts: list[str]) -> np.ndarray:
    """Untruncated length, so the truncation rate can be reported honestly."""
    out = np.zeros(len(texts), dtype=np.int64)
    for i in range(0, len(texts), 256):
        chunk = texts[i:i + 256]
        enc = tok(chunk, add_special_tokens=True, truncation=False,
                  padding=False)["input_ids"]
        out[i:i + len(chunk)] = [len(e) for e in enc]
    return out


def _emb_path(out_dir: Path, model_key: str, variant: str, pool: str) -> Path:
    return Path(out_dir) / f"emb_{model_key}_{variant}_{pool}.npy"


def extract(
    spec: ModelSpec,
    texts: list[str],
    ids: list,
    out_dir: Path,
    ckpt_dir: Path,
    variant: str,
    cache_dir: Path | None = None,
    logger=None,
    force: bool = False,
) -> dict:
    """Extract hidden states for one (model, variant).  Resumable per batch."""
    import torch

    out_dir, ckpt_dir = Path(out_dir), Path(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)

    tok, model = load_model(spec, cache_dir=cache_dir)
    n_layers = model.config.num_hidden_layers + 1   # embeddings + each block
    d = model.config.hidden_size

    lens = token_lengths(tok, texts)
    trunc_rate = float((lens > spec.max_length).mean())
    if logger:
        logger.info("%s/%s: L+1=%d d=%d, truncation rate at %d = %.3f (median len %d, p95 %d)",
                    spec.key, variant, n_layers, d, spec.max_length, trunc_rate,
                    int(np.median(lens)), int(np.percentile(lens, 95)))

    # one memmap per pooling strategy
    paths = {p: _emb_path(out_dir, spec.key, variant, p) for p in spec.pools}
    mm = {}
    for p, path in paths.items():
        if force and path.exists():
            path.unlink()
        if path.exists():
            existing = np.lib.format.open_memmap(path, mode="r+")
            if existing.shape != (n, n_layers, d):
                del existing
                path.unlink()
            else:
                mm[p] = existing
        if p not in mm:
            mm[p] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(n, n_layers, d))

    cp = JsonlCheckpoint(ckpt_dir / f"emb_{spec.key}_{variant}.jsonl")
    if force:
        cp.reset()

    # Length-sorted batching, original order restored by writing to row index.
    order = np.argsort(-lens, kind="stable")
    batches = [order[i:i + spec.batch_size] for i in range(0, n, spec.batch_size)]
    todo = [(bi, b) for bi, b in enumerate(batches) if str(bi) not in cp]
    if logger and len(todo) < len(batches):
        logger.info("resuming %s/%s: %d/%d batches already done",
                    spec.key, variant, len(batches) - len(todo), len(batches))

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    dev = next(model.parameters()).device
    n_clipped = 0
    n_values = 0

    with torch.inference_mode():
        for bi, idx in progress(todo, desc=f"embed {spec.key}/{variant}", total=len(todo)):
            batch = [texts[i] for i in idx]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=spec.max_length, return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask")
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states                      # tuple of [B, T, d]
            attn = enc["attention_mask"]                # [B, T]
            # Pool over *code* tokens: real tokens that are not special tokens.
            code_mask = (attn.bool() & (~special.to(dev).bool())).to(torch.float16)
            denom = code_mask.sum(dim=1, keepdim=True).clamp(min=1.0)     # [B, 1]

            if "mean" in mm:
                # Accumulate in float32, one layer at a time. Summing 1024
                # positions of Qwen's large activations in fp16 overflows to inf
                # (max 65504) and silently poisons the array; and a [B, L+1, T, d]
                # float32 stack does not fit in 8GB.
                m32 = code_mask.float().unsqueeze(-1)                     # [B, T, 1]
                d32 = denom.float()                                       # [B, 1]
                buf = np.empty((len(idx), n_layers, d), dtype=np.float32)
                for li, h in enumerate(hs):
                    buf[:, li, :] = ((h.float() * m32).sum(dim=1) / d32).cpu().numpy()
                n_clipped += int((np.abs(buf) > FP16_MAX).sum())
                n_values += buf.size
                mm["mean"][idx] = np.clip(buf, -FP16_MAX, FP16_MAX).astype(np.float16)
                del buf, m32, d32
            if "last" in mm:
                # padding_side='left' => index -1 is a real token for every row
                buf = torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy()
                n_clipped += int((np.abs(buf) > FP16_MAX).sum())
                n_values += buf.size
                mm["last"][idx] = np.clip(buf, -FP16_MAX, FP16_MAX).astype(np.float16)
                del buf

            del out, hs, enc, code_mask
            cp.put(str(bi), 1)

    for p in mm:
        mm[p].flush()
    cp.close()
    peak = float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": spec.key,
        "hf_id": spec.hf_id,
        "variant": variant,
        "n": n,
        "n_layers": n_layers,
        "hidden_size": d,
        "max_length": spec.max_length,
        "truncation_rate": trunc_rate,
        "median_token_len": int(np.median(lens)),
        "p95_token_len": int(np.percentile(lens, 95)),
        "peak_vram_gb": round(peak, 3),
        "fp16_clipped_values": n_clipped,
        "fp16_clip_rate": (n_clipped / n_values) if n_values else 0.0,
        "fp16_clip_note": ("Pooling is accumulated in float32 and clipped to the fp16 range "
                           "before storage. Qwen carries a few very large activation "
                           "dimensions; the clip rate is reported so the reader can judge "
                           "whether fp16 storage distorted anything."),
        "paths": {p: str(path) for p, path in paths.items()},
        "pools": list(spec.pools),
    }


def load_layer(path: Path, layer: int) -> np.ndarray:
    """One layer's ``[n, d]`` block as float32, read from the memmap."""
    arr = np.load(path, mmap_mode="r")
    return np.asarray(arr[:, layer, :], dtype=np.float32)


def emb_meta(path: Path) -> tuple[int, int, int]:
    arr = np.load(path, mmap_mode="r")
    return arr.shape
