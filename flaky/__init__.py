"""Phase 1 — does a code LM represent test flakiness structurally or lexically?

Library code for the pipeline described in ``plan/``.  Every stage script under
``scripts/`` is a thin CLI wrapper around functions in this package.
"""

__all__ = [
    "common",
    "normalize",
    "javalex",
    "cues",
    "rename",
    "structural",
    "cv",
    "embed",
]
