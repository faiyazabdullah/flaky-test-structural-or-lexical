#!/usr/bin/env python
"""Figures for the write-up, all read from ``out/*.json``.

Four, as 07 requires: the layer curve, within-vs-cross, the transfer drop
comparison, and the minimal-pair distribution.  Nothing is recomputed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flaky.common import add_common_args, ctx_from_args, get_logger, load_json, save_json
from flaky.cv import BASELINE_METHODS

STAGE = "figures"
PALETTE = {"probe": "#2f6f9f", "baseline": "#c4643a", "control": "#8a8a8a",
           "alt": "#4c9a6a", "warn": "#9a4c7a"}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_layer_curve(probe, base, out: Path) -> None:
    models = [(k, p) for k, pools in probe["models"].items() for p in pools]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), squeeze=False)
    swb = base["STRONG_WORD_BASELINE"]["ap_mean"]
    for ax, (key, pool) in zip(axes[0], models):
        e = probe["models"][key][pool]
        xs = e["layers"]
        for tname, colour, label in [
            ("flaky_code", PALETTE["probe"], "flaky | code"),
            ("control", PALETTE["control"], "control (random labels)"),
        ]:
            t = e["targets"].get(tname)
            if not t:
                continue
            m = np.array([c["mean"] if c else np.nan for c in t["ap_by_layer"]])
            s = np.array([c["std"] if c else np.nan for c in t["ap_by_layer"]])
            ax.plot(xs, m, color=colour, lw=1.8, label=label)
            ax.fill_between(xs, m - s, m + s, color=colour, alpha=0.15, lw=0)
        t = e["targets"].get("flaky_code", {})
        if "ap_on_renamed_by_layer" in t:
            m = np.array([c["mean"] for c in t["ap_on_renamed_by_layer"]])
            ax.plot(xs, m, color=PALETTE["alt"], lw=1.8, ls="--",
                    label="flaky | eval on code_renamed")
        ax.axhline(swb, color=PALETTE["baseline"], lw=1.4, ls=":",
                   label=f"STRONG_WORD_BASELINE ({swb:.3f})")
        ax.axhline(probe["class_prior"], color="#bbbbbb", lw=1.0, ls="-",
                   label=f"class prior ({probe['class_prior']:.3f})")
        _style(ax, f"{key} / {pool} pooling", "layer", "cross-project AP")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=True, framealpha=0.85, edgecolor="none",
                  loc="lower right", borderpad=0.5)
    fig.suptitle("Probe AP by layer — full curve, no selected best layer", fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig1_layer_curve.png", dpi=180)
    plt.close(fig)


def fig_within_vs_cross(base, out: Path) -> None:
    methods = [m for m in BASELINE_METHODS
               if "cross" in base["methods"].get(m, {}).get("full", {})]
    w = [base["methods"][m]["full"]["within"]["ap"]["mean"] for m in methods]
    c = [base["methods"][m]["full"]["cross"]["ap"]["mean"] for m in methods]
    we = [base["methods"][m]["full"]["within"]["ap"]["std"] for m in methods]
    ce = [base["methods"][m]["full"]["cross"]["ap"]["std"] for m in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - 0.19, w, 0.36, yerr=we, capsize=3, label="within-project",
           color=PALETTE["probe"], alpha=0.9)
    ax.bar(x + 0.19, c, 0.36, yerr=ce, capsize=3, label="cross-project (grouped)",
           color=PALETTE["baseline"], alpha=0.9)
    ax.axhline(base["class_prior_full"], color="#888", lw=1.0, ls=":",
               label=f"class prior ({base['class_prior_full']:.3f})")
    top = max(a + e for a, e in zip(w, we))
    for i, (a, b, ae, be) in enumerate(zip(w, c, we, ce)):
        ax.annotate(f"gap {a - b:+.3f}", (i, max(a + ae, b + be) + 0.02 * top),
                    ha="center", fontsize=7, color="#444")
    ax.set_ylim(0, top * 1.16)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8, rotation=12, ha="right")
    _style(ax, "Within- vs cross-project AP — the generalisation gap", "", "average precision")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_within_vs_cross.png", dpi=180)
    plt.close(fig)


def fig_transfer(transfer, probe, decision, out: Path) -> None:
    names, code_ap, ren_ap = [], [], []
    for m in BASELINE_METHODS:
        e = transfer["methods"].get(m)
        if not e:
            continue
        names.append(m)
        code_ap.append(e["ap_code"]["mean"])
        ren_ap.append(e["ap_code_renamed"]["mean"])
    s2 = decision["criteria"]["S2"]
    pk = decision["probe"]
    names.append(f"probe\n({pk['model']}/{pk['pool']} L{pk['layer']})")
    code_ap.append(s2["delta_probe_mean"] + 0.0)
    p = decision["criteria"]["S1"]["framings"]
    code_ap[-1] = float(np.mean(p["secondary_on_code"]["a"]))
    ren_ap.append(float(np.mean(p["primary_cue_removed"]["a"])))

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(x - 0.19, code_ap, 0.36, label="trained and evaluated on `code`",
           color=PALETTE["probe"], alpha=0.9)
    ax.bar(x + 0.19, ren_ap, 0.36, label="trained on `code`, evaluated on `code_renamed`",
           color=PALETTE["warn"], alpha=0.9)
    top = max(max(code_ap), max(ren_ap))
    for i, (a, b) in enumerate(zip(code_ap, ren_ap)):
        ax.annotate(f"Δ {a - b:+.3f}", (i, max(a, b) + 0.03 * top), ha="center", fontsize=7,
                    color="#444")
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=12, ha="right")
    _style(ax, "Counterfactual renaming — transfer drop Δ", "", "cross-project AP")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig3_transfer_drop.png", dpi=180)
    plt.close(fig)


def fig_pairs(pairs, out: Path) -> None:
    if not pairs or not pairs.get("n_pairs"):
        return
    scorers = list(pairs["scorers"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={"width_ratios": [1.1, 1.5]})

    ax = axes[0]
    accs = [pairs["scorers"][s]["overall"]["pair_acc"] for s in scorers]
    los = [pairs["scorers"][s]["overall"]["wilson_95"][0] for s in scorers]
    his = [pairs["scorers"][s]["overall"]["wilson_95"][1] for s in scorers]
    y = np.arange(len(scorers))
    ax.errorbar(accs, y, xerr=[np.array(accs) - np.array(los), np.array(his) - np.array(accs)],
                fmt="o", color=PALETTE["probe"], capsize=3, ms=6)
    ax.axvline(0.5, color=PALETTE["baseline"], lw=1.4, ls="--")
    ax.annotate("cue-riding / no signal ←", xy=(0.485, 1.02), xycoords=("data", "axes fraction"),
                ha="right", fontsize=7, color="#666")
    ax.annotate("→ structure", xy=(0.515, 1.02), xycoords=("data", "axes fraction"),
                ha="left", fontsize=7, color="#666")
    ax.set_ylim(-0.7, len(scorers) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(scorers, fontsize=8)
    ax.set_xlim(0, 1)
    _style(ax, f"PairAcc, 95% Wilson (n={pairs['n_pairs']})", "P(s(pre) > s(post))", "")

    ax = axes[1]
    for i, s in enumerate(scorers):
        m = np.asarray(pairs["scorers"][s]["scores_pre"]) - \
            np.asarray(pairs["scorers"][s]["scores_post"])
        ax.hist(m, bins=25, alpha=0.55, label=f"{s} (mean {m.mean():+.3f})",
                color=list(PALETTE.values())[i % len(PALETTE)])
    ax.axvline(0, color="#333", lw=1.2)
    _style(ax, "Margin s(pre) − s(post)", "margin", "pairs")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_minimal_pairs.png", dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)

    out = ctx.out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    def maybe(name):
        p = ctx.out / name
        return load_json(p) if p.exists() else None

    base, transfer = maybe("02_baselines.json"), maybe("03_transfer.json")
    probe, decision, pairs = maybe("06_probe.json"), maybe("07_decision.json"), \
        maybe("05_pairs.json")

    made = []
    if probe and base:
        fig_layer_curve(probe, base, out)
        made.append("fig1_layer_curve.png")
    if base:
        fig_within_vs_cross(base, out)
        made.append("fig2_within_vs_cross.png")
    if transfer and probe and decision:
        fig_transfer(transfer, probe, decision, out)
        made.append("fig3_transfer_drop.png")
    if pairs:
        fig_pairs(pairs, out)
        made.append("fig4_minimal_pairs.png")
    # Every stage writes a JSON result file alongside stdout (00's conventions).
    save_json(ctx.out / "figures.json", {
        "figures": made,
        "dir": str(out.relative_to(ctx.root)),
        "sources": {
            "fig1_layer_curve.png": ["06_probe.json", "02_baselines.json"],
            "fig2_within_vs_cross.png": ["02_baselines.json"],
            "fig3_transfer_drop.png": ["03_transfer.json", "06_probe.json", "07_decision.json"],
            "fig4_minimal_pairs.png": ["05_pairs.json"],
        },
        "skipped": [f for f in ("fig1_layer_curve.png", "fig2_within_vs_cross.png",
                                "fig3_transfer_drop.png", "fig4_minimal_pairs.png")
                    if f not in made],
    })
    log.info("wrote %d figures to %s: %s", len(made), out, ", ".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
