#!/usr/bin/env python
"""Figures and LaTeX tables of the manuscript, generated from the result JSON files.

    python scripts/make_paper_figures.py --results /tmp/results --out docs/paper

Inputs (``--results``): ``pipeline_{tess,glyph,crnn,easyocr}.json`` (with ``records`` for CIs),
``search_ablation.json`` (with per-card confidences), ``seed2.json``, ``seed3.json``,
``no_alignment.json``. Nothing here is typed by hand: every number in the paper's tables and
figures comes from these files.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = ("national_id", "birth_date", "expiry_date")
LABEL = {"national_id": "National id", "birth_date": "Birth date", "expiry_date": "Expiry date"}
SYSTEMS = (
    ("tess", "Tesseract"),
    ("easyocr", "EasyOCR"),
    ("glyph", "Glyph + k-NN"),
    ("crnn", "Proposed"),
)
# Okabe-Ito colour-blind-safe palette
COLOR = {"tess": "#999999", "easyocr": "#E69F00", "glyph": "#56B4E9", "crnn": "#009E73"}
MODES = ("nominal", "per-field", "joint")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def f1_of(exact: np.ndarray, auto: np.ndarray) -> float:
    prec = (exact & auto).sum() / auto.sum() if auto.any() else 0.0
    rec = (exact & auto).sum() / len(exact)
    return float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def outcomes(records: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    exact = np.array([bool(r[name][2]) if name in r else False for r in records])
    auto = np.array([(not r[name][1]) if name in r else False for r in records])
    return exact, auto


def boot_f1(records: list[dict], name: str, draws: int = 2000) -> tuple[float, float, float]:
    exact, auto = outcomes(records, name)
    rng = np.random.default_rng(0)
    vals = [
        f1_of(exact[i], auto[i])
        for i in (rng.integers(0, len(exact), len(exact)) for _ in range(draws))
    ]
    return f1_of(exact, auto), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def prec_rec(records: list[dict], name: str) -> tuple[float, float, float]:
    exact, auto = outcomes(records, name)
    prec = (exact & auto).sum() / auto.sum() if auto.any() else 0.0
    return float(exact.mean()), float(prec), float((exact & auto).sum() / len(exact))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over well-formed readings (conf > 0), equal-width bins."""
    m = conf > 0
    c, y = conf[m], correct[m].astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        s = (c >= lo) & ((c < hi) | (hi == 1.0))
        if s.any():
            total += s.mean() * abs(y[s].mean() - c[s].mean())
    return float(total)


def write_table(path: Path, cols: str, head: str, rows: list[str]) -> None:
    """A complete tabular environment (``\\input`` inside a tabular breaks booktabs rules)."""
    body = "\n".join(rows)
    path.write_text(
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\\\\\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    R = args.results
    recs = {k: json.loads((R / f"pipeline_{k}.json").read_text())["records"] for k, _ in SYSTEMS}
    abl = json.loads((R / "search_ablation.json").read_text())
    extra = {
        k: json.loads((R / f"{k}.json").read_text()) for k in ("seed2", "seed3", "no_alignment")
    }

    # ---- Table 1: main comparison ---
    rows, rows_pr = [], []
    for key, name in SYSTEMS:
        f1c, prc = [], []
        for f in FIELDS:
            f1, lo, hi = boot_f1(recs[key], f)
            acc, prec, rec = prec_rec(recs[key], f)
            f1c.append(f"{f1:.3f} [{lo:.3f}, {hi:.3f}]")
            prc.append(f"{acc:.3f} & {prec:.3f} & {rec:.3f}")
        rows.append(f"{name} & " + " & ".join(f1c) + r" \\")
        rows_pr.append(f"{name} & " + " & ".join(prc) + r" \\")
    head_main = "Method & National id & Birth date & Expiry date"
    head_pr = (
        " & \\multicolumn{3}{c}{National id} & \\multicolumn{3}{c}{Birth date}"
        " & \\multicolumn{3}{c}{Expiry date}\\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
        "Method" + " & Acc & Prec & Rec" * 3
    )
    write_table(args.out / "table_main.tex", "lccc", head_main, rows)
    write_table(args.out / "table_pr.tex", "l" + "c" * 9, head_pr, rows_pr)

    # ---- Fig. 2: F1 with CIs ---
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    w = 0.2
    for j, (key, name) in enumerate(SYSTEMS):
        v, lo, hi = zip(*(boot_f1(recs[key], f) for f in FIELDS), strict=True)
        x = np.arange(3) + (j - 1.5) * w
        ax.bar(x, v, w, color=COLOR[key], label=name, edgecolor="black", linewidth=0.4)
        ax.errorbar(
            x,
            v,
            yerr=[np.array(v) - lo, np.array(hi) - np.array(v)],
            fmt="none",
            ecolor="black",
            capsize=2,
            lw=0.8,
        )
    ax.set_xticks(range(3), [LABEL[f] for f in FIELDS])
    ax.set_ylabel("F1 (95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, loc="upper center", frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.18))
    fig.savefig(fig_dir / "f1_comparison.pdf")
    plt.close(fig)

    # ---- Fig. 3: precision-coverage and Fig. 4: confidence histogram, joint search ---
    joint = abl["joint"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for f, ls in zip(FIELDS, ("-", "--", ":"), strict=True):
        c = np.array(joint[f]["confidence"])
        y = np.array(joint[f]["correct"])
        order = np.argsort(-c)
        cum = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
        cov = np.arange(1, len(y) + 1) / len(y)
        axes[0].plot(cov, cum, ls, color="black", label=LABEL[f], lw=1.2)
    axes[0].set_xlabel("Coverage (fraction auto-accepted)")
    axes[0].set_ylabel("Precision of accepted")
    axes[0].set_ylim(0.8, 1.005)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    conf = np.concatenate([joint[f]["confidence"] for f in FIELDS])
    corr = np.concatenate([joint[f]["correct"] for f in FIELDS]).astype(bool)
    bins = np.linspace(0, 1, 21)
    axes[1].hist(
        [conf[corr], conf[~corr]],
        bins=bins,
        stacked=True,
        color=[COLOR["crnn"], "#D55E00"],
        label=["correct", "incorrect"],
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1].set_xlabel("CTC confidence")
    axes[1].set_ylabel("Fields (3 x 325 cards)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(fig_dir / "confidence.pdf")
    plt.close(fig)

    # ---- Fig. 5: ablation ---
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    variants = [
        ("Nominal window", abl["nominal"]),
        ("Per-field search", abl["per-field"]),
        ("No pseudo-alignment", extra["no_alignment"]["joint"]),
        ("Joint (proposed)", joint),
    ]
    seeds = [abl["joint"], extra["seed2"]["joint"], extra["seed3"]["joint"]]
    for j, (name, d) in enumerate(variants):
        x = np.arange(3) + (j - 1.5) * 0.2
        v = [d[f]["f1"]["value"] for f in FIELDS]
        ax.bar(
            x,
            v,
            0.2,
            label=name,
            color=["#BBBBBB", "#F0E442", "#CC79A7", COLOR["crnn"]][j],
            edgecolor="black",
            linewidth=0.4,
        )
    sd = [np.std([s[f]["f1"]["value"] for s in seeds], ddof=1) for f in FIELDS]
    mean = [np.mean([s[f]["f1"]["value"] for s in seeds]) for f in FIELDS]
    ax.errorbar(
        np.arange(3) + 1.5 * 0.2, mean, yerr=sd, fmt="none", ecolor="black", capsize=2, lw=0.8
    )
    ax.set_xticks(range(3), [LABEL[f] for f in FIELDS])
    ax.set_ylabel("F1 on 325 cards (τ = 0.9)")
    ax.set_ylim(0.5, 0.9)
    ax.legend(ncol=2, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(fig_dir / "ablation.pdf")
    plt.close(fig)

    # ---- Fig. 6: reliability diagram ---
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    edges = np.linspace(0.5, 1.0, 11)
    mid, acc = [], []
    m = conf > 0
    for lo, hi in itertools.pairwise(edges):
        s = m & (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if s.sum() >= 5:
            mid.append(conf[s].mean())
            acc.append(corr[s].mean())
    ax.plot([0.5, 1], [0.5, 1], ":", color="black", lw=0.8)
    ax.plot(mid, acc, "o-", color=COLOR["crnn"], ms=3.5)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(f"ECE = {ece(conf, corr):.3f}", fontsize=8)
    fig.savefig(fig_dir / "reliability.pdf")
    plt.close(fig)

    # ---- Table: ablation and seeds ---
    lines = []
    for name, d in variants:
        v = [d[f]["f1"]["value"] for f in FIELDS]
        p = [d[f]["precision"]["value"] for f in FIELDS]
        lines.append(
            f"{name} & "
            + " & ".join(f"{a:.3f}" for a in v)
            + f" & {np.mean(v):.3f} & {np.mean(p):.3f}"
            + r" \\"
        )
    head_abl = "Variant & id $F_1$ & birth $F_1$ & expiry $F_1$ & macro $F_1$ & macro prec."
    write_table(args.out / "table_ablation.tex", "lccccc", head_abl, lines)
    macro = [np.mean([s[f]["f1"]["value"] for f in FIELDS]) for s in seeds]
    stats = {
        "seed_macro_f1": [float(x) for x in macro],
        "seed_mean": float(np.mean(macro)),
        "seed_std": float(np.std(macro, ddof=1)),
        "ece": ece(conf, corr),
        "auc": {f: joint[f]["auc"]["value"] for f in FIELDS},
        "n_cards": abl["n_cards"],
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
