"""Aggregate runs/*/result.json into summary tables + main figures."""

import glob
import json
import os
import re
import sys

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thascab import data as D

SPLITS = ["iid", "comp_attr", "comp_tmpl", "extrap"]

SYSTEMS = {  # display name -> run-name prefix regex
    "B1 perceiver (frozen)": r"b1frozen_s(\d)$",
    "B1 perceiver (ft)": r"b1ft_s(\d)$",
    "B2 reasoner": r"b2_s(\d)$",
    "B3 interpreter": r"b3_s(\d)$",
    "B4 union frozen": r"b4_s(\d)$",
    "B4.5 union finetune": r"b45_s(\d)$",
    "B5 union scratch": r"b5_s(\d)$",
    "B6 text bridge": r"b6_s(\d)$",
    "B7 ensemble (b1ft)": r"b7_b1ft_s(\d)$",
    "B7 ensemble (b1frozen)": r"b7_b1frozen_s(\d)$",
    "B8 monolith": r"b8_s(\d)$",
}


def collect():
    rows = {}
    for path in glob.glob("runs/*/result.json"):
        name = os.path.basename(os.path.dirname(path))
        for disp, pat in SYSTEMS.items():
            if re.fullmatch(pat, name):
                with open(path) as f:
                    r = json.load(f)
                rows.setdefault(disp, []).append(r)
    return rows


def summarize(rows):
    out = {}
    for disp, rs in rows.items():
        out[disp] = {"n_seeds": len(rs)}
        for sp in SPLITS:
            vals = [r["final"][sp]["overall"] for r in rs if sp in r["final"]]
            if vals:
                out[disp][sp] = (float(np.mean(vals)), float(np.std(vals)))
        # per-family on iid
        fams = {}
        for fam in D.FAMILIES:
            vals = [r["final"]["iid"].get(fam) for r in rs
                    if r["final"].get("iid", {}).get(fam) is not None]
            if vals:
                fams[fam] = (float(np.mean(vals)), float(np.std(vals)))
        out[disp]["families_iid"] = fams
    return out


def print_table(summary):
    hdr = f"{'system':26s} n " + " ".join(f"{sp:>16s}" for sp in SPLITS)
    print(hdr)
    print("-" * len(hdr))
    for disp in SYSTEMS:
        if disp not in summary:
            continue
        s = summary[disp]
        cells = []
        for sp in SPLITS:
            if sp in s:
                m, sd = s[sp]
                cells.append(f"{m*100:6.1f} ±{sd*100:4.1f}  ")
            else:
                cells.append(" " * 16)
        print(f"{disp:26s} {s['n_seeds']} " + " ".join(cells))


def main_bar_chart(summary, path="figures/main_results.png"):
    systems = [s for s in SYSTEMS if s in summary and "iid" in summary[s]]
    x = np.arange(len(systems))
    w = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, sp in enumerate(SPLITS):
        means = [summary[s].get(sp, (np.nan, 0))[0] * 100 for s in systems]
        errs = [summary[s].get(sp, (np.nan, 0))[1] * 100 for s in systems]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, capsize=2, label=sp)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("THASCAB: all systems x all splits (mean ± std over seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    rows = collect()
    summary = summarize(rows)
    os.makedirs("figures", exist_ok=True)
    with open("runs/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_table(summary)
    main_bar_chart(summary)
