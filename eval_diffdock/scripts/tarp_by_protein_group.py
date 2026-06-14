"""
TARP per protein-family group analysis.

Uses pre-computed centroid TARP fractions (K=100) from testset_eval_merged.
Groups the 322 test complexes into 7 biologically meaningful families and
plots per-group ECP curves with bootstrap confidence bands.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from molcalib.tarp import ecp_from_fractions, bootstrap_ecp, plot_ecp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MERGED_DIR = "results/testset_eval_merged"
ANN_CSV    = "data/pdb_annotations.csv"
OUT_DIR    = "results/testset_eval_merged"

# ---------------------------------------------------------------------------
# Group definitions (ordered, first match wins)
# ---------------------------------------------------------------------------
GROUPS = [
    ("Hydrolases",          lambda c: "hydrolase" in c.lower()),
    ("Transferases",        lambda c: "transferase" in c.lower()),
    ("Sugar Binding",       lambda c: "sugar binding" in c.lower()),
    ("Signaling",           lambda c: "signaling" in c.lower()),
    ("Transcription/Gene",  lambda c: any(k in c.lower() for k in
                                         ("transcription", "nuclear protein",
                                          "dna binding", "rna binding",
                                          "gene regulation"))),
    ("Oxidoreductases",     lambda c: "oxidoreductase" in c.lower()),
    ("Other",               lambda c: True),  # catch-all
]

COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


def assign_group(protein_class: str) -> str:
    for name, fn in GROUPS:
        if fn(protein_class):
            return name
    return "Other"


def main():
    # Load data
    names   = np.load(f"{MERGED_DIR}/complex_names.npy", allow_pickle=True)
    f_cent  = np.load(f"{MERGED_DIR}/tarp_fractions_centroid.npy")   # (322, 100)
    ann     = pd.read_csv(ANN_CSV)
    ann_map = dict(zip(ann["pdb_id"], ann["protein_class"].fillna("Unknown")))

    # Assign each complex to a group
    protein_classes  = [ann_map.get(n, "Unknown") for n in names]
    group_labels     = [assign_group(c) for c in protein_classes]
    group_labels_arr = np.array(group_labels)

    # Summary table
    print("\n=== Protein group breakdown ===")
    print(f"{'Group':<22} {'N':>4}")
    print("-" * 28)

    rng = np.random.default_rng(42)
    group_results = {}

    for grp_name, _ in GROUPS:
        mask = group_labels_arr == grp_name
        n = mask.sum()
        if n == 0:
            continue
        f_grp = f_cent[mask]
        ecp, alpha = ecp_from_fractions(f_grp)
        boot = bootstrap_ecp(f_grp, n_bootstrap=500, rng=rng)
        group_results[grp_name] = dict(n=n, f=f_grp, ecp=ecp, alpha=alpha, boot=boot)
        print(f"{grp_name:<22} {n:>4}")

    # Overall for reference
    ecp_all, alpha_all = ecp_from_fractions(f_cent)
    boot_all = bootstrap_ecp(f_cent, n_bootstrap=500, rng=rng)
    print(f"\n{'ALL (baseline)':<22} {len(names):>4}")

    # ---------------------------------------------------------------------------
    # Plot 1: one axes per group (7 panels) + overall
    # ---------------------------------------------------------------------------
    n_groups = len(group_results)
    ncols = 4
    nrows = (n_groups + 1 + ncols - 1) // ncols   # +1 for overall

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.ravel()

    ax = axes[0]
    plot_ecp(ecp_all, alpha_all, ax=ax, label=f"All (N={len(names)})",
             color="black", bootstrap_ecps=boot_all)
    ax.set_title("Overall", fontsize=10)

    for i, (grp_name, color) in enumerate(zip(group_results, COLORS)):
        res = group_results[grp_name]
        ax  = axes[i + 1]
        plot_ecp(res["ecp"], res["alpha"], ax=ax,
                 label=f"{grp_name} (N={res['n']})",
                 color=color, bootstrap_ecps=res["boot"])
        ax.set_title(f"{grp_name}  N={res['n']}", fontsize=9)

    for j in range(i + 2, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("TARP ECP by protein family (centroid, K=100)", fontsize=13, y=1.01)
    fig.tight_layout()
    out1 = f"{OUT_DIR}/tarp_ecp_by_group_panels.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out1}")
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Plot 2: all groups overlaid on one axes
    # ---------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for (grp_name, _), color in zip(group_results.items(), COLORS):
        res = group_results[grp_name]
        ax2.plot(res["alpha"], res["ecp"], color=color, lw=2,
                 label=f"{grp_name} (N={res['n']})")
    ax2.plot(alpha_all, ecp_all, "k--", lw=1.5, label=f"Overall (N={len(names)})")
    ax2.plot([0, 1], [0, 1], "grey", lw=1, ls=":")
    ax2.set_xlabel("Credibility level α", fontsize=12)
    ax2.set_ylabel("Expected coverage probability", fontsize=12)
    ax2.set_title("TARP ECP by protein family (centroid, K=100)", fontsize=11)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8, loc="upper left")
    fig2.tight_layout()
    out2 = f"{OUT_DIR}/tarp_ecp_by_group_overlay.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)

    print("\nDone.")


if __name__ == "__main__":
    main()
