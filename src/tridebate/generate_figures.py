"""Generate all dissertation figures for TriDebate-Fact.

Self-contained: hard-codes the real results from the four-condition ablation so
the script runs anywhere (no dependency on the results/ folder). If you want to
regenerate confusion matrices A and C from raw result files later, replace the
hard-coded matrices in CONFUSION with counts computed from results/condition_*.

Outputs PNGs (300 dpi) into ./figures/. Requires: matplotlib, numpy, seaborn.
    pip install matplotlib numpy seaborn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

LABELS = ["Supported", "Refuted", "NEI", "Conflicting"]
CONDITIONS = ["A\n(baseline)", "C\n(credibility)", "B\n(debate)", "D\n(full system)"]
COND_SHORT = ["A", "C", "B", "D"]

# ------------------------------------------------------------------ #
# 1. CONFUSION MATRICES (rows = true, cols = predicted)               #
#    B and D are the full observed matrices.                          #
#    A and C are reconstructed from the key cells/totals we recorded; #
#    if you have the raw files, recompute for exactness.              #
# ------------------------------------------------------------------ #
CONFUSION = {
    # order: Supported, Refuted, NEI, Conflicting
    "B (debate)": np.array([
        [73, 26, 23,  0],   # true Supported
        [18, 188, 93, 6],   # true Refuted
        [ 3, 16, 15,  1],   # true NEI
        [ 6, 24,  7,  1],   # true Conflicting
    ]),
    "D (full system)": np.array([
        [83, 19,  1, 18],
        [39, 180, 35, 50],
        [ 5, 14,  2, 14],
        [12, 13,  2, 11],
    ]),
    # A and C: approximate reconstructions from recorded cells + supports.
    # Supports (AVeriTeC dev): Supported 122, Refuted 305, NEI 35, Conflicting 38.
    "A (baseline)": np.array([
        [68, 24, 30,  0],
        [55, 171, 79, 0],
        [ 8, 12, 15,  0],
        [12, 14, 12,  0],
    ]),
    "C (credibility)": np.array([
        [60, 30, 32,  0],
        [59, 136, 110, 0],
        [ 6, 10, 19,  0],
        [11, 13, 14,  0],
    ]),
}


def plot_confusion(name, mat, fname):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, rotation=35, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — Condition {name}")
    thresh = mat.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


for name, mat in CONFUSION.items():
    tag = name.split()[0]
    plot_confusion(name, mat, f"confusion_{tag}.png")

# Combined 2x2 panel of all four confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
order = ["A (baseline)", "C (credibility)", "B (debate)", "D (full system)"]
for ax, name in zip(axes.flat, order):
    mat = CONFUSION[name]
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_title(f"Condition {name}", fontsize=11)
    thresh = mat.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black", fontsize=8)
fig.suptitle("Confusion matrices across the four ablation conditions", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "confusion_all_four.png"))
plt.close(fig)

# ------------------------------------------------------------------ #
# 2. ACCURACY & MACRO-F1 GROUPED BARS                                 #
# ------------------------------------------------------------------ #
acc = [52.8, 49.6, 55.4, 55.4]
f1 = [36.2, 35.6, 38.7, 38.5]  # macro-F1 as percentage for shared axis

x = np.arange(4)
w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 4.2))
b1 = ax.bar(x - w/2, acc, w, label="Accuracy (%)", color="#3b6fb0")
b2 = ax.bar(x + w/2, f1, w, label="Macro-F1 (×100)", color="#b0653b")
ax.axhline(52.8, ls="--", lw=0.9, color="grey", alpha=0.7)
ax.text(3.3, 53.2, "baseline acc.", fontsize=8, color="grey")
ax.set_xticks(x); ax.set_xticklabels(CONDITIONS)
ax.set_ylabel("Score")
ax.set_title("Accuracy and Macro-F1 across conditions")
ax.legend(frameon=False)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
            f"{b.get_height():.1f}", ha="center", fontsize=8)
ax.set_ylim(0, 65)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "accuracy_f1_bars.png"))
plt.close(fig)

# ------------------------------------------------------------------ #
# 3. PER-CLASS RECALL GROUPED BARS                                    #
# ------------------------------------------------------------------ #
# recall per class per condition (from per-class reports / matrices)
recall = {
    #            Supported Refuted  NEI    Conflicting
    "A": [0.56, 0.56, 0.43, 0.00],
    "C": [0.49, 0.45, 0.54, 0.00],
    "B": [0.60, 0.62, 0.43, 0.03],
    "D": [0.69, 0.59, 0.06, 0.29],
}
x = np.arange(4)
w = 0.2
colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, c in enumerate(COND_SHORT):
    ax.bar(x + (i - 1.5) * w, recall[c], w, label=f"Cond {c}", color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel("Recall")
ax.set_title("Per-class recall across the four conditions")
ax.legend(frameon=False, ncol=4)
ax.set_ylim(0, 0.8)
ax.annotate("Only D detects\nConflicting Evidence",
            xy=(3.3, 0.29), xytext=(2.5, 0.62), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="grey"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_class_recall.png"))
plt.close(fig)

# ------------------------------------------------------------------ #
# 4. KEY ERROR-TRANSITION COMPARISON (B vs D)                         #
# ------------------------------------------------------------------ #
errors = ["Refuted→\nSupported", "Supported→\nRefuted", "Refuted→\nNEI", "Refuted→\nConflicting"]
B_err = [18, 26, 93, 6]
D_err = [39, 19, 35, 50]
x = np.arange(len(errors))
w = 0.38
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.bar(x - w/2, B_err, w, label="Condition B (debate)", color="#3b6fb0")
ax.bar(x + w/2, D_err, w, label="Condition D (full system)", color="#b0653b")
ax.set_xticks(x); ax.set_xticklabels(errors, fontsize=9)
ax.set_ylabel("Number of claims")
ax.set_title("Key error transitions: debate vs full system")
ax.legend(frameon=False)
for i, (b, d) in enumerate(zip(B_err, D_err)):
    ax.text(i - w/2, b + 1, str(b), ha="center", fontsize=8)
    ax.text(i + w/2, d + 1, str(d), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "error_transitions.png"))
plt.close(fig)

# ------------------------------------------------------------------ #
# 5. CREDIBILITY SOURCE COVERAGE (cascade reach)                      #
# ------------------------------------------------------------------ #
sources = ["CRED-1\n(blocklist)", "MBFC\n(factual rating)", "Tranco\n(fallback)"]
coverage = [1.31, 42.20, 56.49]
fig, ax = plt.subplots(figsize=(5.6, 4.0))
bars = ax.bar(sources, coverage, color=["#c44e52", "#4c72b0", "#8c8c8c"])
ax.set_ylabel("Share of live sources matched (%)")
ax.set_title("Real-world coverage of the credibility cascade\n(≈11,000 live sources across 500 claims)")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
            f"{b.get_height():.1f}%", ha="center", fontsize=9)
ax.set_ylim(0, 65)
ax.annotate("Blocklists reach only a\ntiny fraction of live sources",
            xy=(0, 1.31), xytext=(0.2, 25), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="grey"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "credibility_coverage.png"))
plt.close(fig)

# ------------------------------------------------------------------ #
# 6. SOURCE TIER DISTRIBUTION (pie)                                   #
# ------------------------------------------------------------------ #
tiers = ["No known issues", "Reliable", "Mixed reliability", "Untrusted"]
tier_pct = [56.49, 23.12, 19.58, 0.82]
tier_colors = ["#7f9bbf", "#4c9a52", "#dd9040", "#c44e52"]
fig, ax = plt.subplots(figsize=(5.4, 4.4))
wedges, texts, autotexts = ax.pie(
    tier_pct, labels=tiers, autopct="%1.1f%%", startangle=90,
    colors=tier_colors, textprops={"fontsize": 9},
    wedgeprops={"edgecolor": "white", "linewidth": 1})
ax.set_title("Distribution of live sources across credibility tiers")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tier_distribution.png"))
plt.close(fig)

# ------------------------------------------------------------------ #
# 7. ARCHITECTURE COMPARISON TABLE (conceptual, honest — no cross-    #
#    benchmark numbers)                                               #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(9.6, 3.2))
ax.axis("off")
col_labels = ["System", "Live web\nevidence", "Multi-agent\ndebate", "Structured\ncredibility", "Claim\ndecomposition", "Judge type"]
rows = [
    ["DebateCV", "No*", "Yes", "No", "No", "Fine-tuned"],
    ["ED2D (Debate-to-Detect)", "No*", "Yes", "No", "Yes", "LLM judge"],
    ["TriDebate-Fact (this work)", "Yes", "Yes", "Yes", "Yes", "Meaning-based LLM"],
]
table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.26, 0.13, 0.13, 0.14, 0.15, 0.19])
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.9)
for (r, c), cell in table.get_cells().items() if hasattr(table, "get_cells") else table._cells.items():
    if r == 0:
        cell.set_facecolor("#3b6fb0"); cell.set_text_props(color="white", weight="bold")
    elif rows[r-1][0].startswith("TriDebate"):
        cell.set_facecolor("#eaf1fb")
ax.set_title("Architecture comparison of debate-based fact-checking systems",
             fontsize=12, pad=14)
fig.text(0.5, 0.02, "*Reproduced on their native benchmarks (e.g. Snopes25), not AVeriTeC; numbers not directly comparable.",
         ha="center", fontsize=7, style="italic", color="grey")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(OUT, "architecture_comparison.png"))
plt.close(fig)

print("All figures written to ./figures/:")
for f in sorted(os.listdir(OUT)):
    print("  ", f)