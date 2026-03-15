"""
SMLE Visualization Script
==========================
Generates all charts used in the paper.

Usage:
    python scripts/visualize_segments.py

Outputs (saved to results/):
    - confusion_matrix_verified.png
    - segment_precision_distribution.png
    - segment_population_distribution.png
    - coverage_breakdown.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent
LABELED = ROOT / "data" / "dataset_etiquetado.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("Loading data...")
df = pd.read_csv(LABELED)
real = df[df["Microsegmento_Descubierto"] != "Sin Segmento"]
no_rule = df[df["Microsegmento_Descubierto"] == "Sin Segmento"]

seg_stats = real.groupby("Microsegmento_Descubierto").apply(
    lambda g: pd.Series({
        "count": len(g),
        "positive": (g["Class"] == "positive").sum(),
        "negative": (g["Class"] == "negative").sum(),
        "precision_pct": (g["Class"] == "positive").sum() / len(g) * 100,
    }),
    include_groups=False,
)

TP = (real["Class"] == "positive").sum()
FP = (real["Class"] == "negative").sum()
FN = (no_rule["Class"] == "positive").sum()
TN = (no_rule["Class"] == "negative").sum()

# ─── 1. Confusion Matrix ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm = np.array([[TP, FN], [FP, TN]])
im = ax.imshow(cm, cmap="Blues", vmin=0)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Segment\nAssigned", "No Segment"])
ax.set_yticklabels(["Actual\nPositive", "Actual\nNegative"])
for i in range(2):
    for j in range(2):
        label = ["TP","FN","FP","TN"][i*2+j]
        ax.text(j, i, f"{cm[i,j]}\n({label})", ha="center", va="center",
                fontsize=13, color="white" if cm[i,j] > 200 else "black", fontweight="bold")
ax.set_title("Confusion Matrix — SMLE on Tic-Tac-Toe\nPrecision=100%, Recall=98.2%, F1=99.1%",
             fontsize=10, pad=10)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p = OUT / "confusion_matrix_verified.png"
plt.savefig(p, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ─── 2. Segment Precision Distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
bins = [0, 50, 60, 70, 80, 90, 95, 99, 100, 101]
counts, edges = np.histogram(seg_stats["precision_pct"], bins=bins)
colors = ["#d32f2f","#e64a19","#f57c00","#fbc02d","#388e3c",
          "#1976d2","#303f9f","#006064"]
bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor="white", linewidth=0.8)
ax.set_xticks(range(len(counts)))
ax.set_xticklabels(["0-50","50-60","60-70","70-80","80-90","90-95","95-99","99-100"], fontsize=8)
ax.set_xlabel("Segment Precision (%)")
ax.set_ylabel("Number of Segments")
ax.set_title("Precision Distribution Across 185 Discovered Segments\n(185 segments at 100%)", fontsize=10)
for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(0, max(counts) * 1.15)
plt.tight_layout()
p = OUT / "segment_precision_distribution.png"
plt.savefig(p, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ─── 3. Population per segment (log scale) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
pops = seg_stats["count"].sort_values(ascending=False).values
ax.bar(range(len(pops)), pops, color="#1565c0", alpha=0.8, width=1.0)
ax.set_xlabel("Segment (sorted by population)")
ax.set_ylabel("Instances in Segment")
ax.set_title("Population Distribution Across 185 Boolean Segments\n(Most segments are small — each captures a precise pattern)",
             fontsize=10)
ax.axhline(seg_stats["count"].median(), color="orange", linestyle="--",
           label=f"Median = {seg_stats['count'].median():.0f}")
ax.legend()
plt.tight_layout()
p = OUT / "segment_population_distribution.png"
plt.savefig(p, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ─── 4. Coverage Breakdown Pie ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
labels_pie = ["TP: Positive in\na segment (615)", "FN: Positive\nnot covered (11)",
              "TN: Negative\ncorrectly out (332)"]
sizes = [TP, FN, TN]
colors_pie = ["#1b5e20", "#b71c1c", "#e0e0e0"]
explode = (0.05, 0.1, 0)
wedges, texts, autotexts = ax.pie(sizes, labels=labels_pie, colors=colors_pie,
                                   explode=explode, autopct="%1.1f%%",
                                   startangle=90, textprops={"fontsize": 9})
for at in autotexts:
    at.set_fontweight("bold")
ax.set_title("Dataset Coverage by SMLE Rules\n(958 instances total)", fontsize=10)
plt.tight_layout()
p = OUT / "coverage_breakdown.png"
plt.savefig(p, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

print("\n✅ All visualizations generated in results/")
