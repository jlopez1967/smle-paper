"""
SMLE Results Verification Script
=================================
Reproduces all metrics claimed in the paper from the labeled CSV.

Usage:
    python scripts/verify_results.py

Expected runtime: < 5 seconds
No GPU required. No ML libraries needed. Just pandas.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "tictoc.csv"
DATA_LABELED = ROOT / "data" / "dataset_etiquetado.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("=" * 50)
print("  SMLE RESULTS VERIFICATION")
print("  Paper: Symbolic ML via Exhaustive Boolean Rule Induction")
print("=" * 50)

raw = pd.read_csv(DATA_RAW)
labeled = pd.read_csv(DATA_LABELED)

print(f"\n[DATA] Raw dataset:    {len(raw)} instances, {len(raw.columns)-1} features")
print(f"[DATA] Labeled output: {len(labeled)} instances")

# ─── Class distribution ───────────────────────────────────────────────────────
n_pos = (labeled["Class"] == "positive").sum()
n_neg = (labeled["Class"] == "negative").sum()
print(f"\n[CLASSES] Positive (X wins): {n_pos} ({n_pos/len(labeled)*100:.1f}%)")
print(f"[CLASSES] Negative:          {n_neg} ({n_neg/len(labeled)*100:.1f}%)")

# ─── Segment analysis ─────────────────────────────────────────────────────────
# Separate real segments from "Sin Segmento"
real = labeled[labeled["Microsegmento_Descubierto"] != "Sin Segmento"]
no_rule = labeled[labeled["Microsegmento_Descubierto"] == "Sin Segmento"]

n_segments = labeled["Microsegmento_Descubierto"].nunique() - 1  # exclude "Sin Segmento"
print(f"\n[SEGMENTS] Boolean segments discovered: {n_segments}")

seg_stats = real.groupby("Microsegmento_Descubierto").apply(
    lambda g: pd.Series({
        "count": len(g),
        "positive": (g["Class"] == "positive").sum(),
        "negative": (g["Class"] == "negative").sum(),
        "precision_pct": (g["Class"] == "positive").sum() / len(g) * 100,
    }),
    include_groups=False,
)

n_100_precision = (seg_stats["precision_pct"] == 100).sum()
print(f"[SEGMENTS] Segments with 100% precision: {n_100_precision} / {n_segments}")

# ─── Confusion matrix ─────────────────────────────────────────────────────────
TP = (real["Class"] == "positive").sum()
FP = (real["Class"] == "negative").sum()
FN = (no_rule["Class"] == "positive").sum()
TN = (no_rule["Class"] == "negative").sum()

precision  = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
recall     = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy   = (TP + TN) / (TP + FP + FN + TN) * 100

print(f"\n[CONFUSION MATRIX]")
print(f"  True Positives  (TP): {TP:4d}   — positives correctly segmented")
print(f"  False Positives (FP): {FP:4d}   — negatives wrongly in a segment")
print(f"  False Negatives (FN): {FN:4d}   — positives not covered by any rule")
print(f"  True Negatives  (TN): {TN:4d}   — negatives correctly excluded")

print(f"\n[METRICS]")
print(f"  Precision:  {precision:.1f}%")
print(f"  Recall:     {recall:.1f}%")
print(f"  F1-Score:   {f1:.1f}%")
print(f"  Accuracy:   {accuracy:.1f}%")

# ─── Validation against paper claims ─────────────────────────────────────────
CLAIMS = {
    "Segments discovered == 185":          n_segments == 185,
    "Segments with 100% precision == 185": n_100_precision == 185,
    "False Positives == 0":                FP == 0,
    "Precision == 100.0%":                 abs(precision - 100.0) < 0.1,
    "Recall >= 98.0%":                     recall >= 98.0,
    "F1-Score >= 99.0%":                   f1 >= 99.0,
    "Accuracy >= 98.0%":                   accuracy >= 98.0,
}

print(f"\n[VALIDATION vs PAPER CLAIMS]")
all_pass = True
for claim, passed in CLAIMS.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}  {claim}")
    if not passed:
        all_pass = False

# ─── Save per-segment statistics ──────────────────────────────────────────────
out_path = RESULTS_DIR / "segments_summary.csv"
seg_stats.to_csv(out_path)
print(f"\n[OUTPUT] Segment statistics saved to: {out_path}")

# ─── Final verdict ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
if all_pass:
    print("  ALL METRICS MATCH PAPER CLAIMS. [OK]")
else:
    print("  SOME CLAIMS COULD NOT BE VERIFIED. [FAIL]")
    sys.exit(1)
print("=" * 50)
print()

# ─── Top-10 segments by population ───────────────────────────────────────────
print("[TOP 10 SEGMENTS BY POPULATION]")
top10 = seg_stats.sort_values("count", ascending=False).head(10)
print(top10[["count", "positive", "negative", "precision_pct"]].to_string())
print()
