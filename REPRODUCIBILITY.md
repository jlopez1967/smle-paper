# Reproducibility Guide

> This guide allows any researcher to independently verify all claims in the paper in **under 5 minutes**, without access to any proprietary software.

## What you need

- Python 3.8+
- `pandas`, `numpy`, `matplotlib` (all open-source)
- The files in this repository (no external downloads required)

```bash
pip install -r scripts/requirements.txt
```

---

## Step 1 — Verify the core metrics (30 seconds)

```bash
python scripts/verify_results.py
```

This script reads `data/dataset_etiquetado.csv` and computes:
- Number of segments discovered
- Precision, Recall, F1, Accuracy
- Confusion matrix (TP, FP, FN, TN)

**Expected terminal output:**
```
==================================================
  SMLE RESULTS VERIFICATION
  Paper: Symbolic ML via Exhaustive Boolean Rule Induction
==================================================

[DATA] Raw dataset:    958 instances, 9 features
[DATA] Labeled output: 958 instances

[CLASSES] Positive (X wins): 626 (65.3%)
[CLASSES] Negative:          332 (34.7%)

[SEGMENTS] Boolean segments discovered: 185
[SEGMENTS] Segments with 100% precision: 185 / 185

[CONFUSION MATRIX]
  True Positives  (TP):  615
  False Positives (FP):    0
  False Negatives (FN):   11
  True Negatives  (TN):  332

[METRICS]
  Precision:  100.0%
  Recall:      98.2%
  F1-Score:    99.1%
  Accuracy:    98.9%

[VALIDATION vs PAPER CLAIMS]
  ✅ PASS  Segments discovered == 185
  ✅ PASS  Segments with 100% precision == 185
  ✅ PASS  False Positives == 0
  ✅ PASS  Precision == 100.0%
  ✅ PASS  Recall >= 98.0%
  ✅ PASS  F1-Score >= 99.0%
  ✅ PASS  Accuracy >= 98.0%

==================================================
  ✅ ALL METRICS MATCH PAPER CLAIMS.
==================================================
```

---

## Step 2 — Generate the figures (1 minute)

```bash
python scripts/visualize_segments.py
```

Generates in `results/`:
- `confusion_matrix_verified.png`
- `segment_precision_distribution.png`
- `segment_population_distribution.png`
- `coverage_breakdown.png`

---

## Step 3 — Manual spot-check (optional, 2 minutes)

Open `data/dataset_etiquetado.csv` in any spreadsheet application. Each row is a Tic-Tac-Toe board state. The last column `Microsegmento_Descubierto` shows which Boolean segment (rule) matched that board.

**Pick any row labeled "Segmento N"** and verify manually:
1. Find all rows with the same segment label
2. Confirm all of them belong to the `positive` class (X wins)
3. Verify the board state is consistent with an X-winning position

For example, rows labeled `Segmento 1` should all have `x` in top-left, top-middle, and top-right squares.

---

## Step 4 — Understand the labeled output format

The file `data/dataset_etiquetado.csv` has this structure:

| Column | Description |
|--------|-------------|
| `top-left-square` ... `bottom-right-square` | Original board square values: `x`, `o`, `b` |
| `Class` | Ground-truth class: `positive` (X wins) or `negative` |
| `Microsegmento_Descubierto` | The Boolean segment assigned by SMLE, e.g. `Segmento 28`, or `Sin Segmento` if no rule matched |

Rows labeled `Sin Segmento` ("Without Segment") are instances where no Boolean rule with sufficient precision was discovered. In the Tic-Tac-Toe experiment:
- 11 positive instances were not covered (False Negatives)
- 332 negative instances were correctly excluded (True Negatives)

---

## What is NOT independently verifiable from this repo

| Claim | Status | Reason |
|-------|--------|--------|
| Tic-Tac-Toe metrics | ✅ Fully verifiable | Data + script included |
| Exactor Core algorithm | ✅ Described in detail (Section 4.3 of paper) | Rust source available on request |
| German Credit results | ⏳ Planned | Dataset not included yet |
| Pima Diabetes results | ⏳ Planned | Dataset not included yet |
| Comparison vs baselines (DT, RF, RIPPER) | ⏳ Planned | Approximate literature values used in paper |

---

## Reporting Issues

If you find a discrepancy between the scripts and the paper, please open a GitHub Issue with:
1. The exact command you ran
2. The output you received
3. What you expected based on the paper

We take reproducibility seriously and will respond within 72 hours.
