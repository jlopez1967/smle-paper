# FAQ for Reviewers

Answers to questions a reviewer or researcher is likely to ask when evaluating this work.

---

## About the Results

**Q: Are the reported metrics (Precision=100%, Recall=98.2%, etc.) really from running the algorithm on the data?**

Yes. The file `data/dataset_etiquetado.csv` is the direct output of running Exactor Core on `data/tictoc.csv`. Every metric in the paper is computed from this file. Run `python scripts/verify_results.py` to confirm in under 30 seconds.

---

**Q: The results seem too good. Is this train-set evaluation or test-set?**

This is **training-set evaluation**. The goal of this experiment is to demonstrate that SMLE can perfectly (or near-perfectly) represent the Boolean structure of the training data as logical rules — not to measure generalization to unseen data.

The Tic-Tac-Toe dataset has a known, finite, exact Boolean structure (the rules for X winning). SMLE is finding those rules. Perfect coverage of the training set means it found the rules correctly.

Generalization experiments (cross-validation, test split comparisons) are listed as **Future Work** in Section 9.2.

---

**Q: Why only Tic-Tac-Toe? Isn't one dataset insufficient for a scientific paper?**

For a paper proposing a **new paradigm**, one fully verified result is more scientifically sound than multiple unverified results. The paper is transparent about this — Section 9.2 explicitly lists German Credit and Pima Diabetes as the next replication targets.

The Tic-Tac-Toe choice is deliberate: it has a **logically exact ground truth** (there are known, enumerable winning configurations), making it possible to validate that the discovered rules are correct independently of the algorithm.

---

**Q: Can I reproduce the exact segmentation from scratch (not just verify the CSV)?**

Not yet from this repository alone — Exactor Core is a proprietary Rust implementation. The labeled CSV and verification scripts allow you to verify the *output*. Sharing the full Exactor Core source is planned. Contact the author to request access.

---

**Q: The comparison baseline values (Decision Tree, Random Forest, etc.) — were those actually run?**

No. The comparative table (Table 3 in Section 6.2) uses **approximate literature reference values** for the Tic-Tac-Toe dataset. The paper notes this explicitly. The SMLE result is the only one directly computed. Running the full comparative experiment is listed as Future Work.

---

## About the Algorithm

**Q: Does Exactor use Quine-McCluskey?**

No. See `docs/algorithm_overview.md` for the actual algorithm. The core is a proprietary stack-based reduction using Gray code ordering and a differential mechanism for efficiency.

---

**Q: Is this related to RIPPER, CN2, or other rule learners?**

SMLE is philosophically adjacent (it also discovers rules) but mechanically different:
- RIPPER/CN2 use greedy sequential covering (find best rule, remove covered examples, repeat)
- SMLE uses exhaustive Boolean function reduction — all rules are discovered simultaneously by reducing a joint Boolean function, not sequentially

The key difference is that SMLE can discover XOR relationships, which sequential covering methods structurally cannot.

---

**Q: Is this the same as SAT-based learning or ILP?**

Not exactly:
- SAT/MaxSAT approaches minimize a logical formula over hard constraints
- ILP (Inductive Logic Programming) searches first-order hypothesis spaces
- SMLE synthesizes a DNF Boolean function directly from a truth-table, driven by a proprietary reduction search

The closest classical analog is Boolean function minimization, but SMLE uses a heuristic stack-based search rather than exact minimization (which is NP-hard).

---

## About the Dataset

**Q: Where does `tictoc.csv` come from?**

UCI Machine Learning Repository:
> Aha, D. (1991). Tic-Tac-Toe Endgame. UCI Machine Learning Repository.
> https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame

It is 958 instances representing all possible Tic-Tac-Toe end-game board states. Each instance is labeled `positive` if X wins, `negative` otherwise. It is in the public domain.

---

**Q: What does `dataset_etiquetado.csv` contain?**

It is `tictoc.csv` with one additional column: `Microsegmento_Descubierto`. This column contains the ID of the Boolean segment (rule) that SMLE assigned to each row, or `Sin Segmento` if no rule matched.

---

## Contact

Open a GitHub Issue to report any inconsistency, ask a clarification question, or propose collaboration.

Response guaranteed within 72 hours.
