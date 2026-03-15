# Exactor Core: Algorithm Overview

> This document describes the proprietary algorithms inside Exactor Core that power the Boolean rule discovery in SMLE. It is intended for reviewers and researchers who want to understand *how* the 185 boolean segments were discovered.

## What Exactor Core does

Exactor Core takes as input:
- A **binarized dataset** — a table where every feature has been converted to a 0/1 boolean value
- A **binary target variable** — which class we want to find rules for

It produces as output:
- A set of **Boolean formulas** (in AND/XOR/NOT form) that together cover the positive instances with high precision

## The `reducir` Engine

The heart of Exactor Core is the `reducir` function, implemented in Rust (`function/reduction.rs`). It operates as follows:

### Core Data Structure

Each Boolean function being considered is represented as a `Function` object — essentially a truth table fragment with three possible values per entry:

| Value | Spanish name | Meaning |
|-------|-------------|---------|
| `1` | `Verdad` | This minterm is a positive example → must be covered |
| `0` | `Falso` | This minterm is a negative example → must NOT be covered |
| `?` | `NoImporta` | Don't care — already absorbed into a parent term |

### The Stack

The algorithm maintains a **stack** (`pila`) of Function objects. It pops one, applies reduction operations, and pushes the results back until the stack is empty.

```
Initialize stack with original function
While stack not empty:
    Pop current function f
    Try 4 reduction passes:
        1. gxor pass  → find generalized XOR reductions
        2. xor pass   → find paired XOR reductions
        3. nxor pass  → find negative XOR reductions
        4. AND pass   → find classical conjunctive reductions
    For each valid reduction found:
        Push reduced sub-function onto stack
    If f cannot be reduced further:
        Collect f as a final implicant → add to results
```

### The 4 Reduction Passes

#### Pass 1: `procesar_funcion_gxor` (Generalized XOR)

Scans adjacent minterm pairs in the Boolean hypercube. Two minterms `m₁` and `m₂` that differ in exactly one bit position and are both `Verdad` can be merged — the differing variable is eliminated, and the merged entry is `Verdad` in the reduced function.

This is the fundamental AND-reduction: `x AND NOT x` → eliminated, `(A AND B) OR (A AND NOT B)` → `A`.

#### Pass 2: `procesar_funcion_xor` (Pair XOR)

Tries eliminating **pairs of variables simultaneously**. When two variables `(zᵢ, zⱼ)` can be co-eliminated, the result is an XOR relationship: `zᵢ XOR zⱼ` appears in the output formula as `"zᵢ^zⱼ"`.

#### Pass 3: `procesar_funcion_nxor` (Negative XOR / NXOR)

Handles the complementary case: when a `Verdad` minterm and a `NoImporta` minterm together allow eliminating dimensions via NXOR. Uses `Utility::encuentra_diferencias` to find which bit positions differ.

#### Pass 4: `generar_funcion_reducida` (AND Reduction)

Classical variable elimination: for each variable `zᵢ`, finds its Boolean neighbor (same minterm with `zᵢ` flipped), and if both are `Verdad`, merges them into a reduced function with `zᵢ` removed.

### Termination

A function is fully reduced (collected as a final rule) when it has exactly **one `Verdad` entry** and **zero `NoImporta` entries** — i.e., it represents a single implicant that cannot be further simplified.

---

## Gray Code Ordering

Minterms are stored in **Gray code order** rather than natural binary order. This is a reflected binary code where adjacent entries differ in exactly **one bit**:

```
Binary:    000 001 010 011 100 101 110 111
Gray:      000 001 011 010 110 111 101 100
```

Adjacent Gray code entries are always Boolean neighbors (Hamming distance = 1). This means the reduction algorithm can find merge candidates in O(1) per lookup — it simply checks the next entry in the ordered map.

Minterms are stored as `BTreeMap<u64, ValueStateWithWeight>` keyed by Gray code index.

---

## Differential Mechanism

To avoid recomputing full coverage tables after each variable elimination, Exactor Core uses a **DifferentialMechanism** (`differential.rs`) that:

1. Maintains a **coverage map**: which terms currently cover which minterms
2. Uses **Gray code transitions** (single-bit changes) to apply incremental updates
3. Only re-evaluates terms affected by the changed bit position

This reduces per-step complexity from O(k × n) to O(affected terms).

---

## Output Format

Each discovered rule is returned as a **label string** like:

```
z0 AND z5 AND NOT z12
z3 XOR z7
z1 AND (z4 XOR z9)
```

These labels are then mapped back to human-readable feature names using the binarization dictionary (`binarization_info`), yielding rules like:

```
[top-left == x] AND [bottom-right == x] AND NOT [middle-left == x]
```

---

## What Exactor Core is NOT

- ❌ It does **not** use Quine-McCluskey minimization
- ❌ It does **not** use Petrick's method
- ❌ It does **not** use genetic algorithms for rule discovery (GA is used only for hyperparameter optimization in the outer loop)
- ❌ It does **not** use any continuous mathematics internally
