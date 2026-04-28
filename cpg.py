#!/usr/bin/env python3
"""
CpG-Island Recognition via first-order Markov models + Naive Bayes.

Course:  BIN — Bioinformatics, FEL CTU
Author:  Pimenov L. (pimenol1@cvut.cz)

==============================================================================
Math (exactly as specified by the assignment)
==============================================================================

Alphabet:        Σ = {A, C, G, T, end}        |Σ| = 5
Pseudocount:     +1 per cell, denominator pad = +5  (Laplace smoothing)

Per-class priors over the first symbol:
    P_y(a) = (count_y(a) + 1) / (N_y + 5)
    where N_y is the number of training sequences in class y
    (the count over first-symbol positions sums to N_y).

Per-class transitions (b follows a, OR a is the last symbol → b = "end"):
    P_y(b | a) = (count_y(a, b) + 1) / (count_y(a, ·) + 5)
    where count_y(a, ·) = Σ_{b ∈ Σ} count_y(a, b)
    so the same denominator is used for the four nucleotide successors
    AND the "end" pseudo-successor → P_y(·|a) sums to 1 across Σ.

Sequence likelihood under class y:
    P(x | y) = P_y(x_1) · Π_{i=1..L-1} P_y(x_{i+1} | x_i) · P_y(end | x_L)

Class priors P(y) come from training-file line counts:
    P(y) = N_y / (N_cpg + N_null)

Decision rule (predict label 1 ↔ CpG):
    ŷ = 1   iff   log P(CpG) + log P(x|CpG)  >  log P(null) + log P(x|null)

==============================================================================
Numerics
==============================================================================
Sequences are long, so we work in log-space throughout: every count is turned
into a log-probability once during training, and scoring is a sum.

==============================================================================
Complexity
==============================================================================
Training:  O(total chars in train files)        memory: O(|Σ|^2) = O(25)
Scoring:   O(L) per sequence, O(total test chars) overall.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCLEOTIDES: tuple[str, ...] = ("A", "C", "G", "T")
END: str = "end"
STATES: tuple[str, ...] = (*NUCLEOTIDES, END)   # |Σ| = 5
PSEUDO_DENOM: int = len(STATES)                 # = 5

CPG_TRAIN: str = "cpg_train.txt"
NULL_TRAIN: str = "null_train.txt"
SEQS_TEST: str = "seqs_test.txt"
CLASSES_TEST: str = "classes_test.txt"
PREDICTIONS_OUT: str = "predictions.txt"
ACCURACY_OUT: str = "accuracy.txt"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkovModel:
    """A trained first-order Markov model over Σ = {A, C, G, T, end}.

    All values are stored already in **log space** to avoid underflow.
    """

    log_start: dict[str, float]               # log P(x_1 = a)
    log_trans: dict[str, dict[str, float]]    # log P(b | a),  b ∈ {A,C,G,T}
    log_end: dict[str, float]                 # log P(end | a)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_sequences(path: str) -> list[str]:
    """Read non-empty stripped lines from *path*.

    Tolerates trailing whitespace / blank lines so a stray newline at EOF
    does not produce a phantom empty sequence.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _empty_counts() -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, int]]:
    start: dict[str, int] = {a: 0 for a in NUCLEOTIDES}
    trans: dict[str, dict[str, int]] = {
        a: {b: 0 for b in NUCLEOTIDES} for a in NUCLEOTIDES
    }
    end: dict[str, int] = {a: 0 for a in NUCLEOTIDES}
    return start, trans, end


def train_markov(sequences: Iterable[str]) -> MarkovModel:
    """Fit a first-order Markov model with Laplace smoothing.

    The denominator for every transition row is ``count(a, ·) + 5`` so that
    the four nucleotide successors and the ``end`` pseudo-successor share
    one normalisation — they together sum to 1 in probability space.
    The first-symbol distribution uses ``N_seqs + 5``.
    """
    start_count, trans_count, end_count = _empty_counts()
    n_seqs: int = 0

    for seq in sequences:
        if not seq:
            continue
        n_seqs += 1
        start_count[seq[0]] += 1
        for i in range(len(seq) - 1):
            trans_count[seq[i]][seq[i + 1]] += 1
        end_count[seq[-1]] += 1

    start_denom: float = n_seqs + PSEUDO_DENOM
    log_start: dict[str, float] = {
        a: math.log((start_count[a] + 1) / start_denom) for a in NUCLEOTIDES
    }

    log_trans: dict[str, dict[str, float]] = {}
    log_end: dict[str, float] = {}
    for a in NUCLEOTIDES:
        row_sum: int = sum(trans_count[a].values())
        denom: float = row_sum + PSEUDO_DENOM
        log_trans[a] = {
            b: math.log((trans_count[a][b] + 1) / denom) for b in NUCLEOTIDES
        }
        log_end[a] = math.log((end_count[a] + 1) / denom)

    return MarkovModel(log_start=log_start, log_trans=log_trans, log_end=log_end)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def log_likelihood(seq: str, model: MarkovModel) -> float:
    """Return log P(seq | model). Assumes ``seq`` is non-empty and clean."""
    log_p: float = model.log_start[seq[0]]
    for i in range(len(seq) - 1):
        log_p += model.log_trans[seq[i]][seq[i + 1]]
    log_p += model.log_end[seq[-1]]
    return log_p


def classify(
    seq: str,
    cpg: MarkovModel,
    null: MarkovModel,
    log_prior_cpg: float,
    log_prior_null: float,
) -> int:
    """Return 1 if CpG is more probable than null, else 0."""
    score_cpg: float = log_prior_cpg + log_likelihood(seq, cpg)
    score_null: float = log_prior_null + log_likelihood(seq, null)
    return 1 if score_cpg > score_null else 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    correct: int
    wrong: int
    accuracy: float
    precision: float
    recall: float


def compute_metrics(predictions: list[int], truth: list[int]) -> Metrics:
    """Standard binary-classification metrics with CpG (=1) as positive class.

    Division-by-zero in precision/recall is guarded by returning 0.0, which
    is the conventional safe fallback when the corresponding count is empty.
    """
    if len(predictions) != len(truth):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(truth)} labels"
        )

    tp = fp = tn = fn = 0
    for p, t in zip(predictions, truth):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 0:
            tn += 1
        else:  # p == 0 and t == 1
            fn += 1

    total: int = tp + fp + tn + fn
    correct: int = tp + tn
    wrong: int = fp + fn
    accuracy: float = correct / total if total else 0.0
    precision: float = tp / (tp + fp) if (tp + fp) else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) else 0.0

    return Metrics(
        correct=correct,
        wrong=wrong,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_predictions(path: str, predictions: list[int]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(f"{p}\n" for p in predictions)


def write_accuracy(path: str, m: Metrics) -> None:
    lines = [
        str(m.correct),
        str(m.wrong),
        str(m.accuracy),
        str(m.precision),
        str(m.recall),
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cpg_train = read_sequences(CPG_TRAIN)
    null_train = read_sequences(NULL_TRAIN)
    test_seqs = read_sequences(SEQS_TEST)
    truth = [int(c) for c in read_sequences(CLASSES_TEST)]

    cpg_model = train_markov(cpg_train)
    null_model = train_markov(null_train)

    n_cpg: int = len(cpg_train)
    n_null: int = len(null_train)
    total_train: int = n_cpg + n_null
    # Guarded just in case an empty training file slips through; in practice
    # both files are non-empty so this is mostly defensive.
    log_prior_cpg: float = math.log(n_cpg / total_train) if n_cpg else -math.inf
    log_prior_null: float = math.log(n_null / total_train) if n_null else -math.inf

    predictions: list[int] = [
        classify(seq, cpg_model, null_model, log_prior_cpg, log_prior_null)
        for seq in test_seqs
    ]

    write_predictions(PREDICTIONS_OUT, predictions)

    metrics = compute_metrics(predictions, truth)
    write_accuracy(ACCURACY_OUT, metrics)


if __name__ == "__main__":
    sys.exit(main())
