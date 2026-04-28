#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterable

NUCLEOTIDES: tuple[str, ...] = ("A", "C", "G", "T")
END: str = "end"
STATES: tuple[str, ...] = (*NUCLEOTIDES, END)
ALPHABET_SIZE: int = len(STATES)
PSEUDOCOUNT: int = 1
SMOOTHING_DENOM: int = PSEUDOCOUNT * ALPHABET_SIZE

LABEL_CPG: int = 1
LABEL_NULL: int = 0

CPG_TRAIN: str = "cpg_train.txt"
NULL_TRAIN: str = "null_train.txt"
SEQS_TEST: str = "seqs_test.txt"
CLASSES_TEST: str = "classes_test.txt"
PREDICTIONS_OUT: str = "predictions.txt"
ACCURACY_OUT: str = "accuracy.txt"


@dataclass(frozen=True)
class MarkovModel:
    log_start: dict[str, float]
    log_trans: dict[str, dict[str, float]]
    log_end: dict[str, float]


def read_sequences(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _empty_counts() -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, int]]:
    start: dict[str, int] = {a: 0 for a in NUCLEOTIDES}
    trans: dict[str, dict[str, int]] = {
        a: {b: 0 for b in NUCLEOTIDES} for a in NUCLEOTIDES
    }
    end: dict[str, int] = {a: 0 for a in NUCLEOTIDES}
    return start, trans, end


def train_markov(sequences: Iterable[str]) -> MarkovModel:
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

    start_denom: float = n_seqs + SMOOTHING_DENOM
    log_start: dict[str, float] = {
        a: math.log((start_count[a] + PSEUDOCOUNT) / start_denom) for a in NUCLEOTIDES
    }

    log_trans: dict[str, dict[str, float]] = {}
    log_end: dict[str, float] = {}
    for a in NUCLEOTIDES:
        row_sum: int = sum(trans_count[a].values())
        denom: float = row_sum + SMOOTHING_DENOM
        log_trans[a] = {
            b: math.log((trans_count[a][b] + PSEUDOCOUNT) / denom) for b in NUCLEOTIDES
        }
        log_end[a] = math.log((end_count[a] + PSEUDOCOUNT) / denom)

    return MarkovModel(log_start=log_start, log_trans=log_trans, log_end=log_end)


def log_likelihood(seq: str, model: MarkovModel) -> float:
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
    score_cpg: float = log_prior_cpg + log_likelihood(seq, cpg)
    score_null: float = log_prior_null + log_likelihood(seq, null)
    return LABEL_CPG if score_cpg > score_null else LABEL_NULL


@dataclass(frozen=True)
class Metrics:
    correct: int
    wrong: int
    accuracy: float
    precision: float
    recall: float


def compute_metrics(predictions: list[int], truth: list[int]) -> Metrics:
    if len(predictions) != len(truth):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(truth)} labels"
        )

    tp = fp = tn = fn = 0
    for p, t in zip(predictions, truth):
        if p == LABEL_CPG and t == LABEL_CPG:
            tp += 1
        elif p == LABEL_CPG and t == LABEL_NULL:
            fp += 1
        elif p == LABEL_NULL and t == LABEL_NULL:
            tn += 1
        else:
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
