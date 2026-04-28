#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from collections import Counter
from typing import Iterable

NUCLEOTIDES: tuple[str, ...] = ("A", "C", "G", "T")
END: str = "end"
STATES: tuple[str, ...] = (*NUCLEOTIDES, END)
PSEUDOCOUNT: int = 1
SMOOTHING_DENOM: int = PSEUDOCOUNT * len(STATES)

LABEL_CPG: int = 1
LABEL_NULL: int = 0

MarkovModel = tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]


def read_sequences(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines()


def train_markov(sequences: Iterable[str]) -> MarkovModel:
    start: dict[str, int] = {a: 0 for a in NUCLEOTIDES}
    trans: dict[str, dict[str, int]] = {a: {b: 0 for b in NUCLEOTIDES} for a in NUCLEOTIDES}
    end: dict[str, int] = {a: 0 for a in NUCLEOTIDES}
    n_seqs = 0

    for seq in sequences:
        n_seqs += 1
        start[seq[0]] += 1
        for i in range(len(seq) - 1):
            trans[seq[i]][seq[i + 1]] += 1
        end[seq[-1]] += 1

    start_denom = n_seqs + SMOOTHING_DENOM
    log_start = {a: math.log((start[a] + PSEUDOCOUNT) / start_denom) for a in NUCLEOTIDES}

    log_trans: dict[str, dict[str, float]] = {}
    log_end: dict[str, float] = {}
    for a in NUCLEOTIDES:
        denom = sum(trans[a].values()) + SMOOTHING_DENOM
        log_trans[a] = {b: math.log((trans[a][b] + PSEUDOCOUNT) / denom) for b in NUCLEOTIDES}
        log_end[a] = math.log((end[a] + PSEUDOCOUNT) / denom)

    return log_start, log_trans, log_end


def log_likelihood(seq: str, model: MarkovModel) -> float:
    log_start, log_trans, log_end = model
    log_p = log_start[seq[0]]
    for i in range(len(seq) - 1):
        log_p += log_trans[seq[i]][seq[i + 1]]
    return log_p + log_end[seq[-1]]


def classify(seq: str, cpg: MarkovModel, null: MarkovModel,
             log_prior_cpg: float, log_prior_null: float) -> int:
    score_cpg = log_prior_cpg + log_likelihood(seq, cpg)
    score_null = log_prior_null + log_likelihood(seq, null)
    return LABEL_CPG if score_cpg > score_null else LABEL_NULL


def compute_metrics(predictions: list[int], truth: list[int]) -> tuple[int, int, float, float, float]:
    counts = Counter(zip(predictions, truth))
    tp = counts[(LABEL_CPG, LABEL_CPG)]
    fp = counts[(LABEL_CPG, LABEL_NULL)]
    fn = counts[(LABEL_NULL, LABEL_CPG)]
    tn = counts[(LABEL_NULL, LABEL_NULL)]

    total = len(predictions)
    correct = tp + tn
    wrong = total - correct
    accuracy = correct / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return correct, wrong, accuracy, precision, recall


def main() -> None:
    cpg_train = read_sequences("cpg_train.txt")
    null_train = read_sequences("null_train.txt")
    test_seqs = read_sequences("seqs_test.txt")
    truth = [int(c) for c in read_sequences("classes_test.txt")]

    cpg_model = train_markov(cpg_train)
    null_model = train_markov(null_train)

    total_train = len(cpg_train) + len(null_train)
    log_prior_cpg = math.log(len(cpg_train) / total_train)
    log_prior_null = math.log(len(null_train) / total_train)

    predictions = [classify(s, cpg_model, null_model, log_prior_cpg, log_prior_null)
                   for s in test_seqs]

    with open("predictions.txt", "w", encoding="utf-8") as fh:
        fh.writelines(f"{p}\n" for p in predictions)

    metrics = compute_metrics(predictions, truth)
    with open("accuracy.txt", "w", encoding="utf-8") as fh:
        for v in metrics:
            fh.write(f"{v}\n")


if __name__ == "__main__":
    sys.exit(main())
