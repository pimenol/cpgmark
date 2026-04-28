# CpG-Island Recognition

Assignment 3, BIN (Bioinformatics) — FEL CTU.

Trains two first-order Markov models (CpG vs. null) with Laplace smoothing
and classifies test sequences using a Naive Bayes decision rule in log-space.

## Files

- `cpg.py` — solution (Python 3, stdlib only)
- `compile.sh` — makes `cpg.py` executable, checks Python version
- `cpg.sh` — runs the solution
- `packages.txt` — empty, no external packages needed

## Inputs (CWD)

- `cpg_train.txt`, `null_train.txt` — one sequence per line
- `seqs_test.txt` — test sequences
- `classes_test.txt` — ground-truth labels (1 = CpG, 0 = null)

## Outputs (CWD)

- `predictions.txt` — one label per line, in order of `seqs_test.txt`
- `accuracy.txt` — 5 lines: correct, wrong, accuracy, precision, recall

## Run

```sh
bash compile.sh
bash cpg.sh
```

## Method

Alphabet Σ = {A, C, G, T, end}. Per class:

- prior over first symbol: `P(a) = (count(a) + 1) / (N_seqs + 5)`
- transitions: `P(b | a) = (count(ab) + 1) / (count(a·) + 5)`
- sequence likelihood: `P(x|y) = P_y(x_1) · Π P_y(x_{i+1}|x_i) · P_y(end|x_L)`

Predict CpG iff `log P(CpG) + log P(x|CpG) > log P(null) + log P(x|null)`.

## Result on provided data

59 / 60 correct — accuracy 0.983, precision 0.968, recall 1.0.
