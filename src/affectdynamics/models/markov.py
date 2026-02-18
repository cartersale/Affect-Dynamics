from __future__ import annotations

import numpy as np


def _normalize_rows(A: np.ndarray) -> np.ndarray:
    row_sums = A.sum(axis=1, keepdims=True)
    return A / row_sums


def fit_markov_1(seqs, alpha: float = 1.0):
    """
    Fit a first-order Markov chain to a list of sequences.

    Args:
        seqs: A list of integer arrays (sequences) or a single integer array.
        alpha: Laplace smoothing parameter (additive smoothing).

    Returns:
        A transition probability matrix P where P[i, j] is the probability 
        of transitioning from state i to state j.
    """
    if isinstance(seqs, np.ndarray):
        seqs = [seqs]
    # determine number of states
    max_state = -1
    for s in seqs:
        if len(s) > 0:
            max_state = max(max_state, int(np.max(s)))
    if max_state < 0:
        return np.zeros((0, 0))
    K = max_state + 1
    counts = np.full((K, K), alpha, dtype=float)
    for s in seqs:
        for i in range(len(s) - 1):
            counts[int(s[i]), int(s[i + 1])] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    # avoid division by zero for states with no outgoing counts (shouldn't happen with alpha>0)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums
    return P


def fit_markov_2(prev1_seqs, prev2_seqs, next_seqs, alpha: float = 1.0):
    """
    Fit a second-order (coupled) Markov model.
    Models P(next | prev1, prev2).

    Args:
        prev1_seqs: Sequences of the first predictor variable (e.g., self at t-1).
        prev2_seqs: Sequences of the second predictor variable (e.g., partner at t-1).
        next_seqs: Sequences of the target variable (e.g., self at t).
        alpha: Laplace smoothing parameter.

    Returns:
        A 3D probability array P where P[i, j, k] is the probability of 
        outcome k given antecedents i and j.
    """
    if isinstance(prev1_seqs, np.ndarray):
        prev1_seqs = [prev1_seqs]
        prev2_seqs = [prev2_seqs]
        next_seqs = [next_seqs]
    max1 = -1
    max2 = -1
    maxn = -1
    for a in prev1_seqs:
        if len(a) > 0:
            max1 = max(max1, int(np.max(a)))
    for b in prev2_seqs:
        if len(b) > 0:
            max2 = max(max2, int(np.max(b)))
    for c in next_seqs:
        if len(c) > 0:
            maxn = max(maxn, int(np.max(c)))
    if maxn < 0:
        return np.zeros((max1 + 1, max2 + 1, 0))
    K1, K2, Kn = max1 + 1, max2 + 1, maxn + 1
    counts = np.full((K1, K2, Kn), alpha, dtype=float)
    for a, b, c in zip(prev1_seqs, prev2_seqs, next_seqs, strict=False):
        L = min(len(a), len(b), len(c))
        for i in range(L):
            counts[int(a[i]), int(b[i]), int(c[i])] += 1.0
    sums = counts.sum(axis=2, keepdims=True)
    sums[sums == 0] = 1.0
    P = counts / sums
    return P


def loglik_markov_1(seq, P):
    if len(seq) < 2 or P.size == 0:
        return 0.0
    ll = 0.0
    for i in range(len(seq) - 1):
        p = P[int(seq[i]), int(seq[i + 1])]
        ll += np.log(p)
    return float(ll)


def loglik_markov_2(prev1, prev2, next_, P):
    if len(next_) == 0 or P.size == 0:
        return 0.0
    ll = 0.0
    for i in range(len(next_)):
        p = P[int(prev1[i]), int(prev2[i]), int(next_[i])]
        ll += np.log(p)
    return float(ll)
