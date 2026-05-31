from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without optional accelerator.
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def _em_stats_kernel(
    T: np.ndarray,
    C: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    pT: np.ndarray,
    pC: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = T.shape[0]
    K = pi.shape[0]
    n_obs = pT.shape[1]
    B = np.empty((n, K))
    alpha = np.empty((n, K))
    beta = np.empty((n, K))
    scales = np.empty(n)

    for t in range(n):
        for k in range(K):
            B[t, k] = pT[k, T[t]] * pC[k, C[t]]

    scale = 0.0
    for k in range(K):
        alpha[0, k] = pi[k] * B[0, k]
        scale += alpha[0, k]
    scales[0] = scale
    for k in range(K):
        alpha[0, k] /= scale

    for t in range(1, n):
        scale = 0.0
        for j in range(K):
            prob = 0.0
            for i in range(K):
                prob += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = prob * B[t, j]
            scale += alpha[t, j]
        scales[t] = scale
        for j in range(K):
            alpha[t, j] /= scale

    for k in range(K):
        beta[n - 1, k] = 1.0
    for t in range(n - 2, -1, -1):
        for i in range(K):
            prob = 0.0
            for j in range(K):
                prob += A[i, j] * B[t + 1, j] * beta[t + 1, j]
            beta[t, i] = prob / scales[t + 1]

    pi_exp = np.zeros(K)
    A_exp = np.zeros((K, K))
    T_exp = np.zeros((K, n_obs))
    C_exp = np.zeros((K, n_obs))
    total_ll = 0.0

    for t in range(n):
        total_ll += np.log(scales[t])
        norm = 0.0
        for k in range(K):
            norm += alpha[t, k] * beta[t, k]
        for k in range(K):
            gamma = alpha[t, k] * beta[t, k] / norm
            if t == 0:
                pi_exp[k] = gamma
            T_exp[k, T[t]] += gamma
            C_exp[k, C[t]] += gamma

    for t in range(n - 1):
        norm = 0.0
        for i in range(K):
            for j in range(K):
                norm += alpha[t, i] * A[i, j] * B[t + 1, j] * beta[t + 1, j]
        for i in range(K):
            for j in range(K):
                A_exp[i, j] += (
                    alpha[t, i] * A[i, j] * B[t + 1, j] * beta[t + 1, j] / norm
                )

    return pi_exp, A_exp, T_exp, C_exp, total_ll


@njit(cache=True)
def _posterior_kernel(
    T: np.ndarray,
    C: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    pT: np.ndarray,
    pC: np.ndarray,
) -> tuple[np.ndarray, float]:
    n = T.shape[0]
    K = pi.shape[0]
    B = np.empty((n, K))
    alpha = np.empty((n, K))
    beta = np.empty((n, K))
    scales = np.empty(n)

    for t in range(n):
        for k in range(K):
            B[t, k] = pT[k, T[t]] * pC[k, C[t]]

    scale = 0.0
    for k in range(K):
        alpha[0, k] = pi[k] * B[0, k]
        scale += alpha[0, k]
    scales[0] = scale
    for k in range(K):
        alpha[0, k] /= scale

    for t in range(1, n):
        scale = 0.0
        for j in range(K):
            prob = 0.0
            for i in range(K):
                prob += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = prob * B[t, j]
            scale += alpha[t, j]
        scales[t] = scale
        for j in range(K):
            alpha[t, j] /= scale

    for k in range(K):
        beta[n - 1, k] = 1.0
    for t in range(n - 2, -1, -1):
        for i in range(K):
            prob = 0.0
            for j in range(K):
                prob += A[i, j] * B[t + 1, j] * beta[t + 1, j]
            beta[t, i] = prob / scales[t + 1]

    gamma = np.empty((n, K))
    ll = 0.0
    for t in range(n):
        ll += np.log(scales[t])
        norm = 0.0
        for k in range(K):
            gamma[t, k] = alpha[t, k] * beta[t, k]
            norm += gamma[t, k]
        for k in range(K):
            gamma[t, k] /= norm
    return gamma, ll


@njit(cache=True)
def _score_kernel(
    T: np.ndarray,
    C: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    pT: np.ndarray,
    pC: np.ndarray,
) -> float:
    K = pi.shape[0]
    previous = np.empty(K)
    current = np.empty(K)
    scale = 0.0
    for k in range(K):
        previous[k] = pi[k] * pT[k, T[0]] * pC[k, C[0]]
        scale += previous[k]
    ll = np.log(scale)
    for k in range(K):
        previous[k] /= scale

    for t in range(1, T.shape[0]):
        scale = 0.0
        for j in range(K):
            prob = 0.0
            for i in range(K):
                prob += previous[i] * A[i, j]
            current[j] = prob * pT[j, T[t]] * pC[j, C[t]]
            scale += current[j]
        ll += np.log(scale)
        for j in range(K):
            previous[j] = current[j] / scale
    return ll


@dataclass
class SharedEmissionHMM:
    """
    HMM with one latent chain and two conditionally independent discrete emissions.
    """

    K: int
    n_obs: int = 3
    pi: np.ndarray | None = None
    A: np.ndarray | None = None
    pT: np.ndarray | None = None
    pC: np.ndarray | None = None
    compiled: bool = True
    n_iter_: int = 0

    def init_random(self, rng: np.random.Generator) -> None:
        self.pi = rng.random(self.K)
        self.pi /= self.pi.sum()
        self.A = rng.random((self.K, self.K))
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pT = rng.random((self.K, self.n_obs))
        self.pT /= self.pT.sum(axis=1, keepdims=True)
        self.pC = rng.random((self.K, self.n_obs))
        self.pC /= self.pC.sum(axis=1, keepdims=True)

    def _params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert (
            self.pi is not None
            and self.A is not None
            and self.pT is not None
            and self.pC is not None
        )
        return self.pi, self.A, self.pT, self.pC

    def _log_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(np.log(np.clip(p, 1e-300, 1.0)) for p in self._params())  # type: ignore[return-value]

    def _log_emissions(self, T: np.ndarray, C: np.ndarray) -> np.ndarray:
        _, _, logpT, logpC = self._log_params()
        return logpT[:, T].T + logpC[:, C].T

    def _arrays(self, T: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(T, dtype=np.int64), np.asarray(C, dtype=np.int64)

    def score_sequences(self, seqs: list[tuple[np.ndarray, np.ndarray]]) -> float:
        return float(sum(self._forward_loglik(T, C) for T, C in seqs))

    def _forward_loglik(self, T: np.ndarray, C: np.ndarray) -> float:
        T, C = self._arrays(T, C)
        return float(_score_kernel(T, C, *self._params()))

    def _fb(self, T: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return full posteriors; EM uses `_expected_stats` to avoid storing `xi`."""
        T, C = self._arrays(T, C)
        pi, A, pT, pC = self._params()
        gamma, ll = _posterior_kernel(T, C, pi, A, pT, pC)
        B = pT[:, T].T * pC[:, C].T
        n = len(T)
        alpha = np.empty((n, self.K), dtype=float)
        beta = np.empty((n, self.K), dtype=float)
        scales = np.empty(n, dtype=float)
        alpha[0] = pi * B[0]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            scales[t] = alpha[t].sum()
            alpha[t] /= scales[t]
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = A @ (B[t + 1] * beta[t + 1]) / scales[t + 1]
        xi = np.empty((n - 1, self.K, self.K), dtype=float)
        for t in range(n - 1):
            xi[t] = alpha[t, :, None] * A * (B[t + 1] * beta[t + 1])[None, :]
            xi[t] /= xi[t].sum()
        return gamma, xi, ll

    def _expected_stats(
        self, T: np.ndarray, C: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        T, C = self._arrays(T, C)
        return _em_stats_kernel(T, C, *self._params())

    def fit_em(
        self,
        seqs: list[tuple[np.ndarray, np.ndarray]],
        *,
        n_iter: int = 100,
        tol: float = 1e-4,
        alpha_trans: float = 1.0,
        alpha_emit: float = 1.0,
        seed: int = 0,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        if self.pi is None:
            self.init_random(rng)

        ll_trace: list[float] = []
        prev = -np.inf
        for iteration in range(n_iter):
            pi_exp = np.zeros(self.K, dtype=float)
            A_exp = np.zeros((self.K, self.K), dtype=float)
            T_exp = np.zeros((self.K, self.n_obs), dtype=float)
            C_exp = np.zeros((self.K, self.n_obs), dtype=float)
            total_ll = 0.0

            for T, C in seqs:
                if len(T) < 2:
                    continue
                pi_seq, A_seq, T_seq, C_seq, ll = self._expected_stats(T, C)
                pi_exp += pi_seq
                A_exp += A_seq
                T_exp += T_seq
                C_exp += C_seq
                total_ll += ll

            self.pi = pi_exp + alpha_trans
            self.pi /= self.pi.sum()
            self.A = A_exp + alpha_trans
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.pT = T_exp + alpha_emit
            self.pT /= self.pT.sum(axis=1, keepdims=True)
            self.pC = C_exp + alpha_emit
            self.pC /= self.pC.sum(axis=1, keepdims=True)

            ll_trace.append(float(total_ll))
            self.n_iter_ = iteration + 1
            if iteration > 0 and abs(total_ll - prev) < tol:
                break
            prev = total_ll
        return ll_trace

    def posterior(self, T: np.ndarray, C: np.ndarray) -> np.ndarray:
        T, C = self._arrays(T, C)
        gamma, _ = _posterior_kernel(T, C, *self._params())
        return gamma

    def viterbi(self, T: np.ndarray, C: np.ndarray) -> np.ndarray:
        T, C = self._arrays(T, C)
        logpi, logA, _, _ = self._log_params()
        logB = self._log_emissions(T, C)
        Tlen = len(T)
        delta = np.zeros((Tlen, self.K), dtype=float)
        psi = np.zeros((Tlen, self.K), dtype=int)
        delta[0] = logpi + logB[0]
        for t in range(1, Tlen):
            scores = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = logB[t] + scores[psi[t], np.arange(self.K)]
        path = np.zeros(Tlen, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(Tlen - 2, -1, -1):
            path[t] = int(psi[t + 1, path[t + 1]])
        return path
