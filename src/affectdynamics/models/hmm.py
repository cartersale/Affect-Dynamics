from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def _log_normalize(a: np.ndarray, axis: int = -1) -> np.ndarray:
    return a - logsumexp(a, axis=axis, keepdims=True)


def _onehot_counts(x: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    # x: (T,), weights: (T,)
    out = np.zeros(n, dtype=float)
    for xi, wi in zip(x, weights, strict=False):
        out[int(xi)] += float(wi)
    return out


@dataclass
class SharedEmissionHMM:
    """
    HMM with one latent chain S_t (K states) and two discrete emissions:
      T_t in {0,1,2}, C_t in {0,1,2}
    with conditional independence given S_t:
      p(T,C|S=k) = pT[k,T] * pC[k,C]
    """

    K: int
    n_obs: int = 3
    pi: np.ndarray | None = None  # (K,)
    A: np.ndarray | None = None  # (K,K)
    pT: np.ndarray | None = None  # (K,n_obs)
    pC: np.ndarray | None = None  # (K,n_obs)

    def init_random(self, rng: np.random.Generator) -> None:
        self.pi = rng.random(self.K)
        self.pi /= self.pi.sum()
        self.A = rng.random((self.K, self.K))
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pT = rng.random((self.K, self.n_obs))
        self.pT /= self.pT.sum(axis=1, keepdims=True)
        self.pC = rng.random((self.K, self.n_obs))
        self.pC /= self.pC.sum(axis=1, keepdims=True)

    def _log_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert (
            self.pi is not None
            and self.A is not None
            and self.pT is not None
            and self.pC is not None
        )
        logpi = np.log(np.clip(self.pi, 1e-300, 1.0))
        logA = np.log(np.clip(self.A, 1e-300, 1.0))
        logpT = np.log(np.clip(self.pT, 1e-300, 1.0))
        logpC = np.log(np.clip(self.pC, 1e-300, 1.0))
        return logpi, logA, logpT, logpC

    def _log_emissions(self, T: np.ndarray, C: np.ndarray) -> np.ndarray:
        # returns logB[t,k] = log p(T_t, C_t | S_t=k)
        logpi, logA, logpT, logpC = self._log_params()
        return logpT[:, T].T + logpC[:, C].T  # (Tlen,K)

    def score_sequences(self, seqs: list[tuple[np.ndarray, np.ndarray]]) -> float:
        # total log-likelihood across sequences
        ll = 0.0
        for T, C in seqs:
            ll += self._forward_loglik(T, C)
        return float(ll)

    def _forward_loglik(self, T: np.ndarray, C: np.ndarray) -> float:
        logpi, logA, *_ = self._log_params()
        logB = self._log_emissions(T, C)  # (Tlen,K)

        alpha = logpi + logB[0]  # (K,)
        for t in range(1, len(T)):
            alpha = logB[t] + logsumexp(alpha[:, None] + logA, axis=0)
        return float(logsumexp(alpha))

    def _fb(self, T: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Forward-backward in log space.
        Returns:
          gamma: (Tlen,K) posterior p(S_t=k|x)
          xi: (Tlen-1,K,K) posterior p(S_t=i,S_{t+1}=j|x)
          ll: scalar log-likelihood
        """
        logpi, logA, *_ = self._log_params()
        logB = self._log_emissions(T, C)
        Tlen = len(T)

        # forward
        log_alpha = np.zeros((Tlen, self.K), dtype=float)
        log_alpha[0] = logpi + logB[0]
        for t in range(1, Tlen):
            log_alpha[t] = logB[t] + logsumexp(log_alpha[t - 1][:, None] + logA, axis=0)

        ll = float(logsumexp(log_alpha[-1]))

        # backward
        log_beta = np.zeros((Tlen, self.K), dtype=float)
        log_beta[-1] = 0.0
        for t in range(Tlen - 2, -1, -1):
            log_beta[t] = logsumexp(logA + logB[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

        log_gamma = _log_normalize(log_alpha + log_beta, axis=1)
        gamma = np.exp(log_gamma)

        # xi
        xi = np.zeros((Tlen - 1, self.K, self.K), dtype=float)
        for t in range(Tlen - 1):
            log_xi_t = (
                log_alpha[t][:, None] + logA + logB[t + 1][None, :] + log_beta[t + 1][None, :]
            )
            log_xi_t = log_xi_t - logsumexp(log_xi_t)
            xi[t] = np.exp(log_xi_t)

        return gamma, xi, ll

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
        """
        EM with Dirichlet/Laplace smoothing:
          alpha_trans applied to A counts
          alpha_emit applied to emission counts
        Returns log-likelihood trace.
        """
        rng = np.random.default_rng(seed)
        if self.pi is None:
            self.init_random(rng)

        ll_trace: list[float] = []
        prev = -np.inf

        for it in range(n_iter):
            # expected sufficient statistics
            pi_exp = np.zeros(self.K, dtype=float)
            A_exp = np.zeros((self.K, self.K), dtype=float)
            T_exp = np.zeros((self.K, self.n_obs), dtype=float)
            C_exp = np.zeros((self.K, self.n_obs), dtype=float)

            total_ll = 0.0

            for T, C in seqs:
                if len(T) < 2:
                    continue
                gamma, xi, ll = self._fb(T, C)
                total_ll += ll

                pi_exp += gamma[0]
                A_exp += xi.sum(axis=0)

                # emissions
                for k in range(self.K):
                    T_exp[k] += _onehot_counts(T, gamma[:, k], self.n_obs)
                    C_exp[k] += _onehot_counts(C, gamma[:, k], self.n_obs)

            # M-step with smoothing
            self.pi = pi_exp + alpha_trans
            self.pi /= self.pi.sum()

            self.A = A_exp + alpha_trans
            self.A /= self.A.sum(axis=1, keepdims=True)

            self.pT = T_exp + alpha_emit
            self.pT /= self.pT.sum(axis=1, keepdims=True)

            self.pC = C_exp + alpha_emit
            self.pC /= self.pC.sum(axis=1, keepdims=True)

            ll_trace.append(float(total_ll))
            if it > 0 and abs(total_ll - prev) < tol:
                break
            prev = total_ll

        return ll_trace

    def posterior(self, T: np.ndarray, C: np.ndarray) -> np.ndarray:
        gamma, _, _ = self._fb(T, C)
        return gamma  # (Tlen,K)

    def viterbi(self, T: np.ndarray, C: np.ndarray) -> np.ndarray:
        # MAP state path in log space
        logpi, logA, *_ = self._log_params()
        logB = self._log_emissions(T, C)
        Tlen = len(T)

        delta = np.zeros((Tlen, self.K), dtype=float)
        psi = np.zeros((Tlen, self.K), dtype=int)

        delta[0] = logpi + logB[0]
        psi[0] = 0

        for t in range(1, Tlen):
            scores = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = logB[t] + scores[psi[t], np.arange(self.K)]

        path = np.zeros(Tlen, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(Tlen - 2, -1, -1):
            path[t] = int(psi[t + 1, path[t + 1]])
        return path
