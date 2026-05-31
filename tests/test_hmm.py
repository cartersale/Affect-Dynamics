from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from affectdynamics.models.hmm import SharedEmissionHMM
from affectdynamics.schemas import Session


def test_expected_stats_match_full_posterior() -> None:
    T = np.array([0, 1, 2, 2, 1, 0, 1, 1], dtype=np.int64)
    C = np.array([1, 2, 0, 1, 1, 2, 0, 1], dtype=np.int64)
    model = SharedEmissionHMM(K=3)
    model.init_random(np.random.default_rng(21))

    gamma, xi, ll = model._fb(T, C)
    pi_exp, A_exp, T_exp, C_exp, stats_ll = model._expected_stats(T, C)
    logpi, logA, _, _ = model._log_params()
    logB = model._log_emissions(T, C)
    alpha = logpi + logB[0]
    for t in range(1, len(T)):
        alpha = logB[t] + logsumexp(alpha[:, None] + logA, axis=0)
    reference_ll = float(logsumexp(alpha))

    assert np.allclose(pi_exp, gamma[0], atol=1e-12)
    assert np.allclose(A_exp, xi.sum(axis=0), atol=1e-12)
    for k in range(model.K):
        assert np.allclose(T_exp[k], np.bincount(T, weights=gamma[:, k], minlength=3))
        assert np.allclose(C_exp[k], np.bincount(C, weights=gamma[:, k], minlength=3))
    assert np.isclose(stats_ll, ll, atol=1e-12)
    assert np.allclose(model.posterior(T, C), gamma, atol=1e-12)
    assert np.isclose(model.score_sequences([(T, C)]), ll, atol=1e-12)
    assert np.isclose(ll, reference_ll, atol=1e-12)


def test_fit_script_refits_and_resumes_checkpoints(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    out = tmp_path / "out"
    processed.mkdir()
    rows = []
    sequences = [
        ("001_A", "001", [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 1]),
        ("001_B", "001", [0, 1, 0, 1, 2, 1], [1, 1, 0, 2, 1, 2]),
        ("002_A", "002", [2, 2, 1, 1, 0, 0], [2, 1, 1, 0, 0, 1]),
        ("002_B", "002", [2, 1, 2, 1, 0, 1], [1, 2, 1, 0, 1, 0]),
    ]
    for sid, dyad, therapist, client in sequences:
        filename = f"{sid}.parquet"
        Session(
            session_id=sid,
            dyad_id=dyad,
            therapist_codes=np.asarray(therapist),
            client_codes=np.asarray(client),
            t=np.arange(len(therapist)),
            meta={},
        ).to_parquet(processed / filename)
        rows.append({"file": filename, "dyad_id": dyad})
    pd.DataFrame(rows).to_csv(processed / "processed_manifest.csv", index=False)

    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo / "scripts" / "03_fit_shared_hmm.py"),
        "--processed_dir",
        str(processed),
        "--out_dir",
        str(out),
        "--n_splits",
        "2",
        "--k_min",
        "2",
        "--k_max",
        "2",
        "--n_iter",
        "3",
        "--n_restarts",
        "1",
        "--refit_restarts",
        "1",
        "--n_jobs",
        "2",
    ]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    first = subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True, text=True)
    assert "checkpointed refit restart" in first.stdout

    params = json.loads((out / "best_model.json").read_text())
    run_dir = next((out / "hmm_checkpoints").iterdir())
    refit_path = next(run_dir.glob("refit_*.npz"))
    with np.load(refit_path) as refit:
        assert np.allclose(params["pi"], refit["pi"])
        assert np.allclose(params["A"], refit["A"])
        assert np.isclose(params["refit_train_ll_total"], float(refit["train_ll"]))
    assert len(pd.read_csv(out / "decoded_sessions.csv")) == len(sequences)
    assert len(list(run_dir.glob("cv_*.npz"))) == 2

    second = subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True, text=True)
    assert "resumed completed fold K=2 fold=0" in second.stdout
    assert "checkpointed refit restart" not in second.stdout
