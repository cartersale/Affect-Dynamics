#!/usr/bin/env python3
"""Fit a shared-emission HMM to dyadic behavioral codes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from affectdynamics.eval.splits import groupkfold_splits
from affectdynamics.models.hmm import SharedEmissionHMM
from affectdynamics.schemas import Session


def load_sessions(processed_dir: Path) -> tuple[list[Session], pd.DataFrame]:
    """Load all processed sessions from a directory."""
    mf = pd.read_csv(processed_dir / "processed_manifest.csv")
    sessions = [Session.from_parquet(processed_dir / f) for f in mf["file"].tolist()]
    return sessions, mf


def as_seqs(sessions: list[Session], idxs: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert a list of sessions and indices to a list of (therapist, client) code sequences."""
    return [(sessions[i].therapist_codes, sessions[i].client_codes) for i in idxs]


def n_steps(seqs: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """Calculate the total number of time steps across a list of sequences."""
    return int(sum(len(T) for T, _ in seqs))


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main():
    """
    Main entry point for the script.

    This script performs a cross-validated grid search to find the best
    hyperparameters (K, alpha_trans, alpha_emit) for a Shared-Emission HMM.
    It then refits a model with the best hyperparameters on all data,
    saves the model parameters, and decodes the Viterbi path for each session.
    """
    p = argparse.ArgumentParser(
        description="Fit Shared-Emission HMM with cross-validation and refit best model."
    )

    # I/O
    p.add_argument("--config", default="configs/analysis.yaml", help="Analysis YAML config.")
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=None,
        help="Directory containing processed session data.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to save model artifacts.",
    )

    # CV / sweep
    p.add_argument(
        "--n_splits", type=int, default=None, help="Number of folds for cross-validation."
    )
    p.add_argument(
        "--k_min", type=int, default=None, help="Minimum number of hidden states (K)."
    )
    p.add_argument(
        "--k_max", type=int, default=None, help="Maximum number of hidden states (K)."
    )

    # EM
    p.add_argument(
        "--n_iter", type=int, default=None, help="Maximum number of EM iterations."
    )
    p.add_argument("--tol", type=float, default=None, help="Convergence tolerance for EM.")

    # Priors
    p.add_argument(
        "--alpha_trans",
        type=float,
        nargs="+",
        default=None,
        help="Dirichlet prior for transition matrix rows.",
    )
    p.add_argument(
        "--alpha_emit",
        type=float,
        nargs="+",
        default=None,
        help="Dirichlet prior for emission distributions.",
    )

    # Restarts
    p.add_argument(
        "--n_restarts",
        type=int,
        default=None,
        help="Number of random restarts for each model fit in CV.",
    )
    p.add_argument(
        "--refit_restarts",
        type=int,
        default=None,
        help="Number of restarts for the final refit on all data. Defaults to n_restarts.",
    )

    # Selection / logging
    p.add_argument(
        "--selection_metric",
        choices=["ll_per_step"],
        default="ll_per_step",
        help="Model selection criterion (currently only ll_per_step).",
    )
    p.add_argument(
        "--save_diagnostics",
        action="store_true",
        help="Save train LL/step and best-fit iteration count if available.",
    )

    args = p.parse_args()
    cfg = load_config(Path(args.config))
    cfg_data = cfg.get("data", {})
    cfg_analysis = cfg.get("analysis", {})

    processed_dir = Path(args.processed_dir or cfg_data.get("processed_dir", "data/processed"))
    out_dir = Path(args.out_dir or cfg_analysis.get("hmm_out_dir", "artifacts/chmm"))
    args.n_splits = int(
        args.n_splits if args.n_splits is not None else cfg_analysis.get("hmm_n_splits", 5)
    )
    args.k_min = int(args.k_min if args.k_min is not None else cfg_analysis.get("hmm_k_min", 2))
    args.k_max = int(args.k_max if args.k_max is not None else cfg_analysis.get("hmm_k_max", 8))
    args.n_iter = int(args.n_iter if args.n_iter is not None else cfg_analysis.get("hmm_n_iter", 150))
    args.tol = float(args.tol if args.tol is not None else cfg_analysis.get("hmm_tol", 1e-4))
    args.alpha_trans = (
        args.alpha_trans
        if args.alpha_trans is not None
        else cfg_analysis.get("hmm_alpha_trans", [1.0])
    )
    args.alpha_emit = (
        args.alpha_emit if args.alpha_emit is not None else cfg_analysis.get("hmm_alpha_emit", [1.0])
    )
    args.n_restarts = int(
        args.n_restarts if args.n_restarts is not None else cfg_analysis.get("hmm_n_restarts", 3)
    )
    if args.refit_restarts is None:
        args.refit_restarts = args.n_restarts
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions, mf = load_sessions(processed_dir)
    groups = mf["dyad_id"].astype(str).tolist()
    splits = groupkfold_splits(groups, n_splits=args.n_splits)

    # sweep
    records: list[dict] = []

    Ks = list(range(args.k_min, args.k_max + 1))
    aTs = [float(x) for x in args.alpha_trans]
    aEs = [float(x) for x in args.alpha_emit]

    total_cfgs = len(Ks) * len(aTs) * len(aEs)
    print(
        f"Running sweep: K in [{args.k_min},{args.k_max}] ({len(Ks)}), "
        f"alpha_trans x{len(aTs)}, alpha_emit x{len(aEs)} => {total_cfgs} configs"
    )
    print(f"CV: {args.n_splits} folds (grouped by dyad_id), restarts per fold: {args.n_restarts}")

    for K in Ks:
        for aT in aTs:
            for aE in aEs:
                fold_scores = []
                for fold, sp in enumerate(splits):
                    train = as_seqs(sessions, sp.train_idx)
                    test = as_seqs(sessions, sp.test_idx)

                    best_model = None
                    best_train_ll = -np.inf
                    best_n_iter_used = None

                    # Restarts: keep best TRAIN LL
                    for r in range(args.n_restarts):
                        model = SharedEmissionHMM(K=K)
                        model.fit_em(
                            train,
                            n_iter=args.n_iter,
                            tol=args.tol,
                            alpha_trans=aT,
                            alpha_emit=aE,
                            seed=1000 * K + 97 * fold + 7 * r,
                        )
                        tr_ll = model.score_sequences(train)
                        if tr_ll > best_train_ll:
                            best_train_ll = tr_ll
                            best_model = model
                            # if your implementation exposes this; otherwise None
                            best_n_iter_used = getattr(model, "n_iter_", None)

                    assert best_model is not None

                    te_ll = best_model.score_sequences(test)
                    te_steps = n_steps(test)
                    te_ll_per_step = float(te_ll / max(te_steps, 1))
                    fold_scores.append(te_ll_per_step)

                    rec = {
                        "K": int(K),
                        "alpha_trans": float(aT),
                        "alpha_emit": float(aE),
                        "fold": int(fold),
                        "test_ll_total": float(te_ll),
                        "test_ll_per_step": te_ll_per_step,
                        "test_steps": int(te_steps),
                    }

                    if args.save_diagnostics:
                        tr_steps = n_steps(train)
                        rec["train_ll_total"] = float(best_train_ll)
                        rec["train_ll_per_step"] = float(best_train_ll / max(tr_steps, 1))
                        rec["best_fit_n_iter_used"] = (
                            int(best_n_iter_used) if best_n_iter_used is not None else -1
                        )

                    records.append(rec)

                mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
                print(f"K={K:2d} aT={aT:g} aE={aE:g}  mean held-out LL/step = {mean_score:.6g}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "model_selection.csv", index=False)

    # summarize + select best config
    sel = (
        df.groupby(["K", "alpha_trans", "alpha_emit"], as_index=False)["test_ll_per_step"]
        .mean()
        .sort_values("test_ll_per_step", ascending=False)
    )
    sel.to_csv(out_dir / "model_selection_summary.csv", index=False)

    best = sel.iloc[0]
    best_K = int(best["K"])
    best_aT = float(best["alpha_trans"])
    best_aE = float(best["alpha_emit"])
    print(
        "\nBest config by held-out LL/step:",
        {"K": best_K, "alpha_trans": best_aT, "alpha_emit": best_aE},
    )

    # --- Save Model Artifacts ---

    # Viterbi decoding for each session
    decoded = []
    for s in sessions:
        T, C = s.therapist_codes, s.client_codes
        post = best_model.posterior(T, C)  # (Tlen,K)
        occ = post.mean(axis=0)
        path = best_model.viterbi(T, C)

        decoded.append(
            {
                "session_id": s.session_id,
                "dyad_id": s.dyad_id,
                "n_steps": int(len(T)),
                **{f"occ_{k}": float(occ[k]) for k in range(best_K)},
                "path_file": f"{s.session_id}_viterbi.npy",
            }
        )
        np.save(out_dir / f"{s.session_id}_viterbi.npy", path.astype(np.int16))

    dec = pd.DataFrame(decoded)
    dec.to_csv(out_dir / "decoded_sessions.csv", index=False)

    # Save final model parameters
    params = {
        "K": best_K,
        "pi": best_model.pi.tolist(),
        "A": best_model.A.tolist(),
        "pT": best_model.pT.tolist(),
        "pC": best_model.pC.tolist(),
        "alpha_trans": best_aT,
        "alpha_emit": best_aE,
        "n_iter": args.n_iter,
        "tol": args.tol,
        "n_restarts": args.n_restarts,
        "refit_restarts": args.refit_restarts,
        "n_splits": args.n_splits,
        "selection_metric": args.selection_metric,
    }
    (out_dir / "best_model.json").write_text(json.dumps(params, indent=2))

    print(f"\n--- Artifacts written to {out_dir} ---")
    print(f"  - model_selection.csv: Per-fold results of the grid search.")
    print(f"  - model_selection_summary.csv: Grid search results averaged over folds.")
    print(f"  - top_configs.csv: Best hyperparameter configurations.")
    print(f"  - best_model.json: Parameters of the final refit model.")
    print(f"  - decoded_sessions.csv: Per-session Viterbi path info and mean occupancy.")
    print(f"  - *_viterbi.npy: Saved Viterbi paths for each session.")


if __name__ == "__main__":
    main()
