
# This script performs analyses on therapy session data to understand how therapists and clients interact.
# It uses statistical models to measure how much their behaviors are linked (coupling) and who influences whom (directionality).
# The results are saved as CSV files for further review.

#!/usr/bin/env python3
from __future__ import annotations

# Import necessary libraries for handling files, data, and math
import argparse  
import json      
import sys       
from pathlib import Path  
import hashlib   

# Add the source directory to the system path so we can import custom modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import scientific libraries for data analysis
import numpy as np  
import pandas as pd 
import yaml         

# Import custom functions for splitting data and statistical modeling
from affectdynamics.eval.splits import groupkfold_splits
from affectdynamics.models.markov import (
    fit_markov_1,   # Fits a simple Markov model to sequences
    fit_markov_2,   # Fits a coupled Markov model to pairs of sequences
    loglik_markov_1,# Calculates likelihood for simple Markov model
    loglik_markov_2,# Calculates likelihood for coupled Markov model
)
from affectdynamics.schemas import Session # Represents a therapy session

# A small value to avoid division by zero
EPS = 1e-12

# Loads all session data from the processed directory.
# Returns a list of session objects and a table describing each session.
def load_sessions(processed_dir: Path) -> tuple[list[Session], pd.DataFrame]:
    manifest_path = processed_dir / "processed_manifest.csv"
    # Check if the manifest file exists; this file lists all sessions
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run preprocess first.")
    mf = pd.read_csv(manifest_path)

    # Ensure the manifest has a session ID column
    if "session_id" not in mf.columns:
        raise ValueError("Manifest must contain a 'session_id' column.")

    sessions: list[Session] = []
    # Load each session's data file
    for _, row in mf.iterrows():
        p = processed_dir / row["file"]
        sessions.append(Session.from_parquet(p))

    # Check that the loaded sessions match the manifest order
    bad = []
    for i, s in enumerate(sessions):
        sid_mf = str(mf.iloc[i]["session_id"])
        if str(s.session_id) != sid_mf:
            bad.append((i, sid_mf, str(s.session_id)))
            if len(bad) >= 10:
                break

    print("alignment mismatches:", len(bad))
    if bad:
        print("first mismatches:", bad[:10])

    return sessions, mf



# Combines results for each dyad (pair of therapist and client).
# Calculates weighted averages and counts for each dyad.
def aggregate_by_dyad(df: pd.DataFrame, value_col: str, weight_col: str) -> pd.DataFrame:
    # Calculate weighted mean for each dyad
    wmean = (
        df.groupby("dyad_id")
        .apply(lambda x: np.average(x[value_col], weights=x[weight_col]))
        .rename(f"{value_col}_wmean")
        .reset_index()
    )

    # Count samples and transitions for each dyad
    agg = (
        df.groupby("dyad_id")
        .agg(
            n_samples=(value_col, "size"),
            n_transitions=(weight_col, "sum"),
        )
        .reset_index()
    )

    # Merge the counts and weighted means into one table
    return pd.merge(agg, wmean, on="dyad_id")



# Measures how much therapists and clients influence each other in their sessions.
# Uses statistical models to compare independent and coupled behaviors.
# Returns a table of results for each session.
def run_coupling_test(
    sessions: list[Session], mf: pd.DataFrame, n_splits: int, alpha: float
) -> pd.DataFrame:
    # Split the data into groups so each dyad is kept together
    groups = mf["dyad_id"].astype(str).tolist()
    splits = groupkfold_splits(groups, n_splits=n_splits)
    
    rows: list[dict] = []

    for fold, sp in enumerate(splits):
        train_sessions = [sessions[i] for i in sp.train_idx]

        # Helper: create a unique hash for each sequence
        def arr_md5(x: np.ndarray) -> str:
            x = np.asarray(x, dtype=np.int16)
            return hashlib.md5(x.tobytes()).hexdigest()

        # Helper: create a hash for a group of sequences
        def multiset_hash(arrs: list[np.ndarray]) -> str:
            hs = sorted(arr_md5(a) for a in arrs)
            return hashlib.md5(("|".join(hs)).encode("utf-8")).hexdigest()

        # Get all therapist and client code sequences for training
        T_list = [s.therapist_codes for s in train_sessions]
        C_list = [s.client_codes for s in train_sessions]

        # Print hashes for debugging and reproducibility
        print(f"Fold {fold} multiset hash T:", multiset_hash(T_list))
        print(f"Fold {fold} multiset hash C:", multiset_hash(C_list))

        # Only use sequences with at least one code
        T_seqs = [s.therapist_codes for s in train_sessions if len(s.therapist_codes) >= 1]
        C_seqs = [s.client_codes for s in train_sessions if len(s.client_codes) >= 1]

        # Fit simple Markov models to therapist and client codes
        P_T = fit_markov_1(T_seqs, alpha=alpha)
        P_C = fit_markov_1(C_seqs, alpha=alpha)

        # Prepare sequences for coupled modeling (previous and next codes)
        T_prev = [
            s.therapist_codes[:-1]
            for s in train_sessions
            if len(s.therapist_codes) >= 2 and len(s.client_codes) >= 2
        ]
        C_prev = [
            s.client_codes[:-1]
            for s in train_sessions
            if len(s.therapist_codes) >= 2 and len(s.client_codes) >= 2
        ]
        T_next = [
            s.therapist_codes[1:]
            for s in train_sessions
            if len(s.therapist_codes) >= 2 and len(s.client_codes) >= 2
        ]
        C_next = [
            s.client_codes[1:]
            for s in train_sessions
            if len(s.therapist_codes) >= 2 and len(s.client_codes) >= 2
        ]

        # Count transitions for reporting
        t_trans = sum(max(len(s.therapist_codes) - 1, 0) for s in train_sessions)
        c_trans = sum(max(len(s.client_codes) - 1, 0) for s in train_sessions)

        paired_trans = sum(
            max(min(len(s.therapist_codes), len(s.client_codes)) - 1, 0)
            for s in train_sessions
        )

        print(f"Fold {fold} train transitions: T={t_trans} C={c_trans} paired={paired_trans}")

        # Fit coupled Markov models (therapist influenced by client and vice versa)
        P_T_cpl = fit_markov_2(T_prev, C_prev, T_next, alpha=alpha)
        P_C_cpl = fit_markov_2(C_prev, T_prev, C_next, alpha=alpha)

        # Test the models on the held-out sessions
        for i in sp.test_idx:
            s = sessions[i]
            T, C = s.therapist_codes, s.client_codes
            # Only analyze sessions where therapist and client codes are aligned
            if len(T) != len(C):
                continue
            if len(T) < 2:
                continue

            # Calculate how well the models fit the data
            ll0 = loglik_markov_1(T, P_T) + loglik_markov_1(C, P_C)
            ll1 = loglik_markov_2(T[:-1], C[:-1], T[1:], P_T_cpl) + loglik_markov_2(
                C[:-1], T[:-1], C[1:], P_C_cpl
            )
            n_trans = (len(T) - 1) + (len(C) - 1)

            # Store results for this session
            rows.append(
                {
                    "fold": fold,
                    "session_id": str(mf.iloc[i]["session_id"]),
                    "dyad_id": str(mf.iloc[i]["dyad_id"]),
                    "n_transitions": int(n_trans),
                    "ll0_total": float(ll0),
                    "ll1_total": float(ll1),
                    "ll0_per_trans": float(ll0 / n_trans),
                    "ll1_per_trans": float(ll1 / n_trans),
                    "gain_per_trans": float((ll1 - ll0) / n_trans),
                }
            )

    # Return all results as a table
    return pd.DataFrame(rows)



# Measures who influences whom in the session: therapist to client or client to therapist.
# Uses statistical models to compare independent and coupled behaviors for each direction.
# Returns a table of results for each session.
def run_directionality_test(
    sessions: list[Session], mf: pd.DataFrame, n_splits: int, alpha: float
) -> pd.DataFrame:
    # Split the data so each dyad stays together
    groups = mf["dyad_id"].astype(str).tolist()
    splits = groupkfold_splits(groups, n_splits=n_splits)
    rows: list[dict] = []

    for fold, sp in enumerate(splits):
        train_sessions = [sessions[i] for i in sp.train_idx]

        # Only use sessions with enough therapist codes
        train_sessions = [s for s in train_sessions if len(s.therapist_codes) >= 2]

        # Get all therapist and client code sequences
        T_seqs = [s.therapist_codes for s in train_sessions]
        C_seqs = [s.client_codes for s in train_sessions]

        # Fit independent Markov models
        P_T_ind = fit_markov_1(T_seqs, alpha=alpha)
        P_C_ind = fit_markov_1(C_seqs, alpha=alpha)

        # Prepare sequences for coupled modeling
        T_prev = [t[:-1] for t in T_seqs]
        C_prev = [c[:-1] for c in C_seqs]
        T_next = [t[1:] for t in T_seqs]
        C_next = [c[1:] for c in C_seqs]

        # Fit coupled Markov models for both directions
        P_T_cpl = fit_markov_2(T_prev, C_prev, T_next, alpha=alpha)
        P_C_cpl = fit_markov_2(C_prev, T_prev, C_next, alpha=alpha)

        # Test the models on the held-out sessions
        for i in sp.test_idx:
            s = sessions[i]
            T, C = s.therapist_codes, s.client_codes

            # Only analyze sessions where therapist and client codes are aligned
            if len(T) != len(C):
                continue
            if len(T) < 2:
                continue

            nT = len(T) - 1
            nC = len(C) - 1

            # Calculate how well the models fit the data for each direction
            ll_T_ind = loglik_markov_1(T, P_T_ind)
            ll_T_cpl = loglik_markov_2(T[:-1], C[:-1], T[1:], P_T_cpl)
            ll_C_ind = loglik_markov_1(C, P_C_ind)
            ll_C_cpl = loglik_markov_2(C[:-1], T[:-1], C[1:], P_C_cpl)

            dT_total = ll_T_cpl - ll_T_ind
            dC_total = ll_C_cpl - ll_C_ind

            # Store results for this session
            rows.append(
                {
                    "fold": fold,
                    "session_id": str(mf.iloc[i]["session_id"]),
                    "dyad_id": str(mf.iloc[i]["dyad_id"]),
                    "nT": int(nT),
                    "nC": int(nC),
                    "dT_per_trans": float(dT_total / nT),
                    "dC_per_trans": float(dC_total / nC),
                    "dT_total": float(dT_total),
                    "dC_total": float(dC_total),
                }
            )

    # Return all results as a table
    return pd.DataFrame(rows)



# Saves the coupling test results to files for later review.
# Produces both detailed and summary tables.
def save_coupling_outputs(coupling_res: pd.DataFrame, out_dir: Path) -> None:
    # Save all session-level results
    coupling_res.to_csv(out_dir / "coupling_results.csv", index=False)

    # Summarize results for each dyad
    dyad_coupling = aggregate_by_dyad(coupling_res, "gain_per_trans", "n_transitions")

    # Add extra summary columns (mean and median gain per dyad)
    extras = (
        coupling_res.groupby("dyad_id", as_index=False)
        .agg(
            gain_per_trans_mean=("gain_per_trans", "mean"),
            gain_per_trans_median=("gain_per_trans", "median"),
        )
    )
    dyad_coupling = dyad_coupling.merge(extras, on="dyad_id", how="left")
    # Save the summary table
    dyad_coupling.to_csv(out_dir / "coupling_by_dyad.csv", index=False)



# Saves the directionality test results to files for later review.
# Produces both detailed and summary tables.
def save_directionality_outputs(directionality_res: pd.DataFrame, out_dir: Path) -> None:
    # Save all session-level results
    directionality_res.to_csv(out_dir / "directionality_results.csv", index=False)

    # Summarize results for each dyad, for both directions
    dyad_dir_T = aggregate_by_dyad(directionality_res, "dT_per_trans", "nT")
    dyad_dir_C = aggregate_by_dyad(directionality_res, "dC_per_trans", "nC")
    dyad_dir = pd.merge(
        dyad_dir_T.rename(columns={"n_transitions": "nT"}),
        dyad_dir_C.rename(columns={"n_transitions": "nC"}),
        on=["dyad_id", "n_samples"],
    )
    # Calculate total transitions and average transitions per session
    dyad_dir["n_transitions"] = dyad_dir.get("nT", 0) + dyad_dir.get("nC", 0)
    dyad_dir["avg_trans_per_session"] = dyad_dir["n_transitions"] / dyad_dir["n_samples"].replace(
        0, np.nan
    )

    # Add aliases for compatibility with older notebooks
    dyad_dir["dT_wmean"] = dyad_dir["dT_per_trans_wmean"]
    dyad_dir["dC_wmean"] = dyad_dir["dC_per_trans_wmean"]

    # Save the summary table
    dyad_dir.to_csv(out_dir / "directionality_by_dyad.csv", index=False)



# Creates placeholder files for directionality horizon analysis.
# This is for compatibility with downstream scripts; full analysis can be restored if needed.
def run_directionality_horizon_placeholder(out_dir: Path, horizons: list[int]) -> None:
    summary = pd.DataFrame(index=horizons)
    summary.index.name = "horizon"
    summary.to_csv(out_dir / "directionality_horizon_summary.csv")

    metrics = {
        "horizons": horizons,
        "client_to_therapist": {"half_life": float("nan"), "crossing": float("nan")},
        "therapist_to_client": {"half_life": float("nan"), "crossing": float("nan")},
    }
    (out_dir / "directionality_horizon_metrics.json").write_text(json.dumps(metrics, indent=2))



# Converts a comma-separated string into a list of unique, sorted integers.
# Used for specifying horizons in the analysis.
def parse_horizons(text: str) -> list[int]:
    parts = [int(x.strip()) for x in text.split(",") if x.strip()]
    parts = sorted(set(parts))
    if not parts:
        raise ValueError("At least one horizon must be specified.")
    return parts



# Loads a YAML configuration file.
# Returns an empty dictionary if the file does not exist.
def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}



# Main entry point for running the analysis.
# Handles command-line arguments and configuration, then runs the requested tests.
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified coupling + directionality analysis.")
    parser.add_argument("--config", default="configs/analysis.yaml", help="Analysis YAML config.")
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--n_splits", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--run_coupling_test", action="store_true")
    parser.add_argument("--run_directionality_test", action="store_true")
    parser.add_argument("--run_directionality_horizon", action="store_true")
    parser.add_argument("--horizons", default="1,2,5,10,20,30,60")
    args = parser.parse_args()

    # Load configuration from file
    cfg = load_config(Path(args.config))
    cfg_data = cfg.get("data", {})
    cfg_analysis = cfg.get("analysis", {})

    # Set up directories and parameters, using command-line or config values
    processed_dir = Path(args.processed_dir or cfg_data.get("processed_dir", "data/processed"))
    out_dir = Path(args.out_dir or cfg_analysis.get("out_dir", "artifacts"))
    n_splits = int(args.n_splits if args.n_splits is not None else cfg_analysis.get("n_splits", 5))
    alpha = float(args.alpha if args.alpha is not None else cfg_analysis.get("alpha", 1.0))

    # Decide which tests to run
    run_coupling = args.run_coupling_test or bool(cfg_analysis.get("run_coupling_test", True))
    run_directionality = args.run_directionality_test or bool(
        cfg_analysis.get("run_directionality_test", True)
    )
    run_horizon = args.run_directionality_horizon or bool(
        cfg_analysis.get("run_directionality_horizon", False)
    )

    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all session data
    sessions, mf = load_sessions(processed_dir)
    if not sessions:
        raise RuntimeError("No sessions found in processed data.")

    # Run coupling test if requested
    if run_coupling:
        print("Running coupling test...")
        coupling_res = run_coupling_test(sessions, mf, n_splits, alpha)
        if coupling_res.empty:
            raise RuntimeError("No coupling results computed.")
        save_coupling_outputs(coupling_res, out_dir)
        print(f"Saved coupling outputs to {out_dir}")

    # Run directionality test if requested
    if run_directionality:
        print("Running directionality test...")
        directionality_res = run_directionality_test(sessions, mf, n_splits, alpha)
        if directionality_res.empty:
            raise RuntimeError("No directionality results computed.")
        save_directionality_outputs(directionality_res, out_dir)
        print(f"Saved directionality outputs to {out_dir}")

    # Run directionality horizon summary if requested
    if run_horizon:
        print("Running directionality horizon summary (compatibility mode)...")
        horizons = parse_horizons(args.horizons)
        run_directionality_horizon_placeholder(out_dir, horizons)
        print(f"Saved directionality horizon outputs to {out_dir}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
