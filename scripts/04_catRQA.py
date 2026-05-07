#!/usr/bin/env python3
"""
Performs Categorical Recurrence Quantification Analysis (catRQA) on joint affect
sequences within HMM-defined regimes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from affectdynamics.schemas import Session


def load_sessions(processed_dir: Path) -> tuple[list[Session], pd.DataFrame]:
    """Load all processed sessions from a directory."""
    mf = pd.read_csv(processed_dir / "processed_manifest.csv")
    sessions = [Session.from_parquet(processed_dir / f) for f in mf["file"].tolist()]
    return sessions, mf


def find_episodes(path: np.ndarray, min_len: int) -> list[tuple[int, int, int]]:
    """Find contiguous episodes of the same state."""
    episodes = []
    if len(path) == 0:
        return episodes
    
    current_state = path[0]
    start_idx = 0
    for i in range(1, len(path)):
        if path[i] != current_state:
            if i - start_idx >= min_len:
                episodes.append((current_state, start_idx, i))
            current_state = path[i]
            start_idx = i
    if len(path) - start_idx >= min_len:
        episodes.append((current_state, start_idx, len(path)))
    return episodes


def recurrence_matrix_1d(x: np.ndarray, theiler_w: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build recurrence matrix R for 1D categorical sequence x, and an 'eligible' mask.
    R[i,j]=1 if x[i]==x[j], else 0. Then exclude Theiler band (incl diagonal).
    Returns (R, eligible_mask).
    """
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return np.zeros((0, 0), dtype=np.int8), np.zeros((0, 0), dtype=bool)

    R = (x[:, None] == x[None, :]).astype(np.int8)

    # Eligible cells mask
    eligible = np.ones((n, n), dtype=bool)
    if theiler_w >= 0:
        # exclude |i-j| <= theiler_w
        idx = np.arange(n)
        eligible &= (np.abs(idx[:, None] - idx[None, :]) > theiler_w)

    # Apply exclusion
    R = R * eligible.astype(np.int8)
    return R, eligible

def _run_lengths_1d(binary_1d: np.ndarray) -> list[int]:
    """
    Lengths of contiguous 1-runs in a 1D binary array.
    """
    b = np.asarray(binary_1d, dtype=np.int8)
    if b.size == 0:
        return []
    padded = np.concatenate(([0], b, [0]))
    starts = np.where((padded[:-1] == 0) & (padded[1:] == 1))[0]
    ends   = np.where((padded[:-1] == 1) & (padded[1:] == 0))[0]
    return (ends - starts).tolist()

def line_lengths_diagonal(R: np.ndarray) -> list[int]:
    n = R.shape[0]
    lengths = []
    for k in range(-(n - 1), n):
        lengths.extend(_run_lengths_1d(np.diag(R, k)))
    return lengths

def line_lengths_vertical(R: np.ndarray) -> list[int]:
    n = R.shape[0]
    lengths = []
    for j in range(n):
        lengths.extend(_run_lengths_1d(R[:, j]))
    return lengths

def catrqa_rr_det_lam(x: np.ndarray, theiler_w: int = 1, l_min: int = 2, v_min: int = 2) -> tuple[float, float, float]:
    """
    RR: recurrent points / eligible points
    DET: points in diagonal lines (len>=l_min) / recurrent points
    LAM: points in vertical  lines (len>=v_min) / recurrent points
    """
    R, eligible = recurrence_matrix_1d(x, theiler_w=theiler_w)
    n_rec = int(R.sum())
    n_elig = int(eligible.sum())

    rr = (n_rec / n_elig) if n_elig > 0 else 0.0
    if n_rec == 0:
        return rr, 0.0, 0.0

    diag_lens = [L for L in line_lengths_diagonal(R) if L >= l_min]
    vert_lens = [L for L in line_lengths_vertical(R) if L >= v_min]

    det = (sum(diag_lens) / n_rec) if n_rec > 0 else 0.0
    lam = (sum(vert_lens) / n_rec) if n_rec > 0 else 0.0
    return rr, det, lam

def generate_markov_surrogate(x: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Generate a surrogate sequence from a 1st-order Markov model
    fit to the input sequence `x`.
    """
    states = np.unique(x)
    state_map = {s: i for i, s in enumerate(states)}
    inv_state_map = {i: s for s, i in state_map.items()}
    
    n_states = len(states)
    trans_mat = np.zeros((n_states, n_states))
    
    for i in range(len(x) - 1):
        trans_mat[state_map[x[i]], state_map[x[i+1]]] += 1
        
    # Normalize to get probabilities
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    trans_mat = np.divide(trans_mat, row_sums, where=row_sums != 0)
    
    # Generate surrogate
    surrogate = np.zeros(n_steps, dtype=x.dtype)
    surrogate[0] = x[0]
    
    for i in range(1, n_steps):
        prev_state_idx = state_map[surrogate[i-1]]
        probs = trans_mat[prev_state_idx]
        if np.sum(probs) == 0: # Handle sink states
            next_state_idx = np.random.choice(n_states)
        else:
            # Re-normalize to handle potential floating point inaccuracies
            probs /= probs.sum()
            next_state_idx = np.random.choice(n_states, p=probs)
        surrogate[i] = inv_state_map[next_state_idx]
        
    return surrogate

def catrqa_with_null(
    x: np.ndarray,
    n_null: int,
    theiler_w: int = 1,
    l_min: int = 2,
    v_min: int = 2,
) -> dict[str, float]:
    """
    Computes catRQA metrics, Z-scores, and empirical p-values against a shuffle null distribution.

    Args:
        x: The 1D categorical sequence.
        n_null: The number of null surrogates to generate.
        theiler_w: The Theiler window size for recurrence analysis.
        l_min: The minimum diagonal line length for DET calculation.
        v_min: The minimum vertical line length for LAM calculation.

    Returns:
        A dictionary containing the observed metrics (rr, det, lam), their
        Z-scores (rr_z, det_z, lam_z), and their empirical p-values
        (rr_p, det_p, lam_p) against the shuffle null distribution.
    """
    # Calculate observed metrics
    rr, det, lam = catrqa_rr_det_lam(x, theiler_w, l_min, v_min)

    # Generate null distribution
    null_rr, null_det, null_lam = np.zeros(n_null), np.zeros(n_null), np.zeros(n_null)
    for i in range(n_null):
        x_null = np.random.permutation(x)
        null_rr[i], null_det[i], null_lam[i] = catrqa_rr_det_lam(
            x_null, theiler_w, l_min, v_min
        )

    def z_score(val, null_dist):
        """Calculate the Z-score of a value against a null distribution."""
        mean, std = np.mean(null_dist), np.std(null_dist)
        return (val - mean) / std if std > 0 else 0.0

    def p_emp(val, null_dist):
        """Calculate the empirical p-value of a value against a null distribution."""
        return (1 + np.sum(null_dist >= val)) / (len(null_dist) + 1)

    # Return all metrics in a dictionary
    return {
        "rr": rr,
        "det": det,
        "lam": lam,
        "rr_z": z_score(rr, null_rr),
        "det_z": z_score(det, null_det),
        "lam_z": z_score(lam, null_lam),
        "rr_p": p_emp(rr, null_rr),
        "det_p": p_emp(det, null_det),
        "lam_p": p_emp(lam, null_lam),
    }

def main():
    """
    Main script execution. Parses arguments, loads data, runs catRQA analysis
    on HMM regime episodes, and saves the results.
    """
    # --- Argument Parsing ---
    p = argparse.ArgumentParser(description="catRQA on HMM regime episodes.")
    p.add_argument("--config", default="configs/analysis.yaml", help="Analysis YAML config.")
    args = p.parse_args()

    # --- Configuration Loading ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load configs for this specific analysis
    ja_cfg = cfg["analysis"]["joint_affect_catrqa"]
    hmm_dir = Path(ja_cfg["hmm_dir"])
    processed_data_dir = Path(ja_cfg["processed_data_dir"])
    out_dir = Path(ja_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # catRQA parameters
    n_min = int(ja_cfg.get("n_min", 10))
    plot_n_min = int(ja_cfg.get("plot_n_min", 60))
    theiler_w = int(ja_cfg.get("theiler_w", 1))
    l_min = int(ja_cfg.get("l_min", 2))
    v_min = int(ja_cfg.get("v_min", 2))
    n_example_plots = int(ja_cfg.get("n_example_plots", 3))
    n_null = int(ja_cfg.get("n_null", 200)) # Number of null surrogates for significance testing

    # Set random seed for reproducibility
    random_seed = ja_cfg.get("random_seed", None)
    if random_seed is not None:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    # --- Data Loading ---
    # Load session metadata and map session IDs to session objects
    decoded_sessions = pd.read_csv(hmm_dir / "decoded_sessions.csv")
    sessions, _ = load_sessions(processed_data_dir)
    session_map = {s.session_id: s for s in sessions}

    # --- Main Processing Loop ---
    episode_data: list[dict] = []
    example_plots_data = []
    plots_per_regime = {}  # Tracker for example plots

    # Iterate over each session with a decoded Viterbi path
    for _, row in tqdm(decoded_sessions.iterrows(), total=len(decoded_sessions)):
        session_id = row["session_id"]

        # Load Viterbi path
        path_file = hmm_dir / f"{session_id}_viterbi.npy"
        if not path_file.exists():
            continue
        viterbi_path = np.load(path_file)

        # Load raw affect codes for the session
        session = session_map.get(session_id)
        if session is None:
            continue
        therapist_codes = np.asarray(session.therapist_codes)
        client_codes = np.asarray(session.client_codes)

        # Align all data to the minimum length
        min_len_data = min(len(viterbi_path), len(therapist_codes), len(client_codes))
        if min_len_data <= 0:
            continue
        viterbi_path = viterbi_path[:min_len_data]
        therapist_codes = therapist_codes[:min_len_data]
        client_codes = client_codes[:min_len_data]

        # Create a single joint affect sequence from therapist and client codes
        joint_affect = therapist_codes * 3 + client_codes

        # Find contiguous episodes of the same HMM state (regime)
        episodes = find_episodes(viterbi_path, min_len=n_min)

        # Process each episode
        for regime, start, end in episodes:
            segment = joint_affect[start:end]
            if len(segment) < max(n_min, l_min, v_min):
                continue

            # --- Shuffle Null ---
            # Calculate catRQA metrics and compare against a shuffled null distribution
            shuffle_results = catrqa_with_null(
                segment,
                n_null=n_null,
                theiler_w=theiler_w,
                l_min=l_min,
                v_min=v_min,
            )

            # --- Collect Example Plots ---
            # Save data for a few example recurrence plots for later visualization
            if plots_per_regime.get(regime, 0) < n_example_plots and (end - start) >= plot_n_min:
                R, _ = recurrence_matrix_1d(segment, theiler_w=theiler_w)
                example_plots_data.append({
                    "session_id": session_id,
                    "regime": regime,
                    "start": start,
                    "end": end,
                    "R": R
                })
                plots_per_regime[regime] = plots_per_regime.get(regime, 0) + 1

            # --- Store Episode Results ---
            episode_data.append(
                {
                    "session_id": session_id,
                    "dyad_id": row.get("dyad_id", None),
                    "regime": int(regime),
                    "start": int(start),
                    "end": int(end),
                    "duration": int(end - start),
                    "RR_shuffle": float(shuffle_results["rr"]),
                    "DET_shuffle": float(shuffle_results["det"]),
                    "LAM_shuffle": float(shuffle_results["lam"]),
                    "RR_Z_shuffle": float(shuffle_results["rr_z"]),
                    "DET_Z_shuffle": float(shuffle_results["det_z"]),
                    "LAM_Z_shuffle": float(shuffle_results["lam_z"]),
                    "RR_p_shuffle": float(shuffle_results["rr_p"]),
                    "DET_p_shuffle": float(shuffle_results["det_p"]),
                    "LAM_p_shuffle": float(shuffle_results["lam_p"]),
                }
            )

    # --- Save All Results ---
    # Save episode-level data to a CSV file
    results_df = pd.DataFrame(episode_data)
    results_df.to_csv(out_dir / "catrqa_episode_results.csv", index=False)

    # Save example recurrence plots to files
    plot_dir = out_dir / "example_plots"
    plot_dir.mkdir(exist_ok=True)
    for i, plot_data in enumerate(example_plots_data):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(plot_data["R"], cmap="binary", origin="lower")
        ax.set_title(f"Session {plot_data['session_id']}, Regime {plot_data['regime']}\nTime {plot_data['start']}-{plot_data['end']}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Time")
        fig.savefig(plot_dir / f"example_{i}.png")
        plt.close(fig)

    # Save metadata about the run
    run_info = {
        "date": pd.Timestamp.now().isoformat(),
        "params": {
            "n_min": n_min,
            "theiler_w": theiler_w,
            "l_min": l_min,
            "v_min": v_min,
            "random_seed": random_seed,
            "n_example_plots": n_example_plots,
        },
        "inputs": {
            "hmm_dir": str(hmm_dir),
            "processed_data_dir": str(processed_data_dir),
            "decoded_sessions_csv": str(hmm_dir / "decoded_sessions.csv"),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "episode_results_csv": str(out_dir / "catrqa_episode_results.csv"),
            "example_plots_dir": str(plot_dir),
        },
    }
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Saved {len(results_df)} episodes to {out_dir / 'catrqa_episode_results.csv'}")


if __name__ == "__main__":
    main()
