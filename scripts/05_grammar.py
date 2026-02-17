#!/usr/bin/env python3
"""Grammar-style motif analysis of joint affect sequences within HMM regimes.

This script extracts k-gram motifs from joint affect streams (therapist, client)
inside HMM-identified regimes (contiguous viterbi episodes). For each episode it
computes motif distribution statistics (entropy, effective count, top-mass,
repeat-rate, transition diversity) and compares them to nulls (frequency-shuffled
and first-order Markov surrogates).

Inputs (configured via `configs/analysis.yaml` under `analysis.joint_affect_grammar`):
- `hmm_dir`: folder containing HMM outputs (`decoded_sessions.csv` and `*_viterbi.npy`).
- `processed_data_dir`: folder with processed session Parquet files and manifest.
- `out_dir`: where episode-level CSV results are written.
- `ks`, `top_n`, `n_null`: parameters controlling motif length, toplist size, and null draws.
- `min_null_windows`, `min_null_duration`: thresholds for running null comparisons.

Outputs:
- `grammar_episode_results.csv` in `out_dir` with episode-level metrics and null stats.

"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

from affectdynamics.schemas import Session




def load_sessions(processed_dir: Path) -> tuple[list[Session], pd.DataFrame]:
	"""Load session objects from a processed data directory.

	Expects `processed_manifest.csv` listing parquet filenames under
	`processed_dir`. Returns a list of `Session` dataclass instances and the
	manifest DataFrame. This is a small helper to keep I/O centralized.
	"""
	mf = pd.read_csv(processed_dir / "processed_manifest.csv")
	sessions = [Session.from_parquet(processed_dir / f) for f in mf["file"].tolist()]
	return sessions, mf


def find_episodes(path: np.ndarray, min_len: int) -> list[tuple[int, int, int]]:
	"""Return contiguous regime episodes (state, start_idx, end_idx).

	Episodes shorter than `min_len` are excluded. This mirrors the logic used
	in later analyses and in the catRQA script so episode boundaries are
	consistent across analyses.
	"""
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


def build_joint_affect(therapist_codes: np.ndarray, client_codes: np.ndarray) -> np.ndarray:
	"""Encode joint therapist/client affect into single integer symbols.

	We map `(T_t, C_t)` to `J_t = T_t * 3 + C_t` producing 9 possible joint
	symbols (0..8). This compact encoding enables fast vectorized motif
	extraction below.
	"""
	t_codes = np.asarray(therapist_codes)
	c_codes = np.asarray(client_codes)
	return t_codes * 3 + c_codes


def motif_to_str(motif: Iterable[int]) -> str:
	return "-".join(str(int(v)) for v in motif)


def decode_motif_id(motif_id: int, k: int) -> tuple[int, ...]:
	"""Decode a base-9 motif id back to tuple of symbols of length k."""
	digits: list[int] = []
	rest = int(motif_id)
	for _ in range(k):
		digits.append(rest % 9)
		rest //= 9
	return tuple(reversed(digits))


def motif_ids(seq: np.ndarray, k: int) -> np.ndarray:
	"""Vectorized encoding of length-k motifs to integer IDs.

	Uses `sliding_window_view` to create overlapping windows and treats each
	window as a base-9 number to produce a unique integer id for each motif.
	This avoids Python tuple overhead and significantly speeds up null
	computations for long episodes.
	"""
	if len(seq) < k:
		return np.array([], dtype=np.int64)
	windows = sliding_window_view(seq, k)
	powers = (9 ** np.arange(k - 1, -1, -1, dtype=np.int64)).reshape(1, -1)
	return (windows * powers).sum(axis=1, dtype=np.int64)


def compute_motif_metrics(motif_ids_seq: np.ndarray, k: int, top_n: int) -> dict[str, float | int | list[tuple[tuple[int, ...], float]]]:
	"""Compute motif statistics from encoded motif-id sequence.

	Returns a dictionary with motif entropy, effective motif count (exp(entropy)),
	top-motif probabilities, proportional top-mass, repeat-rate (adjacent motif
	repeats), transition entropy, and a normalized transition-diversity score.
	This function operates on integer IDs and is designed for efficiency.
	"""
	n_windows = int(motif_ids_seq.size)
	if n_windows == 0:
		return {
			"n_windows": 0,
			"n_unique": 0,
			"motif_entropy": 0.0,
			"effective_motif_count": 0.0,
			"top_motif": (),
			"top_motif_prob": 0.0,
			"top_mass": 0.0,
			"repeat_rate": 0.0,
			"transition_entropy": 0.0,
			"motif_transition_diversity": 0.0,
			"top_motifs": [],
		}

	unique_ids, counts = np.unique(motif_ids_seq, return_counts=True)
	probs = counts / n_windows
	motif_entropy = float(-np.sum(probs * np.log(probs)))
	effective_motif_count = float(np.exp(motif_entropy))

	sort_idx = np.argsort(-probs)
	unique_ids_sorted = unique_ids[sort_idx]
	probs_sorted = probs[sort_idx]
	top_mass = float(probs_sorted[:top_n].sum())
	top_motif_id = int(unique_ids_sorted[0])
	top_motif_prob = float(probs_sorted[0])
	top_motifs = [
		(decode_motif_id(int(mid), k), float(p))
		for mid, p in zip(unique_ids_sorted[:top_n], probs_sorted[:top_n])
	]

	repeat_rate = 0.0
	transition_entropy = 0.0
	motif_transition_diversity = 0.0
	if n_windows > 1:
		repeats = np.sum(motif_ids_seq[1:] == motif_ids_seq[:-1])
		repeat_rate = float(repeats / (n_windows - 1))

		base = int(9 ** k)
		trans_ids = motif_ids_seq[:-1] * base + motif_ids_seq[1:]
		trans_unique, trans_counts = np.unique(trans_ids, return_counts=True)
		trans_probs = trans_counts / (n_windows - 1)
		transition_entropy = float(-np.sum(trans_probs * np.log(trans_probs)))
		max_ent = float(np.log(len(trans_unique))) if len(trans_unique) > 0 else 0.0
		motif_transition_diversity = (transition_entropy / max_ent) if max_ent > 0 else 0.0

	return {
		"n_windows": n_windows,
		"n_unique": int(len(unique_ids)),
		"motif_entropy": motif_entropy,
		"effective_motif_count": effective_motif_count,
		"top_motif": decode_motif_id(top_motif_id, k),
		"top_motif_prob": top_motif_prob,
		"top_mass": top_mass,
		"repeat_rate": repeat_rate,
		"transition_entropy": transition_entropy,
		"motif_transition_diversity": motif_transition_diversity,
		"top_motifs": top_motifs,
	}


def freq_matched_shuffle(seq: np.ndarray) -> np.ndarray:
	"""Shuffle sequence to preserve symbol base rates while destroying order."""
	return np.random.permutation(seq)


def z_score(val: float, null_dist: np.ndarray) -> float:
	mean = float(np.mean(null_dist))
	std = float(np.std(null_dist))
	return (val - mean) / std if std > 0 else 0.0


def empirical_p_values(val: float, null_dist: np.ndarray) -> tuple[float, float, float]:
	"""One-sided upper/lower and two-sided empirical p-values."""
	n = len(null_dist)
	if n == 0:
		return float("nan"), float("nan"), float("nan")
	upper = (1 + np.sum(null_dist >= val)) / (n + 1)
	lower = (1 + np.sum(null_dist <= val)) / (n + 1)
	two_sided = min(1.0, 2 * min(upper, lower))
	return float(upper), float(lower), float(two_sided)


def metrics_from_null(null_metrics: list[dict[str, float]], key: str) -> np.ndarray:
	return np.array([m[key] for m in null_metrics], dtype=float)


def generate_markov_surrogate(x: np.ndarray, n_steps: int) -> np.ndarray:
	"""Fit a 1st-order Markov chain to `x` and simulate a surrogate of length `n_steps`.

	The surrogate preserves first-order transition structure (and roughly the
	stationary distribution) but destroys higher-order ordering. Used as an
	alternative null alongside frequency shuffling.
	"""
	states, inv = np.unique(x, return_inverse=True)
	n_states = len(states)
	trans = np.zeros((n_states, n_states), dtype=np.int64)
	for a, b in zip(inv[:-1], inv[1:]):
		trans[a, b] += 1
	row_sums = trans.sum(axis=1, keepdims=True)
	probs = np.divide(trans, row_sums, where=row_sums != 0)

	p0 = np.bincount(inv, minlength=n_states) / len(inv)
	surrogate_idx = np.zeros(n_steps, dtype=np.int64)
	surrogate_idx[0] = np.random.choice(n_states, p=p0)

	for i in range(1, n_steps):
		prev = surrogate_idx[i - 1]
		p = probs[prev]
		if p.sum() == 0:
			surrogate_idx[i] = np.random.randint(0, n_states)
		else:
			p = p / p.sum()
			surrogate_idx[i] = np.random.choice(n_states, p=p)
	return states[surrogate_idx]


def main():
	p = argparse.ArgumentParser(description="Joint affect grammar motifs within HMM regimes")
	p.add_argument("--config", default="configs/analysis.yaml", help="Analysis YAML config.")
	args = p.parse_args()

	with open(args.config) as f:
		cfg = yaml.safe_load(f)

	ga_cfg = cfg["analysis"]["joint_affect_grammar"]

	# These control thresholds and paths used across the analysis.
	hmm_dir = Path(ga_cfg["hmm_dir"])
	processed_data_dir = Path(ga_cfg["processed_data_dir"])
	out_dir = Path(ga_cfg["out_dir"])
	out_dir.mkdir(parents=True, exist_ok=True)

	n_min = int(ga_cfg.get("n_min", 10))
	ks = [int(k) for k in ga_cfg.get("ks", [2, 3])]
	top_n = int(ga_cfg.get("top_n", 5))
	n_null = int(ga_cfg.get("n_null", 200))
	random_seed = ga_cfg.get("random_seed", None)
	if random_seed is not None:
		random.seed(int(random_seed))
		np.random.seed(int(random_seed))

	# Null thresholds (moved to config)
	min_null_windows = int(ga_cfg.get("min_null_windows", 30))
	min_null_duration = int(ga_cfg.get("min_null_duration", 60))

	decoded_sessions = pd.read_csv(hmm_dir / "decoded_sessions.csv")
	sessions, _ = load_sessions(processed_data_dir)
	session_map = {s.session_id: s for s in sessions}

	episode_rows: list[dict] = []

	for _, row in tqdm(decoded_sessions.iterrows(), total=len(decoded_sessions)):
		session_id = row["session_id"]
		path_file = hmm_dir / f"{session_id}_viterbi.npy"
		if not path_file.exists():
			continue

		viterbi_path = np.load(path_file)
		session = session_map.get(session_id)
		if session is None:
			continue

		therapist_codes = np.asarray(session.therapist_codes)
		client_codes = np.asarray(session.client_codes)

		min_len_data = min(len(viterbi_path), len(therapist_codes), len(client_codes))
		if min_len_data <= 0:
			continue

		viterbi_path = viterbi_path[:min_len_data]
		therapist_codes = therapist_codes[:min_len_data]
		client_codes = client_codes[:min_len_data]

		joint_affect = build_joint_affect(therapist_codes, client_codes)
		episodes = find_episodes(viterbi_path, min_len=n_min)

		for regime, start, end in episodes:
			segment = joint_affect[start:end]
			duration = end - start
			if duration < min(ks):
				continue

			for k in ks:
				if duration < k:
					continue

				motif_ids_seq = motif_ids(segment, k)
				metrics = compute_motif_metrics(motif_ids_seq, k, top_n=top_n)
				if metrics["n_windows"] == 0:
					continue

				compute_nulls = metrics["n_windows"] >= min_null_windows and duration >= min_null_duration
				null_metrics_shuffle: list[dict[str, float]] = []
				null_metrics_markov: list[dict[str, float]] = []
				if compute_nulls:
					for _ in range(n_null):
						shuffled = freq_matched_shuffle(segment)
						null_shuffle_ids = motif_ids(shuffled, k)
						null_metrics_shuffle.append(compute_motif_metrics(null_shuffle_ids, k, top_n=top_n))

						markov_seq = generate_markov_surrogate(segment, len(segment))
						null_markov_ids = motif_ids(markov_seq, k)
						null_metrics_markov.append(compute_motif_metrics(null_markov_ids, k, top_n=top_n))

				def add_stat(metrics_list: list[dict[str, float]], key: str) -> tuple[float, float, float, float]:
					if not metrics_list:
						return float("nan"), float("nan"), float("nan"), float("nan")
					null_arr = metrics_from_null(metrics_list, key)
					z = z_score(float(metrics[key]), null_arr)
					p_hi, p_lo, p_two = empirical_p_values(float(metrics[key]), null_arr)
					return z, p_hi, p_lo, p_two

				top_motifs_serialized = json.dumps(
					[
						{"motif": motif_to_str(motif), "prob": prob}
						for motif, prob in metrics["top_motifs"]
					]
				)

				(
					motif_entropy_z_shuffle,
					motif_entropy_p_hi_shuffle,
					motif_entropy_p_lo_shuffle,
					motif_entropy_p_two_shuffle,
				) = add_stat(null_metrics_shuffle, "motif_entropy")
				(
					motif_entropy_z_markov,
					motif_entropy_p_hi_markov,
					motif_entropy_p_lo_markov,
					motif_entropy_p_two_markov,
				) = add_stat(null_metrics_markov, "motif_entropy")

				(
					eff_count_z_shuffle,
					eff_count_p_hi_shuffle,
					eff_count_p_lo_shuffle,
					eff_count_p_two_shuffle,
				) = add_stat(null_metrics_shuffle, "effective_motif_count")
				(
					eff_count_z_markov,
					eff_count_p_hi_markov,
					eff_count_p_lo_markov,
					eff_count_p_two_markov,
				) = add_stat(null_metrics_markov, "effective_motif_count")

				(
					top_mass_z_shuffle,
					top_mass_p_hi_shuffle,
					top_mass_p_lo_shuffle,
					top_mass_p_two_shuffle,
				) = add_stat(null_metrics_shuffle, "top_mass")
				(
					top_mass_z_markov,
					top_mass_p_hi_markov,
					top_mass_p_lo_markov,
					top_mass_p_two_markov,
				) = add_stat(null_metrics_markov, "top_mass")

				(
					repeat_rate_z_shuffle,
					repeat_rate_p_hi_shuffle,
					repeat_rate_p_lo_shuffle,
					repeat_rate_p_two_shuffle,
				) = add_stat(null_metrics_shuffle, "repeat_rate")
				(
					repeat_rate_z_markov,
					repeat_rate_p_hi_markov,
					repeat_rate_p_lo_markov,
					repeat_rate_p_two_markov,
				) = add_stat(null_metrics_markov, "repeat_rate")

				(
					motif_div_z_shuffle,
					motif_div_p_hi_shuffle,
					motif_div_p_lo_shuffle,
					motif_div_p_two_shuffle,
				) = add_stat(null_metrics_shuffle, "motif_transition_diversity")
				(
					motif_div_z_markov,
					motif_div_p_hi_markov,
					motif_div_p_lo_markov,
					motif_div_p_two_markov,
				) = add_stat(null_metrics_markov, "motif_transition_diversity")

				episode_id = f"{session_id}_{regime}_{start}_{end}"

				episode_rows.append(
					{
						"episode_id": episode_id,
						"session_id": session_id,
						"dyad_id": row.get("dyad_id", None),
						"regime": int(regime),
						"start": int(start),
						"end": int(end),
						"duration": int(duration),
						"k": int(k),
						"n_windows": int(metrics["n_windows"]),
						"n_unique_motifs": int(metrics["n_unique"]),
						"motif_entropy": float(metrics["motif_entropy"]),
						"motif_entropy_z_shuffle": float(motif_entropy_z_shuffle),
						"motif_entropy_p_hi_shuffle": float(motif_entropy_p_hi_shuffle),
						"motif_entropy_p_lo_shuffle": float(motif_entropy_p_lo_shuffle),
						"motif_entropy_p_two_shuffle": float(motif_entropy_p_two_shuffle),
						"motif_entropy_z_markov": float(motif_entropy_z_markov),
						"motif_entropy_p_hi_markov": float(motif_entropy_p_hi_markov),
						"motif_entropy_p_lo_markov": float(motif_entropy_p_lo_markov),
						"motif_entropy_p_two_markov": float(motif_entropy_p_two_markov),
						"effective_motif_count": float(metrics["effective_motif_count"]),
						"effective_motif_count_z_shuffle": float(eff_count_z_shuffle),
						"effective_motif_count_p_hi_shuffle": float(eff_count_p_hi_shuffle),
						"effective_motif_count_p_lo_shuffle": float(eff_count_p_lo_shuffle),
						"effective_motif_count_p_two_shuffle": float(eff_count_p_two_shuffle),
						"effective_motif_count_z_markov": float(eff_count_z_markov),
						"effective_motif_count_p_hi_markov": float(eff_count_p_hi_markov),
						"effective_motif_count_p_lo_markov": float(eff_count_p_lo_markov),
						"effective_motif_count_p_two_markov": float(eff_count_p_two_markov),
						"top_motif": motif_to_str(metrics["top_motif"]),
						"top_motif_prob": float(metrics["top_motif_prob"]),
						"top_mass_top_n": float(metrics["top_mass"]),
						"top_mass_top_n_z_shuffle": float(top_mass_z_shuffle),
						"top_mass_top_n_p_hi_shuffle": float(top_mass_p_hi_shuffle),
						"top_mass_top_n_p_lo_shuffle": float(top_mass_p_lo_shuffle),
						"top_mass_top_n_p_two_shuffle": float(top_mass_p_two_shuffle),
						"top_mass_top_n_z_markov": float(top_mass_z_markov),
						"top_mass_top_n_p_hi_markov": float(top_mass_p_hi_markov),
						"top_mass_top_n_p_lo_markov": float(top_mass_p_lo_markov),
						"top_mass_top_n_p_two_markov": float(top_mass_p_two_markov),
						"repeat_rate": float(metrics["repeat_rate"]),
						"repeat_rate_z_shuffle": float(repeat_rate_z_shuffle),
						"repeat_rate_p_hi_shuffle": float(repeat_rate_p_hi_shuffle),
						"repeat_rate_p_lo_shuffle": float(repeat_rate_p_lo_shuffle),
						"repeat_rate_p_two_shuffle": float(repeat_rate_p_two_shuffle),
						"repeat_rate_z_markov": float(repeat_rate_z_markov),
						"repeat_rate_p_hi_markov": float(repeat_rate_p_hi_markov),
						"repeat_rate_p_lo_markov": float(repeat_rate_p_lo_markov),
						"repeat_rate_p_two_markov": float(repeat_rate_p_two_markov),
						"transition_entropy": float(metrics["transition_entropy"]),
						"motif_transition_diversity": float(metrics["motif_transition_diversity"]),
						"motif_transition_diversity_z_shuffle": float(motif_div_z_shuffle),
						"motif_transition_diversity_p_hi_shuffle": float(motif_div_p_hi_shuffle),
						"motif_transition_diversity_p_lo_shuffle": float(motif_div_p_lo_shuffle),
						"motif_transition_diversity_p_two_shuffle": float(motif_div_p_two_shuffle),
						"motif_transition_diversity_z_markov": float(motif_div_z_markov),
						"motif_transition_diversity_p_hi_markov": float(motif_div_p_hi_markov),
						"motif_transition_diversity_p_lo_markov": float(motif_div_p_lo_markov),
						"motif_transition_diversity_p_two_markov": float(motif_div_p_two_markov),
						"top_motifs": top_motifs_serialized,
					}
				)

	results_df = pd.DataFrame(episode_rows)
	results_path = out_dir / "grammar_episode_results.csv"
	results_df.to_csv(results_path, index=False)

	run_info = {
		"date": pd.Timestamp.now().isoformat(),
		"params": {
			"n_min": n_min,
			"ks": ks,
			"top_n": top_n,
			"n_null": n_null,
			"random_seed": random_seed,
			"min_null_windows": min_null_windows,
			"min_null_duration": min_null_duration,
		},
		"inputs": {
			"hmm_dir": str(hmm_dir),
			"processed_data_dir": str(processed_data_dir),
			"decoded_sessions_csv": str(hmm_dir / "decoded_sessions.csv"),
		},
		"outputs": {
			"out_dir": str(out_dir),
			"episode_results_csv": str(results_path),
		},
	}

	with open(out_dir / "run_info.json", "w") as f:
		json.dump(run_info, f, indent=2)

	print(f"Saved {len(results_df)} k-gram rows to {results_path}")


if __name__ == "__main__":
	main()
