#!/usr/bin/env python3
"""Local sliding PH around HMM switch events on posterior trajectories.

For each session this script:
1) Reconstructs posterior vectors p_t from the fitted shared-emission HMM.
2) Builds delay embeddings of posterior trajectories.
3) Computes PH summaries on rolling windows across time.
4) Detects Viterbi switch times and aligns rolling PH series to switches.
5) Produces switch-level before/during/after summaries, jump-size associations,
   and transition-type reorganization summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from tqdm import tqdm

from affectdynamics.models.hmm import SharedEmissionHMM
from affectdynamics.schemas import Session


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def load_sessions(processed_dir: Path) -> tuple[list[Session], pd.DataFrame]:
    mf = pd.read_csv(processed_dir / "processed_manifest.csv")
    sessions = [Session.from_parquet(processed_dir / f) for f in mf["file"].tolist()]
    return sessions, mf


def load_model_from_json(path: Path) -> SharedEmissionHMM:
    params = json.loads(path.read_text())
    k = int(params["K"])
    model = SharedEmissionHMM(K=k)
    model.pi = np.asarray(params["pi"], dtype=float)
    model.A = np.asarray(params["A"], dtype=float)
    model.pT = np.asarray(params["pT"], dtype=float)
    model.pC = np.asarray(params["pC"], dtype=float)
    return model


def posterior_matrix(model: SharedEmissionHMM, session: Session) -> np.ndarray:
    t_codes = np.asarray(session.therapist_codes, dtype=int)
    c_codes = np.asarray(session.client_codes, dtype=int)
    n = min(len(t_codes), len(c_codes))
    if n <= 0:
        return np.empty((0, model.K), dtype=float)
    return model.posterior(t_codes[:n], c_codes[:n])


def posterior_entropy(posterior_row: np.ndarray) -> float:
    p = np.asarray(posterior_row, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def posterior_local_speed(posterior: np.ndarray) -> np.ndarray:
    if posterior.ndim != 2 or len(posterior) == 0:
        return np.empty((0,), dtype=float)
    speed = np.zeros(len(posterior), dtype=float)
    if len(posterior) > 1:
        speed[1:] = np.linalg.norm(np.diff(posterior, axis=0), axis=1)
    return speed


def build_delay_embedding(
    posterior: np.ndarray,
    *,
    m: int,
    tau_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return embedded points and center time index for each point."""
    if posterior.ndim != 2:
        raise ValueError("posterior must have shape (T, K)")
    if m <= 0 or tau_steps <= 0:
        raise ValueError("m and tau_steps must be > 0")

    t_len, k = posterior.shape
    span = (m - 1) * tau_steps
    n_windows = t_len - span
    if n_windows <= 0:
        return np.empty((0, m * k), dtype=float), np.empty((0,), dtype=int)

    starts = np.arange(n_windows, dtype=int)[:, None]
    offsets = (np.arange(m, dtype=int) * tau_steps)[None, :]
    idx = starts + offsets
    embedded = posterior[idx].reshape(n_windows, m * k)

    centers = starts[:, 0] + span // 2
    return embedded, centers


def compute_distance_matrix(x: np.ndarray, *, metric: str) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be 2D for distance computation")
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if metric == "euclidean":
        return squareform(pdist(x, metric="euclidean"))

    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        xn = x / norms
        d = 1.0 - (xn @ xn.T)
        d = np.clip(d, 0.0, 2.0)
        np.fill_diagonal(d, 0.0)
        return d

    raise ValueError(f"Unsupported metric: {metric}")


def max_edge_from_data(d: np.ndarray, quantile: float) -> float:
    tri = d[np.triu_indices_from(d, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return 1.0
    return float(np.quantile(tri, quantile))


def persistence_entropy(diagram: np.ndarray) -> float:
    if diagram.size == 0:
        return 0.0
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    finite = np.isfinite(deaths)
    if not np.any(finite):
        return 0.0
    life = deaths[finite] - births[finite]
    life = life[life > 0]
    if life.size == 0:
        return 0.0
    p = life / life.sum()
    return float(-np.sum(p * np.log(p)))


def total_persistence(diagram: np.ndarray) -> float:
    if diagram.size == 0:
        return 0.0
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    finite = np.isfinite(deaths)
    if not np.any(finite):
        return 0.0
    life = deaths[finite] - births[finite]
    life = life[life > 0]
    return float(np.sum(life)) if life.size > 0 else 0.0


def betti_curve_auc_h1(diagram_h1: np.ndarray, max_edge: float, n_bins: int) -> float:
    grid = np.linspace(0.0, max_edge, max(2, n_bins))
    if diagram_h1.size == 0:
        return 0.0
    births = diagram_h1[:, 0]
    deaths = diagram_h1[:, 1]
    deaths = np.where(np.isfinite(deaths), deaths, max_edge)
    vals = np.zeros_like(grid)
    for i, g in enumerate(grid):
        vals[i] = np.sum((births <= g) & (g < deaths))
    return float(np.trapezoid(vals, grid))


def compute_ph_core(
    x: np.ndarray,
    *,
    metric: str,
    max_edge_mode: str,
    max_edge_quantile: float,
    max_edge_value: float | None,
    betti_bins: int,
) -> dict[str, float]:
    if len(x) < 3:
        return {
            "n_points": float(len(x)),
            "max_edge": float("nan"),
            "total_persistence_h1": 0.0,
            "betti_h1_auc": 0.0,
            "persistence_entropy_h1": 0.0,
        }

    d = compute_distance_matrix(x, metric=metric)
    if max_edge_mode == "fixed":
        max_edge = _safe_float(max_edge_value, 1.0)
    elif max_edge_mode == "quantile":
        max_edge = max_edge_from_data(d, max_edge_quantile)
    else:
        raise ValueError(f"Unknown max_edge_mode: {max_edge_mode}")

    res = ripser(d, distance_matrix=True, maxdim=1, thresh=max_edge)
    dgms = res["dgms"]
    dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2), dtype=float)

    return {
        "n_points": float(len(x)),
        "max_edge": float(max_edge),
        "total_persistence_h1": float(total_persistence(dgm1)),
        "betti_h1_auc": float(betti_curve_auc_h1(dgm1, max_edge=max_edge, n_bins=betti_bins)),
        "persistence_entropy_h1": float(persistence_entropy(dgm1)),
    }


def detect_switches(path: np.ndarray) -> np.ndarray:
    if len(path) < 2:
        return np.empty((0,), dtype=int)
    return np.where(path[1:] != path[:-1])[0] + 1


def phase_label(rel_time: float, half_band: float) -> str:
    if rel_time < -half_band:
        return "before"
    if rel_time > half_band:
        return "after"
    return "during"


def run_phase_transitions(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text()) or {}

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    emb_cfg = cfg.get("embedding", {})
    roll_cfg = cfg.get("rolling", {})
    sw_cfg = cfg.get("switch_alignment", {})
    ph_cfg = cfg.get("ph", {})

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    sample_dt_sec = _safe_float(data_cfg.get("sample_dt_sec", 1.0), 1.0)

    hmm_dir = Path(model_cfg.get("hmm_dir", "artifacts/03_chmm_outputs/chmm_8_full_dataset"))
    model_json = Path(model_cfg.get("model_json", hmm_dir / "best_model.json"))
    decoded_csv = Path(model_cfg.get("decoded_sessions_csv", hmm_dir / "decoded_sessions.csv"))

    out_dir = Path(cfg.get("out_dir", "artifacts/07_transitions"))
    out_dir.mkdir(parents=True, exist_ok=True)

    m = _safe_int(emb_cfg.get("m", 10), 10)
    tau_sec = _safe_float(emb_cfg.get("tau_sec", 1.0), 1.0)
    tau_steps = max(1, int(round(tau_sec / sample_dt_sec)))

    rolling_window_sec = _safe_float(roll_cfg.get("window_sec", 30.0), 30.0)
    rolling_step_sec = _safe_float(roll_cfg.get("step_sec", 5.0), 5.0)
    rolling_window_steps = max(2, int(round(rolling_window_sec / sample_dt_sec)))
    rolling_step_steps = max(1, int(round(rolling_step_sec / sample_dt_sec)))

    switch_window_sec = _safe_float(sw_cfg.get("window_sec", 30.0), 30.0)
    switch_window_steps = max(2, int(round(switch_window_sec / sample_dt_sec)))
    phase_half_band_sec = _safe_float(sw_cfg.get("during_half_band_sec", rolling_step_sec), rolling_step_sec)

    metric = str(ph_cfg.get("metric", "euclidean"))
    max_edge_mode = str(ph_cfg.get("max_edge_mode", "quantile"))
    max_edge_quantile = _safe_float(ph_cfg.get("max_edge_quantile", 0.95), 0.95)
    max_edge_value = ph_cfg.get("max_edge_value", None)
    max_edge_value = float(max_edge_value) if max_edge_value is not None else None
    betti_bins = _safe_int(ph_cfg.get("betti_bins", 64), 64)

    sessions, _ = load_sessions(processed_dir)
    session_map = {s.session_id: s for s in sessions}
    model = load_model_from_json(model_json)

    dec = pd.read_csv(decoded_csv)

    rolling_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    pseudo_aligned_rows: list[dict[str, Any]] = []
    session_max_time_idx: dict[str, int] = {}

    for _, row in tqdm(dec.iterrows(), total=len(dec), desc="Sessions", unit="session"):
        session_id = str(row["session_id"])
        dyad_id = str(row["dyad_id"])

        session = session_map.get(session_id)
        if session is None:
            continue

        path_file = hmm_dir / str(row["path_file"])
        if not path_file.exists():
            continue

        viterbi = np.load(path_file).astype(int)
        posterior = posterior_matrix(model, session)
        t_len = min(len(viterbi), len(posterior))
        if t_len < max(rolling_window_steps, 5):
            continue

        viterbi = viterbi[:t_len]
        posterior = posterior[:t_len]
        entropy_series = np.apply_along_axis(posterior_entropy, 1, posterior)
        local_speed_series = posterior_local_speed(posterior)

        emb, centers = build_delay_embedding(posterior, m=m, tau_steps=tau_steps)
        if len(emb) < 3:
            continue

        half_roll = rolling_window_steps // 2
        center_grid = np.arange(half_roll, t_len - half_roll, rolling_step_steps, dtype=int)

        session_roll_rows: list[dict[str, Any]] = []
        for c in center_grid:
            mask = (centers >= (c - half_roll)) & (centers <= (c + half_roll))
            x = emb[mask]
            ph = compute_ph_core(
                x,
                metric=metric,
                max_edge_mode=max_edge_mode,
                max_edge_quantile=max_edge_quantile,
                max_edge_value=max_edge_value,
                betti_bins=betti_bins,
            )
            rr = {
                "session_id": session_id,
                "dyad_id": dyad_id,
                "time_idx": int(c),
                "time_sec": float(c * sample_dt_sec),
                "window_sec": float(rolling_window_sec),
                "step_sec": float(rolling_step_sec),
                "metric": metric,
                "m": int(m),
                "tau_sec": float(tau_sec),
                "posterior_entropy": float(entropy_series[c]),
                "local_speed_l2": float(local_speed_series[c]),
                **ph,
            }
            session_roll_rows.append(rr)
            rolling_rows.append(rr)

        if not session_roll_rows:
            continue
        df_roll = pd.DataFrame(session_roll_rows)
        session_max_time_idx[session_id] = int(df_roll["time_idx"].max())

        switches = detect_switches(viterbi)

        # Sample pseudo-switches
        n_pseudo = len(switches)
        possible_pseudo = np.setdiff1d(np.arange(t_len), switches)
        if len(possible_pseudo) > n_pseudo and n_pseudo > 0:
            pseudo_switches = np.random.choice(possible_pseudo, n_pseudo, replace=False)
        else:
            pseudo_switches = np.array([])

        for sidx, s in enumerate(switches):
            from_state = int(viterbi[s - 1]) if s - 1 >= 0 else -1
            to_state = int(viterbi[s])
            jump = float(np.linalg.norm(posterior[s] - posterior[s - 1])) if s - 1 >= 0 else np.nan

            # Local segment PH around switch on posterior-derived embedding.
            seg_start = max(0, int(s - switch_window_steps))
            seg_end = min(t_len, int(s + switch_window_steps + 1))
            seg_post = posterior[seg_start:seg_end]
            seg_emb, _ = build_delay_embedding(seg_post, m=m, tau_steps=tau_steps)
            seg_ph = compute_ph_core(
                seg_emb,
                metric=metric,
                max_edge_mode=max_edge_mode,
                max_edge_quantile=max_edge_quantile,
                max_edge_value=max_edge_value,
                betti_bins=betti_bins,
            )

            switch_rows.append(
                {
                    "session_id": session_id,
                    "dyad_id": dyad_id,
                    "switch_index": int(sidx),
                    "switch_time_idx": int(s),
                    "switch_time_sec": float(s * sample_dt_sec),
                    "from_state": from_state,
                    "to_state": to_state,
                    "transition_type": f"{from_state}->{to_state}",
                    "posterior_jump_l2": jump,
                    "segment_window_sec": float(switch_window_sec),
                    **{f"segment_{k}": v for k, v in seg_ph.items()},
                }
            )

            # Align rolling PH time series to this switch.
            rel = df_roll.copy()
            rel["rel_time_idx"] = rel["time_idx"] - int(s)
            rel["rel_time_sec"] = rel["time_sec"] - float(s * sample_dt_sec)
            rel = rel[np.abs(rel["rel_time_sec"]) <= float(switch_window_sec)].copy()
            if rel.empty:
                continue

            rel["switch_index"] = int(sidx) # This is pseudo-switch index
            rel["switch_time_idx"] = int(s)
            rel["from_state"] = from_state
            rel["to_state"] = to_state
            rel["transition_type"] = f"{from_state}->{to_state}"
            rel["posterior_jump_l2"] = jump
            rel["phase"] = rel["rel_time_sec"].apply(lambda x: phase_label(float(x), phase_half_band_sec))
            rel["anchor_kind"] = "real"
            records = cast(list[dict[str, Any]], rel.to_dict(orient="records"))
            aligned_rows.extend(records)

        for sidx, s in enumerate(pseudo_switches):
            # Align rolling PH time series to this pseudo switch.
            rel = df_roll.copy()
            rel["rel_time_idx"] = rel["time_idx"] - int(s)
            rel["rel_time_sec"] = rel["time_sec"] - float(s * sample_dt_sec)
            rel = rel[np.abs(rel["rel_time_sec"]) <= float(switch_window_sec)].copy()
            if rel.empty:
                continue

            rel["switch_index"] = int(sidx) # This is pseudo-switch index
            rel["switch_time_idx"] = int(s)
            rel["from_state"] = -1
            rel["to_state"] = -1
            rel["transition_type"] = "pseudo"
            rel["posterior_jump_l2"] = np.nan
            rel["phase"] = rel["rel_time_sec"].apply(lambda x: phase_label(float(x), phase_half_band_sec))
            rel["anchor_kind"] = "pseudo"
            records = cast(list[dict[str, Any]], rel.to_dict(orient="records"))
            pseudo_aligned_rows.extend(records)


    rolling_df = pd.DataFrame(rolling_rows)
    switch_df = pd.DataFrame(switch_rows)
    aligned_df = pd.DataFrame(aligned_rows)
    pseudo_aligned_df = pd.DataFrame(pseudo_aligned_rows)

    # Combine real and pseudo aligned data
    combined_aligned_df = pd.concat([aligned_df, pseudo_aligned_df], ignore_index=True)

    # Match each real switch to a pseudo anchor with similar normalized session time.
    if not aligned_df.empty and not pseudo_aligned_df.empty:
        real_norm_time = aligned_df[['session_id', 'switch_time_idx']].drop_duplicates().copy()
        pseudo_norm_time = pseudo_aligned_df[['session_id', 'switch_time_idx']].drop_duplicates().copy()

        session_lengths = session_max_time_idx

        real_norm_time['norm_time'] = real_norm_time.apply(
            lambda r: r['switch_time_idx'] / session_lengths.get(r['session_id'], 1), axis=1
        )
        pseudo_norm_time['norm_time'] = pseudo_norm_time.apply(
            lambda r: r['switch_time_idx'] / session_lengths.get(r['session_id'], 1), axis=1
        )

        matched_rows = []
        for _, real_row in real_norm_time.iterrows():
            sid = real_row['session_id']
            pseudos_in_session = pseudo_norm_time[pseudo_norm_time['session_id'] == sid]
            if not pseudos_in_session.empty:
                time_diffs = np.abs(pseudos_in_session['norm_time'] - real_row['norm_time'])
                best_match_idx = time_diffs.idxmin()
                matched_rows.append({
                    'session_id': sid,
                    'real_switch_time_idx': real_row['switch_time_idx'],
                    'pseudo_switch_time_idx': pseudos_in_session.loc[best_match_idx, 'switch_time_idx'],
                    'norm_time_diff': time_diffs.min()
                })
        matched_anchors_df = pd.DataFrame(matched_rows)
    else:
        matched_anchors_df = pd.DataFrame()


    # Aggregates for notebook-ready analyses.
    if not combined_aligned_df.empty:
        feat_cols = ["total_persistence_h1", "betti_h1_auc", "persistence_entropy_h1"]

        aligned_mean = (
            combined_aligned_df.groupby(["anchor_kind", "rel_time_sec"], as_index=False)[feat_cols]
            .mean()
            .sort_values(["anchor_kind", "rel_time_sec"])
        )

        phase_summary = (
            combined_aligned_df.groupby(["anchor_kind", "phase"], as_index=False)[feat_cols]
            .mean()
            .sort_values(["anchor_kind", "phase"])
        )

        # Per-switch phase means for reorganization deltas.
        switch_phase = (
            combined_aligned_df.groupby(["session_id", "switch_index", "anchor_kind", "phase"], as_index=False)[feat_cols]
            .mean()
        )
        wide = switch_phase.pivot_table(
            index=["session_id", "switch_index", "anchor_kind"],
            columns="phase",
            values=feat_cols,
        )
        wide.columns = [f"{f}_{p}" for f, p in wide.columns]
        wide = wide.reset_index()

        if not switch_df.empty:
            wide = wide.merge(
                switch_df[["session_id", "switch_index", "transition_type", "posterior_jump_l2"]],
                on=["session_id", "switch_index"],
                how="left",
            )

        delta_rows: list[pd.DataFrame] = []
        for f in feat_cols:
            col_b = f"{f}_before"
            col_a = f"{f}_after"
            if col_b in wide.columns and col_a in wide.columns:
                delta = wide[col_a] - wide[col_b]
                temp = wide[["session_id", "switch_index", "anchor_kind", "transition_type", "posterior_jump_l2"]].copy()
                temp["feature"] = f
                temp["delta_after_minus_before"] = delta
                delta_rows.append(temp)
        delta_df = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame()

        jump_assoc_rows: list[dict[str, Any]] = []
        if not delta_df.empty:
            real_delta_df = delta_df[delta_df['anchor_kind'] == 'real']
            for f in feat_cols:
                d = real_delta_df[real_delta_df["feature"] == f][["posterior_jump_l2", "delta_after_minus_before"]].dropna()
                if len(d) >= 8:
                    r_val, p_val = spearmanr(d["posterior_jump_l2"], d["delta_after_minus_before"])
                    r_float = float(np.asarray(r_val).item())
                    p_float = float(np.asarray(p_val).item())
                    jump_assoc_rows.append(
                        {
                            "feature": f,
                            "spearman_r": r_float,
                            "p_value": p_float,
                            "n": int(len(d)),
                        }
                    )
        jump_assoc_df = pd.DataFrame(jump_assoc_rows)

        transition_summary = pd.DataFrame()
        if not delta_df.empty:
            real_delta_df = delta_df[delta_df['anchor_kind'] == 'real']
            grouped = real_delta_df.groupby(["transition_type", "feature"])["delta_after_minus_before"].mean()
            transition_summary = grouped.reset_index(name="delta_after_minus_before")
            order = np.lexsort(
                (
                    -transition_summary["delta_after_minus_before"].to_numpy(),
                    transition_summary["feature"].to_numpy(),
                )
            )
            transition_summary = transition_summary.iloc[order].reset_index(drop=True)

    else:
        aligned_mean = pd.DataFrame()
        phase_summary = pd.DataFrame()
        delta_df = pd.DataFrame()
        jump_assoc_df = pd.DataFrame()
        transition_summary = pd.DataFrame()

    rolling_path = out_dir / "rolling_ph_timeseries.csv"
    switch_path = out_dir / "switch_local_segment_ph.csv"
    aligned_path = out_dir / "switch_aligned_rolling_ph.csv"
    combined_aligned_path = out_dir / "combined_aligned_rolling_ph.csv"
    matched_anchors_path = out_dir / "matched_real_pseudo_anchors.csv"
    aligned_mean_path = out_dir / "switch_aligned_mean_by_rel_time.csv"
    phase_summary_path = out_dir / "switch_phase_summary.csv"
    delta_path = out_dir / "switch_reorganization_delta.csv"
    jump_assoc_path = out_dir / "switch_jump_association.csv"
    transition_summary_path = out_dir / "switch_transition_type_reorganization.csv"

    rolling_df.to_csv(rolling_path, index=False)
    switch_df.to_csv(switch_path, index=False)
    aligned_df.to_csv(aligned_path, index=False)
    combined_aligned_df.to_csv(combined_aligned_path, index=False)
    matched_anchors_df.to_csv(matched_anchors_path, index=False)
    aligned_mean.to_csv(aligned_mean_path, index=False)
    phase_summary.to_csv(phase_summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    jump_assoc_df.to_csv(jump_assoc_path, index=False)
    transition_summary.to_csv(transition_summary_path, index=False)

    run_info = {
        "config": str(config_path),
        "processed_dir": str(processed_dir),
        "hmm_dir": str(hmm_dir),
        "model_json": str(model_json),
        "decoded_csv": str(decoded_csv),
        "out_dir": str(out_dir),
        "params": {
            "m": m,
            "tau_sec": tau_sec,
            "tau_steps": tau_steps,
            "rolling_window_sec": rolling_window_sec,
            "rolling_step_sec": rolling_step_sec,
            "switch_window_sec": switch_window_sec,
            "during_half_band_sec": phase_half_band_sec,
            "metric": metric,
            "max_edge_mode": max_edge_mode,
            "max_edge_quantile": max_edge_quantile,
            "max_edge_value": max_edge_value,
            "betti_bins": betti_bins,
        },
        "outputs": {
            "rolling_timeseries": str(rolling_path),
            "switch_local_segment": str(switch_path),
            "switch_aligned": str(aligned_path),
            "combined_aligned": str(combined_aligned_path),
            "matched_anchors": str(matched_anchors_path),
            "aligned_mean": str(aligned_mean_path),
            "phase_summary": str(phase_summary_path),
            "reorganization_delta": str(delta_path),
            "jump_association": str(jump_assoc_path),
            "transition_summary": str(transition_summary_path),
        },
        "counts": {
            "n_rolling_rows": int(len(rolling_df)),
            "n_switch_rows": int(len(switch_df)),
            "n_aligned_rows": int(len(aligned_df)),
            "n_pseudo_aligned_rows": int(len(pseudo_aligned_df)),
            "n_combined_aligned_rows": int(len(combined_aligned_df)),
        },
    }
    (out_dir / "phase_transitions_run_info.json").write_text(json.dumps(run_info, indent=2))

    print("Phase-transition local PH analysis complete.")
    print(f"  Rolling PH time series: {rolling_path}")
    print(f"  Switch-aligned PH rows: {aligned_path}")
    print(f"  Phase summary:          {phase_summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local sliding PH over posterior trajectories aligned to Viterbi switches."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_transitions.yaml"),
        help="Path to phase-transition analysis YAML config.",
    )
    args = parser.parse_args()
    run_phase_transitions(args.config)


if __name__ == "__main__":
    main()
