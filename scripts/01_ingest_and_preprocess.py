#!/usr/bin/env python3
"""
This script handles the initial data ingestion and preprocessing of SPAFF codes.
It performs the following steps:
1.  Loads raw SPAFF code CSV files from a specified directory.
2.  Maps the raw codes to a simplified valence space (e.g., positive, negative, neutral).
3.  Optionally debounces the code sequences to remove short, transient states.
4.  Optionally windows the data into smaller, overlapping segments for analysis.
5.  Saves the processed data as Parquet files, along with a manifest CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from affectdynamics.preprocess.debounce import debounce_short_runs
from affectdynamics.preprocess.window import sliding_windows
from affectdynamics.schemas import Session


def hash_array_triplet(
    therapist_codes: Any,
    client_codes: Any,
    t: Any | None,
) -> str:
    """Stable hash for a session's canonical arrays (T, C, t)."""
    import numpy as np

    h = hashlib.sha256()
    h.update(np.asarray(therapist_codes, dtype=np.int16).tobytes())
    h.update(np.asarray(client_codes, dtype=np.int16).tobytes())
    if t is None:
        h.update(b"NO_T")
    else:
        h.update(np.asarray(t, dtype=np.float64).tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_run_config_hash(cfg: dict, legend_path: Path) -> str:
    payload = {
        "ingestion": cfg.get("ingestion", {}),
        "debounce": cfg.get("debounce", {}),
        "windowing": cfg.get("windowing", {}),
        "legend_file_sha256": file_sha256(legend_path),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_legend(path: Path, collapse_map: dict) -> dict[int, int]:
    """
    Loads a SPAFF code legend from a CSV file and creates a mapping to a collapsed
    valence space.

    Args:
        path: The path to the legend CSV file.
        collapse_map: A dictionary mapping valence strings to integer codes.

    Returns:
        A dictionary mapping raw SPAFF codes to the collapsed valence codes.
    """
    df = pd.read_csv(path)
    mapping = {}
    for _, row in df.iterrows():
        code = int(row["code_number"])
        valence = row["valence"]
        if valence not in collapse_map:
            raise ValueError(f"Unknown valence: {valence}")
        mapping[code] = collapse_map[valence]
    return mapping


def parse_filename(name: str, pattern: str) -> tuple[str, str]:
    """
    Parses a filename to extract the dyad and session identifiers.

    Args:
        name: The filename to parse.
        pattern: A regular expression pattern with 'dyad' and 'session' capture groups.

    Returns:
        A tuple containing the dyad and session IDs.
    """
    m = re.match(pattern, name)
    if not m:
        raise ValueError(f"Filename does not match pattern: {name}")
    return m.group("dyad"), m.group("session")


def main():
    """
    Main function to run the data ingestion and preprocessing pipeline.
    """
    # --- 1. Load Configuration ---
    # The script starts by parsing command-line arguments to get the path to the
    # configuration file. This file contains all the settings needed for the script to run.
    p = argparse.ArgumentParser(description="Ingest and preprocess SPAFF data.")
    p.add_argument("--config", required=True, help="Path to the configuration YAML file.")
    p.add_argument(
        "--clean_output",
        action="store_true",
        help="Delete existing processed outputs before writing new files.",
    )
    args = p.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    # Extract directories for input and output from the config.
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for pth in out_dir.glob("*.parquet"):
            pth.unlink(missing_ok=True)
        for pth in [out_dir / "processed_manifest.csv", out_dir / "processing_run.json"]:
            pth.unlink(missing_ok=True)

    # --- 2. Ingestion ---
    # Load the SPAFF code legend. This maps the detailed SPAFF codes to a simpler
    # set of categories (e.g., positive, negative, neutral).
    legend = load_legend(
        Path(cfg["ingestion"]["legend_path"]),
        cfg["ingestion"]["collapse_map"],
    )
    legend_path = Path(cfg["ingestion"]["legend_path"])

    run_config_hash = compute_run_config_hash(cfg, legend_path)

    # Guardrail: fail fast if existing manifest appears to come from different preprocessing settings.
    manifest_path = out_dir / "processed_manifest.csv"
    if manifest_path.exists() and not args.clean_output:
        existing = pd.read_csv(manifest_path)
        if not existing.empty:
            deb_cfg = cfg.get("debounce", {}) or {}
            win_cfg = cfg.get("windowing", {}) or {}

            deb_enabled_now = bool(deb_cfg.get("enabled", False))
            k_now = int(deb_cfg.get("k", 1))
            win_enabled_now = bool(win_cfg.get("enabled", False))
            win_sec_now = float(win_cfg.get("window_sec", 60))
            overlap_now = float(win_cfg.get("overlap", 0.5))

            row0 = existing.iloc[0]
            deb_enabled_old = bool(row0.get("debounce_enabled", False))
            k_old = int(row0.get("debounce_k", 1)) if pd.notna(row0.get("debounce_k", None)) else 1
            win_enabled_old = bool(row0.get("windowing_enabled", False))
            window_sec_value = row0.get("window_sec", None)
            overlap_value = row0.get("overlap", None)
            win_sec_old = float(window_sec_value) if pd.notna(window_sec_value) else 60.0
            overlap_old = float(overlap_value) if pd.notna(overlap_value) else 0.5

            mismatch = (
                deb_enabled_old != deb_enabled_now
                or k_old != k_now
                or win_enabled_old != win_enabled_now
                or abs(win_sec_old - win_sec_now) > 1e-12
                or abs(overlap_old - overlap_now) > 1e-12
            )

            if mismatch:
                raise RuntimeError(
                    "Existing processed manifest settings do not match current config. "
                    "Re-run with --clean_output to rebuild deterministic outputs."
                )

    ingested_sessions = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        dyad, sess = parse_filename(csv_path.name, cfg["ingestion"]["filename_pattern"])
        session_id = f"{dyad}_{sess}"

        df = pd.read_csv(csv_path)

        # Use explicit column names for therapist and client codes.
        required = {"therapist", "client"}
        cols_lower = {c.lower(): c for c in df.columns}
        missing = required - set(cols_lower.keys())
        if missing:
            raise ValueError(
                f"{csv_path.name} missing columns: {sorted(missing)}. Found: {list(df.columns)}"
            )

        T_raw = pd.to_numeric(df[cols_lower["therapist"]], errors="coerce")
        C_raw = pd.to_numeric(df[cols_lower["client"]], errors="coerce")

        # Map the raw codes to the simplified valence codes.
        T = T_raw.map(legend)
        C = C_raw.map(legend)

        # Ensure all codes were successfully mapped.
        bad_T = sorted(set(T_raw[T.isna() & T_raw.notna()].astype(int).tolist()))
        bad_C = sorted(set(C_raw[C.isna() & C_raw.notna()].astype(int).tolist()))
        if bad_T or bad_C:
            raise ValueError(
                f"Unknown SPAFF codes in {csv_path.name}. "
                f"Therapist unknown: {bad_T} | Client unknown: {bad_C}"
            )
        if T_raw.isna().any() or C_raw.isna().any():
            raise ValueError(
                f"Missing values in {csv_path.name}: "
                f"therapist={int(T_raw.isna().sum())}, client={int(C_raw.isna().sum())}"
            )

        t = None
        if "unnamed: 0" in cols_lower:
            t_raw = pd.to_numeric(df[cols_lower["unnamed: 0"]], errors="coerce")
            if t_raw.isna().any():
                raise ValueError(
                    f"Missing time/index values in {csv_path.name}: {int(t_raw.isna().sum())}"
                )
            t = t_raw.to_numpy()

        T_arr = T.astype(int).to_numpy()
        C_arr = C.astype(int).to_numpy()
        raw_mapped_hash = hash_array_triplet(T_arr, C_arr, t)

        # Create a Session object to hold the ingested data.
        session = Session(
            session_id=session_id,
            dyad_id=str(dyad),
            therapist_codes=T_arr,
            client_codes=C_arr,
            t=t,
            meta={
                "source": csv_path.name,
                "raw_mapped_hash": raw_mapped_hash,
                "run_config_hash": run_config_hash,
            },
        )
        ingested_sessions.append(session)

    print(f"Ingested {len(ingested_sessions)} sessions.")

    # --- 3. Preprocessing ---
    # Now, preprocess each of the ingested sessions.
    manifest = []
    deb_cfg = cfg.get("debounce", {}) or {}
    win_cfg = cfg.get("windowing", {}) or {}

    deb_enabled = bool(deb_cfg.get("enabled", False))
    k = int(deb_cfg.get("k", 1))

    win_enabled = bool(win_cfg.get("enabled", False))
    win_sec = float(win_cfg.get("window_sec", 60))
    overlap = float(win_cfg.get("overlap", 0.5))
    if not (0.0 <= overlap < 1.0):
        raise ValueError("windowing.overlap must be in [0, 1)")

    for s in ingested_sessions:
        T = s.therapist_codes
        C = s.client_codes
        t = s.t

        # Debounce short runs of codes if enabled.
        if deb_enabled:
            T = debounce_short_runs(T, k=k)
            C = debounce_short_runs(C, k=k)

        # If windowing is not enabled, save one file per session.
        if not win_enabled:
            out = Session(
                session_id=s.session_id,
                dyad_id=s.dyad_id,
                therapist_codes=T,
                client_codes=C,
                t=t,
                meta={**(s.meta or {}), "debounce_enabled": deb_enabled, "debounce_k": k},
            )
            out_path = out_dir / f"{s.session_id}.parquet"
            out.to_parquet(out_path)
            processed_hash = hash_array_triplet(T, C, t)
            raw_hash = (s.meta or {}).get("raw_mapped_hash", "")

            manifest.append(
                {
                    "session_id": s.session_id,
                    "dyad_id": s.dyad_id,
                    "window_id": None,
                    "n_steps": int(len(T)),
                    "file": out_path.name,
                    "debounce_enabled": deb_enabled,
                    "debounce_k": k if deb_enabled else None,
                    "windowing_enabled": False,
                    "window_sec": None,
                    "overlap": None,
                    "raw_mapped_hash": raw_hash,
                    "processed_hash": processed_hash,
                    "changed_from_raw": bool(raw_hash and processed_hash != raw_hash),
                    "run_config_hash": run_config_hash,
                }
            )
            print(f"Wrote {out_path}")
            continue

        # If windowing is enabled, create sliding windows.
        dt = 1.0
        if t is not None and len(t) >= 2:
            dt = float(pd.Series(t).diff().median())
            if not (dt > 0):
                dt = 1.0

        win = int(round(win_sec / dt))
        step = int(round(win * (1.0 - overlap)))
        step = max(step, 1)

        spans = sliding_windows(len(T), win=win, step=step)

        for w_i, (a, b) in enumerate(spans):
            sid = f"{s.session_id}__w{w_i:04d}"
            out = Session(
                session_id=sid,
                dyad_id=s.dyad_id,
                therapist_codes=T[a:b],
                client_codes=C[a:b],
                t=(t[a:b] if t is not None else None),
                meta={
                    **(s.meta or {}),
                    "parent_session_id": s.session_id,
                    "window_id": w_i,
                    "a": int(a),
                    "b": int(b),
                    "debounce_enabled": deb_enabled,
                    "debounce_k": k,
                    "window_sec": win_sec,
                    "overlap": overlap,
                    "dt": dt,
                },
            )
            out_path = out_dir / f"{sid}.parquet"
            out.to_parquet(out_path)
            processed_hash = hash_array_triplet(T[a:b], C[a:b], (t[a:b] if t is not None else None))

            manifest.append(
                {
                    "session_id": s.session_id,
                    "dyad_id": s.dyad_id,
                    "window_id": w_i,
                    "n_steps": int(b - a),
                    "file": out_path.name,
                    "debounce_enabled": deb_enabled,
                    "debounce_k": k if deb_enabled else None,
                    "windowing_enabled": True,
                    "window_sec": win_sec,
                    "overlap": overlap,
                    "raw_mapped_hash": (s.meta or {}).get("raw_mapped_hash", ""),
                    "processed_hash": processed_hash,
                    "changed_from_raw": True,
                    "run_config_hash": run_config_hash,
                }
            )
        print(f"Wrote {len(spans)} windows for {s.session_id}")

    # --- 4. Save Manifest ---
    # Finally, save a manifest file that lists all the processed files.
    mf = pd.DataFrame(manifest)
    mf.to_csv(out_dir / "processed_manifest.csv", index=False)

    run_summary = {
        "run_config_hash": run_config_hash,
        "raw_dir": str(raw_dir),
        "processed_dir": str(out_dir),
        "debounce": cfg.get("debounce", {}),
        "windowing": cfg.get("windowing", {}),
        "ingestion": {
            "legend_path": str(legend_path),
            "legend_file_sha256": file_sha256(legend_path),
            "filename_pattern": cfg.get("ingestion", {}).get("filename_pattern"),
        },
        "n_manifest_rows": int(mf.shape[0]),
        "n_unique_sessions": int(mf["session_id"].nunique()) if not mf.empty else 0,
    }
    (out_dir / "processing_run.json").write_text(json.dumps(run_summary, indent=2))

    print(f"\nProcessed {len(ingested_sessions)} canonical sessions")
    print(f"Manifest: {out_dir / 'processed_manifest.csv'}")
    print(f"Run summary: {out_dir / 'processing_run.json'}")
    

if __name__ == "__main__":
    main()
