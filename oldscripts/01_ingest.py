#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from spaffcoord.schemas import Session


def load_legend(path: Path, collapse_map: dict) -> dict[int, int]:
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
    m = re.match(pattern, name)
    if not m:
        raise ValueError(f"Filename does not match pattern: {name}")
    return m.group("dyad"), m.group("session")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    raw_dir = Path(cfg["raw_dir"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    legend = load_legend(
        Path(cfg["legend_path"]),
        cfg["collapse_map"],
    )

    manifest = []

    for csv_path in sorted(raw_dir.glob("*.csv")):
        dyad, sess = parse_filename(csv_path.name, cfg["filename_pattern"])
        session_id = f"{dyad}_{sess}"

        df = pd.read_csv(csv_path)

        # Use explicit column names
        required = {"therapist", "client"}
        cols_lower = {c.lower(): c for c in df.columns}

        dyad = str(dyad)

        t = None
        if "unnamed: 0" in cols_lower:
            t_raw = pd.to_numeric(df[cols_lower["unnamed: 0"]], errors="coerce")
            if t_raw.isna().any():
                raise ValueError(
                    f"Missing time/index values in {csv_path.name}: {int(t_raw.isna().sum())}"
                )
            t = t_raw.to_numpy()

        missing = required - set(cols_lower.keys())
        if missing:
            raise ValueError(
                f"{csv_path.name} missing columns: {sorted(missing)}. Found: {list(df.columns)}"
            )

        T_raw = pd.to_numeric(df[cols_lower["therapist"]], errors="coerce")
        C_raw = pd.to_numeric(df[cols_lower["client"]], errors="coerce")

        T = T_raw.map(legend)
        C = C_raw.map(legend)

        # Hard fail if any unknown SPAFF codes exist
        bad_T = sorted(set(T_raw[T.isna() & T_raw.notna()].astype(int).tolist()))
        bad_C = sorted(set(C_raw[C.isna() & C_raw.notna()].astype(int).tolist()))
        if bad_T or bad_C:
            raise ValueError(
                f"Unknown SPAFF codes in {csv_path.name}. "
                f"Therapist unknown: {bad_T} | Client unknown: {bad_C}"
            )

        # Hard fail on missing data too (you can loosen later)
        if T_raw.isna().any() or C_raw.isna().any():
            raise ValueError(
                f"Missing values in {csv_path.name}: "
                f"therapist={int(T_raw.isna().sum())}, client={int(C_raw.isna().sum())}"
            )

        T = T.astype(int).to_numpy()
        C = C.astype(int).to_numpy()

        session = Session(
            session_id=session_id,
            dyad_id=str(dyad),
            therapist_codes=T,
            client_codes=C,
            t=t,
            meta={"source": csv_path.name},
        )

        out_path = out_dir / f"{session_id}.parquet"
        session.to_parquet(out_path)

        manifest.append(
            {
                "session_id": session_id,
                "dyad_id": dyad,
                "n_steps": len(T),
                "file": out_path.name,
            }
        )

        print(f"Wrote {out_path}")

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(f"\nIngested {len(manifest)} sessions")
    print(f"Manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
