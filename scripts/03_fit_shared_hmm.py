#!/usr/bin/env python3
"""Fit a shared-emission HMM to dyadic behavioral codes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd
import yaml

from affectdynamics.eval.splits import groupkfold_splits
from affectdynamics.models.hmm import NUMBA_AVAILABLE, SharedEmissionHMM
from affectdynamics.schemas import Session


def load_sessions(processed_dir: Path) -> tuple[list[Session], pd.DataFrame]:
    mf = pd.read_csv(processed_dir / "processed_manifest.csv")
    sessions = [Session.from_parquet(processed_dir / f) for f in mf["file"].tolist()]
    return sessions, mf


def as_seqs(sessions: list[Session], idxs: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(sessions[i].therapist_codes, sessions[i].client_codes) for i in idxs]


def n_steps(seqs: list[tuple[np.ndarray, np.ndarray]]) -> int:
    return int(sum(len(T) for T, _ in seqs))


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, df.to_csv(index=False))


def atomic_save_model(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            K=result["K"],
            n_iter_used=result["n_iter_used"],
            train_ll=result["train_ll"],
            pi=result["pi"],
            A=result["A"],
            pT=result["pT"],
            pC=result["pC"],
        )
    tmp.replace(path)


def atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def load_model(path: Path) -> tuple[SharedEmissionHMM, dict[str, Any]]:
    with np.load(path) as data:
        model = SharedEmissionHMM(
            K=int(data["K"]),
            pi=data["pi"],
            A=data["A"],
            pT=data["pT"],
            pC=data["pC"],
        )
        model.n_iter_ = int(data["n_iter_used"])
        metadata = {
            "K": model.K,
            "n_iter_used": model.n_iter_,
            "train_ll": float(data["train_ll"]),
        }
    return model, metadata


def fit_restart(task: dict[str, Any]) -> dict[str, Any]:
    model = SharedEmissionHMM(K=task["K"])
    model.fit_em(
        task["seqs"],
        n_iter=task["n_iter"],
        tol=task["tol"],
        alpha_trans=task["alpha_trans"],
        alpha_emit=task["alpha_emit"],
        seed=task["seed"],
    )
    return {
        "K": task["K"],
        "n_iter_used": model.n_iter_,
        "train_ll": model.score_sequences(task["seqs"]),
        "pi": model.pi,
        "A": model.A,
        "pT": model.pT,
        "pC": model.pC,
    }


def model_from_result(result: dict[str, Any]) -> SharedEmissionHMM:
    model = SharedEmissionHMM(
        K=result["K"], pi=result["pi"], A=result["A"], pT=result["pT"], pC=result["pC"]
    )
    model.n_iter_ = int(result["n_iter_used"])
    return model


def checkpoint_name(
    stage: str, K: int, alpha_trans: float, alpha_emit: float, fold: int | None, restart: int
) -> str:
    fold_text = "" if fold is None else f"_fold{fold}"
    return (
        f"{stage}_K{K}_aT{alpha_trans:g}_aE{alpha_emit:g}"
        f"{fold_text}_restart{restart}.npz"
    )


def signature_for_run(args: argparse.Namespace, sessions: list[Session], mf: pd.DataFrame) -> str:
    observations = hashlib.sha256()
    for session in sessions:
        observations.update(np.asarray(session.therapist_codes, dtype=np.int64).tobytes())
        observations.update(np.asarray(session.client_codes, dtype=np.int64).tobytes())
    payload = {
        "checkpoint_version": 1,
        "files": mf["file"].tolist(),
        "lengths": [int(len(s.therapist_codes)) for s in sessions],
        "dyad_ids": mf["dyad_id"].astype(str).tolist(),
        "observations": observations.hexdigest(),
        "n_splits": args.n_splits,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "n_iter": args.n_iter,
        "tol": args.tol,
        "alpha_trans": list(args.alpha_trans),
        "alpha_emit": list(args.alpha_emit),
        "n_restarts": args.n_restarts,
        "refit_restarts": args.refit_restarts,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def train_or_load_restarts(
    *,
    stage: str,
    seqs: list[tuple[np.ndarray, np.ndarray]],
    K: int,
    alpha_trans: float,
    alpha_emit: float,
    fold: int | None,
    n_restarts: int,
    seed_base: int,
    args: argparse.Namespace,
    run_dir: Path,
    restart_records: list[dict[str, Any]],
) -> list[tuple[SharedEmissionHMM, dict[str, Any]]]:
    completed: list[tuple[SharedEmissionHMM, dict[str, Any]]] = []
    pending: list[tuple[int, Path, dict[str, Any]]] = []
    for restart in range(n_restarts):
        path = run_dir / checkpoint_name(stage, K, alpha_trans, alpha_emit, fold, restart)
        if args.resume and path.exists():
            model, metadata = load_model(path)
            completed.append((model, metadata))
            continue
        task = {
            "seqs": seqs,
            "K": K,
            "n_iter": args.n_iter,
            "tol": args.tol,
            "alpha_trans": alpha_trans,
            "alpha_emit": alpha_emit,
            "seed": seed_base + 7 * restart,
        }
        pending.append((restart, path, task))

    def persist(restart: int, path: Path, result: dict[str, Any]) -> None:
        atomic_save_model(path, result)
        restart_records.append(
            {
                "stage": stage,
                "K": K,
                "alpha_trans": alpha_trans,
                "alpha_emit": alpha_emit,
                "fold": -1 if fold is None else fold,
                "restart": restart,
                "train_ll_total": result["train_ll"],
                "n_iter_used": result["n_iter_used"],
                "checkpoint_file": path.name,
            }
        )
        atomic_write_csv(pd.DataFrame(restart_records), run_dir / "restart_results.csv")
        completed.append((model_from_result(result), result))
        print(f"  checkpointed {stage} restart {restart + 1}/{n_restarts}: {path.name}")

    if args.n_jobs > 1 and len(pending) > 1:
        workers = min(args.n_jobs, len(pending))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fit_restart, task): (restart, path)
                for restart, path, task in pending
            }
            for future in as_completed(futures):
                restart, path = futures[future]
                persist(restart, path, future.result())
    else:
        for restart, path, task in pending:
            persist(restart, path, fit_restart(task))
    return completed


def select_priors_for_k(out_dir: Path, K: int, args: argparse.Namespace) -> tuple[float, float]:
    summary_path = out_dir / "model_selection_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        rows = summary[summary["K"].astype(int) == int(K)]
        if not rows.empty:
            rows = rows.sort_values("test_ll_per_step", ascending=False)
            return float(rows.iloc[0]["alpha_trans"]), float(rows.iloc[0]["alpha_emit"])
    return float(args.alpha_trans[0]), float(args.alpha_emit[0])


def decode_and_save(
    *,
    sessions: list[Session],
    model: SharedEmissionHMM,
    out_dir: Path,
    run_dir: Path,
    K: int,
    suffix: str,
    resume: bool,
) -> None:
    decode_checkpoint = run_dir / f"decoded_sessions{suffix}.csv"
    out_csv = out_dir / f"decoded_sessions{suffix}.csv"
    decoded = (
        pd.read_csv(decode_checkpoint).to_dict("records")
        if resume and decode_checkpoint.exists()
        else []
    )
    decoded_ids = {str(row["session_id"]) for row in decoded}
    for s in sessions:
        viterbi_name = f"{s.session_id}_viterbi{suffix}.npy"
        if str(s.session_id) in decoded_ids and (out_dir / viterbi_name).exists():
            continue
        T, C = s.therapist_codes, s.client_codes
        occ = model.posterior(T, C).mean(axis=0)
        path = model.viterbi(T, C)
        row = {
            "session_id": s.session_id,
            "dyad_id": s.dyad_id,
            "n_steps": int(len(T)),
            **{f"occ_{k}": float(occ[k]) for k in range(K)},
            "path_file": viterbi_name,
        }
        decoded.append(row)
        decoded_ids.add(str(s.session_id))
        atomic_save_array(out_dir / viterbi_name, path.astype(np.int16))
        dec = pd.DataFrame(decoded)
        atomic_write_csv(dec, decode_checkpoint)
        atomic_write_csv(dec, out_csv)
    atomic_write_csv(pd.DataFrame(decoded), out_csv)


def refit_only_k(
    *,
    K: int,
    sessions: list[Session],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    best_model_path = out_dir / "best_model.json"
    base_params = json.loads(best_model_path.read_text()) if best_model_path.exists() else {}
    run_signature = base_params.get("run_signature", f"manual_refit_K{K}")
    run_dir = out_dir / "hmm_checkpoints" / str(run_signature)
    run_dir.mkdir(parents=True, exist_ok=True)
    restart_results_path = run_dir / "restart_results.csv"
    restart_records = (
        pd.read_csv(restart_results_path).to_dict("records")
        if args.resume and restart_results_path.exists()
        else []
    )
    alpha_trans, alpha_emit = select_priors_for_k(out_dir, K, args)
    all_seqs = [(s.therapist_codes, s.client_codes) for s in sessions]
    print(
        f"Refit-only mode: K={K}, alpha_trans={alpha_trans:g}, alpha_emit={alpha_emit:g}, "
        f"restarts={args.refit_restarts}, n_jobs={args.n_jobs}, resume={'on' if args.resume else 'off'}"
    )
    refits = train_or_load_restarts(
        stage="refit",
        seqs=all_seqs,
        K=K,
        alpha_trans=alpha_trans,
        alpha_emit=alpha_emit,
        fold=None,
        n_restarts=args.refit_restarts,
        seed_base=900000 + 1000 * K,
        args=args,
        run_dir=run_dir,
        restart_records=restart_records,
    )
    model, metadata = max(refits, key=lambda fit: fit[1]["train_ll"])
    suffix = f"_K{K}"
    params = {
        **base_params,
        "run_signature": run_signature,
        "K": K,
        "pi": model.pi.tolist(),
        "A": model.A.tolist(),
        "pT": model.pT.tolist(),
        "pC": model.pC.tolist(),
        "alpha_trans": alpha_trans,
        "alpha_emit": alpha_emit,
        "n_iter": args.n_iter,
        "tol": args.tol,
        "refit_restarts": args.refit_restarts,
        "refit_train_ll_total": float(metadata["train_ll"]),
        "refit_n_iter_used": int(metadata["n_iter_used"]),
        "selected_refit_only": True,
    }
    atomic_write_text(out_dir / f"best_model{suffix}.json", json.dumps(params, indent=2))
    decode_and_save(
        sessions=sessions,
        model=model,
        out_dir=out_dir,
        run_dir=run_dir,
        K=K,
        suffix=suffix,
        resume=args.resume,
    )
    print(f"\n--- K={K} refit artifacts written to {out_dir} ---")
    print(f"  - best_model{suffix}.json")
    print(f"  - decoded_sessions{suffix}.csv")
    print(f"  - *_viterbi{suffix}.npy")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fit Shared-Emission HMM with cross-validation and resumable refit."
    )
    p.add_argument("--config", default="configs/analysis.yaml", help="Analysis YAML config.")
    p.add_argument("--processed_dir", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--n_splits", type=int, default=None)
    p.add_argument("--k_min", type=int, default=None)
    p.add_argument("--k_max", type=int, default=None)
    p.add_argument("--n_iter", type=int, default=None)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--alpha_trans", type=float, nargs="+", default=None)
    p.add_argument("--alpha_emit", type=float, nargs="+", default=None)
    p.add_argument("--n_restarts", type=int, default=None)
    p.add_argument("--refit_restarts", type=int, default=None)
    p.add_argument("--n_jobs", type=int, default=None, help="Parallel restart fits to run.")
    p.add_argument(
        "--refit_only_k",
        type=int,
        default=None,
        help="Skip CV and only fit/decode a full-data model for this K.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume completed restart, fold, refit, and decoding checkpoints.",
    )
    p.add_argument(
        "--selection_metric",
        choices=["ll_per_step"],
        default="ll_per_step",
    )
    p.add_argument("--save_diagnostics", action="store_true")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    cfg_data = cfg.get("data", {})
    cfg_analysis = cfg.get("analysis", {})
    processed_dir = Path(args.processed_dir or cfg_data.get("processed_dir", "data/processed"))
    out_dir = Path(args.out_dir or cfg_analysis.get("hmm_out_dir", "artifacts/chmm"))
    args.n_splits = int(args.n_splits or cfg_analysis.get("hmm_n_splits", 5))
    args.k_min = int(args.k_min or cfg_analysis.get("hmm_k_min", 2))
    args.k_max = int(args.k_max or cfg_analysis.get("hmm_k_max", 8))
    args.n_iter = int(args.n_iter or cfg_analysis.get("hmm_n_iter", 150))
    args.tol = float(args.tol or cfg_analysis.get("hmm_tol", 1e-4))
    args.alpha_trans = args.alpha_trans or cfg_analysis.get("hmm_alpha_trans", [1.0])
    args.alpha_emit = args.alpha_emit or cfg_analysis.get("hmm_alpha_emit", [1.0])
    args.n_restarts = int(args.n_restarts or cfg_analysis.get("hmm_n_restarts", 3))
    args.refit_restarts = int(args.refit_restarts or args.n_restarts)
    args.n_jobs = max(1, int(args.n_jobs or cfg_analysis.get("hmm_n_jobs", 1)))
    args.resume = bool(cfg_analysis.get("hmm_resume", True) if args.resume is None else args.resume)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions, mf = load_sessions(processed_dir)
    if args.refit_only_k is not None:
        refit_only_k(K=int(args.refit_only_k), sessions=sessions, out_dir=out_dir, args=args)
        return

    groups = mf["dyad_id"].astype(str).tolist()
    splits = groupkfold_splits(groups, n_splits=args.n_splits)
    signature = signature_for_run(args, sessions, mf)
    run_name = signature if args.resume else f"{signature}_{uuid.uuid4().hex[:8]}"
    run_dir = out_dir / "hmm_checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "signature": signature,
                "resume": args.resume,
                "numba_available": NUMBA_AVAILABLE,
                "n_jobs": args.n_jobs,
            },
            indent=2,
        ),
    )

    existing_records_path = out_dir / "model_selection.csv"
    if args.resume and existing_records_path.exists():
        existing_df = pd.read_csv(existing_records_path)
        records = (
            existing_df[existing_df.get("run_signature", "") == signature].to_dict("records")
            if "run_signature" in existing_df.columns
            else []
        )
    else:
        records = []
    existing_keys = {
        (int(r["K"]), float(r["alpha_trans"]), float(r["alpha_emit"]), int(r["fold"]))
        for r in records
    }
    restart_results_path = run_dir / "restart_results.csv"
    restart_records = (
        pd.read_csv(restart_results_path).to_dict("records")
        if args.resume and restart_results_path.exists()
        else []
    )

    Ks = list(range(args.k_min, args.k_max + 1))
    aTs = [float(x) for x in args.alpha_trans]
    aEs = [float(x) for x in args.alpha_emit]
    print(
        f"Running sweep: K in [{args.k_min},{args.k_max}], "
        f"{len(Ks) * len(aTs) * len(aEs)} configs; "
        f"{args.n_splits} folds; {args.n_restarts} restarts; n_jobs={args.n_jobs}; "
        f"Numba={'on' if NUMBA_AVAILABLE else 'off'}; resume={'on' if args.resume else 'off'}"
    )
    for K in Ks:
        for aT in aTs:
            for aE in aEs:
                fold_scores = []
                for fold, sp in enumerate(splits):
                    key = (K, aT, aE, fold)
                    if key in existing_keys:
                        rec = next(
                            r
                            for r in records
                            if (
                                int(r["K"]),
                                float(r["alpha_trans"]),
                                float(r["alpha_emit"]),
                                int(r["fold"]),
                            )
                            == key
                        )
                        fold_scores.append(float(rec["test_ll_per_step"]))
                        print(f"  resumed completed fold K={K} fold={fold}")
                        continue

                    train = as_seqs(sessions, sp.train_idx)
                    test = as_seqs(sessions, sp.test_idx)
                    fits = train_or_load_restarts(
                        stage="cv",
                        seqs=train,
                        K=K,
                        alpha_trans=aT,
                        alpha_emit=aE,
                        fold=fold,
                        n_restarts=args.n_restarts,
                        seed_base=1000 * K + 97 * fold,
                        args=args,
                        run_dir=run_dir,
                        restart_records=restart_records,
                    )
                    best_model, best_metadata = max(fits, key=lambda fit: fit[1]["train_ll"])
                    te_ll = best_model.score_sequences(test)
                    te_steps = n_steps(test)
                    te_ll_per_step = float(te_ll / max(te_steps, 1))
                    fold_scores.append(te_ll_per_step)
                    rec = {
                        "run_signature": signature,
                        "K": K,
                        "alpha_trans": aT,
                        "alpha_emit": aE,
                        "fold": fold,
                        "test_ll_total": float(te_ll),
                        "test_ll_per_step": te_ll_per_step,
                        "test_steps": te_steps,
                    }
                    if args.save_diagnostics:
                        rec["train_ll_total"] = float(best_metadata["train_ll"])
                        rec["train_ll_per_step"] = float(
                            best_metadata["train_ll"] / max(n_steps(train), 1)
                        )
                        rec["best_fit_n_iter_used"] = int(best_metadata["n_iter_used"])
                    records.append(rec)
                    existing_keys.add(key)
                    atomic_write_csv(pd.DataFrame(records), existing_records_path)
                    print(f"  checkpointed fold K={K} fold={fold}: LL/step={te_ll_per_step:.6g}")
                print(
                    f"K={K:2d} aT={aT:g} aE={aE:g}  "
                    f"mean held-out LL/step = {float(np.mean(fold_scores)):.6g}"
                )

    df = pd.DataFrame(records)
    atomic_write_csv(df, existing_records_path)
    sel = (
        df.groupby(["K", "alpha_trans", "alpha_emit"], as_index=False)["test_ll_per_step"]
        .mean()
        .sort_values("test_ll_per_step", ascending=False)
    )
    atomic_write_csv(sel, out_dir / "model_selection_summary.csv")
    best = sel.iloc[0]
    best_K = int(best["K"])
    best_aT = float(best["alpha_trans"])
    best_aE = float(best["alpha_emit"])
    print(
        "\nBest config by held-out LL/step:",
        {"K": best_K, "alpha_trans": best_aT, "alpha_emit": best_aE},
    )

    all_seqs = [(s.therapist_codes, s.client_codes) for s in sessions]
    refits = train_or_load_restarts(
        stage="refit",
        seqs=all_seqs,
        K=best_K,
        alpha_trans=best_aT,
        alpha_emit=best_aE,
        fold=None,
        n_restarts=args.refit_restarts,
        seed_base=900000 + 1000 * best_K,
        args=args,
        run_dir=run_dir,
        restart_records=restart_records,
    )
    best_model, refit_metadata = max(refits, key=lambda fit: fit[1]["train_ll"])
    params = {
        "run_signature": signature,
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
        "refit_train_ll_total": float(refit_metadata["train_ll"]),
        "refit_n_iter_used": int(refit_metadata["n_iter_used"]),
        "n_splits": args.n_splits,
        "selection_metric": args.selection_metric,
    }
    atomic_write_text(out_dir / "best_model.json", json.dumps(params, indent=2))

    decode_and_save(
        sessions=sessions,
        model=best_model,
        out_dir=out_dir,
        run_dir=run_dir,
        K=best_K,
        suffix="",
        resume=args.resume,
    )

    print(f"\n--- Artifacts written to {out_dir} ---")
    print("  - model_selection.csv / model_selection_summary.csv: Resumable CV results.")
    print("  - best_model.json: Parameters from the full-data refit.")
    print("  - decoded_sessions.csv and *_viterbi.npy: Incrementally saved decoding results.")
    print(f"  - {run_dir}: Completed restart and decode checkpoints.")


if __name__ == "__main__":
    main()
