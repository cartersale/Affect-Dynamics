#!/usr/bin/env python3
"""Exploratory Session A -> Session B prediction from transition topology."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


SOURCE_TABLE = (
    PROJECT_ROOT
    / "artifacts"
    / "alliance_dynamics"
    / "transition_alliance"
    / "tables"
    / "session_level_alliance_transition_features.csv"
)
CATRQA_TABLE = PROJECT_ROOT / "artifacts" / "04_joint_affect_catRQA" / "catrqa_episode_results.csv"
GRAMMAR_TABLE = PROJECT_ROOT / "artifacts" / "05_joint_affect_grammar" / "grammar_episode_results.csv"
OUT_DIR = PROJECT_ROOT / "artifacts" / "07_transitions" / "a_to_b_prediction"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


PREDICTORS = [
    "arc_mean_post_switch_slope",
    "arc_mean_after_before_delta",
    "robust_session_mean_real_minus_pseudo_post_switch_slope",
    "robust_session_mean_real_minus_pseudo_arc_magnitude",
    "growth_arc_magnitude_mean",
    "fluct_near_switch_roughness_rmssd_mean",
]

SESSION_OUTCOMES = [
    "delta_ll",
    "delta_t",
    "delta_c",
    "r_switch",
    "mean_dwell",
    "cv_dwell",
    "h_trans",
    "h_occ",
    "pc1",
    "pc2",
]

TRANSITION_OUTCOMES = [
    "arc_mean_post_switch_slope",
    "arc_mean_after_before_delta",
    "growth_arc_magnitude_mean",
    "growth_posterior_jump_l2_mean",
    "growth_client_valence_delta_std",
    "growth_therapist_valence_delta_std",
    "fluct_near_switch_roughness_rmssd_mean",
    "client_valence_toward_more_negative_prop",
    "therapist_valence_toward_neutral_or_positive_prop",
]


def normalize_dyad_id(x: object) -> str:
    s = str(x).strip().split(".")[0]
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"{int(digits):03d}" if digits else s.upper()


def weighted_session_mean(df: pd.DataFrame, value_cols: list[str], weight_col: str = "duration") -> pd.DataFrame:
    keep = ["session_id", weight_col] + [c for c in value_cols if c in df.columns]
    d = df[keep].copy()
    d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce")
    for c in value_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    rows = []
    for sid, g in d.groupby("session_id"):
        row = {"session_id": sid}
        w = g[weight_col].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
        for c in value_cols:
            if c not in g.columns:
                continue
            x = g[c].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(w)
            row[c] = float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_session_table() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_TABLE)
    df["session_id"] = df["session_id"].astype(str)
    df["dyad_id"] = df["dyad_id"].map(normalize_dyad_id)

    if CATRQA_TABLE.exists():
        rqa = pd.read_csv(CATRQA_TABLE)
        rqa_cols = ["RR_Z_shuffle", "DET_Z_shuffle", "LAM_Z_shuffle"]
        rqa_s = weighted_session_mean(rqa, rqa_cols).rename(columns={c: f"catrqa_{c}" for c in rqa_cols})
        df = df.merge(rqa_s, on="session_id", how="left")

    if GRAMMAR_TABLE.exists():
        gram = pd.read_csv(GRAMMAR_TABLE)
        gram = gram[gram["k"].isin([2, 3])].copy()
        gram_cols = [
            "motif_entropy_z_shuffle",
            "effective_motif_count_z_shuffle",
            "repeat_rate_z_shuffle",
            "motif_transition_diversity_z_shuffle",
        ]
        gram_s = weighted_session_mean(gram, gram_cols)
        gram_s = gram_s.rename(columns={c: f"grammar_{c}" for c in gram_cols})
        df = df.merge(gram_s, on="session_id", how="left")

    return df


def build_a_to_b(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["session_letter"] = d["session_id"].str.extract(r"_([AB])$", expand=False)
    a = d[d["session_letter"] == "A"].copy()
    b = d[d["session_letter"] == "B"].copy()
    pair = a.merge(b, on="dyad_id", suffixes=("_A", "_B"), how="inner")
    pair.insert(0, "dyad_id_norm", pair.pop("dyad_id"))
    return pair


def add_fdr(tab: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if tab.empty:
        return tab
    out = tab.copy()
    out["q_value"] = np.nan
    out["reject_fdr_0.05"] = False
    for _, idx in out.groupby(group_col).groups.items():
        p = out.loc[idx, "p_value"]
        ok = p.notna()
        if ok.any():
            reject, q, _, _ = multipletests(p[ok].to_numpy(), alpha=0.05, method="fdr_bh")
            out.loc[p[ok].index, "q_value"] = q
            out.loc[p[ok].index, "reject_fdr_0.05"] = reject
    return out


def spearman_table(pair: pd.DataFrame, predictors: list[str], outcomes: list[str], target: str) -> pd.DataFrame:
    rows = []
    for pred in predictors:
        pred_col = f"{pred}_A"
        if pred_col not in pair.columns:
            continue
        for outcome in outcomes:
            out_col = f"{outcome}_B" if target == "B_level" else f"{outcome}_change_B_minus_A"
            if out_col not in pair.columns:
                continue
            sub = pair[[pred_col, out_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 10 or sub[pred_col].std(ddof=0) == 0 or sub[out_col].std(ddof=0) == 0:
                continue
            rho, p = spearmanr(sub[pred_col], sub[out_col])
            rows.append(
                {
                    "target": target,
                    "a_predictor": pred,
                    "outcome": outcome,
                    "rho": float(rho),
                    "p_value": float(p),
                    "n_dyads": int(len(sub)),
                }
            )
    return add_fdr(pd.DataFrame(rows), "target")


def loo_ridge_table(pair: pd.DataFrame, predictors: list[str], outcomes: list[str], target: str) -> pd.DataFrame:
    pred_cols = [f"{p}_A" for p in predictors if f"{p}_A" in pair.columns]
    rows = []
    alphas = np.logspace(-3, 3, 25)
    for outcome in outcomes:
        out_col = f"{outcome}_B" if target == "B_level" else f"{outcome}_change_B_minus_A"
        if out_col not in pair.columns:
            continue
        d = pair[pred_cols + [out_col]].apply(pd.to_numeric, errors="coerce").dropna()
        d = d.loc[:, d.std(ddof=0).fillna(0) > 0]
        if out_col not in d.columns:
            continue
        x_cols = [c for c in d.columns if c != out_col]
        if len(d) < 12 or len(x_cols) < 2:
            continue

        y = d[out_col].to_numpy(dtype=float)
        preds = np.zeros_like(y, dtype=float)
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(d):
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            model.fit(d.iloc[train_idx][x_cols], y[train_idx])
            preds[test_idx] = model.predict(d.iloc[test_idx][x_cols])

        rho, p = spearmanr(y, preds) if np.std(preds) > 0 and np.std(y) > 0 else (np.nan, np.nan)
        rows.append(
            {
                "target": target,
                "outcome": outcome,
                "n_dyads": int(len(d)),
                "n_predictors": int(len(x_cols)),
                "loo_r2": float(r2_score(y, preds)),
                "loo_mae": float(mean_absolute_error(y, preds)),
                "pred_obs_spearman_rho": float(rho),
                "pred_obs_p_value": float(p),
                "predictors": ", ".join([c.removesuffix("_A") for c in x_cols]),
            }
        )
    return pd.DataFrame(rows)


def make_change_columns(pair: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    out = pair.copy()
    for c in outcomes:
        a_col = f"{c}_A"
        b_col = f"{c}_B"
        if a_col in out.columns and b_col in out.columns:
            out[f"{c}_change_B_minus_A"] = pd.to_numeric(out[b_col], errors="coerce") - pd.to_numeric(out[a_col], errors="coerce")
    return out


def save_heatmap(tab: pd.DataFrame, value: str, out_path: Path, title: str) -> None:
    if tab.empty:
        return
    heat = tab.pivot(index="a_predictor", columns="outcome", values=value)
    if heat.empty:
        return
    plt.figure(figsize=(max(8, 0.5 * heat.shape[1]), max(4, 0.45 * heat.shape[0])))
    sns.heatmap(heat, cmap="coolwarm", center=0, linewidths=0.3, annot=True, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def main() -> None:
    session_df = load_session_table()
    outcomes = SESSION_OUTCOMES + TRANSITION_OUTCOMES
    extra_outcomes = [
        c
        for c in session_df.columns
        if c.startswith("catrqa_") or c.startswith("grammar_")
    ]
    outcomes = [c for c in outcomes + extra_outcomes if c in session_df.columns]
    predictors = [p for p in PREDICTORS if p in session_df.columns]

    pair = build_a_to_b(session_df)
    pair = make_change_columns(pair, outcomes)

    b_spearman = spearman_table(pair, predictors, outcomes, target="B_level")
    change_spearman = spearman_table(pair, predictors, outcomes, target="B_minus_A_change")
    spearman_all = pd.concat([b_spearman, change_spearman], ignore_index=True)

    b_ridge = loo_ridge_table(pair, predictors, outcomes, target="B_level")
    change_ridge = loo_ridge_table(pair, predictors, outcomes, target="B_minus_A_change")
    ridge_all = pd.concat([b_ridge, change_ridge], ignore_index=True)

    pair.to_csv(TAB_DIR / "session_a_to_b_analysis_table.csv", index=False)
    b_spearman.to_csv(TAB_DIR / "a_predictors_to_b_outcomes_spearman.csv", index=False)
    change_spearman.to_csv(TAB_DIR / "a_predictors_to_b_minus_a_change_spearman.csv", index=False)
    spearman_all.sort_values(["q_value", "p_value"], na_position="last").to_csv(
        TAB_DIR / "a_to_b_spearman_ranked_all.csv", index=False
    )
    ridge_all.sort_values(["target", "loo_r2"], ascending=[True, False]).to_csv(
        TAB_DIR / "a_to_b_ridge_loo_prediction.csv", index=False
    )

    save_heatmap(
        b_spearman,
        "rho",
        FIG_DIR / "session_a_predictors_to_b_outcomes_spearman.svg",
        "Session A transition topology vs Session B outcomes",
    )
    save_heatmap(
        change_spearman,
        "rho",
        FIG_DIR / "session_a_predictors_to_b_minus_a_change_spearman.svg",
        "Session A transition topology vs B-A outcome change",
    )

    print("A->B dyads:", pair["dyad_id_norm"].nunique())
    print("A predictors:", predictors)
    print("Outcomes:", len(outcomes))
    print("\nTop rank correlations:")
    if spearman_all.empty:
        print("No valid correlations.")
    else:
        print(
            spearman_all.sort_values(["q_value", "p_value"], na_position="last")
            [
                [
                    "target",
                    "a_predictor",
                    "outcome",
                    "rho",
                    "p_value",
                    "q_value",
                    "reject_fdr_0.05",
                    "n_dyads",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )
    print("\nBest leave-one-out ridge rows:")
    if ridge_all.empty:
        print("No valid ridge models.")
    else:
        print(
            ridge_all.sort_values("loo_r2", ascending=False)
            [
                [
                    "target",
                    "outcome",
                    "loo_r2",
                    "pred_obs_spearman_rho",
                    "pred_obs_p_value",
                    "n_dyads",
                    "n_predictors",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
