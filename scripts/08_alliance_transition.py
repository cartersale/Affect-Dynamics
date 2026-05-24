#!/usr/bin/env python3
"""Session-level alliance associations with transition-arc summaries."""

from __future__ import annotations

import warnings
from pathlib import Path

import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


ALLIANCE_TABLE = PROJECT_ROOT / "artifacts" / "alliance_dynamics" / "notebook_tables" / "session_level_alliance_dynamics_table.csv"
TRANSITION_TABLE_DIR = PROJECT_ROOT / "artifacts" / "07_transitions" / "notebook_tables"
OUT_DIR = PROJECT_ROOT / "artifacts" / "alliance_dynamics" / "transition_alliance"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


PRIMARY_ALLIANCE = ["Difference (Cl-Ther)", "RRI Total"]
EXPLORATORY_ALLIANCE = [
    "WAIT Task",
    "WAIT Bond",
    "WAIT Goal",
    "WAIC Task",
    "WAIC Bond",
    "WAIC Goal",
    "RRI  Genuine",
    "RRI Realism",
]


def normalize_dyad_id(x: object) -> str:
    s = str(x).strip().split(".")[0]
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"{int(digits):03d}" if digits else s.upper()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _session_numeric_summary(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    d = df[["session_id"] + cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    out = d.groupby("session_id")[cols].agg(["mean", "median", "std"]).reset_index()
    out.columns = [
        "session_id" if c[0] == "session_id" else f"{prefix}_{c[0]}_{c[1]}"
        for c in out.columns.to_flat_index()
    ]
    return out


def _session_category_props(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    base = df[["session_id", "switch_index", col]].dropna().drop_duplicates()
    if base.empty:
        return pd.DataFrame()
    tab = pd.crosstab(base["session_id"], base[col], normalize="index").reset_index()
    tab.columns = ["session_id"] + [f"{prefix}_{str(c)}_prop" for c in tab.columns[1:]]
    return tab


def build_transition_session_features() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    arc = _read_csv(TRANSITION_TABLE_DIR / "session_arc_geometry_analysis_table.csv")
    if not arc.empty:
        keep = [
            "session_id",
            "mean_post_switch_slope",
            "median_post_switch_slope",
            "n_real_switches",
            "mean_after_before_delta",
            "median_after_before_delta",
        ]
        pieces.append(arc[[c for c in keep if c in arc.columns]].add_prefix("arc_").rename(columns={"arc_session_id": "session_id"}))

    robustness = _read_csv(TRANSITION_TABLE_DIR / "session_level_robustness_values.csv")
    if not robustness.empty:
        rob = robustness.pivot_table(index="session_id", columns="metric", values="difference", aggfunc="mean").reset_index()
        rob = rob.rename(columns={c: f"robust_{c}" for c in rob.columns if c != "session_id"})
        pieces.append(rob)

    growth = _read_csv(TRANSITION_TABLE_DIR / "switch_growth_moderation_input.csv")
    growth_cols = [
        "post_switch_slope",
        "pre_switch_slope",
        "after_before_delta",
        "arc_magnitude",
        "posterior_jump_l2",
        "posterior_entropy",
        "local_speed_l2",
        "from_run_len",
        "to_run_len",
        "client_valence_delta",
        "therapist_valence_delta",
    ]
    pieces.append(_session_numeric_summary(growth, growth_cols, "growth"))
    pieces.append(_session_category_props(growth, "client_valence_direction", "client_valence"))
    pieces.append(_session_category_props(growth, "therapist_valence_direction", "therapist_valence"))

    fluct = _read_csv(TRANSITION_TABLE_DIR / "switch_fluctuation_analysis_table.csv")
    if not fluct.empty and "feature" in fluct.columns:
        fluct = fluct[fluct["feature"] == "total_persistence_h1"].copy()
    fluct_cols = [
        "delta_volatility_sd",
        "delta_volatility_rmssd",
        "delta_volatility_mean_abs_diff",
        "near_switch_roughness_sd",
        "near_switch_roughness_rmssd",
        "near_switch_roughness_mean_abs_diff",
        "near_switch_roughness_range",
    ]
    pieces.append(_session_numeric_summary(fluct, fluct_cols, "fluct"))

    pieces = [p for p in pieces if p is not None and not p.empty]
    if not pieces:
        return pd.DataFrame()

    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on="session_id", how="outer")
    return out


def fit_mixedlm(df: pd.DataFrame, outcome: str, predictor: str, covariates: list[str]) -> dict[str, object] | None:
    covariates = [c for c in covariates if c in df.columns and c != predictor]
    cols = ["dyad_id", outcome, predictor] + covariates
    d = df[cols].copy()
    for c in [outcome, predictor] + [c for c in covariates if c in df.columns]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().copy()
    if len(d) < 20 or d["dyad_id"].nunique() < 8 or d[predictor].std(ddof=0) == 0:
        return None

    z_cols = []
    term_map = {}
    for i, c in enumerate([predictor] + [c for c in covariates if c in d.columns]):
        sd = d[c].std(ddof=0)
        if pd.isna(sd) or sd == 0:
            continue
        z = f"z{i}"
        d[z] = (d[c] - d[c].mean()) / sd
        z_cols.append(z)
        term_map[c] = z

    if predictor not in term_map:
        return None

    d["__y"] = d[outcome]
    model = smf.mixedlm("__y ~ " + " + ".join(z_cols), data=d, groups=d["dyad_id"])
    fit = None
    for method in ["lbfgs", "powell", "cg"]:
        try:
            fit = model.fit(reml=False, method=method, maxiter=400, disp=False)
            break
        except Exception:
            fit = None
    if fit is None:
        return None

    term = term_map[predictor]
    conf = fit.conf_int()
    return {
        "alliance_outcome": outcome,
        "transition_predictor": predictor,
        "coef_per_sd_transition": float(fit.fe_params.get(term, np.nan)),
        "se": float(fit.bse_fe.get(term, np.nan)),
        "z": float(fit.tvalues.get(term, np.nan)),
        "p_value": float(fit.pvalues.get(term, np.nan)),
        "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
        "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
        "n_sessions": int(len(d)),
        "n_dyads": int(d["dyad_id"].nunique()),
        "covariates": ", ".join([c for c in covariates if c in term_map]),
    }


def add_fdr(df: pd.DataFrame, group_cols: list[str], p_col: str = "p_value") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["q_value"] = np.nan
    out["reject_fdr_0.05"] = False
    for _, idx in out.groupby(group_cols).groups.items():
        p = out.loc[idx, p_col]
        ok = p.notna()
        if ok.any():
            reject, q, _, _ = multipletests(p[ok].to_numpy(), alpha=0.05, method="fdr_bh")
            out.loc[p[ok].index, "q_value"] = q
            out.loc[p[ok].index, "reject_fdr_0.05"] = reject
    return out


def spearman_screen(df: pd.DataFrame, alliance_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    dyad = df[["dyad_id"] + alliance_cols + feature_cols].copy()
    for c in alliance_cols + feature_cols:
        dyad[c] = pd.to_numeric(dyad[c], errors="coerce")
    dyad = dyad.groupby("dyad_id", as_index=False)[alliance_cols + feature_cols].mean()
    for a in alliance_cols:
        for f in feature_cols:
            sub = dyad[[a, f]].dropna()
            if len(sub) < 8:
                continue
            rho, p = spearmanr(sub[a], sub[f])
            rows.append({"alliance_metric": a, "transition_feature": f, "rho": rho, "p_value": p, "n_dyads": len(sub)})
    out = pd.DataFrame(rows)
    return add_fdr(out, ["alliance_metric"]) if not out.empty else out


def main() -> None:
    alliance = _read_csv(ALLIANCE_TABLE)
    if alliance.empty:
        raise SystemExit(f"Alliance analysis table not found: {ALLIANCE_TABLE}")
    alliance["session_id"] = alliance["session_id"].astype(str)
    alliance["dyad_id"] = alliance["dyad_id"].map(normalize_dyad_id)

    trans = build_transition_session_features()
    if trans.empty:
        raise SystemExit("No transition feature tables were found.")

    df = alliance.merge(trans, on="session_id", how="inner")
    alliance_cols = [c for c in PRIMARY_ALLIANCE + EXPLORATORY_ALLIANCE if c in df.columns]
    feature_cols = [c for c in trans.columns if c != "session_id"]
    feature_cols = [c for c in feature_cols if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 20]

    primary_features = [
        "arc_mean_post_switch_slope",
        "arc_median_post_switch_slope",
        "arc_mean_after_before_delta",
        "arc_median_after_before_delta",
        "robust_session_mean_real_minus_pseudo_post_switch_slope",
        "robust_session_mean_real_minus_pseudo_pre_switch_slope",
        "growth_arc_magnitude_mean",
        "growth_arc_magnitude_median",
        "growth_after_before_delta_mean",
        "fluct_delta_volatility_sd_mean",
        "fluct_near_switch_roughness_rmssd_mean",
    ]
    primary_features = [c for c in primary_features if c in feature_cols]

    covariates = [c for c in ["arc_n_real_switches"] if c in df.columns]
    rows = []
    for outcome in [c for c in PRIMARY_ALLIANCE if c in df.columns]:
        for predictor in primary_features:
            res = fit_mixedlm(df, outcome, predictor, covariates)
            if res is not None:
                res["model_block"] = "primary_transition_arc_features"
                rows.append(res)
    primary_models = add_fdr(pd.DataFrame(rows), ["alliance_outcome"]) if rows else pd.DataFrame()

    rows = []
    for outcome in alliance_cols:
        for predictor in feature_cols:
            res = fit_mixedlm(df, outcome, predictor, covariates)
            if res is not None:
                res["model_block"] = "exploratory_transition_feature_screen"
                rows.append(res)
    exploratory_models = add_fdr(pd.DataFrame(rows), ["alliance_outcome"]) if rows else pd.DataFrame()

    corr = spearman_screen(df, alliance_cols, feature_cols)

    df.to_csv(TAB_DIR / "session_level_alliance_transition_features.csv", index=False)
    primary_models.to_csv(TAB_DIR / "models_primary_alliance_from_transition_features.csv", index=False)
    exploratory_models.to_csv(TAB_DIR / "models_exploratory_alliance_from_transition_features_fdr.csv", index=False)
    corr.to_csv(TAB_DIR / "dyad_collapsed_transition_alliance_spearman_fdr.csv", index=False)

    ranked = exploratory_models.sort_values(["q_value", "p_value"], na_position="last").head(40)
    ranked.to_csv(TAB_DIR / "ranked_transition_alliance_screen_top40.csv", index=False)

    if not corr.empty:
        top_features = (
            corr.assign(abs_rho=corr["rho"].abs())
            .sort_values("abs_rho", ascending=False)["transition_feature"]
            .drop_duplicates()
            .head(30)
            .tolist()
        )
        heat = corr[corr["transition_feature"].isin(top_features)].pivot(
            index="alliance_metric",
            columns="transition_feature",
            values="rho",
        )
        plt.figure(figsize=(max(10, 0.35 * len(top_features)), 5.5))
        sns.heatmap(heat, cmap="coolwarm", center=0, linewidths=0.3)
        plt.title("Alliance vs Transition Features (dyad-collapsed Spearman rho)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "transition_alliance_spearman_top_features.svg", format="svg")
        plt.close()

    print("Merged sessions:", len(df))
    print("Dyads:", df["dyad_id"].nunique())
    print("Transition features tested:", len(feature_cols))
    print("Primary model rows:", len(primary_models))
    print("Exploratory model rows:", len(exploratory_models))
    if not primary_models.empty:
        print("\nPrimary transition-arc models:")
        print(
            primary_models.sort_values("p_value")[
                [
                    "alliance_outcome",
                    "transition_predictor",
                    "coef_per_sd_transition",
                    "p_value",
                    "q_value",
                    "n_sessions",
                    "n_dyads",
                ]
            ].to_string(index=False)
        )
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
