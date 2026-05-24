#!/usr/bin/env python3
"""Predict arriving affective regime/emotion from switch-centered topology."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


PH_TABLE = PROJECT_ROOT / "artifacts" / "07_transitions" / "switch_aligned_rolling_ph.csv"
GROWTH_TABLE = PROJECT_ROOT / "artifacts" / "07_transitions" / "notebook_tables" / "switch_growth_moderation_input.csv"
REGIME_LABEL_TABLE = PROJECT_ROOT / "artifacts" / "regime_geometry" / "regime_raw_spaff_distributions.csv"
OUT_DIR = PROJECT_ROOT / "artifacts" / "07_transitions" / "emotion_prediction"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


TOPO_FEATURES = ["total_persistence_h1", "betti_h1_auc", "persistence_entropy_h1"]
MIN_CLASS_COUNT = 25


def regime_label_map() -> dict[int, str]:
    labels = pd.read_csv(REGIME_LABEL_TABLE)[["regime_sorted_idx", "regime_label"]].drop_duplicates()
    return {
        int(r.regime_sorted_idx): str(r.regime_label)
        for r in labels.itertuples(index=False)
    }


def valence_from_label(label: str, role: str) -> str:
    pattern = r"T(neg|neu|pos)" if role == "therapist" else r"C(neg|neu|pos)"
    m = re.search(pattern, label)
    return m.group(1) if m else "unknown"


def slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def build_analysis_table() -> pd.DataFrame:
    # Use the existing switch-level total-persistence arc table for speed and
    # reproducibility. It contains one row per real switch with pre/post slopes
    # and after-before arc summaries used in the manuscript transition analyses.
    growth = pd.read_csv(GROWTH_TABLE)
    df = growth.copy()

    switch_meta = (
        pd.read_csv(
            PH_TABLE,
            usecols=[
                "session_id",
                "switch_index",
                "anchor_kind",
                "from_state",
                "to_state",
                "transition_type",
            ],
        )
        .query("anchor_kind == 'real'")
        .drop_duplicates(["session_id", "switch_index"])
        .drop(columns=["anchor_kind"])
    )
    df = df.merge(switch_meta, on=["session_id", "switch_index"], how="inner")

    fluct = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "07_transitions" / "notebook_tables" / "switch_fluctuation_analysis_table.csv"
    )
    fluct = fluct[fluct["feature"].eq("total_persistence_h1")].copy()
    fluct_cols = [
        "session_id",
        "switch_index",
        "delta_volatility_sd",
        "delta_volatility_rmssd",
        "near_switch_roughness_sd",
        "near_switch_roughness_rmssd",
        "near_switch_roughness_range",
    ]
    df = df.merge(fluct[[c for c in fluct_cols if c in fluct.columns]], on=["session_id", "switch_index"], how="left")

    labels = regime_label_map()
    df["from_regime_label"] = df["from_state"].map(labels)
    df["to_regime_label"] = df["to_state"].map(labels)
    df["to_therapist_valence"] = df["to_regime_label"].map(lambda x: valence_from_label(str(x), "therapist"))
    df["to_client_valence"] = df["to_regime_label"].map(lambda x: valence_from_label(str(x), "client"))
    df["from_therapist_valence"] = df["from_regime_label"].map(lambda x: valence_from_label(str(x), "therapist"))
    df["from_client_valence"] = df["from_regime_label"].map(lambda x: valence_from_label(str(x), "client"))
    return df


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    pre_boundary = ["pre_switch_slope"]
    full_arc = [
        "pre_switch_slope",
        "post_switch_slope",
        "after_before_delta",
        "arc_magnitude",
        "delta_volatility_sd",
        "delta_volatility_rmssd",
        "near_switch_roughness_sd",
        "near_switch_roughness_rmssd",
        "near_switch_roughness_range",
    ]
    baseline = [
        "from_state",
        "from_therapist_valence",
        "from_client_valence",
        "posterior_jump_l2",
        "posterior_entropy",
        "local_speed_l2",
        "log_from_run_len",
    ]
    sets = {
        "baseline_from_state_and_jump": [c for c in baseline if c in df.columns],
        "pre_boundary_topology_only": [c for c in pre_boundary if c in df.columns],
        "baseline_plus_pre_boundary_topology": [c for c in baseline + pre_boundary if c in df.columns],
        "full_arc_topology_only": [c for c in full_arc if c in df.columns],
        "baseline_plus_full_arc_topology": [c for c in baseline + full_arc if c in df.columns],
    }
    return sets


def make_model(x: pd.DataFrame) -> tuple[ColumnTransformer, LogisticRegression]:
    cat_cols = [c for c in x.columns if x[c].dtype == "object" or c.endswith("_state")]
    num_cols = [c for c in x.columns if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        C=0.25,
    )
    return pre, clf


def cross_validated_scores(df: pd.DataFrame, target: str, features: list[str], groups: pd.Series) -> dict[str, object] | None:
    d = df[features + [target, "session_id"]].copy().dropna(subset=[target])
    d = d.replace([np.inf, -np.inf], np.nan)
    for c in features:
        if c in d.columns and pd.api.types.is_numeric_dtype(d[c]):
            lo, hi = d[c].quantile([0.005, 0.995])
            d[c] = d[c].clip(lo, hi)
    counts = d[target].value_counts()
    keep_classes = counts[counts >= MIN_CLASS_COUNT].index
    d = d[d[target].isin(keep_classes)].copy()
    if d[target].nunique() < 2 or d["session_id"].nunique() < 5:
        return None

    group_counts = d["session_id"].nunique()
    n_splits = min(5, group_counts)
    cv = GroupKFold(n_splits=n_splits)
    y_true_all = []
    y_pred_all = []
    for train_idx, test_idx in cv.split(d[features], d[target], groups=d["session_id"]):
        x_train = d.iloc[train_idx][features]
        y_train = d.iloc[train_idx][target]
        x_test = d.iloc[test_idx][features]
        y_test = d.iloc[test_idx][target]
        if y_train.nunique() < 2:
            continue
        pre, clf = make_model(x_train)
        model = make_pipeline(pre, clf)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())

    if not y_true_all:
        return None
    return {
        "target": target,
        "n_switches": len(y_true_all),
        "n_sessions": int(d["session_id"].nunique()),
        "n_classes": int(pd.Series(y_true_all).nunique()),
        "classes": ", ".join(sorted(pd.Series(y_true_all).astype(str).unique())),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")),
    }


def permutation_scores(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    observed_bal_acc: float,
    n_perm: int = 20,
    seed: int = 7,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    d = df[features + [target, "session_id"]].dropna(subset=[target]).copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    counts = d[target].value_counts()
    keep_classes = counts[counts >= MIN_CLASS_COUNT].index
    d = d[d[target].isin(keep_classes)].copy()
    for _ in range(n_perm):
        dp = d.copy()
        dp[target] = rng.permutation(dp[target].to_numpy())
        res = cross_validated_scores(dp, target, features, dp["session_id"])
        if res is not None:
            scores.append(res["balanced_accuracy"])
    if not scores:
        return np.nan, np.nan
    scores = np.asarray(scores)
    p = (1 + np.sum(scores >= observed_bal_acc)) / (len(scores) + 1)
    return float(np.mean(scores)), float(p)


def run_models(df: pd.DataFrame) -> pd.DataFrame:
    sets = feature_sets(df)
    rows = []
    targets = [
        "to_therapist_valence",
        "to_client_valence",
        "to_regime_label",
        "therapist_valence_direction",
        "client_valence_direction",
    ]
    for target in targets:
        for set_name, feats in sets.items():
            res = cross_validated_scores(df, target, feats, df["session_id"])
            if res is None:
                continue
            res.update(
                {
                    "feature_set": set_name,
                    "n_features": len(feats),
                    "permutation_mean_balanced_accuracy": np.nan,
                    "permutation_p_value": np.nan,
                }
            )
            rows.append(res)
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame) -> None:
    if results.empty:
        return
    plt.figure(figsize=(12, 5.8))
    order = [
        "baseline_from_state_and_jump",
        "pre_boundary_topology_only",
        "baseline_plus_pre_boundary_topology",
        "full_arc_topology_only",
        "baseline_plus_full_arc_topology",
    ]
    sns.barplot(
        data=results,
        x="target",
        y="balanced_accuracy",
        hue="feature_set",
        hue_order=[x for x in order if x in results["feature_set"].unique()],
    )
    plt.axhline(1 / 3, color="black", ls="--", lw=1, alpha=0.45)
    plt.ylabel("Group-CV balanced accuracy")
    plt.xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "transition_emotion_prediction_balanced_accuracy.svg", format="svg")
    plt.close()


def main() -> None:
    df = build_analysis_table()
    results = run_models(df)
    df.to_csv(TAB_DIR / "transition_emotion_prediction_table.csv", index=False)
    results.to_csv(TAB_DIR / "transition_emotion_prediction_cv_results.csv", index=False)
    plot_results(results)

    print("Switch rows:", len(df))
    print("Sessions:", df["session_id"].nunique())
    print("Arriving therapist valence counts:")
    print(df["to_therapist_valence"].value_counts().to_string())
    print("\nArriving client valence counts:")
    print(df["to_client_valence"].value_counts().to_string())
    print("\nCross-validated prediction results:")
    if results.empty:
        print("No valid models.")
    else:
        print(
            results.sort_values(["target", "balanced_accuracy"], ascending=[True, False])[
                [
                    "target",
                    "feature_set",
                    "balanced_accuracy",
                    "macro_f1",
                    "permutation_mean_balanced_accuracy",
                    "permutation_p_value",
                    "n_switches",
                    "n_sessions",
                    "n_classes",
                ]
            ].to_string(index=False)
        )
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
