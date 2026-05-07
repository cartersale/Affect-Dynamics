import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from scipy.stats import chi2, spearmanr, wilcoxon
from statsmodels.regression.mixed_linear_model import MixedLM

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


GROWTH_WINDOWS = {
    "pre_switch_growth": (-20, -5),
    "post_switch_growth": (5, 20),
}

FIGURE_FONT_FAMILY = "serif"
FIGURE_FONT_SERIF = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]


def _safe_quantile_bins(x: pd.Series, labels=("small", "medium", "large")) -> pd.Series:
    good = x.dropna()
    if good.nunique() < 3 or len(good) < 6:
        return pd.Series(["unknown"] * len(x), index=x.index)

    q1, q2 = good.quantile([1 / 3, 2 / 3])

    def _bin(v):
        if pd.isna(v):
            return "unknown"
        if v <= q1:
            return labels[0]
        if v <= q2:
            return labels[1]
        return labels[2]

    return x.map(_bin)


def _compute_run_labels(chmm_dir: Path) -> pd.DataFrame:
    dec_path = chmm_dir / "decoded_sessions.csv"
    if not dec_path.exists():
        return pd.DataFrame()

    dec = pd.read_csv(dec_path)
    path_map = {}
    for _, row in dec.iterrows():
        path_file = chmm_dir / row["path_file"]
        if path_file.exists():
            path_map[str(row["session_id"])] = np.load(path_file).astype(int)

    run_info_rows = []
    for session_id, path in path_map.items():
        if len(path) < 2:
            continue
        run_start = 0
        switch_counter = 0
        for t in range(1, len(path)):
            if path[t] != path[t - 1]:
                from_run_len = t - run_start
                run_end = t + 1
                while run_end < len(path) and path[run_end] == path[t]:
                    run_end += 1
                to_run_len = run_end - t
                run_info_rows.append(
                    {
                        "session_id": session_id,
                        "switch_index": switch_counter,
                        "from_run_len": int(from_run_len),
                        "to_run_len": int(to_run_len),
                    }
                )
                switch_counter += 1
                run_start = t

    df_runs = pd.DataFrame(run_info_rows)
    if df_runs.empty:
        return df_runs

    persistent_thresh = float(df_runs[["from_run_len", "to_run_len"]].stack().quantile(0.75))
    df_runs["from_persistent"] = df_runs["from_run_len"] >= persistent_thresh
    df_runs["to_persistent"] = df_runs["to_run_len"] >= persistent_thresh

    def stability_type(row):
        if row["from_persistent"] and not row["to_persistent"]:
            return "out_of_persistent"
        if not row["from_persistent"] and row["to_persistent"]:
            return "into_persistent"
        if row["from_persistent"] and row["to_persistent"]:
            return "persistent_to_persistent"
        return "nonpersistent_to_nonpersistent"

    df_runs["regime_stability_type"] = df_runs.apply(stability_type, axis=1)
    return df_runs[
        [
            "session_id",
            "switch_index",
            "from_run_len",
            "to_run_len",
            "regime_stability_type",
        ]
    ]


def _state_valence_scores(model_json: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not model_json.exists():
        return None, None

    params = json.loads(model_json.read_text())
    p_t = np.asarray(params.get("pT"), dtype=float)
    p_c = np.asarray(params.get("pC"), dtype=float)

    therapist_score = None
    client_score = None
    if p_t.ndim == 2 and p_t.shape[1] == 3:
        therapist_score = p_t @ np.array([-1.0, 0.0, 1.0])
    if p_c.ndim == 2 and p_c.shape[1] == 3:
        client_score = p_c @ np.array([-1.0, 0.0, 1.0])
    return therapist_score, client_score


def _valence_direction_label(
    from_state: float,
    to_state: float,
    score: np.ndarray | None,
    threshold: float = 0.1,
    neutral_band: float = 0.25,
) -> str:
    if score is None or pd.isna(from_state) or pd.isna(to_state):
        return "unknown"
    if from_state < 0 or to_state < 0:
        return "unknown"

    fr = int(from_state)
    to = int(to_state)
    if fr >= len(score) or to >= len(score):
        return "unknown"

    dv = float(score[to]) - float(score[fr])
    if abs(dv) < threshold:
        return "neutral_or_minimal_change"
    if dv < 0:
        return "toward_more_negative"
    if abs(float(score[fr])) <= neutral_band and abs(float(score[to])) <= neutral_band:
        return "neutral_or_minimal_change"
    return "toward_neutral_or_positive"


def _state_score_value(state: float, score: np.ndarray | None) -> float:
    if score is None or pd.isna(state):
        return np.nan
    if state < 0:
        return np.nan
    idx = int(state)
    if idx >= len(score):
        return np.nan
    return float(score[idx])


def _state_transition_score_delta(
    from_state: float,
    to_state: float,
    score: np.ndarray | None,
) -> float:
    from_value = _state_score_value(from_state, score)
    to_value = _state_score_value(to_state, score)
    if pd.isna(from_value) or pd.isna(to_value):
        return np.nan
    return float(to_value - from_value)


def _phase_window_label(rel_time_sec: float) -> str | None:
    if pd.isna(rel_time_sec):
        return None
    if -30 <= rel_time_sec < -15:
        return "far_before"
    if -15 <= rel_time_sec < -5:
        return "near_before"
    if -5 <= rel_time_sec <= 5:
        return "transition"
    if 5 < rel_time_sec <= 15:
        return "near_after"
    if 15 < rel_time_sec <= 30:
        return "far_after"
    return None


def _add_baseline_centered_features(
    df: pd.DataFrame,
    features: Iterable[str],
    baseline_window=(-15, -5),
) -> pd.DataFrame:
    out = df.copy()
    lo, hi = baseline_window
    for feature in features:
        baseline = (
            out[(out["rel_time_sec"] >= lo) & (out["rel_time_sec"] <= hi)]
            .groupby(["session_id", "switch_index", "anchor_kind"], as_index=False)[feature]
            .mean()
            .rename(columns={feature: f"{feature}_baseline"})
        )
        out = out.merge(baseline, on=["session_id", "switch_index", "anchor_kind"], how="left")
        out[f"{feature}_baseline_centered"] = out[feature] - out[f"{feature}_baseline"]
    return out


def _apply_publication_style() -> None:
    plt.rcParams["font.family"] = FIGURE_FONT_FAMILY
    plt.rcParams["font.serif"] = FIGURE_FONT_SERIF
    # Convert glyphs to paths so the SVG preserves the intended typography.
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def _minimal_axis(ax, show_left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(show_left)
    ax.spines["bottom"].set_linewidth(0.8)
    if show_left:
        ax.spines["left"].set_linewidth(0.8)
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def _save_publication_svg(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight", facecolor="white")


def _slope_summary_with_se(df_growth: pd.DataFrame, feature: str) -> pd.DataFrame:
    d = df_growth[df_growth["feature"] == feature].dropna(subset=["window", "anchor_kind", "slope"]).copy()
    if d.empty:
        return pd.DataFrame()
    out = (
        d.groupby(["window", "anchor_kind"], as_index=False)["slope"]
        .agg(mean_slope="mean", sd_slope="std", n="count")
    )
    out["se_slope"] = out["sd_slope"] / np.sqrt(out["n"].clip(lower=1))
    return out


def _quantile_bin_summary(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_bins: int = 20,
) -> pd.DataFrame:
    d = df[[x_col, y_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    n_unique = d[x_col].nunique()
    if n_unique < 2:
        return pd.DataFrame()

    q = min(n_bins, int(n_unique))
    if q < 2:
        return pd.DataFrame()

    d["bin"] = pd.qcut(d[x_col], q=q, labels=False, duplicates="drop")
    out = (
        d.groupby("bin", as_index=False)
        .agg(
            x_mean=(x_col, "mean"),
            x_min=(x_col, "min"),
            x_max=(x_col, "max"),
            y_mean=(y_col, "mean"),
            y_sd=(y_col, "std"),
            n=(y_col, "count"),
        )
        .sort_values("x_mean")
    )
    out["y_se"] = out["y_sd"] / np.sqrt(out["n"].clip(lower=1))
    out["y_lower"] = out["y_mean"] - 1.96 * out["y_se"]
    out["y_upper"] = out["y_mean"] + 1.96 * out["y_se"]
    return out


def _zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=0)
    denom = sd if pd.notna(sd) and sd > 0 else 1.0
    return (x - x.mean()) / denom


def _likelihood_ratio_row(
    *,
    feature: str,
    full_fit,
    reduced_fit,
    model_type: str,
    spline_df: int,
    n_obs: int,
    n_sessions: int,
) -> dict[str, float | int | str]:
    ll_full = float(getattr(full_fit, "llf", np.nan))
    ll_reduced = float(getattr(reduced_fit, "llf", np.nan))
    df_full = len(getattr(full_fit, "params", []))
    df_reduced = len(getattr(reduced_fit, "params", []))
    df_diff = max(int(df_full - df_reduced), 0)
    lr_stat = float(max(0.0, 2.0 * (ll_full - ll_reduced))) if pd.notna(ll_full) and pd.notna(ll_reduced) else np.nan
    p_value = float(chi2.sf(lr_stat, df_diff)) if pd.notna(lr_stat) and df_diff > 0 else np.nan
    return {
        "feature": feature,
        "model_type": model_type,
        "spline_df_used": int(spline_df),
        "n_obs": int(n_obs),
        "n_sessions": int(n_sessions),
        "ll_full": ll_full,
        "ll_reduced": ll_reduced,
        "lr_stat": lr_stat,
        "df_diff": df_diff,
        "p_value": p_value,
        "aic_full": float(getattr(full_fit, "aic", np.nan)),
        "bic_full": float(getattr(full_fit, "bic", np.nan)),
        "aic_reduced": float(getattr(reduced_fit, "aic", np.nan)),
        "bic_reduced": float(getattr(reduced_fit, "bic", np.nan)),
    }


def _fit_valence_slope_model(df_growth_moderation_input: pd.DataFrame):
    use_cols = [
        "session_id",
        "post_switch_slope",
        "client_valence_direction",
        "therapist_valence_direction",
        "z_posterior_jump_l2",
    ]
    d = df_growth_moderation_input[[col for col in use_cols if col in df_growth_moderation_input.columns]].dropna().copy()
    if len(d) < 30 or d["session_id"].nunique() < 8:
        return pd.DataFrame(), None

    fit = MixedLM.from_formula(
        "post_switch_slope ~ C(client_valence_direction) + C(therapist_valence_direction) + z_posterior_jump_l2",
        data=d,
        groups="session_id",
    ).fit(reml=False, method="lbfgs", maxiter=300)
    params = fit.params
    ci = fit.conf_int()
    se = fit.bse

    term_map = {
        "C(client_valence_direction)[T.toward_more_negative]": ("Client toward negative", "client"),
        "C(client_valence_direction)[T.toward_neutral_or_positive]": ("Client toward neutral/positive", "client"),
        "C(therapist_valence_direction)[T.toward_more_negative]": ("Therapist toward negative", "therapist"),
        "C(therapist_valence_direction)[T.toward_neutral_or_positive]": ("Therapist toward neutral/positive", "therapist"),
    }

    rows = []
    for term, (label, role) in term_map.items():
        if term not in params.index:
            continue
        rows.append(
            {
                "term": term,
                "label": label,
                "role": role,
                "coef": float(params[term]),
                "se": float(se[term]),
                "lower": float(ci.loc[term, 0]),
                "upper": float(ci.loc[term, 1]),
            }
        )
    return pd.DataFrame(rows), fit


def build_analysis_table(
    df_aligned: pd.DataFrame,
    df_switch: pd.DataFrame,
    chmm_dir: Path,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_aligned.empty or df_switch.empty:
        return pd.DataFrame(), pd.DataFrame()

    key_cols = [
        "session_id",
        "switch_index",
        "from_state",
        "to_state",
        "transition_type",
        "posterior_jump_l2",
    ]
    keep_cols = [col for col in key_cols if col in df_switch.columns]
    df_switch_labels = df_switch[keep_cols].drop_duplicates(["session_id", "switch_index"]).copy()
    df_switch_labels["transition_identity"] = df_switch_labels["transition_type"].astype(str)
    df_switch_labels["jump_magnitude_bin"] = _safe_quantile_bins(df_switch_labels["posterior_jump_l2"])

    df_runs = _compute_run_labels(chmm_dir)
    if not df_runs.empty:
        df_switch_labels = df_switch_labels.merge(df_runs, on=["session_id", "switch_index"], how="left")
    else:
        df_switch_labels["regime_stability_type"] = "unknown"
        df_switch_labels["from_run_len"] = np.nan
        df_switch_labels["to_run_len"] = np.nan

    therapist_score, client_score = _state_valence_scores(chmm_dir / "best_model.json")
    df_switch_labels["client_valence_direction"] = df_switch_labels.apply(
        lambda row: _valence_direction_label(row.get("from_state", np.nan), row.get("to_state", np.nan), client_score),
        axis=1,
    )
    df_switch_labels["therapist_valence_direction"] = df_switch_labels.apply(
        lambda row: _valence_direction_label(row.get("from_state", np.nan), row.get("to_state", np.nan), therapist_score),
        axis=1,
    )
    df_switch_labels["client_valence_from"] = df_switch_labels["from_state"].map(lambda s: _state_score_value(s, client_score))
    df_switch_labels["client_valence_to"] = df_switch_labels["to_state"].map(lambda s: _state_score_value(s, client_score))
    df_switch_labels["client_valence_delta"] = df_switch_labels.apply(
        lambda row: _state_transition_score_delta(row.get("from_state", np.nan), row.get("to_state", np.nan), client_score),
        axis=1,
    )
    df_switch_labels["therapist_valence_from"] = df_switch_labels["from_state"].map(
        lambda s: _state_score_value(s, therapist_score)
    )
    df_switch_labels["therapist_valence_to"] = df_switch_labels["to_state"].map(
        lambda s: _state_score_value(s, therapist_score)
    )
    df_switch_labels["therapist_valence_delta"] = df_switch_labels.apply(
        lambda row: _state_transition_score_delta(row.get("from_state", np.nan), row.get("to_state", np.nan), therapist_score),
        axis=1,
    )

    for raw_col, log_col in [("from_run_len", "log_from_run_len"), ("to_run_len", "log_to_run_len")]:
        if raw_col in df_switch_labels.columns:
            df_switch_labels[log_col] = np.log1p(df_switch_labels[raw_col])

    for col in [
        "posterior_jump_l2",
        "from_run_len",
        "to_run_len",
        "log_from_run_len",
        "log_to_run_len",
        "client_valence_from",
        "client_valence_to",
        "client_valence_delta",
        "therapist_valence_from",
        "therapist_valence_to",
        "therapist_valence_delta",
    ]:
        if col in df_switch_labels.columns:
            df_switch_labels[f"z_{col}"] = _zscore_series(df_switch_labels[col])

    df_main = df_aligned.merge(
        df_switch_labels[
            [
                "session_id",
                "switch_index",
                "from_state",
                "to_state",
                "transition_identity",
                "jump_magnitude_bin",
                "regime_stability_type",
                "client_valence_direction",
                "therapist_valence_direction",
                "from_run_len",
                "to_run_len",
                "log_from_run_len",
                "log_to_run_len",
                "client_valence_from",
                "client_valence_to",
                "client_valence_delta",
                "therapist_valence_from",
                "therapist_valence_to",
                "therapist_valence_delta",
                "z_posterior_jump_l2",
                "z_from_run_len",
                "z_to_run_len",
                "z_log_from_run_len",
                "z_log_to_run_len",
                "z_client_valence_from",
                "z_client_valence_to",
                "z_client_valence_delta",
                "z_therapist_valence_from",
                "z_therapist_valence_to",
                "z_therapist_valence_delta",
            ]
        ],
        on=["session_id", "switch_index"],
        how="left",
    )

    for base_col in ["from_state", "to_state"]:
        x_col = f"{base_col}_x"
        y_col = f"{base_col}_y"
        if x_col in df_main.columns or y_col in df_main.columns:
            left = df_main[x_col] if x_col in df_main.columns else pd.Series(np.nan, index=df_main.index)
            right = df_main[y_col] if y_col in df_main.columns else pd.Series(np.nan, index=df_main.index)
            df_main[base_col] = left.combine_first(right)
            drop_cols = [col for col in [x_col, y_col] if col in df_main.columns]
            df_main = df_main.drop(columns=drop_cols)

    for col in ["transition_identity", "jump_magnitude_bin", "regime_stability_type", "client_valence_direction", "therapist_valence_direction"]:
        if col in df_main.columns:
            df_main[col] = df_main[col].fillna("pseudo")

    df_main["phase_window_5"] = df_main["rel_time_sec"].map(_phase_window_label)
    df_main["abs_rel_time_sec"] = df_main["rel_time_sec"].abs()

    if "time_sec" in df_main.columns:
        max_by_session = df_main.groupby("session_id")["time_sec"].transform("max")
        df_main["norm_session_time"] = np.where(max_by_session > 0, df_main["time_sec"] / max_by_session, np.nan)
    else:
        df_main["norm_session_time"] = np.nan

    df_main = _add_baseline_centered_features(df_main, features)
    return df_main, df_switch_labels


def run_qc_overview(
    df_main: pd.DataFrame,
    df_switch_labels: pd.DataFrame,
    fig_dir: Path,
    tab_dir: Path,
) -> dict[str, pd.DataFrame]:
    if df_main.empty:
        print("No analysis table available yet.")
        return {"qc": pd.DataFrame(), "support": pd.DataFrame()}

    qc_rows = {
        "n_rows": len(df_main),
        "n_sessions": df_main["session_id"].nunique(),
        "n_real_switches": df_main.loc[df_main["anchor_kind"] == "real", ["session_id", "switch_index"]].drop_duplicates().shape[0],
        "n_pseudo_switches": df_main.loc[df_main["anchor_kind"] == "pseudo", ["session_id", "switch_index"]].drop_duplicates().shape[0],
        "n_real_rows": int((df_main["anchor_kind"] == "real").sum()),
        "n_pseudo_rows": int((df_main["anchor_kind"] == "pseudo").sum()),
    }
    df_qc = pd.DataFrame([qc_rows])
    display(df_qc)

    support = (
        df_main.groupby(["anchor_kind", "rel_time_sec"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
        .sort_values(["anchor_kind", "rel_time_sec"])
    )
    display(support.head(10))
    support.to_csv(tab_dir / "time_support_by_anchor_kind.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(
        data=df_main.drop_duplicates(["session_id", "switch_index", "anchor_kind"]),
        x="anchor_kind",
        ax=axes[0],
    )
    axes[0].set_title("Anchor count by type")

    sns.lineplot(data=support, x="rel_time_sec", y="n_rows", hue="anchor_kind", ax=axes[1])
    axes[1].axvline(0.0, color="red", linestyle="--", linewidth=1.0)
    axes[1].set_title("Observation support over relative time")

    if "transition_identity" in df_switch_labels.columns and not df_switch_labels.empty:
        top_trans = df_switch_labels["transition_identity"].value_counts().head(12)
        sns.barplot(x=top_trans.values, y=top_trans.index, color="steelblue", ax=axes[2])
        axes[2].set_title("Most common transition identities")
        axes[2].set_xlabel("n switches")
        axes[2].set_ylabel("transition identity")

    plt.tight_layout()
    plt.savefig(fig_dir / "qc_overview.svg", format="svg")
    plt.show()
    return {"qc": df_qc, "support": support}


def _full_rank_design(X: pd.DataFrame, tol: float = 1e-10) -> pd.DataFrame:
    X = X.copy()
    has_const = "const" in X.columns
    const_col = X[["const"]].copy() if has_const else None

    non_const_cols = [c for c in X.columns if c != "const"]
    if non_const_cols:
        Xn = X[non_const_cols]
        keep = Xn.columns[Xn.std(axis=0) > tol]
        Xn = Xn[keep]
        while Xn.shape[1] > 0 and np.linalg.matrix_rank(Xn.to_numpy()) < Xn.shape[1]:
            Xn = Xn.iloc[:, :-1]
    else:
        Xn = pd.DataFrame(index=X.index)

    Xout = pd.concat([const_col, Xn], axis=1) if has_const else Xn
    while Xout.shape[1] > 1 and np.linalg.matrix_rank(Xout.to_numpy()) < Xout.shape[1]:
        if "const" in Xout.columns and Xout.columns[-1] == "const":
            break
        Xout = Xout.iloc[:, :-1]
    return Xout


def summarize_timecourse(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    min_sessions: int = 8,
) -> pd.DataFrame:
    d = df[["session_id", "rel_time_sec", group_col, feature]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    keep = d.groupby(group_col)["session_id"].nunique()
    keep = keep[keep >= min_sessions].index
    d = d[d[group_col].isin(keep)].copy()
    if d.empty:
        return pd.DataFrame()

    by_session = (
        d.groupby([group_col, "session_id", "rel_time_sec"], as_index=False)[feature]
        .mean()
        .rename(columns={feature: "value"})
    )
    out = (
        by_session.groupby([group_col, "rel_time_sec"], as_index=False)["value"]
        .agg(mean="mean", sd="std", n="count")
        .sort_values([group_col, "rel_time_sec"])
    )
    out["sem"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out["lower"] = out["mean"] - 1.96 * out["sem"]
    out["upper"] = out["mean"] + 1.96 * out["sem"]
    out["feature"] = feature
    return out


def _supported_moderator_levels(
    df: pd.DataFrame,
    group_col: str,
    feature: str,
    min_sessions: int = 8,
) -> list[str]:
    d = df[["session_id", group_col, feature]].dropna().copy()
    if d.empty:
        return []
    keep = d.groupby(group_col)["session_id"].nunique()
    keep = keep[keep >= min_sessions].index.tolist()
    return [str(x) for x in keep]


def plot_timecourse_grid(
    df: pd.DataFrame,
    features: list[str],
    group_col: str,
    out_name: str,
    fig_dir: Path,
    tab_dir: Path,
    min_sessions: int = 8,
    palette: dict | None = None,
) -> pd.DataFrame:
    summary_rows = []
    for feature in features:
        s = summarize_timecourse(df, feature, group_col, min_sessions=min_sessions)
        if not s.empty:
            summary_rows.append(s)

    if not summary_rows:
        print(f"No summary rows available for group column {group_col}")
        return pd.DataFrame()

    df_sum = pd.concat(summary_rows, ignore_index=True)
    df_sum.to_csv(tab_dir / f"{out_name}_summary.csv", index=False)

    n_feat = len(features)
    fig, axes = plt.subplots(n_feat, 1, figsize=(11, max(4, 3.4 * n_feat)), sharex=True)
    if n_feat == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        sub = df_sum[df_sum["feature"] == feature].copy()
        for group in sub[group_col].dropna().unique():
            dg = sub[sub[group_col] == group].sort_values("rel_time_sec")
            color = palette.get(group) if palette is not None else None
            ax.plot(dg["rel_time_sec"], dg["mean"], linewidth=2, label=str(group), color=color)
            ax.fill_between(dg["rel_time_sec"], dg["lower"], dg["upper"], alpha=0.18, color=color)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(f"{feature} by {group_col}")
        ax.set_ylabel(feature)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time relative to switch (sec)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{out_name}.svg", format="svg")
    plt.show()
    return df_sum


def run_moderator_timecourse_inference(
    df_real: pd.DataFrame,
    features: list[str],
    *,
    group_col: str,
    moderator_family: str,
    out_name: str,
    fig_dir: Path,
    tab_dir: Path,
    min_sessions: int = 8,
    palette: dict | None = None,
) -> dict[str, pd.DataFrame]:
    if df_real.empty:
        return {
            "summary": pd.DataFrame(),
            "model_summary": pd.DataFrame(),
            "predicted_curves": pd.DataFrame(),
            "retained_levels": pd.DataFrame(),
        }

    df_summary = plot_timecourse_grid(
        df_real,
        features,
        group_col=group_col,
        out_name=out_name,
        fig_dir=fig_dir,
        tab_dir=tab_dir,
        min_sessions=min_sessions,
        palette=palette,
    )

    model_rows = []
    pred_rows = []
    retained_rows = []
    n_feat = len(features)
    fig, axes = plt.subplots(n_feat, 1, figsize=(11, max(4, 3.4 * n_feat)), sharex=True)
    if n_feat == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        allowed_groups = _supported_moderator_levels(df_real, group_col, feature, min_sessions=min_sessions)
        if len(allowed_groups) < 2:
            ax.axis("off")
            continue

        d, fit_bundle = fit_real_only_moderator_spline_interaction(
            df_real,
            feature,
            group_col,
            allowed_groups=allowed_groups,
            spline_df_candidates=(5, 4, 3),
        )
        if fit_bundle is None:
            ax.axis("off")
            continue

        fit = fit_bundle["fit"]
        reduced_fit = fit_bundle["reduced_fit"]
        retained_levels = fit_bundle["group_levels"]
        retained_rows.extend(
            {
                "moderator_family": moderator_family,
                "group_col": group_col,
                "feature": feature,
                "group_level": level,
            }
            for level in retained_levels
        )
        model_row = _likelihood_ratio_row(
            feature=feature,
            full_fit=fit,
            reduced_fit=reduced_fit,
            model_type=fit_bundle["model_type"],
            spline_df=fit_bundle["spline_df"],
            n_obs=len(d),
            n_sessions=d["session_id"].nunique(),
        )
        model_row["moderator_family"] = moderator_family
        model_row["group_col"] = group_col
        model_row["n_levels"] = int(len(retained_levels))
        model_row["retained_levels"] = "|".join(retained_levels)
        model_rows.append(model_row)

        pred = predict_moderator_curves(fit_bundle, d["rel_time_sec"].min(), d["rel_time_sec"].max(), n=200)
        pred["feature"] = feature
        pred["moderator_family"] = moderator_family
        pred["group_col"] = group_col
        pred_rows.append(pred)

        avg = summarize_timecourse(df_real[df_real[group_col].astype(str).isin(retained_levels)], feature, group_col, min_sessions=min_sessions)
        for group in retained_levels:
            dg = pred[pred["group"] == group].sort_values("rel_time_sec")
            color = palette.get(group) if palette is not None else None
            ax.plot(dg["rel_time_sec"], dg["pred"], linewidth=2, color=color, label=str(group))
            ds = avg[(avg["feature"] == feature) & (avg[group_col].astype(str) == group)]
            if not ds.empty:
                ax.scatter(ds["rel_time_sec"], ds["mean"], s=10, alpha=0.35, color=color)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(f"{feature} by {group_col}: fitted moderator curves")
        ax.set_ylabel(feature)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time relative to switch (sec)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{out_name}_model_curves.svg", format="svg")
    plt.show()

    df_model_summary = pd.DataFrame(model_rows)
    df_predicted_curves = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    df_retained_levels = pd.DataFrame(retained_rows)

    if not df_model_summary.empty:
        df_model_summary.to_csv(tab_dir / f"{out_name}_model_summary.csv", index=False)
    if not df_predicted_curves.empty:
        df_predicted_curves.to_csv(tab_dir / f"{out_name}_predicted_curves.csv", index=False)
    if not df_retained_levels.empty:
        df_retained_levels.to_csv(tab_dir / f"{out_name}_retained_levels.csv", index=False)

    return {
        "summary": df_summary,
        "model_summary": df_model_summary,
        "predicted_curves": df_predicted_curves,
        "retained_levels": df_retained_levels,
    }


def run_continuous_moderator_timecourse_inference(
    df_real: pd.DataFrame,
    features: list[str],
    moderator_specs: list[dict[str, str]],
    *,
    fig_dir: Path,
    tab_dir: Path,
    out_name: str = "continuous_moderator_timecourses_real_only",
) -> dict[str, pd.DataFrame]:
    if df_real.empty or not moderator_specs:
        empty = pd.DataFrame()
        return {
            "model_summary": empty,
            "predicted_curves": empty,
            "correlations": empty,
        }

    corr_cols = [spec["moderator_col"] for spec in moderator_specs if spec["moderator_col"] in df_real.columns]
    corr_base = (
        df_real[["session_id", "switch_index", *corr_cols]]
        .drop_duplicates(["session_id", "switch_index"])
        .copy()
        if corr_cols
        else pd.DataFrame()
    )
    if not corr_base.empty:
        corr_matrix = corr_base[corr_cols].corr(numeric_only=True)
        corr_rows = []
        for row_name in corr_matrix.index:
            for col_name in corr_matrix.columns:
                corr_rows.append(
                    {
                        "moderator_row": row_name,
                        "moderator_col": col_name,
                        "correlation": float(corr_matrix.loc[row_name, col_name]),
                    }
                )
        df_correlations = pd.DataFrame(corr_rows)
    else:
        df_correlations = pd.DataFrame()

    n_rows = len(moderator_specs)
    n_cols = len(features)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * max(1, n_cols), 3.5 * max(1, n_rows)),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    model_rows = []
    pred_rows = []
    palette = {
        "Low (-1 SD)": "#33658A",
        "Mean": "#758E4F",
        "High (+1 SD)": "#BC4B51",
    }

    for row_idx, spec in enumerate(moderator_specs):
        moderator_col = spec["moderator_col"]
        moderator_name = spec.get("moderator_name", moderator_col)
        for col_idx, feature in enumerate(features):
            ax = axes[row_idx, col_idx]
            d, fit_bundle = fit_real_only_continuous_moderator_spline_interaction(
                df_real,
                feature,
                moderator_col,
                spline_df_candidates=(5, 4, 3),
            )
            if fit_bundle is None or d is None:
                ax.axis("off")
                continue

            model_row = _likelihood_ratio_row(
                feature=feature,
                full_fit=fit_bundle["fit"],
                reduced_fit=fit_bundle["reduced_fit"],
                model_type=fit_bundle["model_type"],
                spline_df=fit_bundle["spline_df"],
                n_obs=len(d),
                n_sessions=d["session_id"].nunique(),
            )
            model_row["moderator_name"] = moderator_name
            model_row["moderator_col"] = moderator_col
            model_row["moderator_type"] = "continuous"
            model_rows.append(model_row)

            pred = predict_continuous_moderator_curves(
                fit_bundle,
                d["rel_time_sec"].min(),
                d["rel_time_sec"].max(),
                moderator_values=(-1.0, 0.0, 1.0),
                moderator_labels=("Low (-1 SD)", "Mean", "High (+1 SD)"),
                n=200,
            )
            pred["feature"] = feature
            pred["moderator_name"] = moderator_name
            pred["moderator_col"] = moderator_col
            pred_rows.append(pred)

            for label, dg in pred.groupby("moderator_label"):
                dg = dg.sort_values("rel_time_sec")
                ax.plot(
                    dg["rel_time_sec"],
                    dg["pred"],
                    linewidth=2,
                    color=palette.get(label),
                    label=label,
                )
            ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
            ax.set_title(f"{moderator_name}: {feature}")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time relative to switch (sec)")
            if col_idx == 0:
                ax.set_ylabel(feature)
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / f"{out_name}_model_curves.svg", format="svg")
    plt.show()

    df_model_summary = pd.DataFrame(model_rows)
    df_predicted_curves = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    if not df_model_summary.empty:
        df_model_summary.to_csv(tab_dir / f"{out_name}_model_summary.csv", index=False)
        df_model_summary.to_csv(tab_dir / "continuous_moderator_timecourse_omnibus_model_summary.csv", index=False)
    if not df_predicted_curves.empty:
        df_predicted_curves.to_csv(tab_dir / f"{out_name}_predicted_curves.csv", index=False)
        df_predicted_curves.to_csv(tab_dir / "continuous_moderator_timecourse_predicted_curves.csv", index=False)
    if not df_correlations.empty:
        df_correlations.to_csv(tab_dir / "continuous_moderator_correlation_matrix_long.csv", index=False)

    return {
        "model_summary": df_model_summary,
        "predicted_curves": df_predicted_curves,
        "correlations": df_correlations,
    }


def fit_mixed_spline_interaction(
    df: pd.DataFrame,
    feature: str,
    spline_df_candidates=(5, 4, 3),
):
    d = df[["session_id", "rel_time_sec", "anchor_kind", feature]].dropna().copy()
    if len(d) < 60 or d["session_id"].nunique() < 4 or d["anchor_kind"].nunique() < 2:
        return None, None

    d["is_real"] = (d["anchor_kind"] == "real").astype(float)
    last_err = None
    for spline_df in spline_df_candidates:
        try:
            X_basis = dmatrix(
                f"0 + bs(rel_time_sec, df={spline_df}, degree=3, include_intercept=False)",
                d,
                return_type="dataframe",
            )
            design_info = X_basis.design_info

            X_reduced = X_basis.copy()
            X_reduced["is_real"] = d["is_real"].to_numpy()
            X_reduced = sm.add_constant(X_reduced, has_constant="add")
            X_reduced = _full_rank_design(X_reduced)
            if "const" not in X_reduced.columns:
                X_reduced = sm.add_constant(X_reduced, has_constant="add")

            X_full = X_reduced.copy()
            for col in X_basis.columns:
                X_full[f"{col}:is_real"] = X_basis[col].to_numpy() * d["is_real"].to_numpy()
            X_full = _full_rank_design(X_full)
            if "const" not in X_full.columns:
                X_full = sm.add_constant(X_full, has_constant="add")

            if X_reduced.shape[1] < 2 or X_full.shape[1] <= X_reduced.shape[1]:
                continue

            try:
                reduced_fit = MixedLM(endog=d[feature], exog=X_reduced, groups=d["session_id"]).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=300,
                )
                full_fit = MixedLM(endog=d[feature], exog=X_full, groups=d["session_id"]).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=300,
                )
                return d, {
                    "fit": full_fit,
                    "reduced_fit": reduced_fit,
                    "model_type": "MixedLM_time_x_anchor_lr",
                    "spline_df": int(spline_df),
                    "X_cols": list(X_full.columns),
                    "design_info": design_info,
                }
            except Exception as exc:
                last_err = exc

            try:
                reduced_fit = sm.OLS(d[feature], X_reduced).fit()
                full_fit = sm.OLS(d[feature], X_full).fit()
                print(f"Fallback to OLS interaction spline for {feature} (spline_df={spline_df}) due to MixedLM failure.")
                return d, {
                    "fit": full_fit,
                    "reduced_fit": reduced_fit,
                    "model_type": "OLS_time_x_anchor_lr_fallback",
                    "spline_df": int(spline_df),
                    "X_cols": list(X_full.columns),
                    "design_info": design_info,
                }
            except Exception as exc:
                last_err = exc
        except Exception as exc:
            last_err = exc

    print(f"Skipping {feature}: unable to fit time x anchor model ({type(last_err).__name__}: {last_err})")
    return None, None


def fit_real_only_moderator_spline_interaction(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    *,
    allowed_groups: list[str] | None = None,
    spline_df_candidates=(5, 4, 3),
):
    use_cols = ["session_id", "rel_time_sec", group_col, feature]
    d = df[[col for col in use_cols if col in df.columns]].dropna().copy()
    if d.empty or group_col not in d.columns:
        return None, None

    if allowed_groups is None:
        allowed_groups = sorted(d[group_col].dropna().astype(str).unique().tolist())
    if len(allowed_groups) < 2:
        return None, None

    d[group_col] = d[group_col].astype(str)
    d = d[d[group_col].isin(allowed_groups)].copy()
    if len(d) < 60 or d["session_id"].nunique() < 4 or d[group_col].nunique() < 2:
        return None, None

    group_levels = [g for g in allowed_groups if g in set(d[group_col].unique())]
    if len(group_levels) < 2:
        return None, None

    d[group_col] = pd.Categorical(d[group_col], categories=group_levels, ordered=True)
    last_err = None
    for spline_df in spline_df_candidates:
        try:
            X_basis = dmatrix(
                f"0 + bs(rel_time_sec, df={spline_df}, degree=3, include_intercept=False)",
                d,
                return_type="dataframe",
            )
            design_info = X_basis.design_info

            X_group = dmatrix(
                f"0 + C({group_col}, levels={group_levels!r})",
                d,
                return_type="dataframe",
            )
            reference_col = X_group.columns[0]
            group_to_dummy_col = {group_levels[i]: col for i, col in enumerate(X_group.columns)}
            X_group_nonref = X_group.drop(columns=[reference_col], errors="ignore")

            X_reduced = pd.concat([X_basis, X_group_nonref], axis=1)
            X_reduced = sm.add_constant(X_reduced, has_constant="add")
            X_reduced = _full_rank_design(X_reduced)
            if "const" not in X_reduced.columns:
                X_reduced = sm.add_constant(X_reduced, has_constant="add")

            X_full = X_reduced.copy()
            interaction_cols = []
            for gcol in X_group_nonref.columns:
                for bcol in X_basis.columns:
                    name = f"{bcol}:{gcol}"
                    X_full[name] = X_basis[bcol].to_numpy() * X_group_nonref[gcol].to_numpy()
                    interaction_cols.append(name)
            X_full = _full_rank_design(X_full)
            if "const" not in X_full.columns:
                X_full = sm.add_constant(X_full, has_constant="add")

            interaction_cols = [col for col in interaction_cols if col in X_full.columns]
            if X_reduced.shape[1] < 2 or X_full.shape[1] <= X_reduced.shape[1] or not interaction_cols:
                continue

            try:
                reduced_fit = MixedLM(endog=d[feature], exog=X_reduced, groups=d["session_id"]).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=300,
                )
                full_fit = MixedLM(endog=d[feature], exog=X_full, groups=d["session_id"]).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=300,
                )
                return d, {
                    "fit": full_fit,
                    "reduced_fit": reduced_fit,
                    "model_type": "MixedLM_time_x_moderator_lr",
                    "spline_df": int(spline_df),
                    "group_col": group_col,
                    "group_levels": group_levels,
                    "reference_level": group_levels[0],
                    "group_to_dummy_col": group_to_dummy_col,
                    "X_cols": list(X_full.columns),
                    "basis_cols": list(X_basis.columns),
                    "group_dummy_cols": list(X_group_nonref.columns),
                    "design_info": design_info,
                }
            except Exception as exc:
                last_err = exc

            try:
                reduced_fit = sm.OLS(d[feature], X_reduced).fit()
                full_fit = sm.OLS(d[feature], X_full).fit()
                print(
                    f"Fallback to OLS moderator spline for {feature}, {group_col} "
                    f"(spline_df={spline_df}) due to MixedLM failure."
                )
                return d, {
                    "fit": full_fit,
                    "reduced_fit": reduced_fit,
                    "model_type": "OLS_time_x_moderator_lr_fallback",
                    "spline_df": int(spline_df),
                    "group_col": group_col,
                    "group_levels": group_levels,
                    "reference_level": group_levels[0],
                    "group_to_dummy_col": group_to_dummy_col,
                    "X_cols": list(X_full.columns),
                    "basis_cols": list(X_basis.columns),
                    "group_dummy_cols": list(X_group_nonref.columns),
                    "design_info": design_info,
                }
            except Exception as exc:
                last_err = exc
        except Exception as exc:
            last_err = exc

    print(
        f"Skipping {feature} moderator spline for {group_col}: "
        f"unable to fit time x moderator model ({type(last_err).__name__}: {last_err})"
    )
    return None, None


def predict_anchor_curves(fit_bundle, t_min: float, t_max: float, n: int = 200) -> pd.DataFrame:
    fit = fit_bundle["fit"]
    x_cols = fit_bundle["X_cols"]
    design_info = fit_bundle["design_info"]
    rows = []

    for anchor_kind, is_real in [("pseudo", 0.0), ("real", 1.0)]:
        grid = pd.DataFrame({"rel_time_sec": np.linspace(t_min, t_max, n)})
        Xg_basis = build_design_matrices([design_info], grid, return_type="dataframe")[0]
        Xg_main = Xg_basis.copy()
        Xg_main["is_real"] = is_real
        for col in Xg_basis.columns:
            Xg_main[f"{col}:is_real"] = Xg_basis[col].to_numpy() * is_real
        Xg = sm.add_constant(Xg_main, has_constant="add")
        for col in x_cols:
            if col not in Xg.columns:
                Xg[col] = 0.0
        Xg = Xg[x_cols]
        if "const" not in Xg.columns:
            Xg = sm.add_constant(Xg, has_constant="add")
        grid["pred"] = fit.predict(exog=Xg)
        grid["anchor_kind"] = anchor_kind
        rows.append(grid)

    return pd.concat(rows, ignore_index=True)


def predict_moderator_curves(fit_bundle, t_min: float, t_max: float, n: int = 200) -> pd.DataFrame:
    fit = fit_bundle["fit"]
    x_cols = fit_bundle["X_cols"]
    design_info = fit_bundle["design_info"]
    basis_cols = fit_bundle["basis_cols"]
    group_dummy_cols = fit_bundle["group_dummy_cols"]
    reference_level = fit_bundle["reference_level"]
    group_to_dummy_col = fit_bundle["group_to_dummy_col"]
    rows = []

    for group in fit_bundle["group_levels"]:
        grid = pd.DataFrame({"rel_time_sec": np.linspace(t_min, t_max, n)})
        Xg_basis = build_design_matrices([design_info], grid, return_type="dataframe")[0]
        Xg_main = Xg_basis.copy()
        for gcol in group_dummy_cols:
            is_active = float(group_to_dummy_col.get(group) == gcol)
            Xg_main[gcol] = is_active
            for bcol in basis_cols:
                Xg_main[f"{bcol}:{gcol}"] = Xg_basis[bcol].to_numpy() * is_active
        Xg = sm.add_constant(Xg_main, has_constant="add")
        for col in x_cols:
            if col not in Xg.columns:
                Xg[col] = 0.0
        Xg = Xg[x_cols]
        if "const" not in Xg.columns:
            Xg = sm.add_constant(Xg, has_constant="add")
        grid["pred"] = fit.predict(exog=Xg)
        grid["group"] = group
        grid["is_reference"] = bool(group == reference_level)
        rows.append(grid)

    return pd.concat(rows, ignore_index=True)


def fit_real_only_continuous_moderator_spline_interaction(
    df: pd.DataFrame,
    feature: str,
    moderator_col: str,
    *,
    spline_df_candidates=(5, 4, 3),
):
    use_cols = ["session_id", "rel_time_sec", moderator_col, feature]
    d = df[[col for col in use_cols if col in df.columns]].dropna().copy()
    if d.empty or moderator_col not in d.columns:
        return None, None

    d[moderator_col] = pd.to_numeric(d[moderator_col], errors="coerce")
    d = d.dropna(subset=[moderator_col])
    if len(d) < 60 or d["session_id"].nunique() < 4 or d[moderator_col].nunique() < 10:
        return None, None

    last_err = None
    for spline_df in spline_df_candidates:
        try:
            X_basis = dmatrix(
                f"0 + bs(rel_time_sec, df={spline_df}, degree=3, include_intercept=False)",
                d,
                return_type="dataframe",
            )
            design_info = X_basis.design_info
            moderator_values = d[moderator_col].astype(float).to_numpy()

            X_reduced = X_basis.copy()
            X_reduced[moderator_col] = moderator_values
            X_reduced = sm.add_constant(X_reduced, has_constant="add")
            X_reduced = _full_rank_design(X_reduced)
            if "const" not in X_reduced.columns:
                X_reduced = sm.add_constant(X_reduced, has_constant="add")

            X_full = X_reduced.copy()
            interaction_cols = []
            for bcol in X_basis.columns:
                name = f"{bcol}:{moderator_col}"
                X_full[name] = X_basis[bcol].to_numpy() * moderator_values
                interaction_cols.append(name)
            X_full = _full_rank_design(X_full)
            if "const" not in X_full.columns:
                X_full = sm.add_constant(X_full, has_constant="add")

            interaction_cols = [col for col in interaction_cols if col in X_full.columns]
            if X_reduced.shape[1] < 2 or X_full.shape[1] <= X_reduced.shape[1] or not interaction_cols:
                continue

            try:
                reduced_fit = MixedLM(endog=d[feature], exog=X_reduced, groups=d["session_id"]).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=300,
                )
                full_fit = MixedLM(endog=d[feature], exog=X_full, groups=d["session_id"]).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=300,
                )
                return d, {
                    "fit": full_fit,
                    "reduced_fit": reduced_fit,
                    "model_type": "MixedLM_time_x_continuous_moderator_lr",
                    "spline_df": int(spline_df),
                    "moderator_col": moderator_col,
                    "X_cols": list(X_full.columns),
                    "basis_cols": list(X_basis.columns),
                    "design_info": design_info,
                }
            except Exception as exc:
                last_err = exc

            try:
                reduced_fit = sm.OLS(d[feature], X_reduced).fit()
                full_fit = sm.OLS(d[feature], X_full).fit()
                print(
                    f"Fallback to OLS continuous moderator spline for {feature}, {moderator_col} "
                    f"(spline_df={spline_df}) due to MixedLM failure."
                )
                return d, {
                    "fit": full_fit,
                    "reduced_fit": reduced_fit,
                    "model_type": "OLS_time_x_continuous_moderator_lr_fallback",
                    "spline_df": int(spline_df),
                    "moderator_col": moderator_col,
                    "X_cols": list(X_full.columns),
                    "basis_cols": list(X_basis.columns),
                    "design_info": design_info,
                }
            except Exception as exc:
                last_err = exc
        except Exception as exc:
            last_err = exc

    print(
        f"Skipping {feature} continuous moderator spline for {moderator_col}: "
        f"unable to fit time x moderator model ({type(last_err).__name__}: {last_err})"
    )
    return None, None


def predict_continuous_moderator_curves(
    fit_bundle,
    t_min: float,
    t_max: float,
    moderator_values: Iterable[float] = (-1.0, 0.0, 1.0),
    moderator_labels: Iterable[str] = ("Low (-1 SD)", "Mean", "High (+1 SD)"),
    n: int = 200,
) -> pd.DataFrame:
    fit = fit_bundle["fit"]
    x_cols = fit_bundle["X_cols"]
    design_info = fit_bundle["design_info"]
    basis_cols = fit_bundle["basis_cols"]
    moderator_col = fit_bundle["moderator_col"]
    rows = []

    for moderator_value, moderator_label in zip(moderator_values, moderator_labels):
        grid = pd.DataFrame({"rel_time_sec": np.linspace(t_min, t_max, n)})
        Xg_basis = build_design_matrices([design_info], grid, return_type="dataframe")[0]
        Xg = Xg_basis.copy()
        Xg[moderator_col] = float(moderator_value)
        for bcol in basis_cols:
            Xg[f"{bcol}:{moderator_col}"] = Xg_basis[bcol].to_numpy() * float(moderator_value)
        Xg = sm.add_constant(Xg, has_constant="add")
        for col in x_cols:
            if col not in Xg.columns:
                Xg[col] = 0.0
        Xg = Xg[x_cols]
        if "const" not in Xg.columns:
            Xg = sm.add_constant(Xg, has_constant="add")
        grid["pred"] = fit.predict(exog=Xg)
        grid["moderator_value"] = float(moderator_value)
        grid["moderator_label"] = str(moderator_label)
        rows.append(grid)

    return pd.concat(rows, ignore_index=True)


def make_difference_curve(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    wide = pred_df.pivot_table(index="rel_time_sec", columns="anchor_kind", values="pred").reset_index()
    if "real" not in wide.columns or "pseudo" not in wide.columns:
        return pd.DataFrame()
    wide["real_minus_pseudo"] = wide["real"] - wide["pseudo"]
    return wide.sort_values("rel_time_sec")


def summarize_window_effects(df: pd.DataFrame, feature: str, window_col: str = "phase_window_5") -> pd.DataFrame:
    d = df[["session_id", "switch_index", "anchor_kind", window_col, feature]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    sw = (
        d.groupby(["session_id", "switch_index", "anchor_kind", window_col], as_index=False)[feature]
        .mean()
        .rename(columns={feature: "value"})
    )
    out = sw.groupby(["anchor_kind", window_col], as_index=False)["value"].agg(mean="mean", sd="std", n="count")
    out["sem"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out["lower"] = out["mean"] - 1.96 * out["sem"]
    out["upper"] = out["mean"] + 1.96 * out["sem"]
    out["feature"] = feature
    return out


def run_primary_anchor_timecourse(
    df_main: pd.DataFrame,
    topo_features: list[str],
    fig_dir: Path,
    tab_dir: Path,
) -> pd.DataFrame:
    if df_main.empty:
        print("No analysis table available yet.")
        return pd.DataFrame()
    return plot_timecourse_grid(
        df_main,
        topo_features,
        group_col="anchor_kind",
        out_name="primary_real_vs_pseudo_timecourses",
        min_sessions=8,
        palette={"real": "tab:blue", "pseudo": "tab:orange"},
        fig_dir=fig_dir,
        tab_dir=tab_dir,
    )


def run_primary_inferential_analysis(
    df_main: pd.DataFrame,
    topo_features: list[str],
    fig_dir: Path,
    tab_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_main.empty:
        print("No analysis table available yet.")
        return pd.DataFrame(), pd.DataFrame()

    model_rows = []
    diff_rows = []
    n_feat = len(topo_features)
    fig, axes = plt.subplots(n_feat, 2, figsize=(14, max(4, 3.6 * n_feat)), sharex="col")
    if n_feat == 1:
        axes = np.array([axes])

    for i, feature in enumerate(topo_features):
        d, fit_bundle = fit_mixed_spline_interaction(df_main, feature, spline_df_candidates=(5, 4, 3))
        if fit_bundle is None:
            continue

        fit = fit_bundle["fit"]
        reduced_fit = fit_bundle["reduced_fit"]
        model_rows.append(
            _likelihood_ratio_row(
                feature=feature,
                full_fit=fit,
                reduced_fit=reduced_fit,
                model_type=fit_bundle["model_type"],
                spline_df=fit_bundle["spline_df"],
                n_obs=len(d),
                n_sessions=d["session_id"].nunique(),
            )
        )

        pred = predict_anchor_curves(fit_bundle, d["rel_time_sec"].min(), d["rel_time_sec"].max(), n=200)
        pred["feature"] = feature
        pred.to_csv(tab_dir / f"predicted_curves_{feature}.csv", index=False)

        avg = summarize_timecourse(df_main, feature, "anchor_kind", min_sessions=8)
        left_ax = axes[i, 0]
        for kind, color in [("real", "tab:blue"), ("pseudo", "tab:orange")]:
            dg = pred[pred["anchor_kind"] == kind]
            left_ax.plot(dg["rel_time_sec"], dg["pred"], color=color, linewidth=2, label=f"{kind} model")
            ds = avg[(avg["feature"] == feature) & (avg["anchor_kind"] == kind)]
            if not ds.empty:
                left_ax.scatter(ds["rel_time_sec"], ds["mean"], s=10, alpha=0.35, color=color)
        left_ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
        left_ax.set_title(f"{feature}: fitted curves")
        left_ax.set_ylabel(feature)
        left_ax.legend(loc="best", fontsize=8)

        diff = make_difference_curve(pred)
        if not diff.empty:
            diff["feature"] = feature
            diff_rows.append(diff)
            right_ax = axes[i, 1]
            right_ax.plot(diff["rel_time_sec"], diff["real_minus_pseudo"], color="black", linewidth=2)
            right_ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
            right_ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
            right_ax.set_title(f"{feature}: real minus pseudo")
            right_ax.set_ylabel("difference")

    axes[-1, 0].set_xlabel("Time relative to switch (sec)")
    axes[-1, 1].set_xlabel("Time relative to switch (sec)")
    plt.tight_layout()
    plt.savefig(fig_dir / "primary_timecourse_models_and_difference_curves.svg", format="svg")
    plt.show()

    df_primary_model = pd.DataFrame(model_rows)
    df_primary_diff = pd.concat(diff_rows, ignore_index=True) if diff_rows else pd.DataFrame()
    display(df_primary_model)
    if not df_primary_model.empty:
        df_primary_model.to_csv(tab_dir / "primary_time_x_anchor_model_summary.csv", index=False)
    if not df_primary_diff.empty:
        df_primary_diff.to_csv(tab_dir / "primary_real_minus_pseudo_difference_curves.csv", index=False)
    return df_primary_model, df_primary_diff


def run_session_level_robustness(
    df_main: pd.DataFrame,
    df_growth: pd.DataFrame,
    fig_dir: Path,
    tab_dir: Path,
    feature: str = "total_persistence_h1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_main.empty or df_growth.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    session_rows = []

    growth_sub = df_growth[df_growth["feature"] == feature].copy()
    if not growth_sub.empty:
        slope_means = (
            growth_sub.groupby(["session_id", "window", "anchor_kind"], as_index=False)["slope"]
            .mean()
            .rename(columns={"slope": "mean_value"})
        )
        slope_wide = slope_means.pivot_table(
            index=["session_id", "window"],
            columns="anchor_kind",
            values="mean_value",
        ).reset_index()
        if {"real", "pseudo"}.issubset(slope_wide.columns):
            slope_wide["difference"] = slope_wide["real"] - slope_wide["pseudo"]
            metric_map = {
                "pre_switch_growth": "session_mean_real_minus_pseudo_pre_switch_slope",
                "post_switch_growth": "session_mean_real_minus_pseudo_post_switch_slope",
            }
            slope_wide["metric"] = slope_wide["window"].map(metric_map)
            slope_wide = slope_wide.dropna(subset=["metric", "difference"]).copy()
            session_rows.append(slope_wide[["session_id", "metric", "difference"]])

    phase_sub = df_main[
        ["session_id", "switch_index", "anchor_kind", "phase", feature]
    ].dropna(subset=[feature, "phase"]).copy()
    if not phase_sub.empty:
        phase_switch = (
            phase_sub.groupby(["session_id", "switch_index", "anchor_kind", "phase"], as_index=False)[feature]
            .mean()
        )
        phase_wide = phase_switch.pivot_table(
            index=["session_id", "switch_index", "anchor_kind"],
            columns="phase",
            values=feature,
        ).reset_index()
        if {"before", "after"}.issubset(phase_wide.columns):
            phase_wide["arc_magnitude"] = (phase_wide["after"] - phase_wide["before"]).abs()
            arc_means = (
                phase_wide.groupby(["session_id", "anchor_kind"], as_index=False)["arc_magnitude"]
                .mean()
                .pivot_table(index="session_id", columns="anchor_kind", values="arc_magnitude")
                .reset_index()
            )
            if {"real", "pseudo"}.issubset(arc_means.columns):
                arc_means["difference"] = arc_means["real"] - arc_means["pseudo"]
                arc_means["metric"] = "session_mean_real_minus_pseudo_arc_magnitude"
                session_rows.append(arc_means[["session_id", "metric", "difference"]])

    time_sub = df_main[["session_id", "anchor_kind", "rel_time_sec", feature]].dropna().copy()
    if not time_sub.empty:
        session_curve = (
            time_sub.groupby(["session_id", "anchor_kind", "rel_time_sec"], as_index=False)[feature]
            .mean()
        )
        curve_wide = session_curve.pivot_table(
            index=["session_id", "rel_time_sec"],
            columns="anchor_kind",
            values=feature,
        ).reset_index()
        if {"real", "pseudo"}.issubset(curve_wide.columns):
            curve_wide["difference"] = curve_wide["real"] - curve_wide["pseudo"]
            peak = (
                curve_wide.groupby("session_id", as_index=False)["difference"]
                .max()
                .rename(columns={"difference": "difference"})
            )
            peak["metric"] = "session_peak_real_minus_pseudo_total_persistence"
            session_rows.append(peak[["session_id", "metric", "difference"]])

    df_session_robustness = pd.concat(session_rows, ignore_index=True) if session_rows else pd.DataFrame()
    if not df_session_robustness.empty:
        for metric, dmetric in df_session_robustness.groupby("metric"):
            vals = dmetric["difference"].dropna()
            if len(vals) < 8:
                continue
            try:
                stat, p_value = wilcoxon(vals)
            except ValueError:
                stat, p_value = np.nan, np.nan
            summary_rows.append(
                {
                    "metric": metric,
                    "n_sessions": int(vals.shape[0]),
                    "mean_difference": float(vals.mean()),
                    "median_difference": float(vals.median()),
                    "wilcoxon_stat": float(stat) if pd.notna(stat) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                }
            )

    df_session_robustness_summary = pd.DataFrame(summary_rows)
    display(df_session_robustness_summary)
    if not df_session_robustness.empty:
        df_session_robustness.to_csv(tab_dir / "session_level_robustness_values.csv", index=False)
    if not df_session_robustness_summary.empty:
        df_session_robustness_summary.to_csv(tab_dir / "session_level_robustness_wilcoxon.csv", index=False)

        plot_df = df_session_robustness.copy()
        metric_order = df_session_robustness_summary["metric"].tolist()
        plt.figure(figsize=(9, 4.8))
        sns.boxplot(data=plot_df, x="difference", y="metric", order=metric_order, color="lightgray")
        sns.stripplot(data=plot_df, x="difference", y="metric", order=metric_order, color="black", size=2.5, alpha=0.35)
        plt.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
        plt.xlabel("Session-level real minus pseudo difference")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(fig_dir / "session_level_robustness_summary.svg", format="svg")
        plt.show()

    return df_session_robustness, df_session_robustness_summary


def run_posterior_control_analysis(
    df_main: pd.DataFrame,
    df_growth: pd.DataFrame,
    fig_dir: Path,
    tab_dir: Path,
    feature: str = "total_persistence_h1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_main.empty:
        return pd.DataFrame(), pd.DataFrame()

    required_cols = ["posterior_entropy", "local_speed_l2", feature]
    if any(col not in df_main.columns for col in required_cols):
        print("Posterior control covariates are unavailable in df_main.")
        return pd.DataFrame(), pd.DataFrame()

    switch_cov = (
        df_main[df_main["phase"] == "during"][
            ["session_id", "switch_index", "anchor_kind", feature, "posterior_entropy", "local_speed_l2"]
        ]
        .dropna()
        .groupby(["session_id", "switch_index", "anchor_kind"], as_index=False)
        .mean()
    )
    if switch_cov.empty:
        return pd.DataFrame(), pd.DataFrame()

    switch_cov["is_real"] = (switch_cov["anchor_kind"] == "real").astype(float)
    switch_cov["z_total_persistence_h1"] = _zscore_series(switch_cov[feature])
    switch_cov["z_posterior_entropy"] = _zscore_series(switch_cov["posterior_entropy"])
    switch_cov["z_local_speed_l2"] = _zscore_series(switch_cov["local_speed_l2"])

    results_rows = []

    discrim = switch_cov[
        ["session_id", "is_real", "z_total_persistence_h1", "z_posterior_entropy", "z_local_speed_l2"]
    ].dropna().copy()
    if len(discrim) >= 40 and discrim["session_id"].nunique() >= 8 and discrim["is_real"].nunique() >= 2:
        X = sm.add_constant(discrim[["z_total_persistence_h1", "z_posterior_entropy", "z_local_speed_l2"]], has_constant="add")
        try:
            fit = sm.GLM(discrim["is_real"], X, family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={"groups": discrim["session_id"]},
            )
            for term in fit.params.index:
                results_rows.append(
                    {
                        "analysis": "real_vs_pseudo_control_logit",
                        "model_type": "GLM_binomial_clustered",
                        "term": term,
                        "coef": float(fit.params[term]),
                        "se": float(fit.bse[term]),
                        "p_value": float(fit.pvalues[term]),
                        "n_obs": int(len(discrim)),
                        "n_sessions": int(discrim["session_id"].nunique()),
                    }
                )
        except Exception as exc:
            print(f"Could not fit posterior control discrimination model: {exc}")

    growth_real = df_growth[
        (df_growth["anchor_kind"] == "real")
        & (df_growth["feature"] == feature)
        & (df_growth["window"] == "post_switch_growth")
    ][["session_id", "switch_index", "slope"]].rename(columns={"slope": "post_switch_slope"})
    control_real = switch_cov[switch_cov["anchor_kind"] == "real"].copy()
    post_df = control_real.merge(growth_real, on=["session_id", "switch_index"], how="inner")
    if not post_df.empty:
        post_df["z_post_switch_slope"] = _zscore_series(post_df["post_switch_slope"])
        dmod = post_df[
            ["session_id", "post_switch_slope", "z_total_persistence_h1", "z_posterior_entropy", "z_local_speed_l2"]
        ].dropna().copy()
        if len(dmod) >= 30 and dmod["session_id"].nunique() >= 8:
            try:
                fit = MixedLM.from_formula(
                    "post_switch_slope ~ z_total_persistence_h1 + z_posterior_entropy + z_local_speed_l2",
                    data=dmod,
                    groups="session_id",
                ).fit(reml=False, method="lbfgs", maxiter=300)
                for term in fit.params.index:
                    results_rows.append(
                        {
                            "analysis": "post_switch_slope_control_mixedlm",
                            "model_type": "MixedLM",
                            "term": term,
                            "coef": float(fit.params[term]),
                            "se": float(fit.bse[term]),
                            "p_value": float(fit.pvalues.get(term, np.nan)),
                            "n_obs": int(len(dmod)),
                            "n_sessions": int(dmod["session_id"].nunique()),
                        }
                    )
            except Exception as exc:
                print(f"Could not fit posterior control post-switch model: {exc}")

    df_control_input = switch_cov.merge(growth_real, on=["session_id", "switch_index"], how="left")
    df_control_models = pd.DataFrame(results_rows)
    display(df_control_models)
    if not df_control_input.empty:
        df_control_input.to_csv(tab_dir / "posterior_control_analysis_input.csv", index=False)
    if not df_control_models.empty:
        df_control_models.to_csv(tab_dir / "posterior_control_models.csv", index=False)
    return df_control_input, df_control_models


def run_window_phase_analysis(
    df_main: pd.DataFrame,
    topo_features: list[str],
    primary_window_labels: list[str],
    fig_dir: Path,
    tab_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_main.empty:
        print("No analysis table available yet.")
        return pd.DataFrame(), pd.DataFrame()

    window_rows = []
    for feature in topo_features:
        ws = summarize_window_effects(df_main, feature, window_col="phase_window_5")
        if not ws.empty:
            window_rows.append(ws)

    df_window_summary = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    display(df_window_summary.head(12) if not df_window_summary.empty else pd.DataFrame())
    if not df_window_summary.empty:
        df_window_summary.to_csv(tab_dir / "windowed_timecourse_summary.csv", index=False)
        fig, axes = plt.subplots(len(topo_features), 1, figsize=(10, max(4, 3.2 * len(topo_features))), sharex=True)
        if len(topo_features) == 1:
            axes = [axes]
        for ax, feature in zip(axes, topo_features):
            sub = df_window_summary[df_window_summary["feature"] == feature].copy()
            sub["phase_window_5"] = pd.Categorical(sub["phase_window_5"], categories=primary_window_labels, ordered=True)
            sub = sub.sort_values(["phase_window_5", "anchor_kind"])
            sns.pointplot(
                data=sub,
                x="phase_window_5",
                y="mean",
                hue="anchor_kind",
                dodge=0.25,
                join=False,
                errorbar=None,
                ax=ax,
            )
            ax.set_title(f"{feature}: windowed mean by anchor kind")
            ax.set_ylabel(feature)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.savefig(fig_dir / "windowed_real_vs_pseudo_summary.svg", format="svg")
        plt.show()

    phase_summary_rows = []
    for feature in topo_features:
        d = df_main[["session_id", "switch_index", "anchor_kind", "phase", "posterior_jump_l2", feature]].dropna(subset=[feature, "phase"]).copy()
        if d.empty:
            continue
        sw = d.groupby(["session_id", "switch_index", "anchor_kind", "phase"], as_index=False)[feature].mean()
        wide = sw.pivot_table(index=["session_id", "switch_index", "anchor_kind"], columns="phase", values=feature).reset_index()
        jumps = d[d["anchor_kind"] == "real"][["session_id", "switch_index", "posterior_jump_l2"]].drop_duplicates()
        wide = wide.merge(jumps, on=["session_id", "switch_index"], how="left")

        for kind in ["real", "pseudo"]:
            kind_wide = wide[wide["anchor_kind"] == kind]
            for contrast_name, c1, c2 in [
                ("during_minus_before", "before", "during"),
                ("after_minus_before", "before", "after"),
                ("after_minus_during", "during", "after"),
            ]:
                if c1 in kind_wide.columns and c2 in kind_wide.columns:
                    delta_df = kind_wide[[c1, c2, "session_id"]].dropna()
                    delta = delta_df[c2] - delta_df[c1]
                    if len(delta) < 8:
                        continue
                    w_stat, w_p = wilcoxon(delta)
                    try:
                        md = MixedLM.from_formula(
                            "delta ~ 1",
                            data=pd.DataFrame({"delta": delta, "session_id": delta_df["session_id"].to_numpy()}),
                            groups="session_id",
                        ).fit(reml=False)
                        mixed_t = float(md.tvalues.iloc[0])
                        mixed_p = float(md.pvalues.iloc[0])
                    except Exception:
                        mixed_t = np.nan
                        mixed_p = np.nan
                    phase_summary_rows.append(
                        {
                            "feature": feature,
                            "anchor_kind": kind,
                            "contrast": contrast_name,
                            "n_switches": int(len(delta)),
                            "mean_delta": float(np.mean(delta)),
                            "median_delta": float(np.median(delta)),
                            "wilcoxon_stat": float(w_stat),
                            "wilcoxon_p": float(w_p),
                            "mixed_lm_t": mixed_t,
                            "mixed_lm_p": mixed_p,
                        }
                    )

        real_wide = wide[wide["anchor_kind"] == "real"]
        if "before" in real_wide.columns and "after" in real_wide.columns:
            dd = real_wide[["posterior_jump_l2", "before", "after"]].dropna().copy()
            if len(dd) >= 8:
                dd["delta_after_minus_before"] = dd["after"] - dd["before"]
                r, p = spearmanr(dd["posterior_jump_l2"], dd["delta_after_minus_before"])
                phase_summary_rows.append(
                    {
                        "feature": feature,
                        "anchor_kind": "real",
                        "contrast": "jump_vs_after_before_delta",
                        "n_switches": int(len(dd)),
                        "mean_delta": np.nan,
                        "median_delta": np.nan,
                        "wilcoxon_stat": float(r),
                        "wilcoxon_p": float(p),
                        "mixed_lm_t": np.nan,
                        "mixed_lm_p": np.nan,
                    }
                )

    df_phase_stats = pd.DataFrame(phase_summary_rows) if phase_summary_rows else pd.DataFrame()
    display(df_phase_stats)
    if not df_phase_stats.empty:
        df_phase_stats.to_csv(tab_dir / "phase_contrast_and_jump_stats.csv", index=False)
    return df_window_summary, df_phase_stats


def run_jump_stability_moderators(
    df_main: pd.DataFrame,
    topo_features: list[str],
    fig_dir: Path,
    tab_dir: Path,
) -> dict[str, pd.DataFrame]:
    if df_main.empty:
        print("No analysis table available yet.")
        empty = pd.DataFrame()
        return {
            "real": empty,
            "jump": empty,
            "jump_model_summary": empty,
            "jump_predicted_curves": empty,
            "jump_retained_levels": empty,
            "stability": empty,
            "stability_model_summary": empty,
            "stability_predicted_curves": empty,
            "stability_retained_levels": empty,
        }

    df_real = df_main[df_main["anchor_kind"] == "real"].copy()
    display(df_real.head(3) if not df_real.empty else pd.DataFrame())
    if df_real.empty:
        empty = pd.DataFrame()
        return {
            "real": df_real,
            "jump": empty,
            "jump_model_summary": empty,
            "jump_predicted_curves": empty,
            "jump_retained_levels": empty,
            "stability": empty,
            "stability_model_summary": empty,
            "stability_predicted_curves": empty,
            "stability_retained_levels": empty,
        }

    jump_results = run_moderator_timecourse_inference(
        df_real,
        topo_features,
        moderator_family="jump_magnitude",
        group_col="jump_magnitude_bin",
        out_name="moderator_jump_magnitude_timecourses_real_only",
        fig_dir=fig_dir,
        tab_dir=tab_dir,
        min_sessions=8,
        palette=None,
    )
    stability_results = run_moderator_timecourse_inference(
        df_real,
        topo_features,
        moderator_family="regime_stability",
        group_col="regime_stability_type",
        out_name="moderator_regime_stability_timecourses_real_only",
        fig_dir=fig_dir,
        tab_dir=tab_dir,
        min_sessions=8,
        palette=None,
    )
    return {
        "real": df_real,
        "jump": jump_results["summary"],
        "jump_model_summary": jump_results["model_summary"],
        "jump_predicted_curves": jump_results["predicted_curves"],
        "jump_retained_levels": jump_results["retained_levels"],
        "stability": stability_results["summary"],
        "stability_model_summary": stability_results["model_summary"],
        "stability_predicted_curves": stability_results["predicted_curves"],
        "stability_retained_levels": stability_results["retained_levels"],
    }


def _direction_counts(df_real: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df_real[["session_id", "switch_index", col]]
        .drop_duplicates()[col]
        .value_counts()
        .rename_axis(col)
        .reset_index(name="n_switches")
    )


def run_valence_moderators(
    df_main: pd.DataFrame,
    topo_features: list[str],
    fig_dir: Path,
    tab_dir: Path,
    baseline_centered: bool = False,
) -> dict[str, pd.DataFrame]:
    if df_main.empty:
        print("No analysis table available yet.")
        return {}

    df_real = df_main[df_main["anchor_kind"] == "real"].copy()
    if df_real.empty:
        print("No real-switch rows available.")
        return {"real": df_real}

    print("Client valence direction counts:")
    df_client_counts = _direction_counts(df_real, "client_valence_direction")
    display(df_client_counts)

    print("Therapist valence direction counts:")
    df_therapist_counts = _direction_counts(df_real, "therapist_valence_direction")
    display(df_therapist_counts)

    client_palette = {
        "toward_more_negative": "firebrick",
        "toward_neutral_or_positive": "seagreen",
        "neutral_or_minimal_change": "gray",
    }
    therapist_palette = {
        "toward_more_negative": "darkred",
        "toward_neutral_or_positive": "darkgreen",
        "neutral_or_minimal_change": "gray",
    }

    features = topo_features
    suffix = ""
    if baseline_centered:
        features = [f"{feature}_baseline_centered" for feature in topo_features]
        suffix = "_baselined"

    client_results = run_moderator_timecourse_inference(
        df_real,
        features,
        moderator_family="client_valence_direction",
        group_col="client_valence_direction",
        out_name=f"moderator_client_valence_direction_timecourses_real_only{suffix}",
        fig_dir=fig_dir,
        tab_dir=tab_dir,
        min_sessions=8,
        palette=client_palette,
    )
    therapist_results = run_moderator_timecourse_inference(
        df_real,
        features,
        moderator_family="therapist_valence_direction",
        group_col="therapist_valence_direction",
        out_name=f"moderator_therapist_valence_direction_timecourses_real_only{suffix}",
        fig_dir=fig_dir,
        tab_dir=tab_dir,
        min_sessions=8,
        palette=therapist_palette,
    )

    if baseline_centered and "total_persistence_h1_baseline_centered" in df_real.columns:
        std_df = (
            df_real.groupby(["client_valence_direction", "rel_time_sec"])["total_persistence_h1_baseline_centered"]
            .std()
            .reset_index()
        )
        sns.lineplot(data=std_df, x="rel_time_sec", y="total_persistence_h1_baseline_centered", hue="client_valence_direction")
        plt.axvline(0, linestyle="--", color="red")
        plt.title("Std dev over time")
        plt.show()
    else:
        std_df = pd.DataFrame()

    display(client_results["summary"].head(10) if not client_results["summary"].empty else pd.DataFrame())
    display(therapist_results["summary"].head(10) if not therapist_results["summary"].empty else pd.DataFrame())
    return {
        "real": df_real,
        "client_counts": df_client_counts,
        "therapist_counts": df_therapist_counts,
        "client_summary": client_results["summary"],
        "client_model_summary": client_results["model_summary"],
        "client_predicted_curves": client_results["predicted_curves"],
        "client_retained_levels": client_results["retained_levels"],
        "therapist_summary": therapist_results["summary"],
        "therapist_model_summary": therapist_results["model_summary"],
        "therapist_predicted_curves": therapist_results["predicted_curves"],
        "therapist_retained_levels": therapist_results["retained_levels"],
        "std_df": std_df,
    }


def run_transition_identity_analysis(
    df_main: pd.DataFrame,
    df_growth: pd.DataFrame,
    topo_features: list[str],
    fig_dir: Path,
    tab_dir: Path,
) -> dict[str, pd.DataFrame]:
    if df_main.empty:
        print("No analysis table available yet.")
        empty = pd.DataFrame()
        return {"rankings": empty, "pair_summary": empty, "pair_counts": empty, "arc_matrix": empty, "post_slope_matrix": empty}

    rows = []
    top_identity_keep = None
    for feature in topo_features:
        d = df_main[df_main["anchor_kind"] == "real"][["session_id", "switch_index", "transition_identity", "phase", feature]].dropna().copy()
        if d.empty:
            continue
        sw = d.groupby(["session_id", "switch_index", "transition_identity", "phase"], as_index=False)[feature].mean()
        wide = sw.pivot_table(index=["session_id", "switch_index", "transition_identity"], columns="phase", values=feature).reset_index()
        if "before" not in wide.columns or "after" not in wide.columns:
            continue
        wide["delta_after_minus_before"] = wide["after"] - wide["before"]
        agg = (
            wide.groupby("transition_identity")["delta_after_minus_before"]
            .agg(mean_delta="mean", median_delta="median", n_switches="count")
            .reset_index()
        )
        agg["feature"] = feature
        if top_identity_keep is None:
            top_identity_keep = agg.sort_values("n_switches", ascending=False).head(10)["transition_identity"].tolist()
        rows.append(agg)

        plot_df = agg[agg["transition_identity"].isin(top_identity_keep)].sort_values("mean_delta", ascending=False)
        if plot_df.empty:
            continue
        plt.figure(figsize=(9, 5))
        sns.barplot(data=plot_df, x="mean_delta", y="transition_identity", color="steelblue")
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        plt.title(f"Transition identity reorganization ranking ({feature})")
        plt.xlabel("Mean after-before delta")
        plt.ylabel("Transition identity")
        plt.tight_layout()
        plt.savefig(fig_dir / f"transition_identity_delta_rank_{feature}.svg", format="svg")
        plt.show()

    df_transition_rank = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    display(df_transition_rank.head(20) if not df_transition_rank.empty else pd.DataFrame())
    if not df_transition_rank.empty:
        df_transition_rank.to_csv(tab_dir / "transition_identity_reorganization_rankings.csv", index=False)

    feature = "total_persistence_h1"
    real_phase = df_main[
        (df_main["anchor_kind"] == "real")
        & df_main["from_state"].notna()
        & df_main["to_state"].notna()
        & df_main[feature].notna()
    ][["session_id", "switch_index", "from_state", "to_state", "phase", feature]].copy()
    if real_phase.empty:
        return {"rankings": df_transition_rank, "pair_summary": pd.DataFrame(), "pair_counts": pd.DataFrame(), "arc_matrix": pd.DataFrame(), "post_slope_matrix": pd.DataFrame()}

    phase_switch = real_phase.groupby(["session_id", "switch_index", "from_state", "to_state", "phase"], as_index=False)[feature].mean()
    phase_wide = phase_switch.pivot_table(
        index=["session_id", "switch_index", "from_state", "to_state"],
        columns="phase",
        values=feature,
    ).reset_index()
    if "before" in phase_wide.columns and "after" in phase_wide.columns:
        phase_wide["arc_magnitude"] = (phase_wide["after"] - phase_wide["before"]).abs()
    else:
        phase_wide["arc_magnitude"] = np.nan

    pair_summary = phase_wide[["session_id", "switch_index", "from_state", "to_state", "arc_magnitude"]].copy()
    pair_summary["transition_pair"] = (
        pair_summary["from_state"].astype(int).astype(str) + "->" + pair_summary["to_state"].astype(int).astype(str)
    )
    growth_real = df_growth[
        (df_growth["anchor_kind"] == "real")
        & (df_growth["feature"] == feature)
        & (df_growth["window"] == "post_switch_growth")
    ][["session_id", "switch_index", "slope"]].rename(columns={"slope": "post_switch_slope"})
    pair_summary = pair_summary.merge(growth_real, on=["session_id", "switch_index"], how="left")

    pair_long = (
        pair_summary.groupby(["from_state", "to_state"], as_index=False)
        .agg(
            n_switches=("session_id", "size"),
            mean_arc_magnitude=("arc_magnitude", "mean"),
            mean_post_switch_slope=("post_switch_slope", "mean"),
        )
        .sort_values(["from_state", "to_state"])
    )

    states = list(range(8))
    pair_counts = pair_long.pivot_table(index="from_state", columns="to_state", values="n_switches", fill_value=0).reindex(index=states, columns=states)
    arc_matrix = pair_long.pivot_table(index="from_state", columns="to_state", values="mean_arc_magnitude").reindex(index=states, columns=states)
    post_slope_matrix = pair_long.pivot_table(index="from_state", columns="to_state", values="mean_post_switch_slope").reindex(index=states, columns=states)
    support_mask = pair_counts < 50
    for matrix in [arc_matrix, post_slope_matrix]:
        np.fill_diagonal(matrix.values, np.nan)
        matrix[support_mask] = np.nan

    pair_long.to_csv(tab_dir / "transition_pair_heterogeneity_summary.csv", index=False)
    pair_counts.to_csv(tab_dir / "transition_pair_counts_matrix.csv")
    arc_matrix.to_csv(tab_dir / "transition_pair_arc_magnitude_matrix.csv")
    post_slope_matrix.to_csv(tab_dir / "transition_pair_post_switch_slope_matrix.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(arc_matrix, cmap="mako", ax=axes[0], square=True, linewidths=0.5, linecolor="white")
    axes[0].set_title("Mean arc magnitude by transition pair")
    axes[0].set_xlabel("To state")
    axes[0].set_ylabel("From state")
    sns.heatmap(post_slope_matrix, cmap="RdBu_r", center=0.0, ax=axes[1], square=True, linewidths=0.5, linecolor="white")
    axes[1].set_title("Mean post-switch slope by transition pair")
    axes[1].set_xlabel("To state")
    axes[1].set_ylabel("From state")
    plt.tight_layout()
    plt.savefig(fig_dir / "transition_pair_heterogeneity_heatmaps.svg", format="svg")
    plt.show()

    return {
        "rankings": df_transition_rank,
        "pair_summary": pair_long,
        "pair_counts": pair_counts.reset_index().rename(columns={"from_state": "from_state"}),
        "arc_matrix": arc_matrix.reset_index().rename(columns={"from_state": "from_state"}),
        "post_slope_matrix": post_slope_matrix.reset_index().rename(columns={"from_state": "from_state"}),
    }


def build_growth_base_table(df: pd.DataFrame, features: list[str], windows=None) -> pd.DataFrame:
    if windows is None:
        windows = GROWTH_WINDOWS
    bounds = np.array(list(windows.values()), dtype=float)
    global_lo = float(bounds[:, 0].min())
    global_hi = float(bounds[:, 1].max())

    keep_cols = [
        "session_id",
        "switch_index",
        "anchor_kind",
        "rel_time_sec",
        "posterior_jump_l2",
        "posterior_entropy",
        "local_speed_l2",
        "transition_identity",
        "regime_stability_type",
        "jump_magnitude_bin",
        "from_run_len",
        "to_run_len",
        "log_from_run_len",
        "log_to_run_len",
        "client_valence_direction",
        "therapist_valence_direction",
        "client_valence_delta",
        "therapist_valence_delta",
        "z_posterior_jump_l2",
        "z_log_from_run_len",
        "z_log_to_run_len",
        "z_client_valence_delta",
        "z_therapist_valence_delta",
    ] + [feature for feature in features if feature in df.columns]
    keep_cols = [col for col in keep_cols if col in df.columns]

    base = df[keep_cols].copy()
    base = base[(base["rel_time_sec"] >= global_lo) & (base["rel_time_sec"] <= global_hi)].copy()
    if base.empty:
        return pd.DataFrame()

    agg_spec = {feature: "mean" for feature in features if feature in base.columns}
    for col in [
        "posterior_jump_l2",
        "posterior_entropy",
        "local_speed_l2",
        "transition_identity",
        "regime_stability_type",
        "jump_magnitude_bin",
        "from_run_len",
        "to_run_len",
        "log_from_run_len",
        "log_to_run_len",
        "client_valence_direction",
        "therapist_valence_direction",
        "client_valence_delta",
        "therapist_valence_delta",
        "z_posterior_jump_l2",
        "z_log_from_run_len",
        "z_log_to_run_len",
        "z_client_valence_delta",
        "z_therapist_valence_delta",
    ]:
        if col in base.columns:
            agg_spec[col] = "first"

    return (
        base.groupby(["session_id", "switch_index", "anchor_kind", "rel_time_sec"], as_index=False)
        .agg(agg_spec)
        .sort_values(["session_id", "switch_index", "anchor_kind", "rel_time_sec"])
    )


def estimate_switch_growth_slopes(df_growth_base: pd.DataFrame, feature: str, windows=None, min_points: int = 4) -> pd.DataFrame:
    if windows is None:
        windows = GROWTH_WINDOWS
    if df_growth_base.empty or feature not in df_growth_base.columns:
        return pd.DataFrame()

    keep_cols = [
        "session_id",
        "switch_index",
        "anchor_kind",
        "rel_time_sec",
        feature,
        "posterior_jump_l2",
        "posterior_entropy",
        "local_speed_l2",
        "transition_identity",
        "regime_stability_type",
        "jump_magnitude_bin",
        "from_run_len",
        "to_run_len",
        "log_from_run_len",
        "log_to_run_len",
        "client_valence_direction",
        "therapist_valence_direction",
        "client_valence_delta",
        "therapist_valence_delta",
        "z_posterior_jump_l2",
        "z_log_from_run_len",
        "z_log_to_run_len",
        "z_client_valence_delta",
        "z_therapist_valence_delta",
    ]
    keep_cols = [col for col in keep_cols if col in df_growth_base.columns]
    d = df_growth_base[keep_cols].dropna(subset=[feature, "rel_time_sec"]).copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for (session_id, switch_index, anchor_kind), g in d.groupby(["session_id", "switch_index", "anchor_kind"]):
        meta = {
            "session_id": session_id,
            "switch_index": switch_index,
            "anchor_kind": anchor_kind,
            "feature": feature,
        }
        for col in [
            "posterior_jump_l2",
            "posterior_entropy",
            "local_speed_l2",
            "transition_identity",
            "regime_stability_type",
            "jump_magnitude_bin",
            "from_run_len",
            "to_run_len",
            "log_from_run_len",
            "log_to_run_len",
            "client_valence_direction",
            "therapist_valence_direction",
            "client_valence_delta",
            "therapist_valence_delta",
            "z_posterior_jump_l2",
            "z_log_from_run_len",
            "z_log_to_run_len",
            "z_client_valence_delta",
            "z_therapist_valence_delta",
        ]:
            if col in g.columns:
                non_na = g[col].dropna()
                meta[col] = non_na.iloc[0] if len(non_na) > 0 else np.nan

        for window_name, (lo, hi) in windows.items():
            w = g[(g["rel_time_sec"] >= lo) & (g["rel_time_sec"] <= hi)][["rel_time_sec", feature]].copy()
            if w.empty:
                continue
            if len(w) < min_points or w["rel_time_sec"].nunique() < 3:
                slope = np.nan
                intercept = np.nan
                start_value = np.nan
                end_value = np.nan
            else:
                slope, intercept = np.polyfit(w["rel_time_sec"].to_numpy(), w[feature].to_numpy(), 1)
                start_value = float(w[feature].iloc[0])
                end_value = float(w[feature].iloc[-1])
            rows.append(
                {
                    **meta,
                    "window": window_name,
                    "window_start_sec": lo,
                    "window_end_sec": hi,
                    "n_timepoints": int(len(w)),
                    "slope": float(slope) if pd.notna(slope) else np.nan,
                    "intercept": float(intercept) if pd.notna(intercept) else np.nan,
                    "start_value": start_value,
                    "end_value": end_value,
                    "delta_end_minus_start": end_value - start_value if pd.notna(start_value) and pd.notna(end_value) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def mixedlm_params_table(model, model_name: str) -> pd.DataFrame:
    rows = []
    for term in model.params.index:
        se = float(model.bse.get(term, np.nan)) if hasattr(model, "bse") else np.nan
        rows.append(
            {
                "model_name": model_name,
                "term": term,
                "coef": float(model.params[term]),
                "std_error": se,
                "lower": float(model.params[term] - 1.96 * se) if pd.notna(se) else np.nan,
                "upper": float(model.params[term] + 1.96 * se) if pd.notna(se) else np.nan,
                "p_value": float(model.pvalues.get(term, np.nan)),
                "aic": float(getattr(model, "aic", np.nan)),
                "bic": float(getattr(model, "bic", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _model_params_table(model, model_name: str, model_type: str, n_obs: int, n_sessions: int) -> pd.DataFrame:
    rows = []
    for term in model.params.index:
        se = float(model.bse.get(term, np.nan)) if hasattr(model, "bse") else np.nan
        rows.append(
            {
                "model_name": model_name,
                "model_type": model_type,
                "term": term,
                "coef": float(model.params[term]),
                "std_error": se,
                "lower": float(model.params[term] - 1.96 * se) if pd.notna(se) else np.nan,
                "upper": float(model.params[term] + 1.96 * se) if pd.notna(se) else np.nan,
                "p_value": float(model.pvalues.get(term, np.nan)),
                "aic": float(getattr(model, "aic", np.nan)),
                "bic": float(getattr(model, "bic", np.nan)),
                "n_obs": int(n_obs),
                "n_sessions": int(n_sessions),
            }
        )
    return pd.DataFrame(rows)


def _window_fluctuation_stats(values: pd.Series, min_points: int) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < min_points:
        return {
            "sd": np.nan,
            "variance": np.nan,
            "rmssd": np.nan,
            "mean_abs_diff": np.nan,
            "range": np.nan,
            "n_points": int(len(clean)),
        }

    diffs = np.diff(clean.to_numpy())
    return {
        "sd": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "variance": float(clean.var(ddof=1)) if len(clean) > 1 else np.nan,
        "rmssd": float(np.sqrt(np.mean(np.square(diffs)))) if len(diffs) > 0 else np.nan,
        "mean_abs_diff": float(np.mean(np.abs(diffs))) if len(diffs) > 0 else np.nan,
        "range": float(clean.max() - clean.min()),
        "n_points": int(len(clean)),
    }


def compute_switch_fluctuation_table(
    df_main: pd.DataFrame,
    features: list[str],
    *,
    pre_window=(-20, -5),
    post_window=(5, 20),
    near_window=(-5, 5),
    min_points: int = 4,
    near_min_points: int = 3,
) -> pd.DataFrame:
    if df_main.empty:
        return pd.DataFrame()

    df_real = df_main[df_main["anchor_kind"] == "real"].copy()
    if df_real.empty:
        return pd.DataFrame()

    meta_cols = [
        "posterior_jump_l2",
        "posterior_entropy",
        "local_speed_l2",
        "from_run_len",
        "to_run_len",
        "log_from_run_len",
        "log_to_run_len",
        "client_valence_delta",
        "therapist_valence_delta",
        "z_posterior_jump_l2",
        "z_log_from_run_len",
        "z_log_to_run_len",
        "z_client_valence_delta",
        "z_therapist_valence_delta",
    ]
    rows = []

    for (session_id, switch_index), g_switch in df_real.groupby(["session_id", "switch_index"]):
        meta = {"session_id": session_id, "switch_index": switch_index}
        for col in meta_cols:
            if col in g_switch.columns:
                non_na = g_switch[col].dropna()
                meta[col] = non_na.iloc[0] if len(non_na) > 0 else np.nan

        for feature in features:
            if feature not in g_switch.columns:
                continue
            g = g_switch[["rel_time_sec", feature]].dropna().sort_values("rel_time_sec").copy()
            if g.empty:
                continue

            pre_stats = _window_fluctuation_stats(
                g[(g["rel_time_sec"] >= pre_window[0]) & (g["rel_time_sec"] <= pre_window[1])][feature],
                min_points=min_points,
            )
            post_stats = _window_fluctuation_stats(
                g[(g["rel_time_sec"] >= post_window[0]) & (g["rel_time_sec"] <= post_window[1])][feature],
                min_points=min_points,
            )
            near_stats = _window_fluctuation_stats(
                g[(g["rel_time_sec"] >= near_window[0]) & (g["rel_time_sec"] <= near_window[1])][feature],
                min_points=near_min_points,
            )

            row = {
                **meta,
                "feature": feature,
                "pre_n_points": pre_stats["n_points"],
                "post_n_points": post_stats["n_points"],
                "near_switch_n_points": near_stats["n_points"],
                "pre_volatility_sd": pre_stats["sd"],
                "post_volatility_sd": post_stats["sd"],
                "delta_volatility_sd": (
                    post_stats["sd"] - pre_stats["sd"]
                    if pd.notna(post_stats["sd"]) and pd.notna(pre_stats["sd"])
                    else np.nan
                ),
                "pre_volatility_variance": pre_stats["variance"],
                "post_volatility_variance": post_stats["variance"],
                "delta_volatility_variance": (
                    post_stats["variance"] - pre_stats["variance"]
                    if pd.notna(post_stats["variance"]) and pd.notna(pre_stats["variance"])
                    else np.nan
                ),
                "pre_volatility_rmssd": pre_stats["rmssd"],
                "post_volatility_rmssd": post_stats["rmssd"],
                "delta_volatility_rmssd": (
                    post_stats["rmssd"] - pre_stats["rmssd"]
                    if pd.notna(post_stats["rmssd"]) and pd.notna(pre_stats["rmssd"])
                    else np.nan
                ),
                "pre_volatility_mean_abs_diff": pre_stats["mean_abs_diff"],
                "post_volatility_mean_abs_diff": post_stats["mean_abs_diff"],
                "delta_volatility_mean_abs_diff": (
                    post_stats["mean_abs_diff"] - pre_stats["mean_abs_diff"]
                    if pd.notna(post_stats["mean_abs_diff"]) and pd.notna(pre_stats["mean_abs_diff"])
                    else np.nan
                ),
                "near_switch_roughness_sd": near_stats["sd"],
                "near_switch_roughness_variance": near_stats["variance"],
                "near_switch_roughness_rmssd": near_stats["rmssd"],
                "near_switch_roughness_mean_abs_diff": near_stats["mean_abs_diff"],
                "near_switch_roughness_range": near_stats["range"],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def fit_switch_level_formula(
    df: pd.DataFrame,
    formula: str,
    *,
    groups_col: str = "session_id",
    model_name: str,
) -> pd.DataFrame:
    if df.empty or groups_col not in df.columns:
        return pd.DataFrame()

    n_obs = len(df)
    n_sessions = df[groups_col].nunique()
    if n_obs < 30 or n_sessions < 8:
        return pd.DataFrame()

    try:
        fit = MixedLM.from_formula(
            formula,
            data=df,
            groups=groups_col,
        ).fit(reml=False, method="lbfgs", maxiter=300)
        return _model_params_table(fit, model_name, "MixedLM", n_obs=n_obs, n_sessions=n_sessions)
    except Exception as exc:
        print(f"Could not fit mixed model {model_name}; falling back to OLS: {exc}")

    try:
        fit = sm.OLS.from_formula(formula, data=df).fit()
        return _model_params_table(fit, model_name, "OLS_fallback", n_obs=n_obs, n_sessions=n_sessions)
    except Exception as exc:
        print(f"Could not fit fallback OLS model {model_name}: {exc}")
        return pd.DataFrame()


def run_growth_analysis(
    df_main: pd.DataFrame,
    topo_features: list[str],
    fig_dir: Path,
    tab_dir: Path,
) -> dict[str, pd.DataFrame]:
    if df_main.empty:
        print("No analysis table available yet.")
        empty = pd.DataFrame()
        return {
            "df_growth_base": empty,
            "df_growth": empty,
            "df_growth_summary": empty,
            "df_growth_models": empty,
            "df_growth_jump": empty,
            "df_growth_moderation_input": empty,
            "df_growth_moderation_models": empty,
            "df_growth_continuous_dwell_models": empty,
            "df_growth_continuous_transition_models": empty,
            "df_jump_moderator_summary": empty,
            "df_jump_moderator_models": empty,
            "df_stability_moderator_summary": empty,
            "df_stability_moderator_models": empty,
            "df_client_valence_moderator_summary": empty,
            "df_client_valence_moderator_models": empty,
            "df_therapist_valence_moderator_summary": empty,
            "df_therapist_valence_moderator_models": empty,
            "df_moderator_omnibus_models": empty,
            "df_continuous_moderator_models": empty,
            "df_continuous_moderator_predicted_curves": empty,
            "df_continuous_moderator_correlations": empty,
            "df_fluctuation_input": empty,
            "df_fluctuation_models": empty,
        }

    df_growth_base = build_growth_base_table(df_main, topo_features)
    print("Growth base table shape:", df_growth_base.shape)

    growth_frames = []
    growth_summary_rows = []
    growth_model_rows = []
    growth_jump_rows = []

    for feature in topo_features:
        df_growth_f = estimate_switch_growth_slopes(df_growth_base, feature)
        if df_growth_f.empty:
            continue
        growth_frames.append(df_growth_f)
        growth_summary_rows.append(
            df_growth_f.groupby(["feature", "window", "anchor_kind"], as_index=False)["slope"]
            .agg(mean_slope="mean", median_slope="median", sd_slope="std", n_switches="count")
        )

        for window_name in sorted(df_growth_f["window"].dropna().unique()):
            dm = df_growth_f[df_growth_f["window"] == window_name][["session_id", "anchor_kind", "slope"]].dropna().copy()
            if len(dm) >= 20 and dm["session_id"].nunique() >= 4 and dm["anchor_kind"].nunique() >= 2:
                try:
                    model = MixedLM.from_formula(
                        "slope ~ C(anchor_kind, Treatment('pseudo'))",
                        data=dm,
                        groups="session_id",
                    ).fit(reml=False)
                    coef_name = [idx for idx in model.params.index if idx.startswith("C(anchor_kind")]
                    coef_name = coef_name[0] if coef_name else None
                    growth_model_rows.append(
                        {
                            "feature": feature,
                            "window": window_name,
                            "n_obs": int(len(dm)),
                            "n_sessions": int(dm["session_id"].nunique()),
                            "coef_real_vs_pseudo": float(model.params[coef_name]) if coef_name else np.nan,
                            "p_real_vs_pseudo": float(model.pvalues[coef_name]) if coef_name else np.nan,
                            "aic": float(getattr(model, "aic", np.nan)),
                            "bic": float(getattr(model, "bic", np.nan)),
                        }
                    )
                except Exception as exc:
                    print(f"Could not fit growth mixed model for {feature}, {window_name}: {exc}")

        real_post = df_growth_f[
            (df_growth_f["anchor_kind"] == "real") & (df_growth_f["window"] == "post_switch_growth")
        ][["posterior_jump_l2", "slope"]].dropna()
        if len(real_post) >= 8:
            r, p = spearmanr(real_post["posterior_jump_l2"], real_post["slope"])
            growth_jump_rows.append(
                {
                    "feature": feature,
                    "window": "post_switch_growth",
                    "n_switches": int(len(real_post)),
                    "spearman_r": float(r),
                    "p_value": float(p),
                }
            )

    df_growth = pd.concat(growth_frames, ignore_index=True) if growth_frames else pd.DataFrame()
    df_growth_summary = pd.concat(growth_summary_rows, ignore_index=True) if growth_summary_rows else pd.DataFrame()
    df_growth_models = pd.DataFrame(growth_model_rows)
    df_growth_jump = pd.DataFrame(growth_jump_rows)

    display(df_growth_summary)
    display(df_growth_models)
    display(df_growth_jump)

    if not df_growth.empty:
        df_growth.to_csv(tab_dir / "switch_growth_slopes.csv", index=False)
    if not df_growth_summary.empty:
        df_growth_summary.to_csv(tab_dir / "switch_growth_slope_summary.csv", index=False)
    if not df_growth_models.empty:
        df_growth_models.to_csv(tab_dir / "switch_growth_anchor_kind_mixed_model.csv", index=False)
    if not df_growth_jump.empty:
        df_growth_jump.to_csv(tab_dir / "switch_growth_jump_association.csv", index=False)

    df_growth_moderation_input = pd.DataFrame()
    df_growth_moderation_models = pd.DataFrame()
    df_growth_continuous_dwell_models = pd.DataFrame()
    df_growth_continuous_transition_models = pd.DataFrame()
    df_jump_moderator_summary = pd.DataFrame()
    df_jump_moderator_models = pd.DataFrame()
    df_stability_moderator_summary = pd.DataFrame()
    df_stability_moderator_models = pd.DataFrame()
    df_client_valence_moderator_summary = pd.DataFrame()
    df_client_valence_moderator_models = pd.DataFrame()
    df_therapist_valence_moderator_summary = pd.DataFrame()
    df_therapist_valence_moderator_models = pd.DataFrame()
    df_moderator_omnibus_models = pd.DataFrame()
    df_continuous_moderator_models = pd.DataFrame()
    df_continuous_moderator_predicted_curves = pd.DataFrame()
    df_continuous_moderator_correlations = pd.DataFrame()
    df_fluctuation_input = pd.DataFrame()
    df_fluctuation_models = pd.DataFrame()
    df_growth_real_tp = df_growth[(df_growth["anchor_kind"] == "real") & (df_growth["feature"] == "total_persistence_h1")].copy()
    if not df_growth_real_tp.empty:
        slope_wide = (
            df_growth_real_tp.pivot_table(index=["session_id", "switch_index"], columns="window", values="slope")
            .reset_index()
            .rename(columns={"pre_switch_growth": "pre_switch_slope", "post_switch_growth": "post_switch_slope"})
        )
        switch_meta = (
            df_growth_real_tp[
                [
                    "session_id",
                    "switch_index",
                    "posterior_jump_l2",
                    "posterior_entropy",
                    "local_speed_l2",
                    "regime_stability_type",
                    "from_run_len",
                    "to_run_len",
                    "log_from_run_len",
                    "log_to_run_len",
                    "client_valence_direction",
                    "therapist_valence_direction",
                    "client_valence_delta",
                    "therapist_valence_delta",
                    "z_posterior_jump_l2",
                    "z_log_from_run_len",
                    "z_log_to_run_len",
                    "z_client_valence_delta",
                    "z_therapist_valence_delta",
                ]
            ].drop_duplicates(["session_id", "switch_index"])
        )
        phase_tp = df_main[
            (df_main["anchor_kind"] == "real") & df_main["total_persistence_h1"].notna()
        ][["session_id", "switch_index", "phase", "total_persistence_h1"]].copy()
        phase_switch = phase_tp.groupby(["session_id", "switch_index", "phase"], as_index=False)["total_persistence_h1"].mean()
        phase_wide = phase_switch.pivot_table(index=["session_id", "switch_index"], columns="phase", values="total_persistence_h1").reset_index()
        if "before" in phase_wide.columns and "after" in phase_wide.columns:
            phase_wide["after_before_delta"] = phase_wide["after"] - phase_wide["before"]
            phase_wide["arc_magnitude"] = phase_wide["after_before_delta"].abs()
        else:
            phase_wide["after_before_delta"] = np.nan
            phase_wide["arc_magnitude"] = np.nan

        df_growth_moderation_input = slope_wide.merge(switch_meta, on=["session_id", "switch_index"], how="left")
        df_growth_moderation_input = df_growth_moderation_input.merge(
            phase_wide[["session_id", "switch_index", "after_before_delta", "arc_magnitude"]],
            on=["session_id", "switch_index"],
            how="left",
        )

        for raw_col, log_col in [("from_run_len", "log_from_run_len"), ("to_run_len", "log_to_run_len")]:
            if raw_col in df_growth_moderation_input.columns and log_col not in df_growth_moderation_input.columns:
                df_growth_moderation_input[log_col] = np.log1p(df_growth_moderation_input[raw_col])

        for col in [
            "posterior_jump_l2",
            "posterior_entropy",
            "local_speed_l2",
            "log_from_run_len",
            "log_to_run_len",
            "client_valence_delta",
            "therapist_valence_delta",
        ]:
            if col in df_growth_moderation_input.columns:
                df_growth_moderation_input[f"z_{col}"] = _zscore_series(df_growth_moderation_input[col])

        model_frames = []
        continuous_dwell_frames = []
        d_post = df_growth_moderation_input[
            [
                "session_id",
                "post_switch_slope",
                "client_valence_direction",
                "therapist_valence_direction",
                "z_posterior_jump_l2",
            ]
        ].dropna().copy()
        if len(d_post) >= 30 and d_post["session_id"].nunique() >= 8:
            try:
                fit_post = MixedLM.from_formula(
                    "post_switch_slope ~ C(client_valence_direction) + C(therapist_valence_direction) + z_posterior_jump_l2",
                    data=d_post,
                    groups="session_id",
                ).fit(reml=False, method="lbfgs", maxiter=300)
                model_frames.append(mixedlm_params_table(fit_post, "post_switch_slope_valence_plus_jump"))
            except Exception as exc:
                print(f"Could not fit post-switch valence/jump model: {exc}")

        d_pre = df_growth_moderation_input[["session_id", "pre_switch_slope", "regime_stability_type"]].dropna().copy()
        if len(d_pre) >= 30 and d_pre["session_id"].nunique() >= 8 and d_pre["regime_stability_type"].nunique() >= 2:
            try:
                fit_pre = MixedLM.from_formula(
                    "pre_switch_slope ~ C(regime_stability_type)",
                    data=d_pre,
                    groups="session_id",
                ).fit(reml=False, method="lbfgs", maxiter=300)
                model_frames.append(mixedlm_params_table(fit_pre, "pre_switch_slope_by_stability_type"))
            except Exception as exc:
                print(f"Could not fit pre-switch stability model: {exc}")

        d_arc = df_growth_moderation_input[["session_id", "arc_magnitude", "regime_stability_type"]].dropna().copy()
        if len(d_arc) >= 30 and d_arc["session_id"].nunique() >= 8 and d_arc["regime_stability_type"].nunique() >= 2:
            try:
                fit_arc = MixedLM.from_formula(
                    "arc_magnitude ~ C(regime_stability_type)",
                    data=d_arc,
                    groups="session_id",
                ).fit(reml=False, method="lbfgs", maxiter=300)
                model_frames.append(mixedlm_params_table(fit_arc, "arc_magnitude_by_stability_type"))
            except Exception as exc:
                print(f"Could not fit arc-magnitude stability model: {exc}")

        d_pre_cont = df_growth_moderation_input[
            ["session_id", "pre_switch_slope", "z_log_from_run_len", "z_log_to_run_len"]
        ].dropna().copy()
        if len(d_pre_cont) >= 30 and d_pre_cont["session_id"].nunique() >= 8:
            try:
                fit_pre_cont = MixedLM.from_formula(
                    "pre_switch_slope ~ z_log_from_run_len + z_log_to_run_len",
                    data=d_pre_cont,
                    groups="session_id",
                ).fit(reml=False, method="lbfgs", maxiter=300)
                continuous_dwell_frames.append(mixedlm_params_table(fit_pre_cont, "pre_switch_slope_by_continuous_dwell"))
            except Exception as exc:
                print(f"Could not fit continuous dwell pre-switch model: {exc}")

        d_arc_cont = df_growth_moderation_input[
            ["session_id", "arc_magnitude", "z_log_from_run_len", "z_log_to_run_len"]
        ].dropna().copy()
        if len(d_arc_cont) >= 30 and d_arc_cont["session_id"].nunique() >= 8:
            try:
                fit_arc_cont = MixedLM.from_formula(
                    "arc_magnitude ~ z_log_from_run_len + z_log_to_run_len",
                    data=d_arc_cont,
                    groups="session_id",
                ).fit(reml=False, method="lbfgs", maxiter=300)
                continuous_dwell_frames.append(mixedlm_params_table(fit_arc_cont, "arc_magnitude_by_continuous_dwell"))
            except Exception as exc:
                print(f"Could not fit continuous dwell arc model: {exc}")

        continuous_predictor_cols = [
            "z_log_from_run_len",
            "z_log_to_run_len",
            "z_posterior_jump_l2",
            "z_client_valence_delta",
            "z_therapist_valence_delta",
        ]
        continuous_formula_rhs = " + ".join(continuous_predictor_cols)
        continuous_transition_frames = []
        for outcome in ["pre_switch_slope", "post_switch_slope", "arc_magnitude", "after_before_delta"]:
            needed_cols = ["session_id", outcome, *continuous_predictor_cols]
            d_cont = df_growth_moderation_input[needed_cols].dropna().copy()
            fit_table = fit_switch_level_formula(
                d_cont,
                f"{outcome} ~ {continuous_formula_rhs}",
                groups_col="session_id",
                model_name=f"{outcome}_by_continuous_transition_indices",
            )
            if not fit_table.empty:
                fit_table["outcome"] = outcome
                continuous_transition_frames.append(fit_table)

        df_growth_moderation_models = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
        df_growth_continuous_dwell_models = (
            pd.concat(continuous_dwell_frames, ignore_index=True) if continuous_dwell_frames else pd.DataFrame()
        )
        df_growth_continuous_transition_models = (
            pd.concat(continuous_transition_frames, ignore_index=True) if continuous_transition_frames else pd.DataFrame()
        )
        display(df_growth_moderation_input.head(10) if not df_growth_moderation_input.empty else pd.DataFrame())
        display(df_growth_moderation_models)
        display(df_growth_continuous_dwell_models)
        display(df_growth_continuous_transition_models)
        if not df_growth_moderation_input.empty:
            df_growth_moderation_input.to_csv(tab_dir / "switch_growth_moderation_input.csv", index=False)
        if not df_growth_moderation_models.empty:
            df_growth_moderation_models.to_csv(tab_dir / "switch_growth_moderation_mixed_models.csv", index=False)
        if not df_growth_continuous_dwell_models.empty:
            df_growth_continuous_dwell_models.to_csv(tab_dir / "switch_growth_continuous_dwell_mixed_models.csv", index=False)
        if not df_growth_continuous_transition_models.empty:
            df_growth_continuous_transition_models.to_csv(
                tab_dir / "switch_growth_continuous_transition_mixed_models.csv",
                index=False,
            )
    else:
        df_growth_continuous_dwell_models = pd.DataFrame()
        df_growth_continuous_transition_models = pd.DataFrame()

    moderator_real = df_main[df_main["anchor_kind"] == "real"].copy() if not df_main.empty else pd.DataFrame()
    if not moderator_real.empty:
        jump_stability_results = run_jump_stability_moderators(df_main, topo_features, fig_dir, tab_dir)
        valence_results = run_valence_moderators(df_main, topo_features, fig_dir, tab_dir, baseline_centered=False)
        continuous_moderator_specs = [
            {"moderator_name": "Departing dwell", "moderator_col": "z_log_from_run_len"},
            {"moderator_name": "Arriving dwell", "moderator_col": "z_log_to_run_len"},
            {"moderator_name": "Posterior jump", "moderator_col": "z_posterior_jump_l2"},
            {"moderator_name": "Client valence delta", "moderator_col": "z_client_valence_delta"},
            {"moderator_name": "Therapist valence delta", "moderator_col": "z_therapist_valence_delta"},
        ]
        continuous_moderator_results = run_continuous_moderator_timecourse_inference(
            moderator_real,
            topo_features,
            continuous_moderator_specs,
            fig_dir=fig_dir,
            tab_dir=tab_dir,
            out_name="continuous_moderator_timecourses_real_only",
        )

        df_jump_moderator_summary = jump_stability_results.get("jump", pd.DataFrame())
        df_jump_moderator_models = jump_stability_results.get("jump_model_summary", pd.DataFrame())
        df_stability_moderator_summary = jump_stability_results.get("stability", pd.DataFrame())
        df_stability_moderator_models = jump_stability_results.get("stability_model_summary", pd.DataFrame())
        df_client_valence_moderator_summary = valence_results.get("client_summary", pd.DataFrame())
        df_client_valence_moderator_models = valence_results.get("client_model_summary", pd.DataFrame())
        df_therapist_valence_moderator_summary = valence_results.get("therapist_summary", pd.DataFrame())
        df_therapist_valence_moderator_models = valence_results.get("therapist_model_summary", pd.DataFrame())
        df_continuous_moderator_models = continuous_moderator_results.get("model_summary", pd.DataFrame())
        df_continuous_moderator_predicted_curves = continuous_moderator_results.get("predicted_curves", pd.DataFrame())
        df_continuous_moderator_correlations = continuous_moderator_results.get("correlations", pd.DataFrame())

        omnibus_frames = [
            df_jump_moderator_models,
            df_stability_moderator_models,
            df_client_valence_moderator_models,
            df_therapist_valence_moderator_models,
            df_continuous_moderator_models,
        ]
        df_moderator_omnibus_models = pd.concat(
            [df for df in omnibus_frames if not df.empty],
            ignore_index=True,
        ) if any(not df.empty for df in omnibus_frames) else pd.DataFrame()
        display(df_moderator_omnibus_models)
        if not df_moderator_omnibus_models.empty:
            df_moderator_omnibus_models.to_csv(tab_dir / "moderator_timecourse_omnibus_model_summary.csv", index=False)

        df_fluctuation_input = compute_switch_fluctuation_table(df_main, topo_features)
        if not df_fluctuation_input.empty:
            fluctuation_frames = []
            fluctuation_predictor_cols = [
                "z_log_from_run_len",
                "z_log_to_run_len",
                "z_posterior_jump_l2",
                "z_client_valence_delta",
                "z_therapist_valence_delta",
            ]
            fluctuation_outcomes = [
                "delta_volatility_sd",
                "delta_volatility_variance",
                "delta_volatility_rmssd",
                "delta_volatility_mean_abs_diff",
                "near_switch_roughness_sd",
                "near_switch_roughness_variance",
                "near_switch_roughness_rmssd",
                "near_switch_roughness_mean_abs_diff",
                "near_switch_roughness_range",
            ]
            fluctuation_formula_rhs = " + ".join(fluctuation_predictor_cols)
            for feature in topo_features:
                for outcome in fluctuation_outcomes:
                    needed_cols = ["session_id", "feature", outcome, *fluctuation_predictor_cols]
                    d_fluct = df_fluctuation_input[needed_cols].copy()
                    d_fluct = d_fluct[d_fluct["feature"] == feature].dropna().copy()
                    fit_table = fit_switch_level_formula(
                        d_fluct,
                        f"{outcome} ~ {fluctuation_formula_rhs}",
                        groups_col="session_id",
                        model_name=f"{feature}_{outcome}_by_continuous_transition_indices",
                    )
                    if not fit_table.empty:
                        fit_table["feature"] = feature
                        fit_table["outcome"] = outcome
                        fluctuation_frames.append(fit_table)
            df_fluctuation_models = (
                pd.concat(fluctuation_frames, ignore_index=True) if fluctuation_frames else pd.DataFrame()
            )
            df_fluctuation_input.to_csv(tab_dir / "switch_fluctuation_analysis_table.csv", index=False)
            if not df_fluctuation_models.empty:
                df_fluctuation_models.to_csv(tab_dir / "switch_fluctuation_mixed_models.csv", index=False)
        display(df_continuous_moderator_models)
        display(df_continuous_moderator_correlations)
        display(df_fluctuation_models)

    if not df_growth.empty:
        window_order = ["pre_switch_growth", "post_switch_growth"]
        fig, axes = plt.subplots(len(topo_features), len(window_order), figsize=(12, max(4, 3.6 * len(topo_features))), sharex=False)
        if len(topo_features) == 1:
            axes = np.array([axes])
        for i, feature in enumerate(topo_features):
            for j, window_name in enumerate(window_order):
                ax = axes[i, j]
                sub = df_growth[(df_growth["feature"] == feature) & (df_growth["window"] == window_name)].copy()
                if sub.empty:
                    ax.axis("off")
                    continue
                sns.boxplot(data=sub, x="anchor_kind", y="slope", order=["pseudo", "real"], ax=ax, color="lightgray")
                sns.stripplot(data=sub, x="anchor_kind", y="slope", order=["pseudo", "real"], ax=ax, color="black", size=2.5, alpha=0.25)
                ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
                ax.set_title(f"{feature}: {window_name}")
                ax.set_xlabel("anchor kind")
                ax.set_ylabel("slope per sec")
        plt.tight_layout()
        plt.savefig(fig_dir / "switch_growth_slopes_by_anchor.svg", format="svg")
        plt.show()

        real_post_all = df_growth[(df_growth["anchor_kind"] == "real") & (df_growth["window"] == "post_switch_growth")].copy()
        if not real_post_all.empty and real_post_all["posterior_jump_l2"].notna().any():
            fig, axes = plt.subplots(len(topo_features), 1, figsize=(8, max(4, 3.4 * len(topo_features))), sharex=False)
            if len(topo_features) == 1:
                axes = [axes]
            for ax, feature in zip(axes, topo_features):
                sub = real_post_all[real_post_all["feature"] == feature].dropna(subset=["posterior_jump_l2", "slope"]).copy()
                if sub.empty:
                    ax.axis("off")
                    continue
                sns.regplot(
                    data=sub,
                    x="posterior_jump_l2",
                    y="slope",
                    scatter_kws={"s": 14, "alpha": 0.35},
                    line_kws={"color": "black"},
                    ax=ax,
                )
                ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
                ax.set_title(f"{feature}: post-switch growth vs posterior jump")
                ax.set_xlabel("posterior jump L2")
                ax.set_ylabel("post-switch slope per sec")
            plt.tight_layout()
            plt.savefig(fig_dir / "switch_growth_vs_jump_scatter.svg", format="svg")
            plt.show()

    return {
        "df_growth_base": df_growth_base,
        "df_growth": df_growth,
        "df_growth_summary": df_growth_summary,
        "df_growth_models": df_growth_models,
        "df_growth_jump": df_growth_jump,
        "df_growth_moderation_input": df_growth_moderation_input,
        "df_growth_moderation_models": df_growth_moderation_models,
        "df_growth_continuous_dwell_models": df_growth_continuous_dwell_models,
        "df_growth_continuous_transition_models": df_growth_continuous_transition_models,
        "df_jump_moderator_summary": df_jump_moderator_summary,
        "df_jump_moderator_models": df_jump_moderator_models,
        "df_stability_moderator_summary": df_stability_moderator_summary,
        "df_stability_moderator_models": df_stability_moderator_models,
        "df_client_valence_moderator_summary": df_client_valence_moderator_summary,
        "df_client_valence_moderator_models": df_client_valence_moderator_models,
        "df_therapist_valence_moderator_summary": df_therapist_valence_moderator_summary,
        "df_therapist_valence_moderator_models": df_therapist_valence_moderator_models,
        "df_moderator_omnibus_models": df_moderator_omnibus_models,
        "df_continuous_moderator_models": df_continuous_moderator_models,
        "df_continuous_moderator_predicted_curves": df_continuous_moderator_predicted_curves,
        "df_continuous_moderator_correlations": df_continuous_moderator_correlations,
        "df_fluctuation_input": df_fluctuation_input,
        "df_fluctuation_models": df_fluctuation_models,
    }


def make_publication_figures(
    df_main: pd.DataFrame,
    df_growth: pd.DataFrame,
    df_growth_moderation_input: pd.DataFrame,
    df_session_arc_analysis: pd.DataFrame,
    fig_dir: Path,
    tab_dir: Path,
    feature: str = "total_persistence_h1",
    df_continuous_moderator_models: pd.DataFrame | None = None,
    df_continuous_moderator_predicted_curves: pd.DataFrame | None = None,
    df_growth_continuous_transition_models: pd.DataFrame | None = None,
    df_fluctuation_models: pd.DataFrame | None = None,
    df_client_valence_moderator_summary: pd.DataFrame | None = None,
    df_therapist_valence_moderator_summary: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    _apply_publication_style()

    if df_continuous_moderator_models is None:
        df_continuous_moderator_models = pd.DataFrame()
    if df_continuous_moderator_predicted_curves is None:
        df_continuous_moderator_predicted_curves = pd.DataFrame()
    if df_growth_continuous_transition_models is None:
        df_growth_continuous_transition_models = pd.DataFrame()
    if df_fluctuation_models is None:
        df_fluctuation_models = pd.DataFrame()
    if df_client_valence_moderator_summary is None:
        df_client_valence_moderator_summary = pd.DataFrame()
    if df_therapist_valence_moderator_summary is None:
        df_therapist_valence_moderator_summary = pd.DataFrame()

    outputs = {
        "figure1_timecourse_summary": pd.DataFrame(),
        "figure1_slope_summary": pd.DataFrame(),
        "figure2_stability_summary": pd.DataFrame(),
        "figure2_valence_coefficients": pd.DataFrame(),
        "figure3_session_scatter": pd.DataFrame(),
        "figure2_continuous_curves": pd.DataFrame(),
        "figure3_effect_summary": pd.DataFrame(),
        "figureS1_affect_client_summary": pd.DataFrame(),
        "figureS1_affect_therapist_summary": pd.DataFrame(),
    }

    timecourse = summarize_timecourse(df_main, feature, "anchor_kind", min_sessions=8)
    slope_summary = _slope_summary_with_se(df_growth, feature)
    outputs["figure1_timecourse_summary"] = timecourse
    outputs["figure1_slope_summary"] = slope_summary
    if not timecourse.empty:
        timecourse.to_csv(tab_dir / "figure1_total_persistence_timecourse_summary.csv", index=False)
    if not slope_summary.empty:
        slope_summary.to_csv(tab_dir / "figure1_total_persistence_slope_summary.csv", index=False)

    real_post_scatter = df_growth[
        (df_growth["feature"] == feature)
        & (df_growth["anchor_kind"] == "real")
        & (df_growth["window"] == "post_switch_growth")
    ][["posterior_jump_l2", "slope"]].dropna().copy()
    real_post_bins = _quantile_bin_summary(real_post_scatter, "posterior_jump_l2", "slope", n_bins=20)

    if not timecourse.empty and not real_post_bins.empty:
        colors = {"real": "#2C7FB8", "pseudo": "#F28E2B"}
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.4, 1.0]})
        label_fs = 12
        tick_fs = 10
        title_fs = 12
        panel_fs = 13

        left = axes[0]
        sub = timecourse[timecourse["anchor_kind"].isin(["real", "pseudo"])].copy()
        for anchor_kind in ["real", "pseudo"]:
            dg = sub[sub["anchor_kind"] == anchor_kind].sort_values("rel_time_sec")
            if dg.empty:
                continue
            left.plot(dg["rel_time_sec"], dg["mean"], color=colors[anchor_kind], linewidth=2.3)
            left.fill_between(dg["rel_time_sec"], dg["lower"], dg["upper"], color=colors[anchor_kind], alpha=0.18)
        left.axvline(0.0, color="#C43C39", linestyle="--", linewidth=1.2)
        left.set_xlim(-30, 30)
        left.set_xlabel("Time relative to switch (sec)", fontsize=label_fs)
        left.set_ylabel("Total persistence", fontsize=label_fs)
        left.set_title("Real vs pseudo time-course", fontsize=title_fs)
        left.tick_params(axis="both", labelsize=tick_fs)
        left.text(-0.1, 1.03, "A", transform=left.transAxes, fontsize=panel_fs, fontweight="bold")
        _minimal_axis(left)

        right = axes[1]
        right.errorbar(
            real_post_bins["x_mean"],
            real_post_bins["y_mean"],
            xerr=[
                real_post_bins["x_mean"] - real_post_bins["x_min"],
                real_post_bins["x_max"] - real_post_bins["x_mean"],
            ],
            yerr=1.96 * real_post_bins["y_se"],
            fmt="o",
            markersize=5.8,
            capsize=2.8,
            linewidth=1.2,
            color=colors["real"],
            markerfacecolor=colors["real"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=colors["real"],
        )
        rho, p_value = spearmanr(real_post_scatter["posterior_jump_l2"], real_post_scatter["slope"])
        slope, intercept = np.polyfit(real_post_bins["x_mean"].to_numpy(), real_post_bins["y_mean"].to_numpy(), 1)
        x_line = np.linspace(real_post_bins["x_min"].min(), real_post_bins["x_max"].max(), 200)
        right.plot(x_line, intercept + slope * x_line, color="#303030", linewidth=1.8)
        right.axhline(0.0, color="#4A4A4A", linewidth=1.4, zorder=0)
        right.set_xlabel("Posterior jump L2", fontsize=label_fs)
        right.set_ylabel("Post-switch slope", fontsize=label_fs)
        right.set_title("Jump magnitude vs post-switch slope", fontsize=title_fs)
        right.tick_params(axis="both", labelsize=tick_fs)
        right.text(-0.14, 1.03, "B", transform=right.transAxes, fontsize=panel_fs, fontweight="bold")
        right.text(
            0.03,
            0.97,
            f"Spearman $\\rho$ = {rho:.2f}\n$p$ = {p_value:.3g}",
            transform=right.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
        _minimal_axis(right)

        fig.tight_layout(w_pad=2.2)
        _save_publication_svg(fig, fig_dir / "figure1_total_persistence_primary.svg")
        plt.show()

    stability_palette = {
        "nonpersistent_to_nonpersistent": "#8C1D18",
        "out_of_persistent": "#E67E22",
        "into_persistent": "#2AA198",
        "persistent_to_persistent": "#1F4E79",
    }
    stability_order = [
        "nonpersistent_to_nonpersistent",
        "out_of_persistent",
        "into_persistent",
        "persistent_to_persistent",
    ]
    df_real = df_main[df_main["anchor_kind"] == "real"].copy()
    stability_summary = summarize_timecourse(df_real, feature, "regime_stability_type", min_sessions=8)
    outputs["figure2_stability_summary"] = stability_summary
    if not stability_summary.empty:
        stability_summary.to_csv(tab_dir / "figure2_regime_stability_timecourse_summary.csv", index=False)

    valence_coef_df, _ = _fit_valence_slope_model(df_growth_moderation_input)
    outputs["figure2_valence_coefficients"] = valence_coef_df
    if not valence_coef_df.empty:
        valence_coef_df.to_csv(tab_dir / "figure2_valence_direction_coefficients.csv", index=False)

    if not stability_summary.empty and not valence_coef_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

        left = axes[0]
        for label in stability_order:
            dg = stability_summary[stability_summary["regime_stability_type"] == label].sort_values("rel_time_sec")
            if dg.empty:
                continue
            left.plot(dg["rel_time_sec"], dg["mean"], color=stability_palette[label], linewidth=2.2)
            left.fill_between(dg["rel_time_sec"], dg["lower"], dg["upper"], color=stability_palette[label], alpha=0.14)
        left.axvline(0.0, color="#C43C39", linestyle="--", linewidth=1.2)
        left.set_xlim(-30, 30)
        left.set_xlabel("Time relative to switch (sec)")
        left.set_ylabel("Total persistence")
        left.set_title("Regime stability type")
        _minimal_axis(left)

        right = axes[1]
        coef_order = [
            "Client toward negative",
            "Client toward neutral/positive",
            "Therapist toward negative",
            "Therapist toward neutral/positive",
        ]
        role_colors = {"client": "#5C6BC0", "therapist": "#00897B"}
        coef_plot = valence_coef_df.set_index("label").reindex(coef_order).reset_index()
        y_pos = np.arange(len(coef_plot))[::-1]
        for y, row in zip(y_pos, coef_plot.to_dict("records")):
            if pd.isna(row["coef"]):
                continue
            color = role_colors[row["role"]]
            right.hlines(y, row["lower"], row["upper"], color=color, linewidth=2.0)
            right.plot(row["coef"], y, "o", color=color, markersize=7)
        right.axvline(0.0, color="#7A7A7A", linestyle="--", linewidth=1.0)
        right.set_yticks(y_pos, coef_plot["label"])
        right.set_xlabel("Mixed-model estimate")
        right.set_title("Valence direction coefficients")
        _minimal_axis(right, show_left=False)

        fig.tight_layout(w_pad=2.0)
        _save_publication_svg(fig, fig_dir / "figure2_moderators_total_persistence.svg")
        plt.show()

    continuous_mod_order = ["Posterior jump", "Departing dwell", "Arriving dwell"]
    continuous_palette = {
        "Low (-1 SD)": "#33658A",
        "Mean": "#758E4F",
        "High (+1 SD)": "#BC4B51",
    }
    continuous_curves = df_continuous_moderator_predicted_curves[
        (df_continuous_moderator_predicted_curves.get("feature") == feature)
        & (df_continuous_moderator_predicted_curves.get("moderator_name").isin(continuous_mod_order))
    ].copy() if not df_continuous_moderator_predicted_curves.empty else pd.DataFrame()
    outputs["figure2_continuous_curves"] = continuous_curves
    if not continuous_curves.empty:
        continuous_curves.to_csv(tab_dir / "figure2_continuous_moderator_curves_total_persistence.csv", index=False)

    if not continuous_curves.empty and not df_continuous_moderator_models.empty:
        fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.85]})
        for idx, moderator_name in enumerate(continuous_mod_order):
            ax = axes[idx]
            sub = continuous_curves[continuous_curves["moderator_name"] == moderator_name].copy()
            for label in ["Low (-1 SD)", "Mean", "High (+1 SD)"]:
                dg = sub[sub["moderator_label"] == label].sort_values("rel_time_sec")
                if dg.empty:
                    continue
                ax.plot(
                    dg["rel_time_sec"],
                    dg["pred"],
                    linewidth=2.2,
                    color=continuous_palette[label],
                    label=label,
                )
            ax.axvline(0.0, color="#C43C39", linestyle="--", linewidth=1.1)
            ax.set_xlim(-30, 30)
            ax.set_title(moderator_name)
            ax.set_xlabel("Time relative to switch (sec)")
            if idx == 0:
                ax.set_ylabel("Total persistence")
            _minimal_axis(ax, show_left=(idx == 0))
            if idx == 2:
                ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.text(-0.14, 1.03, chr(ord("A") + idx), transform=ax.transAxes, fontsize=13, fontweight="bold")

        summary_ax = axes[3]
        omnibus_plot = df_continuous_moderator_models[
            (df_continuous_moderator_models["feature"] == feature)
            & (df_continuous_moderator_models["moderator_name"].isin(continuous_mod_order))
        ].copy()
        omnibus_plot["neg_log10_p"] = -np.log10(omnibus_plot["p_value"].clip(lower=1e-300))
        omnibus_plot["y"] = np.arange(len(omnibus_plot))[::-1]
        for _, row in omnibus_plot.iterrows():
            summary_ax.hlines(row["y"], 0.0, row["neg_log10_p"], color="#4E5D6C", linewidth=2.4)
            summary_ax.plot(row["neg_log10_p"], row["y"], "o", color="#1F4E79", markersize=6.5)
        summary_ax.set_yticks(omnibus_plot["y"], omnibus_plot["moderator_name"])
        summary_ax.set_xlabel(r"$-\log_{10}(p)$")
        summary_ax.set_title("Omnibus support")
        _minimal_axis(summary_ax, show_left=False)
        summary_ax.axvline(-np.log10(0.05), color="#7A7A7A", linestyle="--", linewidth=1.0)
        summary_ax.text(-0.18, 1.03, "D", transform=summary_ax.transAxes, fontsize=13, fontweight="bold")

        fig.tight_layout(w_pad=1.5)
        _save_publication_svg(fig, fig_dir / "figure2_continuous_moderators_total_persistence.svg")
        plt.show()

    predictor_labels = {
        "z_posterior_jump_l2": "Posterior jump",
        "z_log_from_run_len": "Departing dwell",
        "z_log_to_run_len": "Arriving dwell",
    }
    outcome_labels = {
        "pre_switch_slope": "Pre-switch slope",
        "post_switch_slope": "Post-switch slope",
        "arc_magnitude": "Arc magnitude",
        "after_before_delta": "After-before delta",
        "near_switch_roughness_mean_abs_diff": "Near-switch roughness (MAD)",
        "near_switch_roughness_rmssd": "Near-switch roughness (RMSSD)",
        "near_switch_roughness_range": "Near-switch roughness (range)",
        "delta_volatility_mean_abs_diff": "Post-pre volatility (MAD)",
        "delta_volatility_rmssd": "Post-pre volatility (RMSSD)",
    }
    transition_outcomes_keep = ["pre_switch_slope", "post_switch_slope", "arc_magnitude", "after_before_delta"]
    fluctuation_outcomes_keep = [
        "near_switch_roughness_mean_abs_diff",
        "near_switch_roughness_rmssd",
        "near_switch_roughness_range",
        "delta_volatility_mean_abs_diff",
        "delta_volatility_rmssd",
    ]
    effect_frames = []
    if not df_growth_continuous_transition_models.empty:
        d_transition = df_growth_continuous_transition_models[
            df_growth_continuous_transition_models["term"].isin(predictor_labels)
            & df_growth_continuous_transition_models["outcome"].isin(transition_outcomes_keep)
        ].copy()
        d_transition["panel"] = "Switch-level outcomes"
        effect_frames.append(d_transition)
    if not df_fluctuation_models.empty:
        d_fluct = df_fluctuation_models[
            (df_fluctuation_models["feature"] == feature)
            & df_fluctuation_models["term"].isin(predictor_labels)
            & df_fluctuation_models["outcome"].isin(fluctuation_outcomes_keep)
        ].copy()
        d_fluct["panel"] = "Fluctuation outcomes"
        effect_frames.append(d_fluct)
    effect_summary = pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame()
    outputs["figure3_effect_summary"] = effect_summary
    if not effect_summary.empty:
        effect_summary.to_csv(tab_dir / "figure3_continuous_effect_summary_total_persistence.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), gridspec_kw={"width_ratios": [1.0, 1.15]})
        panel_order = ["Switch-level outcomes", "Fluctuation outcomes"]
        for idx, (ax, panel_name) in enumerate(zip(axes, panel_order)):
            sub = effect_summary[effect_summary["panel"] == panel_name].copy()
            if sub.empty:
                ax.axis("off")
                continue
            sub["predictor_label"] = sub["term"].map(predictor_labels)
            sub["outcome_label"] = sub["outcome"].map(outcome_labels)
            pred_order = [predictor_labels[x] for x in predictor_labels]
            out_order = [outcome_labels[x] for x in (transition_outcomes_keep if panel_name == "Switch-level outcomes" else fluctuation_outcomes_keep)]
            heat = (
                sub.pivot_table(index="outcome_label", columns="predictor_label", values="coef")
                .reindex(index=out_order, columns=pred_order)
            )
            annot = (
                sub.assign(star=sub["p_value"].map(lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))))
                .pivot_table(index="outcome_label", columns="predictor_label", values="star", aggfunc="first")
                .reindex(index=out_order, columns=pred_order)
            )
            sns.heatmap(
                heat,
                cmap="RdBu_r",
                center=0.0,
                linewidths=0.6,
                linecolor="white",
                cbar=(idx == 1),
                annot=annot,
                fmt="",
                annot_kws={"fontsize": 10, "fontweight": "bold"},
                ax=ax,
            )
            ax.set_title(panel_name)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=25)
            ax.tick_params(axis="y", rotation=0)
            ax.text(-0.15, 1.03, chr(ord("A") + idx), transform=ax.transAxes, fontsize=13, fontweight="bold")
        fig.tight_layout(w_pad=2.0)
        _save_publication_svg(fig, fig_dir / "figure3_transition_fluctuation_summary.svg")
        plt.show()

    affect_palette = {
        "toward_more_negative": "#8C1D18",
        "neutral_or_minimal_change": "#7A7A7A",
        "toward_neutral_or_positive": "#2A9D8F",
    }
    client_affect = df_client_valence_moderator_summary[
        df_client_valence_moderator_summary.get("feature") == feature
    ].copy() if not df_client_valence_moderator_summary.empty else pd.DataFrame()
    therapist_affect = df_therapist_valence_moderator_summary[
        df_therapist_valence_moderator_summary.get("feature") == feature
    ].copy() if not df_therapist_valence_moderator_summary.empty else pd.DataFrame()
    outputs["figureS1_affect_client_summary"] = client_affect
    outputs["figureS1_affect_therapist_summary"] = therapist_affect
    if not client_affect.empty:
        client_affect.to_csv(tab_dir / "figureS1_client_affect_direction_total_persistence.csv", index=False)
    if not therapist_affect.empty:
        therapist_affect.to_csv(tab_dir / "figureS1_therapist_affect_direction_total_persistence.csv", index=False)

    if not client_affect.empty and not therapist_affect.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
        for idx, (ax, sub, group_col, title) in enumerate([
            (axes[0], client_affect, "client_valence_direction", "Client affect direction"),
            (axes[1], therapist_affect, "therapist_valence_direction", "Therapist affect direction"),
        ]):
            for level in ["toward_more_negative", "neutral_or_minimal_change", "toward_neutral_or_positive"]:
                dg = sub[sub[group_col] == level].sort_values("rel_time_sec")
                if dg.empty:
                    continue
                ax.plot(dg["rel_time_sec"], dg["mean"], linewidth=2.2, color=affect_palette[level], label=level.replace("_", " "))
                ax.fill_between(dg["rel_time_sec"], dg["lower"], dg["upper"], alpha=0.14, color=affect_palette[level])
            ax.axvline(0.0, color="#C43C39", linestyle="--", linewidth=1.1)
            ax.set_xlim(-30, 30)
            ax.set_title(title)
            ax.set_xlabel("Time relative to switch (sec)")
            if idx == 0:
                ax.set_ylabel("Total persistence")
            _minimal_axis(ax, show_left=(idx == 0))
            ax.text(-0.12, 1.03, chr(ord("A") + idx), transform=ax.transAxes, fontsize=13, fontweight="bold")
        axes[1].legend(loc="upper right", fontsize=8, frameon=False)
        fig.tight_layout(w_pad=1.6)
        _save_publication_svg(fig, fig_dir / "figureS1_affect_direction_total_persistence.svg")
        plt.show()

    session_cols = ["session_id", "PC1", "mean_post_switch_slope"]
    session_plot = df_session_arc_analysis[[col for col in session_cols if col in df_session_arc_analysis.columns]].dropna().copy()
    outputs["figure3_session_scatter"] = session_plot
    if not session_plot.empty:
        session_plot.to_csv(tab_dir / "figure3_session_level_pc1_scatter.csv", index=False)
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.8))
        norm = TwoSlopeNorm(vmin=session_plot["PC1"].min(), vcenter=0.0, vmax=session_plot["PC1"].max())
        scatter = ax.scatter(
            session_plot["PC1"],
            session_plot["mean_post_switch_slope"],
            c=session_plot["PC1"],
            cmap="RdBu_r",
            norm=norm,
            s=42,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        sns.regplot(
            data=session_plot,
            x="PC1",
            y="mean_post_switch_slope",
            scatter=False,
            ci=95,
            line_kws={"color": "#303030", "linewidth": 1.8},
            ax=ax,
        )
        ax.set_xlabel("Session volatility (PC1)")
        ax.set_ylabel("Mean post-switch topological slope")
        ax.set_title("Session-level moderation")
        _minimal_axis(ax)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("PC1")
        _save_publication_svg(fig, fig_dir / "figure3_session_level_moderation.svg")
        plt.show()

    return outputs


def extract_video_num_from_dyad_id(dyad_id: str):
    match = re.match(r"^(\d+)", str(dyad_id))
    return int(match.group(1)) if match else None


def build_therapist_lookup(mapping_csv: Path) -> pd.DataFrame:
    if not mapping_csv.exists():
        return pd.DataFrame(columns=["video_num", "therapist_code"])
    mapping = pd.read_csv(mapping_csv)
    required = {"therapist_code", "video_code"}
    if not required.issubset(mapping.columns):
        return pd.DataFrame(columns=["video_num", "therapist_code"])
    out = mapping[["therapist_code", "video_code"]].copy()
    out["video_num"] = pd.to_numeric(out["video_code"], errors="coerce")
    out = out.dropna(subset=["video_num"]).copy()
    out["video_num"] = out["video_num"].astype(int)
    out["therapist_code"] = out["therapist_code"].astype(str).str.strip()
    return out[["video_num", "therapist_code"]].drop_duplicates()


def build_session_geometry_scores(decoded_csv: Path) -> pd.DataFrame:
    if not decoded_csv.exists():
        return pd.DataFrame()

    df_sessions = pd.read_csv(decoded_csv)
    occ_cols = [c for c in df_sessions.columns if c.startswith("occ_")]
    K = len(occ_cols) if occ_cols else 8
    rows = []
    for _, row in df_sessions.iterrows():
        path_file = decoded_csv.parent / row["path_file"]
        if not path_file.exists():
            continue
        path = np.load(path_file)
        if len(path) < 2:
            continue

        counts = np.bincount(path, minlength=K)
        occ = counts / len(path)
        n_switches = int(np.sum(path[:-1] != path[1:]))
        switching_rate = n_switches / max(len(path) - 1, 1)

        loc_run_start = np.empty(len(path), dtype=bool)
        loc_run_start[0] = True
        np.not_equal(path[:-1], path[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        run_lengths = np.diff(np.append(run_starts, len(path)))

        mean_dwell = float(np.mean(run_lengths))
        cv_dwell = float(np.std(run_lengths) / mean_dwell) if mean_dwell > 0 else 0.0

        trans_counts = np.zeros((K, K), dtype=float)
        for t in range(len(path) - 1):
            trans_counts[path[t], path[t + 1]] += 1.0
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        trans_probs = np.divide(trans_counts, row_sums, out=np.zeros_like(trans_counts), where=row_sums != 0)
        row_ent = []
        for i in range(K):
            p = trans_probs[i]
            p = p[p > 0]
            row_ent.append(-np.sum(p * np.log2(p)) if p.size > 0 else 0.0)
        trans_entropy = float(np.sum(occ * np.asarray(row_ent)))

        occ_nonzero = occ[occ > 0]
        occ_entropy = float(-np.sum(occ_nonzero * np.log2(occ_nonzero))) if occ_nonzero.size > 0 else 0.0

        rec = {
            "session_id": str(row["session_id"]),
            "dyad_id": str(row["dyad_id"]),
            "length": int(len(path)),
            "switching_rate": float(switching_rate),
            "mean_dwell": mean_dwell,
            "cv_dwell": cv_dwell,
            "trans_entropy": trans_entropy,
            "occ_entropy": occ_entropy,
        }
        for k in range(K):
            rec[f"occ_{k}"] = float(occ[k])
        rows.append(rec)

    df_regime = pd.DataFrame(rows)
    if df_regime.empty:
        return df_regime

    pca_cols = ["switching_rate", "mean_dwell", "cv_dwell", "trans_entropy", "occ_entropy"]
    X = df_regime[pca_cols].to_numpy(dtype=float)
    X_mean = np.nanmean(X, axis=0, keepdims=True)
    X_sd = np.nanstd(X, axis=0, ddof=0, keepdims=True)
    X_sd = np.where(X_sd > 0, X_sd, 1.0)
    X_scaled = (X - X_mean) / X_sd
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    _, _, vt = np.linalg.svd(X_scaled, full_matrices=False)
    X_pca = X_scaled @ vt.T[:, :3]
    df_regime["PC1"] = X_pca[:, 0]
    df_regime["PC2"] = X_pca[:, 1]
    df_regime["PC3"] = X_pca[:, 2]
    return df_regime


def summarize_session_arc(df_growth: pd.DataFrame, df_main: pd.DataFrame, feature: str = "total_persistence_h1") -> pd.DataFrame:
    if df_growth.empty or df_main.empty:
        return pd.DataFrame()

    growth_real = df_growth[
        (df_growth["anchor_kind"] == "real")
        & (df_growth["feature"] == feature)
        & (df_growth["window"] == "post_switch_growth")
    ].copy()
    slope_summary = growth_real.groupby("session_id", as_index=False)["slope"].agg(
        mean_post_switch_slope="mean",
        median_post_switch_slope="median",
        n_real_switches="count",
    )

    d = df_main[(df_main["anchor_kind"] == "real")][["session_id", "switch_index", "phase", feature]].dropna().copy()
    sw = d.groupby(["session_id", "switch_index", "phase"], as_index=False)[feature].mean()
    wide = sw.pivot_table(index=["session_id", "switch_index"], columns="phase", values=feature).reset_index()
    if "before" in wide.columns and "after" in wide.columns:
        wide["delta_after_minus_before"] = wide["after"] - wide["before"]
        delta_summary = wide.groupby("session_id", as_index=False)["delta_after_minus_before"].agg(
            mean_after_before_delta="mean",
            median_after_before_delta="median",
        )
    else:
        delta_summary = pd.DataFrame(columns=["session_id", "mean_after_before_delta", "median_after_before_delta"])

    return slope_summary.merge(delta_summary, on="session_id", how="outer")


def run_session_geometry_analysis(
    df_main: pd.DataFrame,
    df_growth: pd.DataFrame,
    chmm_dir: Path,
    therapist_map_path: Path,
    fig_dir: Path,
    tab_dir: Path,
) -> dict[str, pd.DataFrame]:
    if df_main.empty or df_growth.empty:
        print("Need df_main and df_growth before running session-level moderation analysis.")
        empty = pd.DataFrame()
        return {
            "df_session_arc": empty,
            "df_session_geometry": empty,
            "df_session_arc_analysis": empty,
            "df_session_arc_models": empty,
        }

    df_session_geometry = build_session_geometry_scores(chmm_dir / "decoded_sessions.csv")
    df_session_arc = summarize_session_arc(df_growth, df_main, feature="total_persistence_h1")
    therapist_lookup = build_therapist_lookup(therapist_map_path)
    if not df_session_geometry.empty:
        df_session_geometry["video_num"] = df_session_geometry["dyad_id"].map(extract_video_num_from_dyad_id)
        df_session_geometry = df_session_geometry.merge(therapist_lookup, on="video_num", how="left")
        df_session_geometry["therapist_group"] = np.where(
            df_session_geometry["therapist_code"].notna(),
            "THER_" + df_session_geometry["therapist_code"].astype(str),
            "UNMAPPED_" + df_session_geometry["dyad_id"].astype(str),
        )

    keep_cols = [
        "session_id",
        "dyad_id",
        "PC1",
        "PC2",
        "PC3",
        "therapist_group",
        "switching_rate",
        "mean_dwell",
        "cv_dwell",
        "trans_entropy",
        "occ_entropy",
    ]
    df_session_arc_analysis = df_session_arc.merge(
        df_session_geometry[[col for col in keep_cols if col in df_session_geometry.columns]],
        on="session_id",
        how="left",
    )
    for col in ["PC1", "PC2"]:
        if col in df_session_arc_analysis.columns:
            sd = df_session_arc_analysis[col].std(ddof=0)
            df_session_arc_analysis[f"z_{col}"] = (
                df_session_arc_analysis[col] - df_session_arc_analysis[col].mean()
            ) / (sd if pd.notna(sd) and sd > 0 else 1.0)
    if "z_PC1" in df_session_arc_analysis.columns:
        df_session_arc_analysis["z_PC1_sq"] = df_session_arc_analysis["z_PC1"] ** 2

    model_rows = []
    for outcome in ["mean_post_switch_slope", "mean_after_before_delta"]:
        use_cols = ["session_id", "therapist_group", outcome, "z_PC1", "z_PC2", "z_PC1_sq"]
        dmod = df_session_arc_analysis[[col for col in use_cols if col in df_session_arc_analysis.columns]].dropna().copy()
        if len(dmod) < 25 or dmod["therapist_group"].nunique() < 5:
            continue
        for model_name, fixed_cols in [
            ("linear", ["z_PC1", "z_PC2"]),
            ("quadratic_pc1", ["z_PC1", "z_PC1_sq", "z_PC2"]),
        ]:
            X = sm.add_constant(dmod[fixed_cols], has_constant="add")
            try:
                fit = MixedLM(endog=dmod[outcome], exog=X, groups=dmod["therapist_group"]).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=300,
                )
                row = {
                    "outcome": outcome,
                    "model_name": model_name,
                    "n_sessions": int(len(dmod)),
                    "n_therapists": int(dmod["therapist_group"].nunique()),
                    "aic": float(getattr(fit, "aic", np.nan)),
                    "bic": float(getattr(fit, "bic", np.nan)),
                }
                for coef in fit.params.index:
                    row[f"coef_{coef}"] = float(fit.params[coef])
                    row[f"p_{coef}"] = float(fit.pvalues.get(coef, np.nan))
                model_rows.append(row)
            except Exception as exc:
                print(f"Could not fit session-level moderation model for {outcome}, {model_name}: {exc}")

    df_session_arc_models = pd.DataFrame(model_rows)
    display(df_session_arc_analysis.head(10) if not df_session_arc_analysis.empty else pd.DataFrame())
    display(df_session_arc_models)
    if not df_session_arc_analysis.empty:
        df_session_arc_analysis.to_csv(tab_dir / "session_arc_geometry_analysis_table.csv", index=False)
    if not df_session_arc_models.empty:
        df_session_arc_models.to_csv(tab_dir / "session_arc_geometry_mixed_models.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if "mean_post_switch_slope" in df_session_arc_analysis.columns:
            sns.regplot(
                data=df_session_arc_analysis,
                x="PC1",
                y="mean_post_switch_slope",
                scatter_kws={"s": 30, "alpha": 0.6},
                line_kws={"color": "black"},
                ax=axes[0],
            )
            axes[0].set_title("Session arc magnitude vs PC1")
            axes[0].set_xlabel("PC1 (volatility)")
            axes[0].set_ylabel("Mean post-switch slope")
        if "mean_after_before_delta" in df_session_arc_analysis.columns:
            sns.regplot(
                data=df_session_arc_analysis,
                x="PC2",
                y="mean_after_before_delta",
                scatter_kws={"s": 30, "alpha": 0.6},
                line_kws={"color": "black"},
                ax=axes[1],
            )
            axes[1].set_title("Session arc magnitude vs PC2")
            axes[1].set_xlabel("PC2 (irregularity)")
            axes[1].set_ylabel("Mean after-before delta")
        plt.tight_layout()
        plt.savefig(fig_dir / "session_arc_vs_geometry.svg", format="svg")
        plt.show()

    return {
        "df_session_arc": df_session_arc,
        "df_session_geometry": df_session_geometry,
        "df_session_arc_analysis": df_session_arc_analysis,
        "df_session_arc_models": df_session_arc_models,
    }


def build_next_switch_outcomes(chmm_dir: Path) -> pd.DataFrame:
    dec_path = chmm_dir / "decoded_sessions.csv"
    if not dec_path.exists():
        return pd.DataFrame()

    dec = pd.read_csv(dec_path)
    rows = []
    for _, row in dec.iterrows():
        path_file = chmm_dir / row["path_file"]
        if not path_file.exists():
            continue
        session_id = str(row["session_id"])
        path = np.load(path_file).astype(int)
        if len(path) < 2:
            continue

        switch_events = []
        switch_counter = 0
        for t in range(1, len(path)):
            if path[t] != path[t - 1]:
                from_state = int(path[t - 1])
                to_state = int(path[t])
                run_end = t + 1
                while run_end < len(path) and path[run_end] == path[t]:
                    run_end += 1
                switch_events.append(
                    {
                        "session_id": session_id,
                        "switch_index": switch_counter,
                        "switch_time_idx": int(t),
                        "from_state": from_state,
                        "to_state": to_state,
                        "next_dwell_steps": int(run_end - t),
                    }
                )
                switch_counter += 1

        for i, event in enumerate(switch_events):
            next_event = switch_events[i + 1] if i + 1 < len(switch_events) else None
            rows.append(
                {
                    **event,
                    "has_next_switch": bool(next_event is not None),
                    "next_transition_identity": f"{next_event['from_state']}->{next_event['to_state']}" if next_event is not None else np.nan,
                    "next_to_state": int(next_event["to_state"]) if next_event is not None else np.nan,
                    "next_switch_index": int(next_event["switch_index"]) if next_event is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_subsequent_dynamics_analysis(
    df_main: pd.DataFrame,
    df_growth: pd.DataFrame,
    df_session_geometry: pd.DataFrame,
    chmm_dir: Path,
    fig_dir: Path,
    tab_dir: Path,
) -> dict[str, pd.DataFrame]:
    if df_main.empty or df_growth.empty:
        print("Need df_main and df_growth before running subsequent-dynamics analysis.")
        empty = pd.DataFrame()
        return {
            "df_next_outcomes": empty,
            "df_dwell_analysis": empty,
            "df_dwell_models": empty,
            "df_next_transition_summary": empty,
            "df_next_state_models": empty,
        }

    df_next_outcomes = build_next_switch_outcomes(chmm_dir)
    post_growth = df_growth[
        (df_growth["anchor_kind"] == "real")
        & (df_growth["window"] == "post_switch_growth")
        & (df_growth["feature"] == "total_persistence_h1")
    ][["session_id", "switch_index", "slope", "delta_end_minus_start", "posterior_jump_l2"]].copy()
    post_growth = post_growth.rename(
        columns={
            "slope": "post_switch_growth_slope",
            "delta_end_minus_start": "post_switch_growth_delta",
        }
    )
    pre_growth = df_growth[
        (df_growth["anchor_kind"] == "real")
        & (df_growth["window"] == "pre_switch_growth")
        & (df_growth["feature"] == "total_persistence_h1")
    ][["session_id", "switch_index", "slope"]].copy().rename(columns={"slope": "pre_switch_growth_slope"})

    df_dwell_analysis = df_next_outcomes.merge(post_growth, on=["session_id", "switch_index"], how="left")
    df_dwell_analysis = df_dwell_analysis.merge(pre_growth, on=["session_id", "switch_index"], how="left")
    if not df_session_geometry.empty:
        df_dwell_analysis = df_dwell_analysis.merge(
            df_session_geometry[[col for col in ["session_id", "PC1", "PC2", "therapist_group"] if col in df_session_geometry.columns]],
            on="session_id",
            how="left",
        )
    if "next_dwell_steps" in df_dwell_analysis.columns:
        df_dwell_analysis["log_next_dwell_steps"] = np.log1p(df_dwell_analysis["next_dwell_steps"])

    for col in ["post_switch_growth_slope", "post_switch_growth_delta", "posterior_jump_l2", "log_next_dwell_steps"]:
        if col in df_dwell_analysis.columns:
            sd = df_dwell_analysis[col].std(ddof=0)
            df_dwell_analysis[f"z_{col}"] = (
                df_dwell_analysis[col] - df_dwell_analysis[col].mean()
            ) / (sd if pd.notna(sd) and sd > 0 else 1.0)

    model_rows = []
    model_specs = [
        ("dwell_vs_post_slope", ["z_post_switch_growth_slope"]),
        ("dwell_vs_post_slope_plus_jump", ["z_post_switch_growth_slope", "z_posterior_jump_l2"]),
        ("dwell_vs_post_delta_plus_jump", ["z_post_switch_growth_delta", "z_posterior_jump_l2"]),
    ]
    dmod_base = df_dwell_analysis[df_dwell_analysis["has_next_switch"]].copy() if "has_next_switch" in df_dwell_analysis.columns else pd.DataFrame()
    for model_name, predictors in model_specs:
        use_cols = ["session_id", "log_next_dwell_steps"] + predictors
        dmod = dmod_base[[col for col in use_cols if col in dmod_base.columns]].dropna().copy()
        if len(dmod) < 30 or dmod["session_id"].nunique() < 8:
            continue
        X = sm.add_constant(dmod[predictors], has_constant="add")
        try:
            fit = MixedLM(endog=dmod["log_next_dwell_steps"], exog=X, groups=dmod["session_id"]).fit(
                reml=False,
                method="lbfgs",
                maxiter=300,
            )
            row = {
                "model_name": model_name,
                "n_switches": int(len(dmod)),
                "n_sessions": int(dmod["session_id"].nunique()),
                "aic": float(getattr(fit, "aic", np.nan)),
                "bic": float(getattr(fit, "bic", np.nan)),
            }
            for coef in fit.params.index:
                row[f"coef_{coef}"] = float(fit.params[coef])
                row[f"p_{coef}"] = float(fit.pvalues.get(coef, np.nan))
            model_rows.append(row)
        except Exception as exc:
            print(f"Could not fit dwell model {model_name}: {exc}")
    df_dwell_models = pd.DataFrame(model_rows)

    if "next_transition_identity" in df_dwell_analysis.columns:
        df_next_transition_summary = (
            df_dwell_analysis.dropna(subset=["next_transition_identity"])
            .groupby("next_transition_identity", as_index=False)
            .agg(
                n_switches=("session_id", "size"),
                mean_pre_switch_growth_slope=("pre_switch_growth_slope", "mean"),
                mean_post_switch_growth_slope=("post_switch_growth_slope", "mean"),
                mean_next_dwell_steps=("next_dwell_steps", "mean"),
                mean_log_next_dwell_steps=("log_next_dwell_steps", "mean"),
            )
            .sort_values("n_switches", ascending=False)
        )
    else:
        df_next_transition_summary = pd.DataFrame()

    if "next_to_state" in df_dwell_analysis.columns:
        state_df = df_dwell_analysis.dropna(subset=["next_to_state", "pre_switch_growth_slope", "posterior_jump_l2"]).copy()
        keep_states = state_df["next_to_state"].value_counts()
        keep_states = keep_states[keep_states >= 20].index
        state_df = state_df[state_df["next_to_state"].isin(keep_states)].copy()
        if len(keep_states) >= 2 and len(state_df) >= 60:
            sd = state_df["pre_switch_growth_slope"].std(ddof=0)
            state_df["z_pre_switch_growth_slope"] = (
                state_df["pre_switch_growth_slope"] - state_df["pre_switch_growth_slope"].mean()
            ) / (sd if pd.notna(sd) and sd > 0 else 1.0)
            try:
                X_state = sm.add_constant(state_df[["z_pre_switch_growth_slope", "z_posterior_jump_l2"]], has_constant="add")
                mn_fit = sm.MNLogit(state_df["next_to_state"].astype(int), X_state).fit(method="newton", maxiter=200, disp=False)
                mn_rows = []
                for outcome in mn_fit.params.columns:
                    row = {"next_to_state_level": str(outcome)}
                    for coef in mn_fit.params.index:
                        row[f"coef_{coef}"] = float(mn_fit.params.loc[coef, outcome])
                        row[f"p_{coef}"] = float(mn_fit.pvalues.loc[coef, outcome])
                    mn_rows.append(row)
                df_next_state_models = pd.DataFrame(mn_rows)
            except Exception as exc:
                print(f"Could not fit exploratory next-state model: {exc}")
                df_next_state_models = pd.DataFrame()
        else:
            df_next_state_models = pd.DataFrame()
    else:
        df_next_state_models = pd.DataFrame()

    display(df_dwell_analysis.head(10) if not df_dwell_analysis.empty else pd.DataFrame())
    display(df_dwell_models)
    display(df_next_transition_summary.head(15) if not df_next_transition_summary.empty else pd.DataFrame())
    display(df_next_state_models)

    if not df_dwell_analysis.empty:
        df_dwell_analysis.to_csv(tab_dir / "switch_subsequent_dynamics_analysis_table.csv", index=False)
    if not df_dwell_models.empty:
        df_dwell_models.to_csv(tab_dir / "switch_subsequent_dynamics_mixed_models.csv", index=False)
    if not df_next_transition_summary.empty:
        df_next_transition_summary.to_csv(tab_dir / "next_transition_topology_summary.csv", index=False)
    if not df_next_state_models.empty:
        df_next_state_models.to_csv(tab_dir / "next_state_prediction_multinomial.csv", index=False)

    if not df_dwell_analysis.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        scatter_df = df_dwell_analysis.dropna(subset=["post_switch_growth_slope", "log_next_dwell_steps"]).copy()
        if not scatter_df.empty:
            sns.regplot(
                data=scatter_df,
                x="post_switch_growth_slope",
                y="log_next_dwell_steps",
                scatter_kws={"s": 18, "alpha": 0.35},
                line_kws={"color": "black"},
                ax=axes[0],
            )
            axes[0].set_title("Log next dwell vs post-switch slope")
            axes[0].set_xlabel("Post-switch slope")
            axes[0].set_ylabel("log1p(next dwell steps)")
        jump_df = df_dwell_analysis.dropna(subset=["posterior_jump_l2", "log_next_dwell_steps"]).copy()
        if not jump_df.empty:
            sns.regplot(
                data=jump_df,
                x="posterior_jump_l2",
                y="log_next_dwell_steps",
                scatter_kws={"s": 18, "alpha": 0.35},
                line_kws={"color": "black"},
                ax=axes[1],
            )
            axes[1].set_title("Log next dwell vs posterior jump")
            axes[1].set_xlabel("Posterior jump L2")
            axes[1].set_ylabel("log1p(next dwell steps)")
        plt.tight_layout()
        plt.savefig(fig_dir / "switch_subsequent_dynamics_scatter.svg", format="svg")
        plt.show()

    if not df_next_transition_summary.empty:
        plot_df = df_next_transition_summary.head(12).copy()
        plt.figure(figsize=(9, 5))
        sns.barplot(data=plot_df, x="mean_log_next_dwell_steps", y="next_transition_identity", color="steelblue")
        plt.title("Mean log next dwell by next transition identity")
        plt.xlabel("Mean log1p(next dwell steps)")
        plt.ylabel("Next transition identity")
        plt.tight_layout()
        plt.savefig(fig_dir / "next_transition_topology_summary.svg", format="svg")
        plt.show()

    return {
        "df_next_outcomes": df_next_outcomes,
        "df_dwell_analysis": df_dwell_analysis,
        "df_dwell_models": df_dwell_models,
        "df_next_transition_summary": df_next_transition_summary,
        "df_next_state_models": df_next_state_models,
    }
