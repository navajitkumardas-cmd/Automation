"""
Per-SKU model selector v2: more folds, recency-weighted, with a safety gate.

Improvements over run_model_selector.py:
  * 5 rolling-origin folds (4 CV + 1 honest held-out eval), each 13 weeks
  * CV folds weighted by recency (exponential: w = 2**i, newest fold heaviest)
  * Safety gate: switch away from the default model (Croston) only if a
    candidate beats it by >= GATE_MARGIN percentage points sMAPE in CV.
    Otherwise stay on Croston.

The default is Croston because it was the best single-model average in
every prior benchmark; the gate prevents the selector from making the
"chase noise" mistake we saw in v1.

Usage:
    python run_model_selector_v2.py                # 4 CV + 1 eval, gate=5pp
    python run_model_selector_v2.py --gate 3 --n-folds 6
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, CrostonOptimized
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

warnings.filterwarnings("ignore")

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out/selector_v2")
SEASON_W = 52
DEFAULT_MODEL = "Croston"
MODELS = ["Prophet", "AutoARIMA", "AutoETS", "Croston", "LightGBM"]


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df = df.dropna(subset=["createdAt"])
    df["date"] = df["createdAt"].dt.normalize()
    return df


def build_weekly_panel(df: pd.DataFrame, products: list[str], metric: str) -> pd.DataFrame:
    sub = df[df["productName"].isin(products)].copy()
    daily = sub.groupby(["productName", "date"], as_index=False)[metric].sum()
    daily.columns = ["unique_id", "ds", "y"]
    frames = []
    for sku, g in daily.groupby("unique_id"):
        full = pd.date_range(g["ds"].min(), g["ds"].max(), freq="D")
        gr = g.set_index("ds").reindex(full, fill_value=0).rename_axis("ds").reset_index()
        gr["unique_id"] = sku
        frames.append(gr)
    daily = pd.concat(frames, ignore_index=True)
    weekly = (daily.set_index("ds")
              .groupby("unique_id")
              .resample("W-MON", label="left", closed="left")["y"].sum()
              .reset_index())
    return weekly[["unique_id", "ds", "y"]]


def err_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(pred, 0, None)
    err = pred - actual
    abs_err = np.abs(err)
    denom = (np.abs(actual) + np.abs(pred)) / 2
    mask = denom > 0
    s_act, s_pred = actual.sum(), pred.sum()
    return dict(
        MAE=float(abs_err.mean()),
        RMSE=float(np.sqrt((err ** 2).mean())),
        sMAPE=float((abs_err[mask] / denom[mask]).mean() * 100) if mask.any() else float("nan"),
        ActualSum=float(s_act),
        ForecastSum=float(s_pred),
        SumPctErr=float((s_pred - s_act) / s_act * 100) if s_act > 0 else float("nan"),
    )


def prophet_weekly(train_w: pd.DataFrame, h: int) -> np.ndarray:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.05)
    m.fit(train_w[["ds", "y"]])
    future = m.make_future_dataframe(periods=h, freq="W-MON")
    return m.predict(future).tail(h)["yhat"].to_numpy()


def statsforecast_one(train_w: pd.DataFrame, h: int) -> dict[str, np.ndarray]:
    sf_df = train_w[["ds", "y"]].assign(unique_id="s1")[["unique_id", "ds", "y"]]
    sf = StatsForecast(models=[AutoARIMA(season_length=SEASON_W),
                               AutoETS(season_length=SEASON_W),
                               CrostonOptimized()], freq="W-MON", n_jobs=1)
    f = sf.forecast(df=sf_df, h=h)
    return {"AutoARIMA": f["AutoARIMA"].to_numpy(),
            "AutoETS": f["AutoETS"].to_numpy(),
            "Croston": f["CrostonOptimized"].to_numpy()}


def lightgbm_global(panel: pd.DataFrame, cutoff: pd.Timestamp, h: int) -> pd.DataFrame:
    train = panel[panel["ds"] < cutoff].copy()
    lgbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                             min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbose=-1)
    mf = MLForecast(
        models={"LightGBM": lgbm}, freq="W-MON",
        lags=[1, 2, 4, 8, 13, 26, 52],
        lag_transforms={1: [RollingMean(window_size=4), RollingMean(window_size=13)],
                        4: [RollingMean(window_size=4)]},
        date_features=["month", "quarter", "year"],
    )
    mf.fit(train)
    return mf.predict(h=h)


def make_folds(last_date: pd.Timestamp, n_folds: int, horizon_weeks: int) -> list[pd.Timestamp]:
    cutoffs = []
    for i in range(n_folds, 0, -1):
        c = last_date - pd.Timedelta(weeks=i * horizon_weeks - 1)
        cutoffs.append(c)
    return cutoffs


def score_sku_fold(sku: str, panel: pd.DataFrame, cutoff: pd.Timestamp,
                   lgb_fcst: pd.DataFrame, horizon_weeks: int) -> dict[str, dict]:
    sku_panel = panel[panel["unique_id"] == sku].sort_values("ds")
    train_w = sku_panel[sku_panel["ds"] < cutoff][["ds", "y"]].assign(unique_id=sku)[
        ["unique_id", "ds", "y"]
    ]
    hold = sku_panel[sku_panel["ds"] >= cutoff].head(horizon_weeks)
    if len(hold) < 2 or len(train_w) < 30:
        return {}
    actual = hold["y"].to_numpy()
    h_sku = len(actual)

    preds: dict[str, np.ndarray] = {}
    try:
        preds["Prophet"] = prophet_weekly(train_w, h_sku)
    except Exception:
        preds["Prophet"] = np.full(h_sku, train_w["y"].tail(13).mean())
    try:
        preds.update(statsforecast_one(train_w, h_sku))
    except Exception:
        mean = train_w["y"].tail(13).mean()
        preds["AutoARIMA"] = np.full(h_sku, mean)
        preds["AutoETS"] = np.full(h_sku, mean)
        preds["Croston"] = np.full(h_sku, mean)

    lgb_sku = lgb_fcst[lgb_fcst["unique_id"] == sku].sort_values("ds")
    preds["LightGBM"] = lgb_sku["LightGBM"].to_numpy()[:h_sku]

    for k, p in preds.items():
        p = np.asarray(p, dtype=float)
        if len(p) < h_sku:
            p = np.concatenate([p, np.full(h_sku - len(p), p[-1] if len(p) else 0.0)])
        preds[k] = p[:h_sku]

    return {name: err_metrics(actual, yhat) for name, yhat in preds.items()}


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--train-top", type=int, default=30)
    ap.add_argument("--score-top", type=int, default=10)
    ap.add_argument("--n-folds", type=int, default=5,
                    help="Total folds incl. held-out eval (last). 5 → 4 CV + 1 eval")
    ap.add_argument("--horizon-weeks", type=int, default=13)
    ap.add_argument("--gate", type=float, default=5.0,
                    help="pp sMAPE margin a candidate must beat the default by")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    revenue = df.groupby("productName")["totalValue"].sum().sort_values(ascending=False)
    train_skus = revenue.head(args.train_top).index.tolist()
    score_skus = revenue.head(args.score_top).index.tolist()

    panels = {m: build_weekly_panel(df, train_skus, m) for m in ["totalValue", "qty"]}
    last_date = panels["totalValue"]["ds"].max()
    cutoffs = make_folds(last_date, args.n_folds, args.horizon_weeks)
    cv_cutoffs = cutoffs[:-1]
    eval_cutoff = cutoffs[-1]
    print(f"Last date: {last_date.date()}")
    print(f"CV folds ({len(cv_cutoffs)}): {[c.date() for c in cv_cutoffs]}")
    print(f"Held-out:                 {eval_cutoff.date()}")
    print(f"Default = {DEFAULT_MODEL}, gate margin = {args.gate} pp\n")

    all_rows = []
    for metric in ["totalValue", "qty"]:
        print(f"=== Metric: {metric} ===")
        panel = panels[metric]
        lgb_per_fold = {}
        for c in cutoffs:
            print(f"  LightGBM global @ {c.date()}", end=" ", flush=True)
            lgb_per_fold[c] = lightgbm_global(panel, c, args.horizon_weeks)
            print("ok")

        for c_idx, c in enumerate(cutoffs):
            kind = "EVAL" if c == eval_cutoff else f"CV-{c_idx}"
            for rank, sku in enumerate(score_skus, start=1):
                scores = score_sku_fold(sku, panel, c, lgb_per_fold[c], args.horizon_weeks)
                if not scores:
                    continue
                for model_name, mm in scores.items():
                    all_rows.append({"rank": rank, "product": sku, "metric": metric,
                                     "fold": kind, "fold_idx": c_idx,
                                     "cutoff": c.date().isoformat(),
                                     "model": model_name, **mm})

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / "_all_fold_metrics.csv", index=False)

    # === Recency-weighted CV selection
    cv = df_all[df_all["fold"].str.startswith("CV")].copy()
    n_cv = cv["fold_idx"].nunique()
    # weights: newest CV fold heaviest. fold_idx is 0..n_cv-1 oldest→newest
    weight_map = {i: 2 ** i for i in range(n_cv)}
    cv["weight"] = cv["fold_idx"].map(weight_map)
    cv["wsMAPE"] = cv["sMAPE"] * cv["weight"]

    grouped = cv.groupby(["rank", "product", "metric", "model"]).agg(
        wsum=("wsMAPE", "sum"), w=("weight", "sum")
    )
    grouped["weighted_sMAPE"] = grouped["wsum"] / grouped["w"]
    weighted = grouped["weighted_sMAPE"].reset_index()
    weighted.to_csv(OUT_DIR / "_cv_weighted_sMAPE.csv", index=False)

    # === Gated choice
    chosen_rows = []
    for (rank, sku, metric), grp in weighted.groupby(["rank", "product", "metric"]):
        scores = dict(zip(grp["model"], grp["weighted_sMAPE"]))
        default_score = scores.get(DEFAULT_MODEL, float("inf"))
        best_model = min(scores, key=scores.get)
        best_score = scores[best_model]
        if best_model == DEFAULT_MODEL or (default_score - best_score) < args.gate:
            chosen = DEFAULT_MODEL
            reason = f"gate ({default_score - best_score:.1f}pp < {args.gate})"
        else:
            chosen = best_model
            reason = f"beats {DEFAULT_MODEL} by {default_score - best_score:.1f}pp"
        chosen_rows.append({
            "rank": rank, "product": sku, "metric": metric,
            "chosen_model": chosen,
            "default_cv_sMAPE": default_score,
            "best_cv_sMAPE": best_score, "best_cv_model": best_model,
            "reason": reason,
        })
    chosen = pd.DataFrame(chosen_rows).sort_values(["rank", "metric"])
    chosen.to_csv(OUT_DIR / "_chosen_models.csv", index=False)
    print("\n=== Chosen model per (SKU, metric) — gated selector ===")
    print(chosen.to_string(index=False))

    # === Honest evaluation on the held-out fold
    eval_df = df_all[df_all["fold"] == "EVAL"]
    sel_rows = []
    for _, row in chosen.iterrows():
        match = eval_df[(eval_df["product"] == row["product"]) &
                       (eval_df["metric"] == row["metric"]) &
                       (eval_df["model"] == row["chosen_model"])]
        if len(match) == 1:
            r = match.iloc[0].to_dict()
            r["model"] = "Selector_v2"
            sel_rows.append(r)
    selector_eval = pd.DataFrame(sel_rows)
    eval_all = pd.concat([eval_df, selector_eval], ignore_index=True)
    eval_all.to_csv(OUT_DIR / "_eval_fold_metrics.csv", index=False)

    print("\n=== Holdout fold — by model (lower is better) ===")
    avg = (eval_all.groupby("model")
           .agg(mean_sMAPE=("sMAPE", "mean"),
                median_sMAPE=("sMAPE", "median"),
                mean_absSumPctErr=("SumPctErr", lambda x: x.abs().mean()),
                median_absSumPctErr=("SumPctErr", lambda x: x.abs().median()))
           .round(1).sort_values("mean_sMAPE"))
    print(avg.to_string())
    avg.to_csv(OUT_DIR / "_holdout_summary.csv")

    # Selector vs always-Croston, side by side
    base = (eval_df[eval_df["model"] == DEFAULT_MODEL]
            [["rank", "product", "metric", "sMAPE", "SumPctErr"]]
            .rename(columns={"sMAPE": "croston_sMAPE", "SumPctErr": "croston_SumErr"}))
    sel = (selector_eval[["rank", "product", "metric", "sMAPE", "SumPctErr"]]
           .rename(columns={"sMAPE": "selector_sMAPE", "SumPctErr": "selector_SumErr"}))
    cmp = base.merge(sel, on=["rank", "product", "metric"]).merge(
        chosen[["rank", "product", "metric", "chosen_model", "reason"]],
        on=["rank", "product", "metric"]
    )
    cmp["delta_sMAPE"] = cmp["selector_sMAPE"] - cmp["croston_sMAPE"]
    cmp["delta_absSumErr"] = cmp["selector_SumErr"].abs() - cmp["croston_SumErr"].abs()
    cmp.to_csv(OUT_DIR / "_selector_vs_croston.csv", index=False)
    print("\n=== Selector vs always-Croston on held-out fold ===")
    print(cmp.to_string(index=False))

    # Plot: per-SKU sMAPE on eval fold
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, metric in zip(axes, ["totalValue", "qty"]):
        sub = eval_all[eval_all["metric"] == metric]
        pivot = sub.pivot(index="product", columns="model", values="sMAPE")
        order = revenue.head(args.score_top).index.tolist()
        pivot = pivot.loc[[p for p in order if p in pivot.index]]
        pivot.plot(kind="bar", ax=ax, width=0.85)
        ax.set_title(f"Held-out fold sMAPE — {metric}  (gate={args.gate}pp, {n_cv} CV folds)")
        ax.set_xlabel("")
        ax.set_ylabel("sMAPE %")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "_holdout_sMAPE_by_sku.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Choice count
    print("\n=== Selector picks ===")
    print(chosen["chosen_model"].value_counts().to_string())


if __name__ == "__main__":
    main()
