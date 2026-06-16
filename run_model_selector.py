"""
Per-SKU model selector with rolling-origin cross-validation.

Setup (each fold = 13 weeks ≈ 90 days):
  - Fold 0 (oldest CV)  : train [...-39w), test [-39w, -26w)
  - Fold 1 (mid CV)     : train [...-26w), test [-26w, -13w)
  - Fold 2 (newest CV)  : train [...-13w), test [-13w, end)            <-- "holdout fold"
We use folds 0+1 for model selection and fold 2 as the held-out
honest evaluation, so the selector never sees the data it is graded on.

For each (SKU, metric) we score 5 base models on the CV folds, pick the
model with the lowest mean sMAPE, then evaluate the SELECTOR vs every
individual model on the held-out fold.

Models: Prophet, AutoARIMA, AutoETS, CrostonOptimized, LightGBM_global.
LightGBM is trained globally per (metric, fold) over all top-K SKUs.

Usage:
    python run_model_selector.py                   # 3 folds (2 CV + 1 eval)
    python run_model_selector.py --n-folds 4 --horizon-weeks 13
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
OUT_DIR = Path("/home/user/Automation/forecast_out/selector")
SEASON_W = 52
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


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def make_folds(last_date: pd.Timestamp, n_folds: int, horizon_weeks: int) -> list[pd.Timestamp]:
    """Return cutoff dates (oldest to newest). Each fold tests the
    horizon_weeks immediately at/after the cutoff."""
    cutoffs = []
    for i in range(n_folds, 0, -1):
        c = last_date - pd.Timedelta(weeks=i * horizon_weeks - 1)
        cutoffs.append(c)
    return cutoffs  # cutoffs[0] = oldest, cutoffs[-1] = most recent


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

    preds = {"Prophet": prophet_weekly(train_w, h_sku)}
    preds.update(statsforecast_one(train_w, h_sku))
    lgb_sku = lgb_fcst[lgb_fcst["unique_id"] == sku].sort_values("ds")
    preds["LightGBM"] = lgb_sku["LightGBM"].to_numpy()[:h_sku]

    # Align lengths
    for k, p in preds.items():
        p = np.asarray(p)
        if len(p) < h_sku:
            p = np.concatenate([p, np.full(h_sku - len(p), p[-1] if len(p) else 0.0)])
        preds[k] = p[:h_sku]

    return {name: err_metrics(actual, yhat) for name, yhat in preds.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--train-top", type=int, default=30)
    ap.add_argument("--score-top", type=int, default=10)
    ap.add_argument("--n-folds", type=int, default=3,
                    help="Total CV folds incl. the held-out eval fold (last one)")
    ap.add_argument("--horizon-weeks", type=int, default=13)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    revenue = df.groupby("productName")["totalValue"].sum().sort_values(ascending=False)
    train_skus = revenue.head(args.train_top).index.tolist()
    score_skus = revenue.head(args.score_top).index.tolist()

    # Build panels once per metric
    panels = {m: build_weekly_panel(df, train_skus, m) for m in ["totalValue", "qty"]}
    last_date = panels["totalValue"]["ds"].max()
    cutoffs = make_folds(last_date, args.n_folds, args.horizon_weeks)
    cv_cutoffs = cutoffs[:-1]
    eval_cutoff = cutoffs[-1]
    print(f"Last date: {last_date.date()}")
    print(f"CV folds:  {[c.date() for c in cv_cutoffs]}")
    print(f"Held-out:  {eval_cutoff.date()}")

    all_rows = []

    for metric in ["totalValue", "qty"]:
        print(f"\n=== Metric: {metric} ===")
        panel = panels[metric]
        # Pre-fit global LightGBM per fold
        lgb_per_fold: dict[pd.Timestamp, pd.DataFrame] = {}
        for c in cutoffs:
            print(f"  LightGBM global fit @ cutoff {c.date()}...", end=" ", flush=True)
            lgb_per_fold[c] = lightgbm_global(panel, c, args.horizon_weeks)
            print("done")

        for c_idx, c in enumerate(cutoffs):
            kind = "EVAL" if c == eval_cutoff else f"CV-{c_idx}"
            print(f"\n  Fold {kind} (cutoff {c.date()}):")
            for rank, sku in enumerate(score_skus, start=1):
                scores = score_sku_fold(sku, panel, c, lgb_per_fold[c], args.horizon_weeks)
                if not scores:
                    continue
                for model_name, mm in scores.items():
                    all_rows.append({
                        "rank": rank, "product": sku, "metric": metric,
                        "fold": kind, "cutoff": c.date().isoformat(),
                        "model": model_name, **mm,
                    })

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / "_all_fold_metrics.csv", index=False)

    # === Selection: average sMAPE across CV folds per (sku, metric, model)
    cv = df_all[df_all["fold"].str.startswith("CV")]
    cv_avg = cv.groupby(["rank", "product", "metric", "model"])["sMAPE"].mean().reset_index()
    chosen = (cv_avg.sort_values("sMAPE")
              .groupby(["rank", "product", "metric"])
              .first()[["model", "sMAPE"]]
              .rename(columns={"model": "chosen_model", "sMAPE": "cv_mean_sMAPE"})
              .reset_index())
    chosen.to_csv(OUT_DIR / "_chosen_models.csv", index=False)
    print("\n=== Chosen model per (SKU, metric) — selected on CV folds ===")
    print(chosen.to_string(index=False))

    # === Honest evaluation on the held-out fold
    eval_df = df_all[df_all["fold"] == "EVAL"]
    # Selector row: pick chosen_model's eval performance
    sel_rows = []
    for _, row in chosen.iterrows():
        match = eval_df[(eval_df["product"] == row["product"]) &
                       (eval_df["metric"] == row["metric"]) &
                       (eval_df["model"] == row["chosen_model"])]
        if len(match) == 1:
            r = match.iloc[0].to_dict()
            r["model"] = "Selector"
            sel_rows.append(r)
    selector_eval = pd.DataFrame(sel_rows)
    eval_all = pd.concat([eval_df, selector_eval], ignore_index=True)
    eval_all.to_csv(OUT_DIR / "_eval_fold_metrics.csv", index=False)

    print("\n=== Holdout fold — by model ===")
    avg = (eval_all.groupby("model")
           .agg(mean_sMAPE=("sMAPE", "mean"),
                median_sMAPE=("sMAPE", "median"),
                mean_absSumPctErr=("SumPctErr", lambda x: x.abs().mean()),
                median_absSumPctErr=("SumPctErr", lambda x: x.abs().median()))
           .round(1).sort_values("mean_sMAPE"))
    print(avg.to_string())
    avg.to_csv(OUT_DIR / "_holdout_summary.csv")

    print("\n=== Selector vs best single model on holdout, per SKU/metric ===")
    base = (eval_df.sort_values("sMAPE")
            .groupby(["rank", "product", "metric"])
            .first()[["model", "sMAPE", "SumPctErr"]]
            .rename(columns={"model": "best_single",
                             "sMAPE": "best_single_sMAPE",
                             "SumPctErr": "best_single_SumErr"})
            .reset_index())
    sel = (selector_eval[["rank", "product", "metric", "sMAPE", "SumPctErr"]]
           .rename(columns={"sMAPE": "selector_sMAPE",
                            "SumPctErr": "selector_SumErr"}))
    merged = base.merge(sel, on=["rank", "product", "metric"]).merge(
        chosen[["rank", "product", "metric", "chosen_model"]], on=["rank", "product", "metric"]
    )
    merged["delta_sMAPE"] = merged["selector_sMAPE"] - merged["best_single_sMAPE"]
    print(merged.to_string(index=False))
    merged.to_csv(OUT_DIR / "_selector_vs_best.csv", index=False)

    # Plot: model accuracy by SKU on the eval fold
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, metric in zip(axes, ["totalValue", "qty"]):
        m_eval = eval_all[eval_all["metric"] == metric].copy()
        pivot = m_eval.pivot(index="product", columns="model", values="sMAPE")
        order = revenue.head(args.score_top).index.tolist()
        pivot = pivot.loc[[p for p in order if p in pivot.index]]
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"sMAPE on held-out fold — {metric}")
        ax.set_xlabel("")
        ax.set_ylabel("sMAPE %")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "_holdout_sMAPE_by_sku.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
