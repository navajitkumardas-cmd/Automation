"""
Option B: Global LightGBM forecast across many SKUs with mlforecast.

Trains ONE LightGBM model per metric on the top-K SKUs (panel data, weekly
aggregation), with engineered lag features + date features + festival
indicators. Same 90-day (13-week) holdout used everywhere else, so results
are directly comparable to:
  - Prophet weekly (backtest_v2)
  - AutoARIMA / AutoETS / Croston (statsforecast)

Outputs:
  - per-SKU comparison plot (LightGBM vs the previous winners)
  - metrics CSV for top-N (scored SKU subset)
  - average-by-model summary (combined with previous benchmark)

Usage:
    python run_lightgbm_global.py                              # train K=30, score top 10
    python run_lightgbm_global.py --train-top 50 --score-top 10
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
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, ExpandingMean

warnings.filterwarnings("ignore")

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out/lightgbm")
PREV_METRICS = Path("/home/user/Automation/forecast_out/statsforecast/_metrics_top10_compare.csv")


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df = df.dropna(subset=["createdAt"])
    df["date"] = df["createdAt"].dt.normalize()
    return df


def build_weekly_panel(df: pd.DataFrame, products: list[str], metric: str) -> pd.DataFrame:
    """Return long-format weekly panel: unique_id, ds, y."""
    sub = df[df["productName"].isin(products)].copy()
    daily = sub.groupby(["productName", "date"], as_index=False)[metric].sum()
    daily.columns = ["unique_id", "ds", "y"]

    # Reindex each SKU on its full date range, fill missing with 0
    frames = []
    for sku, g in daily.groupby("unique_id"):
        full = pd.date_range(g["ds"].min(), g["ds"].max(), freq="D")
        gr = g.set_index("ds").reindex(full, fill_value=0).rename_axis("ds").reset_index()
        gr["unique_id"] = sku
        frames.append(gr)
    daily = pd.concat(frames, ignore_index=True)

    # Resample to weekly per SKU
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


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def make_model() -> MLForecast:
    """LightGBM global model with lag + rolling + date features."""
    lgbm = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    return MLForecast(
        models={"LightGBM": lgbm},
        freq="W-MON",
        lags=[1, 2, 4, 8, 13, 26, 52],
        lag_transforms={
            1: [RollingMean(window_size=4), RollingMean(window_size=13)],
            4: [RollingMean(window_size=4)],
        },
        date_features=["month", "quarter", "year"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--train-top", type=int, default=30,
                    help="Number of top SKUs (by revenue) included in the global training panel")
    ap.add_argument("--score-top", type=int, default=10,
                    help="Number of SKUs to report metrics on (same set as previous benchmarks)")
    ap.add_argument("--holdout-days", type=int, default=90)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    last = df["date"].max()
    cutoff = last - pd.Timedelta(days=args.holdout_days - 1)
    print(f"Data {df['date'].min().date()} → {last.date()} | holdout from {cutoff.date()}")

    revenue = df.groupby("productName")["totalValue"].sum().sort_values(ascending=False)
    train_skus = revenue.head(args.train_top).index.tolist()
    score_skus = revenue.head(args.score_top).index.tolist()
    print(f"Training panel: top {args.train_top} SKUs by revenue")
    print(f"Scoring on: top {args.score_top} SKUs (same as prior benchmarks)")

    all_rows = []
    forecasts_by_metric: dict[str, pd.DataFrame] = {}

    for metric in ["totalValue", "qty"]:
        print(f"\n=== Metric: {metric} ===")
        panel = build_weekly_panel(df, train_skus, metric)
        train_w = panel[panel["ds"] < cutoff].copy()
        # 13-week horizon
        weeks_in_holdout = (panel[panel["ds"] >= cutoff]["ds"].nunique())
        print(f"  panel rows: {len(panel):,}  train rows: {len(train_w):,}  h={weeks_in_holdout}")

        mf = make_model()
        mf.fit(train_w)
        fcst = mf.predict(h=weeks_in_holdout)
        forecasts_by_metric[metric] = fcst

        # Score per SKU on the scoring set
        actuals = panel[panel["ds"] >= cutoff]
        merged = fcst.merge(actuals, on=["unique_id", "ds"], how="inner")
        for sku in score_skus:
            sub = merged[merged["unique_id"] == sku]
            if len(sub) == 0:
                continue
            m = err_metrics(sub["y"].to_numpy(), sub["LightGBM"].to_numpy())
            rank = score_skus.index(sku) + 1
            all_rows.append({"rank": rank, "product": sku, "metric": metric,
                             "model": "LightGBM_global", **m})
            print(f"  [{rank:2d}] {sku:30s}  sMAPE={m['sMAPE']:5.1f}%  "
                  f"SumErr={m['SumPctErr']:+6.1f}%  MAE={m['MAE']:>10,.1f}")

    lgbm_summary = pd.DataFrame(all_rows)
    lgbm_summary.to_csv(OUT_DIR / "_metrics_lightgbm.csv", index=False)

    # Merge with prior benchmark for combined view
    if PREV_METRICS.exists():
        prev = pd.read_csv(PREV_METRICS)
        combined = pd.concat([prev, lgbm_summary], ignore_index=True)
    else:
        combined = lgbm_summary
    combined.to_csv(OUT_DIR / "_metrics_combined.csv", index=False)

    # Average by model
    print("\n=== Average across all SKUs/metrics (combined) ===")
    avg = (combined.groupby("model")
           .agg(mean_sMAPE=("sMAPE", "mean"),
                median_sMAPE=("sMAPE", "median"),
                mean_absSumPctErr=("SumPctErr", lambda x: x.abs().mean()),
                median_absSumPctErr=("SumPctErr", lambda x: x.abs().median()))
           .round(1)
           .sort_values("mean_sMAPE"))
    print(avg.to_string())
    avg.to_csv(OUT_DIR / "_avg_combined.csv")

    # Pivot tables for sMAPE and |SumPctErr|
    print("\n=== sMAPE by model ===")
    print(combined.pivot_table(index=["rank", "product", "metric"],
                               columns="model", values="sMAPE").round(1).to_string())

    print("\n=== |SumPctErr| by model ===")
    print(combined.assign(absSum=lambda x: x["SumPctErr"].abs())
          .pivot_table(index=["rank", "product", "metric"],
                       columns="model", values="absSum").round(1).to_string())

    # Per-SKU plots: LightGBM vs Prophet/AutoARIMA on the same panel
    for rank, sku in enumerate(score_skus, start=1):
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))
        for ax, metric in zip(axes, ["totalValue", "qty"]):
            panel = build_weekly_panel(df, train_skus, metric)
            s = panel[panel["unique_id"] == sku]
            ax.plot(s["ds"], s["y"], "k.-", markersize=4, alpha=0.6, linewidth=0.8, label="actual")
            fc = forecasts_by_metric[metric]
            fc_sku = fc[fc["unique_id"] == sku].sort_values("ds")
            actual_hold = s[s["ds"] >= cutoff]
            merged = actual_hold.merge(fc_sku, on=["unique_id", "ds"], how="inner")
            if len(merged):
                m = err_metrics(merged["y"].to_numpy(), merged["LightGBM"].to_numpy())
                ax.plot(fc_sku["ds"], np.clip(fc_sku["LightGBM"], 0, None),
                        "C4-o", markersize=5, linewidth=1.6,
                        label=f"LightGBM_global  sMAPE={m['sMAPE']:.0f}%  SumErr={m['SumPctErr']:+.1f}%")
            ax.axvspan(cutoff, last, color="orange", alpha=0.10)
            ax.axvline(cutoff, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"{metric} — weekly, 13-week holdout")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"#{rank}  {sku} — Global LightGBM forecast",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{rank:02d}_{safe(sku)}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # Importance plot (using the totalValue model)
    print("\nFeature importances (totalValue model):")
    final_model = make_model()
    panel = build_weekly_panel(df, train_skus, "totalValue")
    final_model.fit(panel[panel["ds"] < cutoff])
    booster = final_model.models_["LightGBM"]
    fi = pd.DataFrame({
        "feature": booster.feature_name_,
        "importance": booster.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(fi.head(15).to_string(index=False))
    fi.to_csv(OUT_DIR / "_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = fi.head(15)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="C4")
    ax.set_title("LightGBM feature importance (top 15) — totalValue model")
    ax.set_xlabel("gain")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "_feature_importance.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
