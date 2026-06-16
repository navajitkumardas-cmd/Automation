"""
Ensemble backtest: combine Prophet, AutoARIMA, AutoETS, Croston, and the
global LightGBM into ensemble forecasters and score them on the same
13-week (90-day) holdout used for every other comparison.

Ensembles tested:
  - mean_all      : mean of all 5 base models
  - median_all    : median of all 5
  - mean_no_prophet   : mean of ARIMA, ETS, Croston, LightGBM
  - median_no_prophet : median of those four
  - mean_classical    : mean of ARIMA, ETS, Croston (no ML)
  - median_classical  : median of those three

Usage:
    python run_ensemble_backtest.py                 # top 10, train panel 30
    python run_ensemble_backtest.py --train-top 50 --score-top 10
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
OUT_DIR = Path("/home/user/Automation/forecast_out/ensemble")
SEASON_W = 52


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
    m.fit(train_w.rename(columns={"unique_id": "uid"})[["ds", "y"]])
    future = m.make_future_dataframe(periods=h, freq="W-MON")
    return m.predict(future).tail(h)["yhat"].to_numpy()


def statsforecast_one(train_w: pd.DataFrame, h: int) -> dict[str, np.ndarray]:
    sf_df = train_w.rename(columns={"unique_id": "unique_id"})[["unique_id", "ds", "y"]]
    sf_df["unique_id"] = "s1"
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
    return mf.predict(h=h)  # cols: unique_id, ds, LightGBM


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--train-top", type=int, default=30)
    ap.add_argument("--score-top", type=int, default=10)
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

    all_rows = []

    for metric in ["totalValue", "qty"]:
        print(f"\n=== Metric: {metric} ===")
        panel = build_weekly_panel(df, train_skus, metric)
        h = panel[panel["ds"] >= cutoff]["ds"].nunique()
        print(f"  panel rows: {len(panel):,}  horizon: {h} weeks")

        # Train global LightGBM ONCE on the whole panel
        print("  training LightGBM global...", end=" ", flush=True)
        lgb_fcst = lightgbm_global(panel, cutoff, h)
        print("done")

        for rank, sku in enumerate(score_skus, start=1):
            sku_panel = panel[panel["unique_id"] == sku].sort_values("ds")
            train_w = sku_panel[sku_panel["ds"] < cutoff][["ds", "y"]].assign(unique_id=sku)[
                ["unique_id", "ds", "y"]
            ]
            actual = sku_panel[sku_panel["ds"] >= cutoff]["y"].to_numpy()
            h_sku = len(actual)
            if h_sku < 2:
                print(f"  [{rank:2d}] {sku}: holdout too short ({h_sku} weeks), skipping")
                continue

            preds = {}
            preds["Prophet"] = prophet_weekly(train_w, h_sku)
            sf = statsforecast_one(train_w, h_sku)
            preds.update(sf)
            lgb_sku = lgb_fcst[lgb_fcst["unique_id"] == sku].sort_values("ds")
            preds["LightGBM"] = lgb_sku["LightGBM"].to_numpy()[:h_sku]

            # Pad/truncate any short ones to h_sku
            for k in list(preds):
                p = np.asarray(preds[k])
                if len(p) < h_sku:
                    pad = np.full(h_sku - len(p), p[-1] if len(p) else 0.0)
                    preds[k] = np.concatenate([p, pad])
                else:
                    preds[k] = p[:h_sku]

            # Build ensembles
            mat = np.vstack([preds["Prophet"], preds["AutoARIMA"], preds["AutoETS"],
                             preds["Croston"], preds["LightGBM"]])
            mat_nop = np.vstack([preds["AutoARIMA"], preds["AutoETS"],
                                 preds["Croston"], preds["LightGBM"]])
            mat_classic = np.vstack([preds["AutoARIMA"], preds["AutoETS"], preds["Croston"]])

            preds["mean_all"]        = mat.mean(axis=0)
            preds["median_all"]      = np.median(mat, axis=0)
            preds["mean_no_prophet"] = mat_nop.mean(axis=0)
            preds["median_no_prophet"] = np.median(mat_nop, axis=0)
            preds["mean_classical"]  = mat_classic.mean(axis=0)
            preds["median_classical"] = np.median(mat_classic, axis=0)

            for name, yhat in preds.items():
                m = err_metrics(actual, yhat)
                all_rows.append({"rank": rank, "product": sku, "metric": metric,
                                 "model": name, **m})

            best = min(preds.items(), key=lambda kv: err_metrics(actual, kv[1])["sMAPE"])
            print(f"  [{rank:2d}] {sku:30s}  best={best[0]:18s} "
                  f"sMAPE={err_metrics(actual, best[1])['sMAPE']:.1f}%")

            # Per-SKU plot
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(sku_panel["ds"], sku_panel["y"], "k.-", markersize=4, alpha=0.6,
                    linewidth=0.8, label="actual")
            hold_dates = sku_panel[sku_panel["ds"] >= cutoff]["ds"].iloc[:h_sku]

            singles = ["Prophet", "AutoARIMA", "AutoETS", "Croston", "LightGBM"]
            ensembles = ["median_all", "median_no_prophet", "median_classical"]
            colors_s = ["C0", "C1", "C2", "C3", "C4"]
            colors_e = ["C6", "C7", "C8"]
            for name, c in zip(singles, colors_s):
                ax.plot(hold_dates, np.clip(preds[name], 0, None), "-", color=c,
                        alpha=0.45, linewidth=1.0, label=name)
            for name, c in zip(ensembles, colors_e):
                ax.plot(hold_dates, np.clip(preds[name], 0, None), "-o", color=c,
                        markersize=4, linewidth=1.8,
                        label=f"{name}  sMAPE={err_metrics(actual, preds[name])['sMAPE']:.0f}%")
            ax.axvspan(cutoff, last, color="orange", alpha=0.10)
            ax.axvline(cutoff, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"#{rank}  {sku} — {metric}  (13-week holdout)")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{rank:02d}_{safe(sku)}_{metric}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)

    summary = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / f"_metrics_top{args.score_top}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nMetrics → {csv_path}")

    print("\n=== Average across all SKUs/metrics ===")
    avg = (summary.groupby("model")
           .agg(mean_sMAPE=("sMAPE", "mean"),
                median_sMAPE=("sMAPE", "median"),
                mean_absSumPctErr=("SumPctErr", lambda x: x.abs().mean()),
                median_absSumPctErr=("SumPctErr", lambda x: x.abs().median()))
           .round(1).sort_values("mean_sMAPE"))
    print(avg.to_string())
    avg.to_csv(OUT_DIR / "_avg_models.csv")

    print("\n=== Winners by sMAPE per SKU/metric ===")
    winners = (summary.sort_values("sMAPE")
               .groupby(["rank", "product", "metric"])
               .first()[["model", "sMAPE", "SumPctErr"]]
               .reset_index())
    print(winners.to_string(index=False))
    winners.to_csv(OUT_DIR / "_winners.csv", index=False)

    # Win count by model
    print("\n=== Number of SKU×metric wins by model ===")
    counts = winners["model"].value_counts()
    print(counts.to_string())


if __name__ == "__main__":
    main()
