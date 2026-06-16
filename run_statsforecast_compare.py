"""
Option A: weekly StatsForecast benchmark vs Prophet.

Fits AutoARIMA, AutoETS, and Croston (CrostonOptimized) per top-N SKU on
weekly-aggregated data. Same 90-day holdout (~13 weeks) as the Prophet
backtest in run_prophet_backtest_v2.py.

Output: per-SKU comparison plot + a metrics CSV with sMAPE / SumPctErr /
MAE / RMSE per (SKU, metric, model). Includes the Prophet weekly numbers
for direct comparison.

Usage:
    python run_statsforecast_compare.py                  # top 10, 90d (13w)
    python run_statsforecast_compare.py --top 5 --holdout-days 90
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, CrostonOptimized

warnings.filterwarnings("ignore")

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out/statsforecast")
SEASON_W = 52  # yearly seasonality at weekly granularity


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df = df.dropna(subset=["createdAt"])
    df["date"] = df["createdAt"].dt.normalize()
    return df


def daily_series(df: pd.DataFrame, product: str, metric: str) -> pd.DataFrame:
    sub = df[df["productName"] == product]
    s = sub.groupby("date", as_index=False)[metric].sum()
    s.columns = ["ds", "y"]
    full = pd.date_range(s["ds"].min(), s["ds"].max(), freq="D")
    return s.set_index("ds").reindex(full, fill_value=0).rename_axis("ds").reset_index()


def to_weekly(d: pd.DataFrame) -> pd.DataFrame:
    return (d.set_index("ds")
            .resample("W-MON", label="left", closed="left")["y"].sum()
            .reset_index())


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


def fit_prophet_weekly(train_w: pd.DataFrame, h: int) -> np.ndarray:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=0.05)
    m.fit(train_w)
    future = m.make_future_dataframe(periods=h, freq="W-MON")
    f = m.predict(future).tail(h)
    return f["yhat"].to_numpy()


def fit_statsforecast(train_w: pd.DataFrame, h: int) -> dict[str, np.ndarray]:
    """Returns {model_name: yhat_array_of_len_h}."""
    sf_df = train_w.rename(columns={"ds": "ds", "y": "y"}).assign(unique_id="s1")[
        ["unique_id", "ds", "y"]
    ]
    models = [
        AutoARIMA(season_length=SEASON_W),
        AutoETS(season_length=SEASON_W),
        CrostonOptimized(),
    ]
    sf = StatsForecast(models=models, freq="W-MON", n_jobs=1)
    fcst = sf.forecast(df=sf_df, h=h)
    return {
        "AutoARIMA": fcst["AutoARIMA"].to_numpy(),
        "AutoETS": fcst["AutoETS"].to_numpy(),
        "Croston": fcst["CrostonOptimized"].to_numpy(),
    }


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--holdout-days", type=int, default=90)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    last = df["date"].max()
    cutoff = last - pd.Timedelta(days=args.holdout_days - 1)
    print(f"Data {df['date'].min().date()} → {last.date()} | holdout from {cutoff.date()}")

    ranking = (df.groupby("productName")["totalValue"].sum()
               .sort_values(ascending=False).head(args.top))

    rows = []
    for rank, (product, _) in enumerate(ranking.items(), start=1):
        print(f"\n[{rank}/{args.top}] {product}")
        sname = safe(product)

        fig, axes = plt.subplots(2, 1, figsize=(14, 9))

        for ax, metric in zip(axes, ["totalValue", "qty"]):
            d = daily_series(df, product, metric)
            w = to_weekly(d)
            train_w = w[w["ds"] < cutoff].copy()
            hold_w = w[w["ds"] >= cutoff].copy()
            h = len(hold_w)
            if h == 0 or len(train_w) < 30:
                print(f"  {metric}: insufficient data")
                continue
            actual = hold_w["y"].to_numpy()

            preds = {}
            preds["Prophet"] = fit_prophet_weekly(train_w, h)
            try:
                preds.update(fit_statsforecast(train_w, h))
            except Exception as e:
                print(f"  statsforecast failed: {e}")
                continue

            ax.plot(w["ds"], w["y"], "k.-", markersize=4, alpha=0.5, linewidth=0.8, label="actual")
            colors = {"Prophet": "C0", "AutoARIMA": "C1", "AutoETS": "C2", "Croston": "C3"}
            for name, yhat in preds.items():
                m = err_metrics(actual, yhat)
                rows.append({"rank": rank, "product": product, "metric": metric,
                             "model": name, **m})
                ax.plot(hold_w["ds"], np.clip(yhat, 0, None), "-o",
                        color=colors[name], markersize=4, linewidth=1.4,
                        label=f"{name}  sMAPE={m['sMAPE']:.0f}%  SumErr={m['SumPctErr']:+.1f}%")
                print(f"  {metric:11s} {name:10s}  "
                      f"sMAPE={m['sMAPE']:5.1f}%  SumErr={m['SumPctErr']:+6.1f}%  "
                      f"MAE={m['MAE']:>10,.1f}")

            ax.axvspan(cutoff, last, color="orange", alpha=0.10)
            ax.axvline(cutoff, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"{metric} — weekly, 13-week holdout")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

        fig.suptitle(f"#{rank}  {product} — StatsForecast vs Prophet (weekly)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{rank:02d}_{sname}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"_metrics_top{args.top}_compare.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nMetrics → {csv_path}")

    print("\n=== sMAPE by model (lower is better) ===")
    pivot_s = summary.pivot_table(index=["rank", "product", "metric"],
                                  columns="model", values="sMAPE")
    print(pivot_s.round(1).to_string())

    print("\n=== |SumPctErr| by model (lower is better) ===")
    pivot_a = (summary.assign(absSum=lambda x: x["SumPctErr"].abs())
               .pivot_table(index=["rank", "product", "metric"],
                            columns="model", values="absSum"))
    print(pivot_a.round(1).to_string())

    # Average by model
    print("\n=== Average across all SKUs/metrics ===")
    avg = summary.groupby("model").agg(
        mean_sMAPE=("sMAPE", "mean"),
        median_sMAPE=("sMAPE", "median"),
        mean_absSumPctErr=("SumPctErr", lambda x: x.abs().mean()),
        median_absSumPctErr=("SumPctErr", lambda x: x.abs().median()),
    ).round(1)
    print(avg.to_string())
    avg.to_csv(OUT_DIR / f"_avg_top{args.top}.csv")

    # Per-row winner
    winners = (summary.assign(absSum=lambda x: x["SumPctErr"].abs())
               .sort_values("sMAPE")
               .groupby(["rank", "product", "metric"])
               .first()[["model", "sMAPE", "absSum"]]
               .reset_index())
    winners.to_csv(OUT_DIR / f"_winners_top{args.top}.csv", index=False)
    print(f"\nWinners by sMAPE → {OUT_DIR}/_winners_top{args.top}.csv")
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
