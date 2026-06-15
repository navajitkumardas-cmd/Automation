"""
Backtest: hold out the last 90 days, train Prophet on everything before,
predict the held-out window, and compute error metrics vs actuals.

For each top-N SKU (ranked by historical totalValue), runs the backtest on
both metrics: totalValue and qty. Produces per-SKU plots and a metrics CSV.

Metrics (over the holdout window, daily granularity):
  - MAE   : mean absolute error
  - RMSE  : root mean squared error
  - MAPE  : mean abs % error (only on days with actual > 0)
  - sMAPE : symmetric MAPE (safe for zeros)
  - Bias  : mean(forecast - actual)
  - Sum%  : (sum forecast - sum actual) / sum actual  -> aggregate accuracy

Usage:
    python run_prophet_backtest.py                  # top 10, 90-day holdout
    python run_prophet_backtest.py --top 5 --holdout 60
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out/backtest")


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df = df.dropna(subset=["createdAt"])
    df["date"] = df["createdAt"].dt.normalize()
    return df


def daily_series(df: pd.DataFrame, product: str, metric: str) -> pd.DataFrame:
    sub = df[df["productName"] == product]
    daily = sub.groupby("date", as_index=False)[metric].sum()
    daily.columns = ["ds", "y"]
    full = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    return daily.set_index("ds").reindex(full, fill_value=0).rename_axis("ds").reset_index()


def fit_predict(train: pd.DataFrame, periods: int):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name="IN")
    m.fit(train)
    future = m.make_future_dataframe(periods=periods, freq="D")
    return m, m.predict(future)


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(pred, 0, None)  # negative forecasts aren't meaningful here
    err = pred - actual
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    nz = actual > 0
    mape = float((abs_err[nz] / actual[nz]).mean() * 100) if nz.any() else float("nan")
    denom = (np.abs(actual) + np.abs(pred)) / 2
    smape_mask = denom > 0
    smape = float((abs_err[smape_mask] / denom[smape_mask]).mean() * 100) if smape_mask.any() else float("nan")
    bias = float(err.mean())
    s_act, s_pred = actual.sum(), pred.sum()
    sum_pct = float((s_pred - s_act) / s_act * 100) if s_act > 0 else float("nan")
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, sMAPE=smape, Bias=bias,
                ActualSum=float(s_act), ForecastSum=float(s_pred), SumPctErr=sum_pct)


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--holdout", type=int, default=90)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    last_date = df["date"].max()
    cutoff = last_date - pd.Timedelta(days=args.holdout - 1)
    print(f"Data: {df['date'].min().date()} → {last_date.date()} "
          f"| holdout = last {args.holdout}d (from {cutoff.date()})")

    ranking = (df.groupby("productName")["totalValue"].sum()
               .sort_values(ascending=False).head(args.top))
    print(f"\nTop {args.top} SKUs by historical totalValue:")
    print(ranking.to_string())

    rows = []
    for rank, (product, _) in enumerate(ranking.items(), start=1):
        print(f"\n[{rank}/{args.top}] {product}")
        sname = safe(product)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for ax, metric, unit in [(axes[0], "totalValue", "INR"), (axes[1], "qty", "units")]:
            daily = daily_series(df, product, metric)
            train = daily[daily["ds"] < cutoff].copy()
            holdout = daily[daily["ds"] >= cutoff].copy()

            if len(train) < 30:
                print(f"  {metric}: train too small ({len(train)}d) — skipping")
                continue

            m, fcst = fit_predict(train, len(holdout))
            pred_hold = fcst[fcst["ds"] >= cutoff][["ds", "yhat", "yhat_lower", "yhat_upper"]]
            merged = holdout.merge(pred_hold, on="ds", how="left")

            mtr = metrics(merged["y"].to_numpy(), merged["yhat"].to_numpy())
            rows.append({"rank": rank, "product": product, "metric": metric, **mtr})
            print(f"  {metric:11s} MAE={mtr['MAE']:>10,.1f}  RMSE={mtr['RMSE']:>10,.1f}  "
                  f"MAPE={mtr['MAPE']:>6.1f}%  sMAPE={mtr['sMAPE']:>6.1f}%  "
                  f"SumErr={mtr['SumPctErr']:>+6.1f}%")

            # Plot: full actual + prediction overlay + holdout shading
            ax.plot(daily["ds"], daily["y"], "k.", markersize=2, alpha=0.4, label="actual")
            ax.plot(fcst["ds"], fcst["yhat"], "C0-", linewidth=1.2, label="yhat (train+holdout)")
            ax.fill_between(merged["ds"], merged["yhat_lower"], merged["yhat_upper"],
                            color="C0", alpha=0.2)
            ax.axvspan(cutoff, last_date, color="orange", alpha=0.12, label="holdout")
            ax.axvline(cutoff, color="red", linestyle="--", alpha=0.6)
            ax.set_title(f"{metric} ({unit})  •  MAE={mtr['MAE']:,.1f}  "
                         f"sMAPE={mtr['sMAPE']:.1f}%  SumErr={mtr['SumPctErr']:+.1f}%")
            ax.set_xlabel("date")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

            merged.to_csv(OUT_DIR / f"{rank:02d}_{sname}__{metric}__holdout.csv", index=False)

        fig.suptitle(f"#{rank}  {product}  (backtest, {args.holdout}-day holdout)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{rank:02d}_{sname}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / f"_metrics_top{args.top}_holdout{args.holdout}d.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nMetrics → {summary_path}\n")

    pretty = summary[["rank", "product", "metric", "MAE", "RMSE", "MAPE",
                      "sMAPE", "Bias", "ActualSum", "ForecastSum", "SumPctErr"]]
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
