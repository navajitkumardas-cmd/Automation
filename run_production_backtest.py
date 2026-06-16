"""
Production backtest: validate the exact production model (weekly Croston
with +30% safety margin) on the most recent 13 weeks.

Trains Croston per SKU on data BEFORE the last 13 weeks, predicts those
13 weeks, then compares to actuals. Produces:

  forecast_out/production_backtest/weekly_actual_vs_forecast.csv
        long format: (rank, product, metric, week_start, actual,
                      point, with_safety_30pct, point_err, safety_err)
  forecast_out/production_backtest/summary.csv
        one row per (SKU, metric): actual_sum, point_sum, safety_sum,
        sMAPE, SumPctErr, safety_SumPctErr, safety_coverage
  forecast_out/production_backtest/<SKU>_<metric>.png
        per-SKU dual panel: history + holdout actual + point + safety
  forecast_out/production_backtest/_dashboard.png
        10-SKU revenue dashboard

Usage:
    python run_production_backtest.py
    python run_production_backtest.py --top 10 --holdout 13 --safety 0.30
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized

warnings.filterwarnings("ignore")

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out/production_backtest")


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df = df.dropna(subset=["createdAt"])
    df["date"] = df["createdAt"].dt.normalize()
    return df


def weekly_series(df: pd.DataFrame, product: str, metric: str) -> pd.DataFrame:
    sub = df[df["productName"] == product]
    daily = sub.groupby("date", as_index=False)[metric].sum()
    daily.columns = ["ds", "y"]
    full = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full, fill_value=0).rename_axis("ds").reset_index()
    return (daily.set_index("ds").resample("W-MON", label="left", closed="left")["y"]
            .sum().reset_index())


def croston_forecast(train: pd.DataFrame, h: int) -> np.ndarray:
    sf_df = train.assign(unique_id="s1")[["unique_id", "ds", "y"]]
    sf = StatsForecast(models=[CrostonOptimized()], freq="W-MON", n_jobs=1)
    f = sf.forecast(df=sf_df, h=h)
    return np.clip(f["CrostonOptimized"].to_numpy(), 0, None)


def err_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    abs_err = np.abs(err)
    denom = (np.abs(actual) + np.abs(pred)) / 2
    mask = denom > 0
    s_act, s_pred = actual.sum(), pred.sum()
    return dict(
        MAE=float(abs_err.mean()),
        sMAPE=float((abs_err[mask] / denom[mask]).mean() * 100) if mask.any() else float("nan"),
        SumPctErr=float((s_pred - s_act) / s_act * 100) if s_act > 0 else float("nan"),
    )


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--holdout", type=int, default=13, help="Weeks to hold out")
    ap.add_argument("--safety", type=float, default=0.30)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    ranking = (df.groupby("productName")["totalValue"].sum()
               .sort_values(ascending=False).head(args.top))
    products = ranking.index.tolist()
    print(f"Data {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Backtest model: weekly CrostonOptimized + {int(args.safety*100)}% safety")
    print(f"Holdout: last {args.holdout} weeks per SKU\n")

    weekly_rows: list[dict] = []
    summary_rows: list[dict] = []

    n = len(products)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    dash, dash_axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    dash_axes = dash_axes.flatten() if n > 1 else [dash_axes]

    for rank, product in enumerate(products, start=1):
        print(f"[{rank:2d}/{args.top}] {product}")
        per_metric = {}

        for metric, unit in [("totalValue", "INR"), ("qty", "units")]:
            w = weekly_series(df, product, metric)
            h = args.holdout
            train = w.iloc[:-h]
            hold = w.iloc[-h:]
            actual = hold["y"].to_numpy()
            point = croston_forecast(train, h)
            safety = point * (1 + args.safety)

            m_point = err_metrics(actual, point)
            m_safety = err_metrics(actual, safety)
            coverage = float((safety >= actual).mean() * 100)
            per_metric[metric] = (w, hold, point, safety)

            for i, (ds, a, p, s) in enumerate(zip(hold["ds"], actual, point, safety), start=1):
                weekly_rows.append({
                    "rank": rank, "product": product, "metric": metric,
                    "week_index": i, "week_start": ds.date().isoformat(),
                    "actual": float(a),
                    "point": float(p),
                    f"with_safety_{int(args.safety*100)}pct": float(s),
                    "point_err": float(p - a),
                    "safety_err": float(s - a),
                })

            summary_rows.append({
                "rank": rank, "product": product, "metric": metric, "unit": unit,
                "actual_sum": float(actual.sum()),
                "point_sum": float(point.sum()),
                "safety_sum": float(safety.sum()),
                "point_sMAPE": m_point["sMAPE"],
                "point_MAE": m_point["MAE"],
                "point_SumPctErr": m_point["SumPctErr"],
                "safety_SumPctErr": m_safety["SumPctErr"],
                "safety_coverage_pct": coverage,
            })

            print(f"    {metric:11s}  actual={actual.sum():>12,.0f}  "
                  f"point={point.sum():>12,.0f} ({m_point['SumPctErr']:+5.1f}%)  "
                  f"safety={safety.sum():>12,.0f} ({m_safety['SumPctErr']:+5.1f}%)  "
                  f"sMAPE={m_point['sMAPE']:5.1f}%  cov={coverage:5.1f}%")

        # Per-SKU plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for ax, (metric, unit) in zip(axes, [("totalValue", "INR"), ("qty", "units")]):
            w, hold, point, safety = per_metric[metric]
            train = w.iloc[:-args.holdout]
            cutoff = hold["ds"].iloc[0]
            ax.plot(w["ds"], w["y"], "k.-", markersize=4, alpha=0.5, linewidth=0.8,
                    label="actual (full history)")
            ax.plot(hold["ds"], point, "C2-o", markersize=5, linewidth=1.8,
                    label="Croston point forecast")
            ax.fill_between(hold["ds"], point, safety, color="C2", alpha=0.18,
                            label=f"+{int(args.safety*100)}% safety")
            ax.plot(hold["ds"], hold["y"], "r-x", markersize=8, linewidth=0.0,
                    markeredgewidth=2, label="holdout actual")
            ax.axvline(cutoff, color="red", linestyle="--", alpha=0.5)
            err = (point.sum() - hold["y"].sum()) / hold["y"].sum() * 100 if hold["y"].sum() > 0 else 0
            ax.set_title(f"{metric} ({unit})  •  holdout {args.holdout}w  "
                         f"actual={hold['y'].sum():,.0f}  point={point.sum():,.0f} ({err:+.1f}%)")
            ax.set_xlabel("week start")
            ax.set_ylabel(unit)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"#{rank}  {product}  — production-model backtest",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{rank:02d}_{safe(product)}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        # Dashboard tile: revenue only
        ax = dash_axes[rank - 1]
        w, hold, point, safety = per_metric["totalValue"]
        ax.plot(w["ds"], w["y"], "k.-", markersize=3, alpha=0.5, linewidth=0.7)
        ax.plot(hold["ds"], point, "C2-o", markersize=4, linewidth=1.3)
        ax.fill_between(hold["ds"], point, safety, color="C2", alpha=0.2)
        ax.plot(hold["ds"], hold["y"], "rx", markersize=8, markeredgewidth=2)
        ax.axvline(hold["ds"].iloc[0], color="red", linestyle="--", alpha=0.5)
        err = (point.sum() - hold["y"].sum()) / hold["y"].sum() * 100 if hold["y"].sum() > 0 else 0
        ax.set_title(f"#{rank}  {product}\nactual={hold['y'].sum():,.0f}  "
                     f"point={point.sum():,.0f} ({err:+.1f}%)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)

    for j in range(n, len(dash_axes)):
        dash_axes[j].axis("off")
    dash.suptitle(f"Production-model backtest — weekly Croston, "
                  f"{args.holdout}-week holdout (revenue panel)",
                  fontsize=13, fontweight="bold")
    dash.tight_layout()
    dash.savefig(OUT_DIR / "_dashboard.png", dpi=110, bbox_inches="tight")
    plt.close(dash)

    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df.to_csv(OUT_DIR / "weekly_actual_vs_forecast.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n" + "=" * 100)
    print(f"BACKTEST SUMMARY — weekly Croston, {args.holdout}-week holdout, +{int(args.safety*100)}% safety")
    print("=" * 100)

    print("\nRevenue (INR):")
    rev = summary_df[summary_df["metric"] == "totalValue"][
        ["rank", "product", "actual_sum", "point_sum", "safety_sum",
         "point_sMAPE", "point_SumPctErr", "safety_SumPctErr", "safety_coverage_pct"]
    ]
    print(rev.to_string(index=False, formatters={
        "actual_sum": "{:>12,.0f}".format,
        "point_sum": "{:>12,.0f}".format,
        "safety_sum": "{:>12,.0f}".format,
        "point_sMAPE": "{:>6.1f}%".format,
        "point_SumPctErr": "{:>+6.1f}%".format,
        "safety_SumPctErr": "{:>+6.1f}%".format,
        "safety_coverage_pct": "{:>5.1f}%".format,
    }))

    print("\nQuantity (units):")
    q = summary_df[summary_df["metric"] == "qty"][
        ["rank", "product", "actual_sum", "point_sum", "safety_sum",
         "point_sMAPE", "point_SumPctErr", "safety_SumPctErr", "safety_coverage_pct"]
    ]
    print(q.to_string(index=False, formatters={
        "actual_sum": "{:>10,.0f}".format,
        "point_sum": "{:>10,.0f}".format,
        "safety_sum": "{:>10,.0f}".format,
        "point_sMAPE": "{:>6.1f}%".format,
        "point_SumPctErr": "{:>+6.1f}%".format,
        "safety_SumPctErr": "{:>+6.1f}%".format,
        "safety_coverage_pct": "{:>5.1f}%".format,
    }))

    print("\nAggregate across all 10 SKUs:")
    for metric, label in [("totalValue", "Revenue (INR)"), ("qty", "Quantity (units)")]:
        sub = summary_df[summary_df["metric"] == metric]
        a, p, s = sub["actual_sum"].sum(), sub["point_sum"].sum(), sub["safety_sum"].sum()
        print(f"  {label:18s}  actual={a:>15,.0f}  point={p:>15,.0f} ({(p-a)/a*100:+.1f}%)  "
              f"safety={s:>15,.0f} ({(s-a)/a*100:+.1f}%)")

    print(f"\nMean sMAPE across all 20 (SKU,metric): "
          f"{summary_df['point_sMAPE'].mean():.1f}%")
    print(f"Median point |SumPctErr|: {summary_df['point_SumPctErr'].abs().median():.1f}%")
    print(f"Safety-band coverage (% of holdout weeks where safety >= actual): "
          f"{summary_df['safety_coverage_pct'].mean():.1f}%")

    print(f"\nFiles written → {OUT_DIR}")


if __name__ == "__main__":
    main()
