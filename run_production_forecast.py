"""
Production forecast: weekly Croston on ALL data for top-N SKUs, both metrics.

This is the recommended v1 production strategy from the handout:
  - CrostonOptimized (statsforecast)
  - Weekly W-MON aggregation
  - Refit per SKU on the full history
  - 13-week horizon (~90 days)
  - +30% safety margin (matches measured mean |SumPctErr| of 35.5%)

Outputs:
  forecast_out/production/forecast_top{N}.csv       # per (SKU, week, metric)
  forecast_out/production/forecast_summary.csv      # total per SKU per metric
  forecast_out/production/weekly_calendar.csv       # wide format by week
  forecast_out/production/<SKU>_<metric>.png        # per-SKU forecast plot
  forecast_out/production/_dashboard.png            # 10-SKU dashboard

Usage:
    python run_production_forecast.py
    python run_production_forecast.py --top 10 --horizon 13 --safety 0.30
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
OUT_DIR = Path("/home/user/Automation/forecast_out/production")
SEASON_W = 52


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
    weekly = (daily.set_index("ds").resample("W-MON", label="left", closed="left")["y"]
              .sum().reset_index())
    return weekly


def fit_croston(history: pd.DataFrame, h: int) -> np.ndarray:
    sf_df = history.assign(unique_id="s1")[["unique_id", "ds", "y"]]
    sf = StatsForecast(models=[CrostonOptimized()], freq="W-MON", n_jobs=1)
    f = sf.forecast(df=sf_df, h=h)
    return np.clip(f["CrostonOptimized"].to_numpy(), 0, None)


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=13, help="Weeks to forecast")
    ap.add_argument("--safety", type=float, default=0.30,
                    help="Safety margin (0.30 = +30%% on the point forecast)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    last_day = df["date"].max()
    print(f"Data {df['date'].min().date()} → {last_day.date()}")

    ranking = (df.groupby("productName")["totalValue"].sum()
               .sort_values(ascending=False).head(args.top))
    products = ranking.index.tolist()
    print(f"\nTraining on top {args.top} SKUs by historical revenue:")
    for i, p in enumerate(products, 1):
        print(f"  {i:2d}. {p}")

    long_rows: list[dict] = []
    summary_rows: list[dict] = []

    n = len(products)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    dash, dash_axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    dash_axes = dash_axes.flatten() if n > 1 else [dash_axes]

    for rank, product in enumerate(products, start=1):
        print(f"\n[{rank}/{args.top}] {product}")
        per_metric = {}

        for metric, unit in [("totalValue", "₹"), ("qty", "units")]:
            history = weekly_series(df, product, metric)
            yhat = fit_croston(history, args.horizon)

            # Build forecast frame
            last_w = history["ds"].max()
            future_ds = pd.date_range(last_w + pd.Timedelta(weeks=1),
                                      periods=args.horizon, freq="W-MON")
            fcst = pd.DataFrame({
                "ds": future_ds,
                "week_index": np.arange(1, args.horizon + 1),
                "yhat": yhat,
                "yhat_with_safety": yhat * (1 + args.safety),
            })
            per_metric[metric] = (history, fcst)

            for _, r in fcst.iterrows():
                long_rows.append({
                    "rank": rank, "product": product, "metric": metric,
                    "week_start": r["ds"].date().isoformat(),
                    "week_index": int(r["week_index"]),
                    "point_forecast": float(r["yhat"]),
                    f"with_safety_{int(args.safety*100)}pct": float(r["yhat_with_safety"]),
                })

            total_point = float(yhat.sum())
            total_safe = float(yhat.sum() * (1 + args.safety))
            avg_per_week = float(yhat.mean())
            recent_13w_actual = float(history["y"].tail(13).sum())
            recent_13w_avg = float(history["y"].tail(13).mean())

            summary_rows.append({
                "rank": rank, "product": product, "metric": metric, "unit": unit,
                "horizon_weeks": args.horizon,
                "forecast_total_point": total_point,
                "forecast_total_with_safety": total_safe,
                "forecast_avg_per_week": avg_per_week,
                "recent_13w_actual_total": recent_13w_actual,
                "recent_13w_actual_avg_per_week": recent_13w_avg,
                "vs_recent_13w_pct": ((total_point - recent_13w_actual) / recent_13w_actual * 100
                                      if recent_13w_actual > 0 else float("nan")),
            })

            print(f"  {metric:11s}  next-13w point = {total_point:>12,.0f}  "
                  f"with +{int(args.safety*100)}% safety = {total_safe:>12,.0f}  "
                  f"avg/wk = {avg_per_week:>10,.0f}")

        # Per-SKU plot (both metrics)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for ax, (metric, unit) in zip(axes, [("totalValue", "₹"), ("qty", "units")]):
            history, fcst = per_metric[metric]
            ax.plot(history["ds"], history["y"], "k.-", markersize=4, alpha=0.6,
                    linewidth=0.9, label="actual (weekly)")
            ax.plot(fcst["ds"], fcst["yhat"], "C2-o", markersize=5, linewidth=1.8,
                    label="Croston point forecast")
            ax.fill_between(fcst["ds"], fcst["yhat"], fcst["yhat_with_safety"],
                            color="C2", alpha=0.18,
                            label=f"+{int(args.safety*100)}% safety band")
            ax.axvline(last_day, color="red", linestyle="--", alpha=0.5,
                       label=f"forecast start ({(last_day + pd.Timedelta(days=1)).date()})")
            ax.set_title(f"{metric} ({unit}) — next {args.horizon} weeks "
                         f"= {fcst['yhat'].sum():,.0f} point, "
                         f"{fcst['yhat_with_safety'].sum():,.0f} with safety")
            ax.set_xlabel("week start")
            ax.set_ylabel(unit)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"#{rank}  {product}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{rank:02d}_{safe(product)}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        # Dashboard tile: revenue only
        ax = dash_axes[rank - 1]
        history, fcst = per_metric["totalValue"]
        ax.plot(history["ds"], history["y"], "k.-", markersize=3, alpha=0.5, linewidth=0.7)
        ax.plot(fcst["ds"], fcst["yhat"], "C2-o", markersize=4, linewidth=1.4)
        ax.fill_between(fcst["ds"], fcst["yhat"], fcst["yhat_with_safety"],
                        color="C2", alpha=0.2)
        ax.axvline(last_day, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"#{rank}  {product}\n13-w ₹ = {fcst['yhat'].sum():,.0f}",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)

    for j in range(n, len(dash_axes)):
        dash_axes[j].axis("off")
    dash.suptitle(f"Production forecast — weekly Croston, next {args.horizon} weeks, "
                  f"top {args.top} SKUs by revenue", fontsize=13, fontweight="bold")
    dash.tight_layout()
    dash.savefig(OUT_DIR / "_dashboard.png", dpi=110, bbox_inches="tight")
    plt.close(dash)

    # Long-format table (rank, product, metric, week, point, safety)
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT_DIR / f"forecast_top{args.top}.csv", index=False)

    # Wide weekly calendar (one row per SKU/metric, one column per week)
    safety_col = f"with_safety_{int(args.safety*100)}pct"
    pivot_point = long_df.pivot_table(index=["rank", "product", "metric"],
                                      columns="week_start", values="point_forecast")
    pivot_safe = long_df.pivot_table(index=["rank", "product", "metric"],
                                     columns="week_start", values=safety_col)
    pivot_point.to_csv(OUT_DIR / "weekly_calendar_point.csv")
    pivot_safe.to_csv(OUT_DIR / "weekly_calendar_with_safety.csv")

    # Summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "forecast_summary.csv", index=False)

    # Print headline summary
    print("\n" + "=" * 78)
    print(f"PRODUCTION FORECAST — next {args.horizon} weeks "
          f"({(last_day + pd.Timedelta(days=1)).date()} onward)")
    print("=" * 78)

    print("\nRevenue (₹) — point forecast and with +{:d}% safety margin:".format(int(args.safety*100)))
    rev = summary_df[summary_df["metric"] == "totalValue"][
        ["rank", "product", "forecast_total_point", "forecast_total_with_safety",
         "forecast_avg_per_week", "vs_recent_13w_pct"]
    ]
    print(rev.to_string(index=False, formatters={
        "forecast_total_point": "₹{:>12,.0f}".format,
        "forecast_total_with_safety": "₹{:>12,.0f}".format,
        "forecast_avg_per_week": "₹{:>10,.0f}".format,
        "vs_recent_13w_pct": "{:+.1f}%".format,
    }))

    print("\nQuantity (units) — point forecast and with +{:d}% safety margin:".format(int(args.safety*100)))
    qq = summary_df[summary_df["metric"] == "qty"][
        ["rank", "product", "forecast_total_point", "forecast_total_with_safety",
         "forecast_avg_per_week", "vs_recent_13w_pct"]
    ]
    print(qq.to_string(index=False, formatters={
        "forecast_total_point": "{:>10,.0f}".format,
        "forecast_total_with_safety": "{:>10,.0f}".format,
        "forecast_avg_per_week": "{:>8,.0f}".format,
        "vs_recent_13w_pct": "{:+.1f}%".format,
    }))

    # Totals
    print("\nTotals across top {} SKUs:".format(args.top))
    rev_total = rev["forecast_total_point"].sum()
    rev_safe = rev["forecast_total_with_safety"].sum()
    qty_total = qq["forecast_total_point"].sum()
    qty_safe = qq["forecast_total_with_safety"].sum()
    print(f"  Revenue (₹) point   : {rev_total:>15,.0f}")
    print(f"  Revenue (₹) safe    : {rev_safe:>15,.0f}")
    print(f"  Quantity   point   : {qty_total:>15,.0f}")
    print(f"  Quantity   safe    : {qty_safe:>15,.0f}")

    print(f"\nFiles written → {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
