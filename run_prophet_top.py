"""
Forecast the top-N products (by total PO value) with Meta Prophet.

For each top product, fits Prophet on its daily `totalValue` series and
writes per-product forecast CSV + plot. Also writes a combined summary.

Usage:
    python run_prophet_top.py                  # top 10, 90-day horizon
    python run_prophet_top.py --top 5 --periods 60
    python run_prophet_top.py --metric qty     # rank/forecast by qty instead
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out/top")


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
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range, fill_value=0).rename_axis("ds").reset_index()
    return daily


def forecast_one(daily: pd.DataFrame, periods: int):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name="IN")
    m.fit(daily)
    future = m.make_future_dataframe(periods=periods, freq="D")
    fcst = m.predict(future)
    return m, fcst


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--periods", type=int, default=90)
    ap.add_argument("--metric", choices=["totalValue", "qty"], default="totalValue")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    print(f"Loaded {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}")

    ranking = (
        df.groupby("productName")[args.metric].sum().sort_values(ascending=False).head(args.top)
    )
    print(f"\nTop {args.top} products by {args.metric}:")
    print(ranking.to_string())

    summary_rows = []
    combined_fig, combined_ax = plt.subplots(figsize=(14, 7))

    for rank, (product, total) in enumerate(ranking.items(), start=1):
        print(f"\n[{rank}/{args.top}] {product}  (historical {args.metric} = {total:,.0f})")
        daily = daily_series(df, product, args.metric)
        print(f"  series: {len(daily)} days, nonzero days = {(daily['y'] > 0).sum()}")

        m, fcst = forecast_one(daily, args.periods)

        sname = safe(product)
        # Per-product forecast CSV (history + future, last `periods` rows are the forecast)
        out_csv = OUT_DIR / f"{rank:02d}_{sname}.csv"
        fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_csv, index=False)

        # Per-product plot
        fig = m.plot(fcst)
        fig.suptitle(f"{product} — Prophet forecast ({args.metric})")
        fig.savefig(OUT_DIR / f"{rank:02d}_{sname}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        # Combined plot: history + future yhat line per product
        combined_ax.plot(fcst["ds"], fcst["yhat"], label=f"{rank}. {product}", linewidth=1.5)

        future_tail = fcst.tail(args.periods)
        summary_rows.append(
            {
                "rank": rank,
                "product": product,
                "historical_total": total,
                "forecast_horizon_days": args.periods,
                "forecast_sum": float(future_tail["yhat"].clip(lower=0).sum()),
                "forecast_mean_daily": float(future_tail["yhat"].clip(lower=0).mean()),
            }
        )

    # Mark forecast cutoff
    cutoff = df["date"].max()
    combined_ax.axvline(cutoff, color="black", linestyle="--", alpha=0.5, label="forecast start")
    combined_ax.set_title(f"Top {args.top} products by {args.metric} — Prophet yhat")
    combined_ax.set_xlabel("date")
    combined_ax.set_ylabel(args.metric)
    combined_ax.legend(loc="upper left", fontsize=8)
    combined_ax.grid(alpha=0.3)
    combined_fig.tight_layout()
    combined_path = OUT_DIR / f"_combined_top{args.top}_{args.metric}.png"
    combined_fig.savefig(combined_path, dpi=120, bbox_inches="tight")
    plt.close(combined_fig)

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / f"_summary_top{args.top}_{args.metric}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nWrote per-product forecasts + plots → {OUT_DIR}")
    print(f"Combined chart → {combined_path}")
    print(f"Summary → {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
