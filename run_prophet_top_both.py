"""
Forecast the top-N SKUs (ranked by totalValue) with Prophet on BOTH metrics:
totalValue (revenue) AND qty (units). Produces, per product, a side-by-side
plot of the two forecasts and a combined summary table.

Usage:
    python run_prophet_top_both.py                     # top 10, 90-day horizon
    python run_prophet_top_both.py --top 5 --periods 180
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out/top_both")


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
    daily = daily.set_index("ds").reindex(full, fill_value=0).rename_axis("ds").reset_index()
    return daily


def fit_predict(daily: pd.DataFrame, periods: int):
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
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    print(f"Loaded {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}")

    ranking = (
        df.groupby("productName")
        .agg(totalValue=("totalValue", "sum"), qty=("qty", "sum"))
        .sort_values("totalValue", ascending=False)
        .head(args.top)
    )
    print(f"\nTop {args.top} SKUs by totalValue:")
    print(ranking.to_string())

    rows = []
    cutoff = df["date"].max()

    for rank, (product, hist) in enumerate(ranking.iterrows(), start=1):
        print(f"\n[{rank}/{args.top}] {product}")
        sname = safe(product)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        per_product = {"rank": rank, "product": product}

        for ax, metric, unit in [(axes[0], "totalValue", "INR"), (axes[1], "qty", "units")]:
            daily = daily_series(df, product, metric)
            m, fcst = fit_predict(daily, args.periods)
            future_tail = fcst.tail(args.periods)
            fsum = float(future_tail["yhat"].clip(lower=0).sum())
            fmean = float(future_tail["yhat"].clip(lower=0).mean())
            per_product[f"hist_{metric}"] = float(hist[metric])
            per_product[f"forecast_{metric}_sum"] = fsum
            per_product[f"forecast_{metric}_mean_daily"] = fmean

            ax.plot(daily["ds"], daily["y"], "k.", markersize=2, alpha=0.5, label="actual")
            ax.plot(fcst["ds"], fcst["yhat"], "C0-", linewidth=1.5, label="yhat")
            ax.fill_between(fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"],
                            color="C0", alpha=0.2, label="80% CI")
            ax.axvline(cutoff, color="red", linestyle="--", alpha=0.6)
            ax.set_title(f"{metric} ({unit})  •  next {args.periods}d = {fsum:,.0f}")
            ax.set_xlabel("date")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

            # Save raw forecast for the metric
            fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
                OUT_DIR / f"{rank:02d}_{sname}__{metric}.csv", index=False
            )

        fig.suptitle(f"#{rank}  {product}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{rank:02d}_{sname}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        rows.append(per_product)
        print(f"  hist value={per_product['hist_totalValue']:>12,.0f}  "
              f"forecast value={per_product['forecast_totalValue_sum']:>10,.0f}")
        print(f"  hist qty  ={per_product['hist_qty']:>12,.0f}  "
              f"forecast qty  ={per_product['forecast_qty_sum']:>10,.0f}")

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / f"_summary_top{args.top}_both_{args.periods}d.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nWrote per-SKU plots + CSVs → {OUT_DIR}")
    print(f"Summary → {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
