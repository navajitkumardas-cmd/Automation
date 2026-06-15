"""
Forecast Bakingo PO quantities with Meta Prophet.

Usage:
    python run_prophet.py                          # daily total qty across all products
    python run_prophet.py --product "Choco Truffle Cake"
    python run_prophet.py --product-id 4359 --periods 60
"""
import argparse
from pathlib import Path

import pandas as pd
from prophet import Prophet

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out")


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df = df.dropna(subset=["createdAt", "qty"])
    df["date"] = df["createdAt"].dt.normalize()
    return df


def aggregate(df: pd.DataFrame, product: str | None, product_id: int | None) -> pd.DataFrame:
    if product_id is not None:
        df = df[df["productId"] == product_id]
    elif product is not None:
        df = df[df["productName"].str.casefold() == product.casefold()]
    daily = df.groupby("date", as_index=False)["qty"].sum()
    daily.columns = ["ds", "y"]
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range, fill_value=0).rename_axis("ds").reset_index()
    return daily


def forecast(daily: pd.DataFrame, periods: int) -> tuple[Prophet, pd.DataFrame]:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name="IN")
    m.fit(daily)
    future = m.make_future_dataframe(periods=periods, freq="D")
    fcst = m.predict(future)
    return m, fcst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--product", default=None, help="Filter by productName (exact, case-insensitive)")
    ap.add_argument("--product-id", type=int, default=None)
    ap.add_argument("--periods", type=int, default=90, help="Days to forecast")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    print(f"Loaded {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Unique products: {df['productName'].nunique()}, stores: {df['fromStoreName'].nunique()}")

    daily = aggregate(df, args.product, args.product_id)
    label = args.product or (f"productId={args.product_id}" if args.product_id else "ALL_PRODUCTS")
    print(f"Series '{label}': {len(daily)} days, total qty = {daily['y'].sum():,.0f}")

    m, fcst = forecast(daily, args.periods)

    tail = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(args.periods)
    safe = label.replace("/", "_").replace(" ", "_")
    csv_out = OUT_DIR / f"forecast_{safe}.csv"
    tail.to_csv(csv_out, index=False)
    print(f"Saved forecast → {csv_out}")

    fig1 = m.plot(fcst)
    fig1.savefig(OUT_DIR / f"forecast_{safe}.png", dpi=120, bbox_inches="tight")
    fig2 = m.plot_components(fcst)
    fig2.savefig(OUT_DIR / f"components_{safe}.png", dpi=120, bbox_inches="tight")
    print(f"Saved plots → {OUT_DIR}/forecast_{safe}.png, components_{safe}.png")

    print("\nLast 7 days of forecast:")
    print(tail.tail(7).to_string(index=False))


if __name__ == "__main__":
    main()
