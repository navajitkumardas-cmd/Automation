"""
Backtest v2: compare three Prophet configurations on the same 90-day holdout:
  A. daily, default changepoint_prior_scale (0.05)        — baseline
  B. daily, tuned cps in {0.01, 0.05, 0.1, 0.5} via mini grid on a
     pre-holdout validation window (last 90d of training)
  C. weekly aggregation, default cps (holdout = 13 weeks)

For each top-N SKU (ranked by historical totalValue) and each metric
(totalValue, qty), reports sum-level and pointwise error metrics so we can
see which configuration generalizes best.

Usage:
    python run_prophet_backtest_v2.py                 # top 10, 90-day holdout
    python run_prophet_backtest_v2.py --top 5 --holdout 90
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

warnings.filterwarnings("ignore")

CSV_PATH = "/root/.claude/uploads/910b4aa1-540b-5e47-818d-7d589334ca73/3a63942d-bakingo_po_20240101_to_20260613.csv"
OUT_DIR = Path("/home/user/Automation/forecast_out/backtest_v2")
CPS_GRID = [0.01, 0.05, 0.1, 0.5]


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df = df.dropna(subset=["createdAt"])
    df["date"] = df["createdAt"].dt.normalize()
    return df


def daily(df: pd.DataFrame, product: str, metric: str) -> pd.DataFrame:
    sub = df[df["productName"] == product]
    s = sub.groupby("date", as_index=False)[metric].sum()
    s.columns = ["ds", "y"]
    full = pd.date_range(s["ds"].min(), s["ds"].max(), freq="D")
    return s.set_index("ds").reindex(full, fill_value=0).rename_axis("ds").reset_index()


def to_weekly(d: pd.DataFrame) -> pd.DataFrame:
    w = d.set_index("ds").resample("W-MON", label="left", closed="left")["y"].sum().reset_index()
    return w


def fit_one(train: pd.DataFrame, periods: int, freq: str, cps: float):
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(freq == "D"),
        daily_seasonality=False,
        changepoint_prior_scale=cps,
    )
    if freq == "D":
        m.add_country_holidays(country_name="IN")
    m.fit(train)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    return m, m.predict(future)


def err_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(pred, 0, None)
    err = pred - actual
    abs_err = np.abs(err)
    denom = (np.abs(actual) + np.abs(pred)) / 2
    smape_mask = denom > 0
    s_act, s_pred = actual.sum(), pred.sum()
    return dict(
        MAE=float(abs_err.mean()),
        RMSE=float(np.sqrt((err ** 2).mean())),
        sMAPE=float((abs_err[smape_mask] / denom[smape_mask]).mean() * 100) if smape_mask.any() else float("nan"),
        ActualSum=float(s_act),
        ForecastSum=float(s_pred),
        SumPctErr=float((s_pred - s_act) / s_act * 100) if s_act > 0 else float("nan"),
    )


def tune_daily_cps(daily_series: pd.DataFrame, holdout_days: int) -> tuple[float, dict]:
    """Pick cps that minimizes |SumPctErr| on a validation window of length
    holdout_days that ends just before the real holdout."""
    n = len(daily_series)
    val_end = n - holdout_days
    val_start = val_end - holdout_days
    if val_start < 60:
        return 0.05, {}
    train = daily_series.iloc[:val_start]
    val = daily_series.iloc[val_start:val_end]
    results = {}
    best, best_score = 0.05, float("inf")
    for cps in CPS_GRID:
        _, fcst = fit_one(train, len(val), "D", cps)
        pred = fcst.iloc[-len(val):]["yhat"].to_numpy()
        m = err_metrics(val["y"].to_numpy(), pred)
        score = abs(m["SumPctErr"]) + 0.5 * m["sMAPE"]
        results[cps] = m
        if score < best_score:
            best, best_score = cps, score
    return best, results


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
    last = df["date"].max()
    cutoff = last - pd.Timedelta(days=args.holdout - 1)
    print(f"Data {df['date'].min().date()} → {last.date()} | holdout last {args.holdout}d from {cutoff.date()}")

    ranking = (df.groupby("productName")["totalValue"].sum()
               .sort_values(ascending=False).head(args.top))

    rows = []
    for rank, (product, _) in enumerate(ranking.items(), start=1):
        print(f"\n[{rank}/{args.top}] {product}")
        sname = safe(product)
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

        for ax, metric in zip(axes, ["totalValue", "qty"]):
            d = daily(df, product, metric)
            train_d = d[d["ds"] < cutoff].copy()
            hold_d = d[d["ds"] >= cutoff].copy()

            # --- A: daily default cps=0.05
            _, fcst_a = fit_one(train_d, len(hold_d), "D", 0.05)
            pred_a = fcst_a.iloc[-len(hold_d):]["yhat"].to_numpy()
            m_a = err_metrics(hold_d["y"].to_numpy(), pred_a)

            # --- B: daily tuned cps
            best_cps, _ = tune_daily_cps(train_d, args.holdout)
            _, fcst_b = fit_one(train_d, len(hold_d), "D", best_cps)
            pred_b = fcst_b.iloc[-len(hold_d):]["yhat"].to_numpy()
            m_b = err_metrics(hold_d["y"].to_numpy(), pred_b)

            # --- C: weekly
            w = to_weekly(d)
            train_w = w[w["ds"] < cutoff].copy()
            hold_w = w[w["ds"] >= cutoff].copy()
            _, fcst_c = fit_one(train_w, len(hold_w), "W-MON", 0.05)
            pred_c = fcst_c.iloc[-len(hold_w):]["yhat"].to_numpy()
            m_c = err_metrics(hold_w["y"].to_numpy(), pred_c)

            for cfg, mm, extra in [("A_daily_cps0.05", m_a, {}),
                                   ("B_daily_tuned_cps", m_b, {"best_cps": best_cps}),
                                   ("C_weekly_cps0.05", m_c, {})]:
                rows.append({"rank": rank, "product": product, "metric": metric,
                             "config": cfg, **extra, **mm})

            print(f"  {metric:11s}  A SumErr={m_a['SumPctErr']:+6.1f}% sMAPE={m_a['sMAPE']:.0f}%  "
                  f"|  B(cps={best_cps}) SumErr={m_b['SumPctErr']:+6.1f}% sMAPE={m_b['sMAPE']:.0f}%  "
                  f"|  C(weekly) SumErr={m_c['SumPctErr']:+6.1f}% sMAPE={m_c['sMAPE']:.0f}%")

            # plot
            ax.plot(d["ds"], d["y"], "k.", markersize=2, alpha=0.4, label="actual (daily)")
            ax.plot(fcst_a["ds"].iloc[-len(hold_d):], pred_a, "C0-", linewidth=1.2, label=f"A daily cps=0.05")
            ax.plot(fcst_b["ds"].iloc[-len(hold_d):], pred_b, "C1-", linewidth=1.2, label=f"B daily cps={best_cps}")
            ax.plot(fcst_c["ds"].iloc[-len(hold_w):], pred_c, "C2o-", markersize=3, linewidth=1.2,
                    label="C weekly cps=0.05")
            ax.axvspan(cutoff, last, color="orange", alpha=0.10)
            ax.axvline(cutoff, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"{metric}: A SumErr {m_a['SumPctErr']:+.1f}%  |  "
                         f"B(cps={best_cps}) {m_b['SumPctErr']:+.1f}%  |  C(weekly) {m_c['SumPctErr']:+.1f}%")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

        fig.suptitle(f"#{rank}  {product}  — 90-day holdout backtest comparison",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{rank:02d}_{sname}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / f"_metrics_top{args.top}_holdout{args.holdout}d_compare.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nMetrics → {summary_path}")

    # Pivot for at-a-glance comparison of |SumPctErr|
    print("\n=== |SumPctErr| (lower is better) by config ===")
    pivot_sum = summary.assign(absSum=lambda x: x["SumPctErr"].abs()).pivot_table(
        index=["rank", "product", "metric"], columns="config", values="absSum"
    )
    print(pivot_sum.round(1).to_string())

    print("\n=== sMAPE (lower is better) by config ===")
    pivot_smape = summary.pivot_table(
        index=["rank", "product", "metric"], columns="config", values="sMAPE"
    )
    print(pivot_smape.round(1).to_string())

    # Winners
    winners = (summary.assign(absSum=lambda x: x["SumPctErr"].abs())
               .sort_values("absSum")
               .groupby(["rank", "product", "metric"])
               .first()[["config", "absSum", "sMAPE"]]
               .reset_index())
    winners_path = OUT_DIR / f"_winners_top{args.top}.csv"
    winners.to_csv(winners_path, index=False)
    print(f"\nWinners (by |SumPctErr|) → {winners_path}")
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
