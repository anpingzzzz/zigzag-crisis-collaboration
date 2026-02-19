#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def roll_mean(x: np.ndarray, w: int = 120) -> np.ndarray:
    return pd.Series(x, dtype=float).rolling(window=w, min_periods=1).mean().values


def spearman_corr_no_scipy(x: np.ndarray, y: np.ndarray) -> float:
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    valid = sx.notna() & sy.notna()
    if valid.sum() < 2:
        return np.nan

    rx = sx[valid].rank(method="average").to_numpy(dtype=float)
    ry = sy[valid].rank(method="average").to_numpy(dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0:
        return np.nan
    return float((rx * ry).sum() / denom)


def compute_daily_eta(contrib_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(contrib_csv, parse_dates=["start_dates"])
    df = df.sort_values("start_dates")

    co_occur = defaultdict(int)
    results = []

    for date, day_df in tqdm(df.groupby("start_dates"), total=df["start_dates"].nunique()):
        for project_id, g in day_df.groupby("project_id"):
            users = list(g["user_id"].unique())
            n = len(users)
            t_s = n

            if n < 2:
                t_c = 0.0
            else:
                pair_count = sum(
                    1
                    for u1, u2 in combinations(users, 2)
                    if co_occur[tuple(sorted((u1, u2)))] > 0
                )
                t_c = pair_count / (n * (n - 1) / 2)

            user_agree = (
                g.dropna(subset=["simple_agreement_score"])
                .groupby("user_id")["simple_agreement_score"]
                .mean()
            )
            t_f = float(user_agree.mean()) if len(user_agree) else 0.0
            t_f = float(np.clip(t_f, 0.0, 1.0))

            results.append(
                {
                    "start_dates": date,
                    "project_id": project_id,
                    "t_s": t_s,
                    "t_c": t_c,
                    "t_f": t_f,
                }
            )

        for _, g in day_df.groupby("project_id"):
            users = list(g["user_id"].unique())
            for u1, u2 in combinations(users, 2):
                co_occur[tuple(sorted((u1, u2)))] += 1

    df_effectiveness = pd.DataFrame(results)
    df_effectiveness["eta"] = (
        df_effectiveness["t_s"] * df_effectiveness["t_c"] * df_effectiveness["t_f"]
    )
    return df_effectiveness.groupby("start_dates")["eta"].mean().reset_index()


def spearman_all_dims(
    betti_curves: dict,
    eff_series: pd.Series,
    win: int = 30,
    dims: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    eff = np.asarray(eff_series.values, dtype=float)
    rows = []
    for key, bc in betti_curves.items():
        for d in dims:
            dim_name = f"dim_{d}"
            if dim_name not in bc:
                continue
            x = np.asarray(bc[dim_name], dtype=float)
            n = min(len(x), len(eff))
            if n == 0:
                continue
            xr = normalize(roll_mean(x[:n], win))
            er = normalize(roll_mean(eff[:n], win))
            rho = spearman_corr_no_scipy(xr, er)
            rows.append(
                {
                    "key": key,
                    "dim": dim_name,
                    "spearman_r": float(rho) if pd.notna(rho) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["key", "dim"]).reset_index(drop=True)


def plot_violin(result_df: pd.DataFrame, output_path: Path) -> None:
    df = result_df[["dim", "spearman_r"]].dropna().copy()
    order = ["dim_0", "dim_1"]
    df["dim"] = pd.Categorical(df["dim"], categories=order, ordered=True)
    data = [df.loc[df["dim"] == k, "spearman_r"].values for k in order]

    fig, ax = plt.subplots(figsize=(3.8, 3))
    parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)

    for body in parts["bodies"]:
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([r"$\beta_0$", r"$\beta_1$"], fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title(r"Betti Curves vs. $\eta$ from MapSwipe", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MapSwipe topological analysis"
    )
    parser.add_argument(
        "--contrib-csv",
        type=Path,
        default=Path("data/mapswipe_user_contributions_by_date_201909_202510.csv"),
    )
    parser.add_argument(
        "--betti-pkl",
        type=Path,
        default=Path("results/mapswipe_data_results/betti_curve_mapswipe.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/betti_effectiveness_correlation_mapswipe.png"),
    )
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    daily_eta = compute_daily_eta(args.contrib_csv)
    betti_curves = pd.read_pickle(args.betti_pkl)
    result_df = spearman_all_dims(
        betti_curves=betti_curves,
        eff_series=daily_eta["eta"],
        win=args.window,
        dims=(0, 1),
    )
    plot_violin(result_df, args.output)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
