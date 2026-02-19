#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import warnings
from collections import defaultdict
from pathlib import Path

PROJECT_MPLCONFIG = Path(__file__).resolve().parents[1] / ".mplconfig"
PROJECT_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_MPLCONFIG))
os.environ.setdefault("USE_PYGEOS", "0")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore", message=r"The Shapely GEOS version .*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Shapely 2.0 is installed.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"The geopandas.dataset module is deprecated.*", category=FutureWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    direct = p.resolve()
    if direct.exists():
        return direct
    root_relative = (ROOT_DIR / p).resolve()
    if root_relative.exists():
        return root_relative
    return root_relative


def get_cmap_obj(cmap_name: str | mcolors.Colormap) -> mcolors.Colormap:
    if isinstance(cmap_name, mcolors.Colormap):
        return cmap_name
    if hasattr(plt, "colormaps"):
        return plt.colormaps.get_cmap(cmap_name)
    return cm.get_cmap(cmap_name)



def build_locations_with_groups(locations_csv: Path, contrib_csv: Path) -> pd.DataFrame:
    locations_csv = resolve_path(locations_csv)
    contrib_csv = resolve_path(contrib_csv)
    df_locations = pd.read_csv(locations_csv)
    df_contrib = pd.read_csv(contrib_csv)

    group_df = (
        df_contrib.groupby("project_id")["user_id"]
        .agg(lambda s: set(s.dropna()))
        .reset_index()
        .rename(columns={"user_id": "groups"})
    )

    merged = df_locations.merge(group_df, on="project_id", how="inner")
    merged["group_size"] = merged["groups"].apply(len)

    required = {"longitude", "latitude"}
    missing = required.difference(merged.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {locations_csv}: {sorted(missing)}. "
            "Expected longitude/latitude for plotting."
        )
    return merged


def plot_projects_global(
    df: pd.DataFrame,
    overlap_metric: str = "jaccard",
    overlap_threshold: float = 0.1,
    min_group_size: int = 1,
    max_edges: int | None = None,
    node_size_min: float = 8,
    node_size_max: float = 80,
    edge_alpha: float = 0.12,
    edge_lw: float = 0.5,
    edge_color: str = "#FF0000",
    cmap_name: str | mcolors.Colormap = "viridis",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (8, 6),
    focus_bbox: tuple[float, float, float, float] | None = None,
    subset_only: bool = True,
    clip_basemap: bool = True,
):
    import geopandas as gpd
    from shapely.geometry import LineString, Point, box

    df = df.dropna(subset=["longitude", "latitude", "groups", "group_size"]).copy()
    df = df[df["group_size"] >= min_group_size].copy()
    if df.empty:
        raise ValueError("No projects to plot after filtering.")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    records = gdf[["project_id", "longitude", "latitude", "groups", "group_size"]].reset_index(drop=True)
    edges = []
    for i in range(len(records)):
        gi = records.at[i, "groups"]
        pi = records.at[i, "project_id"]
        xi, yi = records.at[i, "longitude"], records.at[i, "latitude"]
        for j in range(i + 1, len(records)):
            gj = records.at[j, "groups"]
            pj = records.at[j, "project_id"]
            xj, yj = records.at[j, "longitude"], records.at[j, "latitude"]
            inter = len(gi & gj)
            if overlap_metric == "jaccard":
                uni = len(gi | gj)
                if uni == 0:
                    continue
                score = inter / uni
            elif overlap_metric == "intersection":
                score = inter
            else:
                raise ValueError("overlap_metric must be 'jaccard' or 'intersection'.")
            if score >= overlap_threshold:
                edges.append(
                    {
                        "src": pi,
                        "dst": pj,
                        "weight": score,
                        "geometry": LineString([(xi, yi), (xj, yj)]),
                    }
                )

    edges_gdf = (
        gpd.GeoDataFrame(edges, crs="EPSG:4326")
        if edges
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )
    if max_edges is not None and len(edges_gdf) > max_edges:
        edges_gdf = edges_gdf.sort_values("weight", ascending=False).head(max_edges).copy()

    if focus_bbox is not None:
        lon_min, lon_max, lat_min, lat_max = focus_bbox
        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
        if subset_only:
            gdf = gdf[gdf.geometry.within(bbox_poly)].copy()
            if not edges_gdf.empty:
                edges_gdf = edges_gdf[edges_gdf.geometry.intersects(bbox_poly)].copy()
        world_plot = world.clip(bbox_poly) if clip_basemap else world
    else:
        world_plot = world

    if gdf.empty:
        raise ValueError("No points left after bbox filtering.")

    sizes = gdf["group_size"].astype(float).values
    s_min, s_max = sizes.min(), sizes.max()
    norm = mcolors.Normalize(vmin=s_min, vmax=s_max)
    cmap = get_cmap_obj(cmap_name)
    colors = [cmap(norm(v)) for v in sizes]
    if s_max == s_min:
        marker_sizes = np.full_like(sizes, (node_size_min + node_size_max) / 2.0)
    else:
        sz = (np.sqrt(sizes) - np.sqrt(s_min)) / (np.sqrt(s_max) - np.sqrt(s_min))
        marker_sizes = node_size_min + sz * (node_size_max - node_size_min)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    world_plot.plot(ax=ax, color="#F9FAFB", edgecolor="#BDBDBD", linewidth=0.5)
    if not edges_gdf.empty:
        edges_gdf.plot(ax=ax, color=edge_color, linewidth=edge_lw, alpha=edge_alpha, zorder=1)
    gdf.plot(
        ax=ax,
        markersize=marker_sizes,
        color=colors,
        edgecolor="black",
        linewidth=0.2,
        alpha=0.95,
        zorder=2,
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.6)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Group Size", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    if focus_bbox is not None:
        ax.set_xlim(focus_bbox[0], focus_bbox[1])
        ax.set_ylim(focus_bbox[2], focus_bbox[3])
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_projects_global_pairwise_highorder(
    df: pd.DataFrame,
    overlap_metric: str = "jaccard",
    overlap_threshold: float = 0.1,
    min_group_size: int = 1,
    max_edges_each: int | None = None,
    node_size_min: float = 8,
    node_size_max: float = 80,
    node_size_uniform: float | None = None,
    cmap_name: str | mcolors.Colormap = "Blues",
    pair_color: str = "#BD4B4B",
    pair_lw: float = 1.2,
    pair_alpha: float = 0.8,
    high_color: str = "#EFE867",
    high_lw: float = 2.0,
    high_alpha: float = 0.9,
    bg_edge_alpha: float = 0.02,
    bg_edge_lw: float = 0.4,
    bg_edge_color: str = "#9CA3AF",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (8, 6),
):
    import geopandas as gpd
    from matplotlib.lines import Line2D
    from shapely.geometry import LineString, Point

    df = df.dropna(subset=["longitude", "latitude", "groups", "group_size"]).copy()
    df = df[df["group_size"] >= min_group_size].copy()
    if df.empty:
        raise ValueError("No projects to plot after filtering.")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    sizes = gdf["group_size"].astype(float).values
    s_min, s_max = sizes.min(), sizes.max()
    norm = mcolors.Normalize(vmin=s_min, vmax=s_max)
    cmap = get_cmap_obj(cmap_name)
    colors = [cmap(norm(v)) for v in sizes]
    if node_size_uniform is not None:
        marker_sizes = np.full_like(sizes, float(node_size_uniform))
    elif s_max == s_min:
        marker_sizes = np.full_like(sizes, (node_size_min + node_size_max) / 2.0)
    else:
        sz = (np.sqrt(sizes) - np.sqrt(s_min)) / (np.sqrt(s_max) - np.sqrt(s_min))
        marker_sizes = node_size_min + sz * (node_size_max - node_size_min)

    records = gdf[["project_id", "longitude", "latitude", "groups"]].reset_index(drop=True)
    edges_all = []
    for i in range(len(records)):
        gi = set(records.at[i, "groups"])
        pi = records.at[i, "project_id"]
        xi, yi = float(records.at[i, "longitude"]), float(records.at[i, "latitude"])
        for j in range(i + 1, len(records)):
            gj = set(records.at[j, "groups"])
            pj = records.at[j, "project_id"]
            xj, yj = float(records.at[j, "longitude"]), float(records.at[j, "latitude"])
            inter = len(gi & gj)
            if overlap_metric == "jaccard":
                uni = len(gi | gj)
                if uni == 0:
                    continue
                score = inter / uni
            elif overlap_metric == "intersection":
                score = inter
            else:
                raise ValueError("overlap_metric must be 'jaccard' or 'intersection'.")
            if score >= overlap_threshold:
                edges_all.append(
                    {"src": pi, "dst": pj, "weight": float(score), "geometry": LineString([(xi, yi), (xj, yj)])}
                )

    neighbors = defaultdict(set)
    for e in edges_all:
        neighbors[e["src"]].add(e["dst"])
        neighbors[e["dst"]].add(e["src"])

    pair_edges, high_edges = [], []
    for e in edges_all:
        common = neighbors[e["src"]].intersection(neighbors[e["dst"]])
        if common:
            high_edges.append(e)
        else:
            pair_edges.append(e)

    if max_edges_each is not None:
        pair_edges = sorted(pair_edges, key=lambda x: x["weight"], reverse=True)[:max_edges_each]
        high_edges = sorted(high_edges, key=lambda x: x["weight"], reverse=True)[:max_edges_each]

    pair_gdf = (
        gpd.GeoDataFrame(pair_edges, crs="EPSG:4326")
        if pair_edges
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )
    high_gdf = (
        gpd.GeoDataFrame(high_edges, crs="EPSG:4326")
        if high_edges
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )
    all_gdf = (
        gpd.GeoDataFrame(edges_all, crs="EPSG:4326")
        if edges_all
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    world.plot(ax=ax, color="#F9FAFB", edgecolor="#BDBDBD", linewidth=0.5)
    if not all_gdf.empty:
        all_gdf.plot(ax=ax, color=bg_edge_color, linewidth=bg_edge_lw, alpha=bg_edge_alpha, zorder=0)
    if not pair_gdf.empty:
        pair_gdf.plot(ax=ax, color=pair_color, linewidth=pair_lw, alpha=pair_alpha, zorder=1)
    if not high_gdf.empty:
        high_gdf.plot(ax=ax, color=high_color, linewidth=high_lw, alpha=high_alpha, zorder=2)
    gdf.plot(ax=ax, markersize=marker_sizes, color=colors, edgecolor="white", linewidth=0.3, alpha=0.95, zorder=3)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Group Size")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.35)
    ax.legend(
        handles=[
            Line2D([0], [0], color=pair_color, lw=pair_lw, alpha=pair_alpha, label="Pairwise-only"),
            Line2D([0], [0], color=high_color, lw=high_lw, alpha=high_alpha, label="Higher-order"),
        ],
        loc="lower left",
        frameon=True,
    )

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax


def parse_bbox(bbox_str: str | None) -> tuple[float, float, float, float] | None:
    if bbox_str is None:
        return None
    vals = [float(v.strip()) for v in bbox_str.split(",")]
    if len(vals) != 4:
        raise ValueError("--bbox must be 'lon_min,lon_max,lat_min,lat_max'.")
    return vals[0], vals[1], vals[2], vals[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MapSwipe network visualizations from notebook logic.")
    parser.add_argument(
        "--locations-csv",
        type=Path,
        default=Path("data/completed_projects_with_coords.csv"),
        help="CSV with at least project_id, longitude, latitude.",
    )
    parser.add_argument(
        "--contrib-csv",
        type=Path,
        default=Path("data/mapswipe_user_contributions_by_date_201909_202510.csv"),
    )
    parser.add_argument(
        "--mode",
        choices=["global", "zoom", "both", "highorder"],
        default="both",
    )
    parser.add_argument("--overlap-metric", choices=["jaccard", "intersection"], default="jaccard")
    parser.add_argument("--overlap-threshold", type=float, default=0.1)
    parser.add_argument("--min-group-size", type=int, default=0)
    parser.add_argument("--bbox", type=str, default="10,50,-40,20")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--prefix", type=str, default="mapswipe")
    args = parser.parse_args()

    df = build_locations_with_groups(args.locations_csv, args.contrib_csv)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", ["#60A5FA", "#152D73"])

    if args.mode in ("global", "both"):
        out = args.output_dir / f"{args.prefix}_global.png"
        plot_projects_global(
            df,
            overlap_metric=args.overlap_metric,
            overlap_threshold=args.overlap_threshold,
            min_group_size=args.min_group_size,
            max_edges=None,
            cmap_name=cmap,
            figsize=(8, 6),
            edge_alpha=0.02,
            edge_lw=0.5,
            edge_color="#9CA3AF",
            focus_bbox=None,
            subset_only=True,
            save_path=out,
        )
        plt.close("all")
        print(f"Saved: {out}")

    if args.mode in ("zoom", "both"):
        out = args.output_dir / f"{args.prefix}_zoom.png"
        plot_projects_global(
            df,
            overlap_metric=args.overlap_metric,
            overlap_threshold=args.overlap_threshold,
            min_group_size=args.min_group_size,
            max_edges=None,
            cmap_name=cmap,
            figsize=(4, 6),
            edge_alpha=0.1,
            edge_lw=0.5,
            edge_color="#9CA3AF",
            focus_bbox=parse_bbox(args.bbox),
            subset_only=True,
            save_path=out,
        )
        plt.close("all")
        print(f"Saved: {out}")

    if args.mode == "highorder":
        out = args.output_dir / f"{args.prefix}_highorder.png"
        plot_projects_global_pairwise_highorder(
            df,
            overlap_metric=args.overlap_metric,
            overlap_threshold=args.overlap_threshold,
            min_group_size=args.min_group_size,
            save_path=out,
        )
        plt.close("all")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
