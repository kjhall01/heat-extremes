"""Plot H/M/F/C maps from an existing by_lead_map .nc cache file.

Standalone re-plot: doesn't rerun the notebook or touch dask/the cluster --
just loads the cached NetCDF (`{model_source}_by_lead_map_..._..._..._..._....nc`,
written by Step 4 of deterministic_verification_metrics.ipynb) and draws the
same H/M/F/C figure(s) Step 4 already produces. Useful for re-plotting with
different lead days / region zoom without recomputing anything.

Usage:
    python plot_hmfc.py aifs_ens_mean_by_lead_map_2022_2022_west_africa_t2m_max_6h_abs35.0_rel0.95.nc
    python plot_hmfc.py <path.nc> --lead-days 1 3 5 7 --region-bounds 0 25 -20 20
"""
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROW_LABELS = {
    "hits": "Hits (H)",
    "misses": "Misses (M)",
    "false_alarms": "False alarms (F)",
    "correct_negatives": "Correct negatives (C)",
}


def select_by_lead_day(scores: xr.Dataset, lead_days: list[int]) -> xr.Dataset:
    """Same as _build_notebook.py's Step 4 helper -- nearest available
    prediction_timedelta bin to each nominal lead day, relabeled to that
    clean nominal day."""
    lead_timedeltas = np.array(lead_days).astype("timedelta64[D]").astype("timedelta64[ns]")
    selected = scores.sel(
        prediction_timedelta=lead_timedeltas, method="nearest", tolerance=np.timedelta64(12, "h")
    )
    return selected.assign_coords(prediction_timedelta=lead_timedeltas)


def plot_metric_grid(data_by_row, lead_days, suptitle, region_bounds=None):
    """Same as _build_notebook.py's Step 4/7 helper -- one row per metric,
    one column per lead day."""
    n_rows, n_cols = len(data_by_row), len(lead_days)
    figure, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False,
    )
    for row, (row_label, (data, plot_kwargs)) in enumerate(data_by_row.items()):
        for col, lead in enumerate(lead_days):
            axis = axes[row, col]
            data.sel(prediction_timedelta=np.timedelta64(lead, "D")).plot(
                ax=axis, x="longitude", y="latitude", transform=ccrs.PlateCarree(),
                add_colorbar=(col == n_cols - 1),
                cbar_kwargs={"label": row_label} if col == n_cols - 1 else None,
                **plot_kwargs,
            )
            if region_bounds is not None:
                south, north, west, east = region_bounds
                west = -180.0 if west is None else west
                east = 180.0 if east is None else east
                axis.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            else:
                axis.set_global()
            axis.coastlines(linewidth=0.5)
            axis.set_title(f"{row_label}, lead={lead}d" if row == 0 else f"lead={lead}d")
    figure.suptitle(suptitle)
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nc_path", type=Path, help="by_lead_map .nc cache file (from Step 4)")
    parser.add_argument("--lead-days", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument(
        "--region-bounds", type=float, nargs=4, default=None, metavar=("SOUTH", "NORTH", "WEST", "EAST"),
        help="zoom the map to this box instead of the whole globe (use 'nan' for west/east to leave that axis full-width)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output PNG path prefix (default: derived from nc_path)")
    args = parser.parse_args()

    region_bounds = None
    if args.region_bounds is not None:
        south, north, west, east = args.region_bounds
        region_bounds = (south, north, None if np.isnan(west) else west, None if np.isnan(east) else east)

    out_prefix = args.out if args.out is not None else args.nc_path.with_suffix("")

    by_lead_map = xr.open_dataset(args.nc_path)
    by_lead_map = by_lead_map.assign_coords(
        prediction_timedelta=(by_lead_map["prediction_timedelta"].values * np.timedelta64(1, "D")).astype(
            "timedelta64[ns]"
        )
    )

    for prefix in ("absolute", "relative"):
        columns = [f"{prefix}_{name}" for name in ROW_LABELS]
        if not all(column in by_lead_map for column in columns):
            print(f"No {prefix}_* columns in {args.nc_path} -- skipping.")
            continue
        hmfc_by_lead = select_by_lead_day(by_lead_map[columns], args.lead_days)
        figure = plot_metric_grid(
            {
                label: (hmfc_by_lead[f"{prefix}_{name}"], {"cmap": "viridis"})
                for name, label in ROW_LABELS.items()
            },
            args.lead_days,
            f"{prefix} threshold -- H/M/F/C ({args.nc_path.name})",
            region_bounds=region_bounds,
        )
        out_path = out_prefix.with_name(f"{out_prefix.name}_{prefix}_hmfc.png")
        figure.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
