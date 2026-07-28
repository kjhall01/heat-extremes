"""Loader for the deterministic GraphCast reforecast.

Adapted from `aifs_singlev2.open_aifs_singlev2()` -- same pattern (per-
initialization `init_YYYYMMDDT00.zarr` stores, `start_year`/`end_year`
filename-based filtering to avoid opening the whole archive's metadata just
to restrict a year range, single/deterministic -- no ensemble `"number"`
dimension, matching the `-1` ensemble-member count given for this model).

CAVEAT (more so than `aifs_singlev2.py`'s own caveat): this has NOT been run
against the real store, and unlike AIFS-single-v2's `ls`-confirmed filename
pattern, GraphCast's real directory listing hasn't been seen at all here.
Everything below is inferred from the one-line spec given for this model
(years 2000-2025, `-1` ensemble members, ~weekly initializations, 50-day
lead time, the variable list below, path
`/net/monsoon/marchakitus/reforecast/forecasts_graphcast_e2s`) plus the
assumption that, since this lives under the same
`/net/monsoon/marchakitus/reforecast/` parent as AIFS-single-v2's own store,
it was very likely produced by the same pipeline and follows the same
`init_YYYYMMDDT00.zarr`-per-initialization, 6-hourly-step convention. Confirm
against the real store (e.g. `ls` the path, `xr.open_zarr` one store and
inspect its repr) before trusting any of this -- the glob pattern, the
short-name-to-descriptive rename map, the assumed 6-hourly step chunking,
and even whether `lat`/`lon` are the real coordinate names, are all
unconfirmed guesses by analogy, not verified facts, the way AIFS-single-v2's
now are (after this project's own real-repr confirmation).

**No `2m_dewpoint_temperature` in GraphCast's variable list** (only `2t`,
2m temperature -- no `2d`) -- wet-bulb temperature (`wetbulb.py`) CANNOT be
computed for GraphCast with the variables given. If dewpoint turns out to
be available under some other name in the real store, add it to `wanted`
and the rename map below; until then, don't wire a `t_wb_2m_*` variable
option in a notebook that uses this loader.

Variable list given (short names, presumed to match the store's real
internal names the way `2d`/`2t`/`tp` did for AIFS):
`2t` (2m temperature), `mslp` (mean sea-level pressure), `q_1000`/`q_850`/
`q_925` (specific humidity at 1000/850/925 hPa), `t_1000`/`t_850`/`t_925`
(temperature at 1000/850/925 hPa), `tp` (total precipitation), `u_200`/
`u_50`/`u_850` and `v_200`/`v_50`/`v_850` (wind components at 200/50/850 hPa),
`z_200`/`z_500`/`z_850` (geopotential at 200/500/850 hPa). Only `2t`/`tp` are
renamed to match this pipeline's existing `2m_temperature`/
`total_precipitation` naming convention (so `daily_aifs_aggregates_calendar_aligned`-
style helpers could be pointed at them unchanged) -- the pressure-level
fields are left under their native short names since this pipeline has no
established descriptive-name convention for them yet.
"""

import re
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

# Assumed, not confirmed for this store specifically -- see module docstring.
_INIT_STORE_FILENAME = re.compile(r"^init_(\d{4})(\d{2})(\d{2})T\d{2}\.zarr$")

# Native short variable names assumed present in each store -- see module
# docstring for what each one is and the wet-bulb/dewpoint caveat.
GRAPHCAST_VARIABLES = [
    "2t", "mslp",
    "q_1000", "q_850", "q_925",
    "t_1000", "t_850", "t_925",
    "tp",
    "u_200", "u_50", "u_850",
    "v_200", "v_50", "v_850",
    "z_200", "z_500", "z_850",
]

# Only variables with an existing descriptive-name convention elsewhere in
# this pipeline get renamed; everything else keeps its native short name.
_RENAME = {"2t": "2m_temperature", "tp": "total_precipitation", "lat": "latitude", "lon": "longitude"}


def _init_year(path: Path) -> int:
    """Parse the initialization year from an `init_YYYYMMDDT00.zarr` store name.

    Same helper as `aifs_singlev2._init_year`, duplicated here (not imported)
    so this module stays self-contained -- see that module's docstring for
    why loader modules in this project each own their own small amount of
    shared-shaped logic rather than depending on each other.
    """
    match = _INIT_STORE_FILENAME.match(path.name)
    if not match:
        raise ValueError(
            f"Unrecognized GraphCast store filename: {path.name!r} "
            "(expected 'init_YYYYMMDDT00.zarr' -- unconfirmed for this store, see module docstring)"
        )
    return int(match.group(1))


def open_graphcast(
    start_year: int | None = None,
    end_year: int | None = None,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Open the deterministic GraphCast reforecast (2000-2025, no ensemble dimension).

    Parameters
    ----------
    start_year, end_year : int, optional
        If given, only initializations with `start_year <= year <= end_year`
        are opened (each bound independently optional). Parsed from each
        store's filename *before* calling `xr.open_mfdataset`, avoiding
        opening zarr metadata for stores outside the requested range --
        same approach as `aifs_singlev2.open_aifs_singlev2`. Omit both to
        open every available initialization.
    variables : list of str, optional
        Which raw short variable names (from `GRAPHCAST_VARIABLES`) to
        actually load -- e.g. `["2t"]` if a caller only needs 2m temperature,
        rather than paying the IO cost of all 18 variables (12 of them
        pressure-level fields this pipeline doesn't otherwise use). Defaults
        to `GRAPHCAST_VARIABLES` (everything) if omitted.

    See this module's docstring for the (substantial) caveats: this has not
    been run against, or even had its directory listing seen against, the
    real store.
    """
    root = Path("/net/monsoon/marchakitus/reforecast/forecasts_graphcast_e2s")
    paths = sorted(root.glob("*.zarr"))
    for path in paths:
        assert path.is_dir(), f"{path} does not exist?"

    if start_year is not None or end_year is not None:
        paths = [
            path
            for path in paths
            if (start_year is None or _init_year(path) >= start_year)
            and (end_year is None or _init_year(path) <= end_year)
        ]
        if not paths:
            raise FileNotFoundError(
                f"No GraphCast initialization stores found under {root} "
                f"for start_year={start_year}, end_year={end_year}"
            )

    wanted = GRAPHCAST_VARIABLES if variables is None else variables

    with ProgressBar():
        ds = xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="nested",
            concat_dim="time",
            preprocess=lambda x: x[wanted],
            chunks={
                "time": 1,  # unavoidable: one time per store
                "prediction_timedelta": 24,  # assumed 6-hourly steps, unconfirmed -- see module docstring
                "latitude": 180,
                "longitude": 180,
            },
            parallel=True,
            data_vars="all",
            coords="minimal",
            compat="override",
            join="override",
            combine_attrs="override",
            consolidated=None,
        )
    # Only rename keys actually present (e.g. "tp" isn't loaded at all if the
    # caller restricted variables=["2t"]) -- xr.Dataset.rename() raises if
    # asked to rename something that doesn't exist.
    rename_map = {name: renamed for name, renamed in _RENAME.items() if name in ds.variables}
    return ds.rename(rename_map)
