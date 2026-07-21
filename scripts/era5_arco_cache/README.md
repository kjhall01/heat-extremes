# ECMWF ERA5 ARCO surface cache on DSI

This cache replaces the WeatherBench2 source with ECMWF's authenticated ERA5
Analysis Ready, Cloud Optimised (ARCO) Zarr archive. It uses ECMWF's
time-chunked store, which is the appropriate layout for full-globe data over a
short time span.

The local cache contains these canonical variables:

- `2m_temperature` — K, sampled at 00, 06, 12, and 18 UTC
- `2m_dewpoint_temperature` — K, sampled at 00, 06, 12, and 18 UTC
- `surface_pressure` — Pa, sampled at 00, 06, 12, and 18 UTC
- `total_precipitation` — m accumulated over the six hours ending at each timestamp

The native ARCO variables are `t2m`, `d2m`, `sp`, and `tp`. ECMWF provides
hourly `tp` accumulations, so the cache sums each contiguous six-hour window;
it does not drop five of every six precipitation observations.

Stores are 0.25-degree global fields (`latitude=721`, `longitude=1440`) with
target chunks `time=244`, `latitude=45`, `longitude=90`. Each year is a
separate consolidated Zarr v2 store with an `_SUCCESS` marker:

```text
/net/monsoon/kylehall/ERA5/era5_arco_6h_surface/era5_arco_6h_surface_YYYY.zarr
```

## One-time CDS setup

1. Register with the [Climate Data Store](https://cds.climate.copernicus.eu/),
   accept the ERA5 single-levels licence, and get the personal API token from
   the CDS profile page.
2. Make that token available on the compute nodes either as `CDSAPI_KEY` or in
   `~/.cdsapirc`:

   ```text
   url: https://cds.climate.copernicus.eu/api
   key: <personal-token>
   ```

   Do not put the token in the Slurm script or commit it to this repository.
3. Copy this directory to the cluster and create its environment:

   ```bash
   cd ~/era5_arco_cache
   conda env create -f environment.yml
   ```

## Smoke test

Create the output and log directories once, then run a recent complete year in
an interactive job. ARCO is updated continually, so use a year that is already
complete rather than the current year.

```bash
mkdir -p /net/monsoon/kylehall/ERA5/logs
mkdir -p /net/monsoon/kylehall/ERA5/era5_arco_6h_surface_TEST

conda activate era5-arco-cache
python -u cache_era5_arco_year.py \
    --year 2025 \
    --output-root /net/monsoon/kylehall/ERA5/era5_arco_6h_surface_TEST \
    --workers 8
```

## Submit and validate

The supplied array is `2000-2026%3`; the 2026 task automatically stops at the
latest complete six-hour ARCO timestep. Three concurrent jobs is intentionally
conservative for remote ARCO access. Re-run the current-year task with
`OVERWRITE_CURRENT_YEAR=1` when you want a newer partial-year snapshot.

```bash
cd ~/era5_arco_cache
sbatch submit_era5_arco_cache.sbatch
python validate_era5_arco_cache.py --end-year 2026 --allow-partial-final-year
```

Refresh only the current-year snapshot later with:

```bash
OVERWRITE_CURRENT_YEAR=1 sbatch --array=26 submit_era5_arco_cache.sbatch
```

Completed stores are skipped. Recreate a year deliberately with
`cache_era5_arco_year.py --year YYYY --overwrite`; it will replace only that
year's store. Failed staging stores are discarded before a rerun.

## Open the cache

```python
from heatextremes import daily_era5_aggregates, open_cached_era5

ds = open_cached_era5(start_year=2000, end_year=2024)
daily = daily_era5_aggregates(ds)
```

The `daily_era5_aggregates` helper shifts the end-labelled six-hour
precipitation blocks before summing. This yields actual UTC-day totals. Its
last day is missing until the next 00 UTC cache timestep is present, preventing
a silent partial-day precipitation total. Temperature summaries remain based on
four daily samples, so `t2m_max_6h` is not a true hourly daily maximum.
