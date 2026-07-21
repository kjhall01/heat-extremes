# WeatherBench2 ERA5 surface cache on DSI

This caches the following variables from the public WeatherBench2 0.25-degree,
6-hourly ERA5 dataset:

- `2m_temperature`
- `2m_dewpoint_temperature`
- `surface_pressure`

Default period: 2001 through the source endpoint on 2023-01-10.

Output location:

```text
/net/monsoon/kylehall/ERA5/wb2_era5_6h_surface/
```

Each year is a separate, consolidated Zarr v2 store. A completed store contains
an `_SUCCESS` marker. The cache preserves the official WeatherBench2 variable
names and Kelvin/Pa units.

## Why one store per year?

- Slurm tasks never write to the same Zarr arrays.
- Failed or preempted years can be rerun independently.
- Each year is staged, validated, and atomically renamed.
- Twenty-three stores are still easy to open as one lazy Xarray dataset.

Target chunks are:

```text
time=244, latitude=45, longitude=90
```

At 6-hourly resolution, 244 time steps are 61 days. A full variable chunk is
about 3.8 MiB uncompressed. This is a reasonable compromise for time-series
climatology and global daily aggregation.

## 1. Put these scripts on the cluster

For example:

```bash
mkdir -p ~/wb2_era5_cache
cd ~/wb2_era5_cache
# Copy the files from this package here.
```

## 2. Create the conda environment

```bash
cd ~/wb2_era5_cache
conda env create -f environment.yml
```

If you use a different environment name, set it before submission:

```bash
export WB2_CACHE_ENV=my_environment
```

## 3. Create output and log directories

Run this once on the login node:

```bash
mkdir -p /net/monsoon/kylehall/ERA5/logs
mkdir -p /net/monsoon/kylehall/ERA5/wb2_era5_6h_surface
```

Check available space and write permission:

```bash
df -h /net/monsoon/kylehall/ERA5
touch /net/monsoon/kylehall/ERA5/.write_test
rm /net/monsoon/kylehall/ERA5/.write_test
```

## 4. Smoke-test the short 2023 segment

The WB2 source ends on 2023-01-10, so 2023 is a cheap test.

Start an interactive CPU session, then run:

```bash
srun -p general --qos=interactive \
    --cpus-per-task=8 --mem=16G --time=00:45:00 --pty bash

conda activate wb2-cache

python -u ~/wb2_era5_cache/cache_wb2_era5_year.py \
    --year 2023 \
    --output-root /net/monsoon/kylehall/ERA5/wb2_era5_6h_surface_TEST \
    --workers 8
```

Inspect it:

```bash
du -sh /net/monsoon/kylehall/ERA5/wb2_era5_6h_surface_TEST/*
```

Then remove the test store:

```bash
rm -rf /net/monsoon/kylehall/ERA5/wb2_era5_6h_surface_TEST
```

## 5. Submit the full array

```bash
cd ~/wb2_era5_cache
sbatch submit_wb2_era5_cache.sbatch
```

The array is `2001-2023%3`, meaning no more than three yearly transfers run at
once. This is intentionally conservative for the shared internet connection
and `/net` storage. Increase to `%4` only after observing good behavior.

Monitor:

```bash
squeue -u "$USER"
tail -f /net/monsoon/kylehall/ERA5/logs/wb2_era5_cache_<JOBID>_2001.out
```

Inspect accounting after completion:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

## 6. Rerun failed years

Completed stores are skipped automatically.

Examples:

```bash
sbatch --array=2007 submit_wb2_era5_cache.sbatch
sbatch --array=2007,2014,2021 submit_wb2_era5_cache.sbatch
```

Force replacement of a completed year by running the Python script manually
with `--overwrite`, or remove that yearly store before resubmission.

## 7. Validate the complete cache

```bash
conda activate wb2-cache
python validate_wb2_era5_cache.py
```

## 8. Open the cache

```python
from open_cached_era5 import open_cached_era5

ds = open_cached_era5()
print(ds)
```

The dataset stays lazy. Example UTC-day summaries:

```python
import xarray as xr

daily = xr.Dataset({
    "t2m_mean_6h": ds["2m_temperature"].resample(time="1D").mean(),
    "t2m_max_6h": ds["2m_temperature"].resample(time="1D").max(),
    "t2d_mean_6h": ds["2m_dewpoint_temperature"].resample(time="1D").mean(),
    "sp_mean_6h": ds["surface_pressure"].resample(time="1D").mean(),
})
```

These are based on 00, 06, 12, and 18 UTC samples. `t2m_max_6h` is therefore
the maximum of four daily samples, not a true hourly daily maximum.
