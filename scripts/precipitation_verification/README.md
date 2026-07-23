# AIFS ENS v2 daily precipitation verification pipeline

This is the precipitation counterpart to `../t2m_verification`.  It handles
the two different temporal representations explicitly:

- AIFS `tp` has a `prediction_timedelta_daily` dimension with daily totals
  labelled 1--50 days.  The forecast stage selects those values directly; it
  does **not** sum them again.
- Cached ERA5 `total_precipitation` is a 6-hour accumulation labelled by its
  end time.  For forecast day *n*, the ERA5 stage samples the four values at
  leads `(n - 1) * 24 + [6, 12, 18, 24]` h and sums them.  Thus day 1 is the
  0--24 h accumulation and has no date/accumulation-label shift error.

Each selected day requires all four ERA5 values.  A missing value gives a
missing daily total, never a silently partial accumulation.

## Default run and outputs

The supplied batch scripts cover forecast years 2022--2025 and days 1, 3, 5,
7, and 9.  Change `FORECAST_DAYS` to select any labels available in the model,
including `1 2 ... 50`.

```text
/net/monsoon/kylehall/heat_extremes/precipitation_verification/
├── aifs_daily_precipitation.zarr/
├── era5_daily_precipitation_matched.zarr/
├── precipitation_case_metrics.zarr/
├── precipitation_metric_summary.zarr/
├── figures/
└── logs/
```

The default metrics job writes 90% central-interval coverage only.  No
precipitation PoE threshold is assumed.  Add thresholds in millimetres to
enable member-count PoE, Brier score, and the compact hit/miss/false-positive
contingency diagnostics.  The AIFS and ERA5 precipitation values are assumed
to be metres by default, so these thresholds are converted from mm to m.

## Submit

```bash
cd /path/to/heat-extremes/scripts/precipitation_verification
bash submit_precipitation_verification.sh
```

For all 50 model days, submit with:

```bash
FORECAST_DAYS="$(seq 1 50)" bash submit_precipitation_verification.sh
```

For example, to add 1, 10, and 25 mm PoE diagnostics with a 50% forecast
probability decision threshold:

```bash
OVERWRITE=1 PRECIP_THRESHOLDS_MILLIMETERS="1 10 25" \
POE_DECISION_THRESHOLDS="0.5" \
sbatch 03_calculate_precipitation_metrics.sbatch
```

Then remake the summary and figures with:

```bash
OVERWRITE=1 sbatch 04_aggregate_and_plot_precipitation_metrics.sbatch
```

As in the T2M pipeline, completed Zarr stores have `_SUCCESS` markers and
stages refuse to overwrite an existing store without `OVERWRITE=1`.
