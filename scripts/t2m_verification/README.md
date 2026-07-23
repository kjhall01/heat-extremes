# AIFS ENS v2 T2M verification pipeline

This is the batch version of the top-level daily-temperature notebook.  It
writes restart-safe intermediate Zarr stores, each with an `_SUCCESS` marker,
so the expensive case-wise data can be subset and re-aggregated later without
recalculating ensemble quantiles or member counts.

The supplied configuration is deliberately the requested small first run:

- forecast years 2022--2025;
- forecast days 1, 3, 5, 7, and 9;
- 90% central-interval coverage;
- member-count probability of exceeding 35 °C and its Brier score;
- 50%-PoE hit, miss, and false-positive rates;
- T2M minimum, mean, and maximum in every intermediate.

## Forecast-day convention

Forecast day *n* is the completed 24-hour window containing leads
`(n - 1) * 24 + [6, 12, 18, 24]` hours.  In particular:

| Forecast day | Leads used |
| --- | --- |
| 1 | 6, 12, 18, 24 h |
| 3 | 54, 60, 66, 72 h |
| 5 | 102, 108, 114, 120 h |
| 7 | 150, 156, 162, 168 h |
| 9 | 198, 204, 210, 216 h |

This intentionally excludes the initialization analysis (0 h).  Grouping
timedeltas by `.days` would instead put 6, 12, and 18 h in a day-0 group and
is not used here.

## Submit

Create the default log directory and submit the dependent chain from a login
node:

```bash
cd /path/to/heat-extremes/scripts/t2m_verification
bash submit_t2m_verification.sh
```

The scripts expect the `heat-extremes` Conda environment at
`/home/kylehall/miniconda3/envs/heat-extremes`, like the rest of this project.
Override that or the storage location when needed:

```bash
PIPELINE_ROOT=/net/monsoon/kylehall/another_run \
HEAT_EXTREMES_ENV=heat-extremes \
bash submit_t2m_verification.sh
```

The stages are:

1. `aifs_t2m_daily.zarr` — ensemble T2M min/mean/max by initialization,
   forecast day, member, and grid point.
2. `era5_t2m_daily_matched.zarr` — ERA5 matching the exact four valid times
   used for each forecast case and aggregation.
3. `t2m_case_metrics.zarr` — case-wise coverage, PoE, and Brier score.  This
   is the store to use for later regional or seasonal subsetting.  It also
   holds a signed-byte PoE contingency category for each case.
4. `t2m_metric_summary.zarr` plus three global forecast-day line plots in
   `figures/`.

Each stage refuses to replace an existing store.  To intentionally recreate a
stage, submit that stage (and its downstream stages) with `OVERWRITE=1`.
Incomplete stores have no `_SUCCESS` marker and must also be rerun with that
flag.

## Expand the metric grid later

The metric stage already accepts multiple values.  The supplied Slurm script
uses environment variables, so the eventual full requested grid is:

```bash
OVERWRITE=1 COVERAGE_PERCENTILES="10 20 30 40 50 60 70 80 90" \
THRESHOLDS_CELSIUS="29 32 35" \
POE_DECISION_THRESHOLDS="0.1 0.3 0.5 0.7 0.9" \
sbatch 03_calculate_t2m_metrics.sbatch
```

Run stage 4 with `OVERWRITE=1` afterward to make summaries and plots for those
additional variables.  The AIFS and ERA5 intermediates do not need rebuilding.

Both source datasets are in kelvin, so a named threshold of 35 °C is evaluated
as 308.15 K; the output attributes retain both values.  An event is defined as
strictly `T > threshold`, and PoE is the fraction of valid ensemble members
that meet it.  For the rate diagnostics, a forecast event occurs when PoE is
at least `POE_DECISION_THRESHOLDS` (0.5 by default).  Hit and miss rates are
conditioned on an observed event; the false-positive rate is conditioned on an
observed non-event.  The compact contingency coding is `0=true negative`,
`1=hit`, `2=miss`, `3=false positive`, and `-1=missing`.
