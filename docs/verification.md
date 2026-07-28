# Modular heat-forecast verification

## Architecture

`heatextremes.verification` separates source access from science and output:

- `models/` adapts a source to canonical initialization, forecast day,
  latitude, longitude, ensemble-mean temperature, event probability, and
  observed-event fields;
- `alignment.py` is the vectorized valid-date lookup from the verified heat
  notebooks; it does not stack forecast cases or lose a MultiIndex;
- `case_cache.py` writes restartable canonical local-solar verification cases
  as consolidated Zarr v2 stores, one month/lead at a time;
- metric modules derive additive sufficient statistics from those cases;
- `runner.py` computes one monthly partition and one forecast day at a time,
  from either a source adapter or the case cache;
- `aggregation.py` combines partial numerators and denominators exactly;
- `plotting.py` reads aggregate files only.  It never imports an adapter or
  opens raw forecast/ERA5 stores.

The current adapter is `aifs_ens_v2`.  A compatible compact deterministic or
ensemble model can use the `compact_heat` adapter with a different YAML
mapping.  Core metrics do not refer to AIFS source variables.  A deterministic
model has `ensemble: false`; interval metrics then explicitly receive an
`unavailable` status rather than an invented ensemble.

`standard_reforecast_raw` is a second adapter for standard raw directories
such as `forecasts_<model>/init_*.zarr`. It reuses the verified local-solar
daily-mean and q95/onset definitions directly from raw member temperatures.
The preferred workflow commits a canonical case cache: daily ensemble-mean or
deterministic temperature, q95-event probabilities, aligned ERA5 temperature
and event flags, and selected interval quantiles. It does not save native
forecast timesteps or member-temperature cubes.

For a deterministic source, `ensemble: false` makes the daily-temperature
forecast its own deterministic mean. Its q95 hot-day and event-onset
"probabilities" are valid 0/1 forecasts, so Brier score is the weighted
event-error rate and reliability diagrams use only the zero and one bins.
Interval coverage is correctly unavailable. Temperature, conditional, spatial,
and event Brier metrics remain comparable with ensembles.

## Current input convention

Raw AIFS ENS v2 stores use source variable `2t` and dimensions
`time, number, prediction_timedelta, latitude, longitude`.  The existing
monthly heat processor renames temperature to `2m_temperature`, builds
approximate local-solar daily means, and writes compact fields:

- `t2m_daily_mean_ensemble_mean`;
- `hot_day_q95_probability`;
- `heatwave_start_q95_2d_probability`;
- `heatwave_start_q95_3d_probability`;
- `valid_date(time, forecast_day, longitude)`.

`forecast_day` in these compact stores is zero-based (`0..14`): it is the
existing local-day label and is deliberately not reinterpreted as the older
one-based UTC completed-day convention in `heatextremes.forecast_days`.

ERA5 daily local-solar temperature is `t2m_daily_mean`.  ERA5 hazard products
are `hot_day_q95`, `heatwave_start_q95_2d`, and
`heatwave_start_q95_3d`.  The threshold source is the 1991–2020, 15-day-window
calendar-day field `t2m_daily_mean_calendar_day_percentile` selected at q95.
Events use strict threshold exceedance (`T > q95`), matching the notebooks.

Paths and field mappings live in
[`configs/verification/aifs_ens_v2.yaml`](../configs/verification/aifs_ens_v2.yaml).
Every path may use environment expansion; set
`HEAT_VERIFICATION_RESULTS_ROOT` to redirect only generated results.  The
specific source overrides are `HEAT_AIFS_RAW_ROOT`,
`HEAT_AIFS_COMPACT_MONTHLY_STORE_PATTERN`,
`HEAT_ERA5_DAILY_TEMPERATURE_STORE`, `HEAT_ERA5_HAZARD_STORE`, and
`HEAT_INTERVAL_QUANTILE_FILE_PATTERN`.

## Scientific definitions

Temperature error is `ensemble_mean_forecast - ERA5`.  The deterministic
output includes RMSE, MAE, and mean bias for all cases, observed q95-hot days,
and observed non-hot days.  Conditioning always uses the ERA5 `hot_day_q95`
target—not a forecast classification.

All regional scalar metrics use cosine-latitude weights.  Conditional scores
use one case-weighted reduction over initialization, latitude, and longitude:

```text
sum(weight * metric * valid_condition) / sum(weight * valid_condition)
```

They are not averages of per-initialization conditional values.  Every
conditional row includes weighted and unweighted support.

For hot days and 2-/3-day onset events, the pipeline stores weighted Brier
numerator, weighted mean-probability numerator, observed-frequency numerator,
and probability-frequency-bias numerator.  Reliability bins store weighted
and unweighted counts plus weighted sums of probability and observation;
those values permit exact aggregation across months and models.

POD and FAR use a configured probability decision cutoff, with a positive
forecast defined as `P(event) >= cutoff`. Their partial rows retain weighted
hits, misses, false alarms, and score numerators/denominators, so aggregation
never averages monthly ratios. Deterministic models naturally use 0/1
probabilities.

Spatial products are unweighted initialization sums/counts at selected leads:
temperature bias, squared error, hot-day Brier error, forecast probability,
and observed-event frequency.  Aggregation derives the displayed fields from
these sums.

## Partitions, restart behavior, and result layout

The default Slurm array is exactly 16 tasks: 2022–2025 × JJAS.  Inside each
task the runner selects the requested month, scores one `forecast_day` at a
time, computes bounded reductions, atomically replaces its small tables, and
then releases references.  It does not call `.persist()` and never makes a
cross-lead Dask graph.  A failed task has no `completion.json`; existing lead
rows are resumed safely.  A valid completion marker causes a rerun to skip
unless `--overwrite` is supplied.

Every bounded lead reduction runs within `dask.diagnostics.ProgressBar()`, so
the active task's `.out` log displays Dask progress while data are computing.
The contexts do not use `.persist()` or join metrics across leads.

```text
verification_results/
  aifs_ens_v2/
    run_metadata.json
    case_cache/
      2022-06/
        forecast_day_000.zarr               # canonical cases, consolidated Zarr v2
        ...
        completion.json
    partial/
      2022-06/
        deterministic.parquet                 # or .csv
        probability.parquet
        interval_coverage.parquet
        probability_reliability.parquet
        maps.nc                               # only when map leads are configured
        completion.json
    aggregated/
      deterministic_by_lead_region.parquet
      probability_by_lead_region.parquet
      interval_coverage_by_lead_region.parquet
      probability_reliability_by_lead_region_bin.parquet
      spatial_metrics.nc
      metadata.json
```

Partial rows retain `numerator` and `denominator`. The canonical case cache
permits a later metric pass with a new configured region set, reliability-bin
definition, or probability decision cutoff without reopening raw model data.
The final RMSE is `sqrt(total_squared_error / total_weight)`, not an average
of monthly RMSE values. Output replacement/deletion is guarded: targets must
be resolved children of the configured results root and that root itself is
refused.

Each `forecast_day_*.zarr` cache product has canonical fields
`forecast_temperature`, `observation_temperature`, `forecast_probability`,
`observed_event`, and validity masks. Event is an explicit coordinate with
hot-day, 2-day-onset, and 3-day-onset values. Ensemble products also contain
`forecast_temperature_quantile` when the adapter can supply the configured
central-interval quantiles. Stores are consolidated Zarr **format 2** and are
written to a temporary sibling directory before a same-filesystem rename.

## Dry run and smoke test

These commands are safe off-cluster; dry run does not open an input store.

```bash
python scripts/verification/compute_verification_partition.py \
  --config configs/verification/aifs_ens_v2.yaml --year 2022 --month 6 --dry-run

# On the cluster, build one cache lead to check memory before the array:
python scripts/verification/compute_verification_partition.py \
  --config configs/verification/aifs_ens_v2.yaml --year 2022 --month 6 \
  --forecast-days 0 --stage case_cache --resume

# Calculate selected metrics later from that cache, without raw forecast input:
python scripts/verification/compute_verification_partition.py \
  --config configs/verification/aifs_ens_v2.yaml --year 2022 --month 6 \
  --forecast-days 0 --regions global --probability-bins 0 0.2 0.5 0.8 1 \
  --decision-thresholds 0.2 0.5 \
  --stage cached_metrics --resume
```

`--regions`, `--forecast-days`, `--probability-bins`,
`--decision-thresholds`, `--stage`, `--overwrite`, `--resume`, and `--dry-run`
are explicit. An aggregation can likewise filter `--years`, `--months`,
`--regions`, and `--forecast-days`.  By default it stops on missing expected
partitions; `--allow-missing` writes a clearly marked aggregate of completed
partitions for inspection.

## Slurm submission

The supplied scripts directly execute `HEAT_EXTREMES_PYTHON` (defaulting to
`/home/kylehall/miniconda3/envs/heat-extremes/bin/python`), avoiding interactive
Conda activation.  They use `SLURM_SUBMIT_DIR` or `REPOSITORY_ROOT`, never
`BASH_SOURCE`, because jobs run from spool directories.

Create the log directory once and submit from the repository root:

```bash
mkdir -p /net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/logs
sbatch slurm/verification/submit_aifs_verification_array.sbatch

# Intentional complete rerun of already-complete partitions:
OVERWRITE=1 sbatch slurm/verification/submit_aifs_verification_array.sbatch

# After the array succeeds:
sbatch slurm/verification/submit_aifs_aggregation.sbatch

# Or submit the complete overnight chain: compute -> aggregate -> figures.
bash slurm/verification/submit_aifs_verification_workflow.sh

squeue -u "$USER" -j <array_job_id>
sacct -j <array_job_id> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
python scripts/verification/validate_verification_outputs.py \
  --config configs/verification/aifs_ens_v2.yaml
```

The array’s task `0` is 2022-06, then July, August, September; task `15` is
2025-09.  Resource requests are deliberately conservative starting points and
are grouped at the top of the sbatch file for adjustment after the one-lead
smoke test.  The dependent plotting job writes PNG figures and their exact
filtered input tables under `aifs_ens_v2/figures/` before the workflow exits.

## Aggregate-only plotting and comparisons

```bash
python scripts/verification/plot_verification.py \
  --result-dirs /net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/aifs_ens_v2 \
  --output-directory /net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/aifs_ens_v2/figures \
  --reliability-forecast-days 0 5 10 13

# Later, overlay model directories without reopening any raw source:
python scripts/verification/plot_model_comparison.py \
  --result-dirs /path/to/results/model_a /path/to/results/model_b \
  --output-directory /path/to/results/model_a/figures/comparison_with_model_b
```

The figure suite covers RMSE/MAE/bias by all/hot/non-hot subset; event Brier
scores; forecast probability versus observed frequency; reliability curves;
interval coverage against nominal coverage and by lead; regional hot-day RMSE;
and aggregate spatial maps.  Each filtered tabular input used by a figure is
saved next to it in `figures/data/`.

## Interval coverage prerequisite

Central 50%, 80%, 90%, and 95% intervals require member temperatures or
selected ensemble quantiles.  The normal compact AIFS monthly store has only
ensemble mean and event probabilities, so it cannot reconstruct intervals.
The result table consequently reports `status=unavailable` with a reason.

If raw AIFS members are accessible, uncomment/configure
`paths.interval_quantile_file_pattern` beneath the verification-results root
and run the bounded preprocessor before verification:

```bash
python scripts/verification/precompute_interval_quantiles.py \
  --config /path/to/aifs_with_quantiles.yaml --year 2022 --month 6
```

It reuses the verified existing local-solar AIFS daily aggregation, processes
one lead at a time, and writes only the lower/upper quantiles necessary for
the configured intervals.  It does not save member temperatures.  Inclusion
is documented and implemented as `lower <= ERA5 <= upper`.

## Adding another model

Create a model YAML with a unique `model.name`, its own results root, source
path patterns, `variables` mappings, event mappings, capability flags, and
the shared region file.  Use `adapter: compact_heat` when its compact product
already supplies canonical-equivalent temperature/probability/valid-date
fields.  Add a small adapter under `heatextremes/verification/models/` only if
the source requires a different opening/alignment transformation.  Do not put
model-specific names in deterministic, probabilistic, interval, or plotting
modules.  Aggregate each model separately, then pass both result directories
to the comparison plot command.

For a compatible compact model, copy
`configs/verification/compact_model_template.yaml`, set its paths and variable
names, then submit its own dependency chain.  The `--years` and `--months`
arguments must match the YAML selection exactly:

```bash
bash slurm/verification/submit_verification_workflow.sh \
  --config configs/verification/my_model.yaml \
  --model my_model \
  --result-root /net/monsoon/kylehall/verification_results \
  --years "2022 2023 2024 2025" \
  --months "6 7 8 9" \
  --max-concurrent 2
```

## Inventory and run all standard raw reforecasts

Use the all-model launcher when raw model directories differ only by model
name, for example
`/net/monsoon/marchakitus/reforecast/forecasts_aurora_e2s/init_*.zarr`. It
uses `Rossby Model Storage Locations - Sheet1.csv` as its authoritative model
registry, scans only the registered directory/store names, writes its manifest
and generated per-model YAML files under the results root, then submits two
independent model/month arrays: canonical case cache, followed by cache-backed
metrics. The registry's `N Members` field selects
deterministic versus ensemble behavior; Gencast is explicitly excluded.
Even when a raw source carries a longer horizon, inventory and processing are
hard-capped to the first 15 local-solar forecast days (labels 0--14).
Defaults select available 2022–2025 JJAS initializations:

```bash
bash slurm/verification/submit_all_reforecasts_workflow.sh \
  --reforecast-root /net/monsoon/marchakitus/reforecast \
  --result-root /net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results \
  --years "2022 2023 2024 2025" \
  --months "6 7 8 9" \
  --max-concurrent 1 \
  --probability-bins "0 0.2 0.5 0.8 1" \
  --decision-thresholds "0.2 0.5"
```

Use a replacement registry explicitly with `--metadata-csv /path/to/models.csv`.

The registry's AIFS-ENS-v2 entry is deliberately included even though its raw
path is outside the generic `reforecast` root. It is the established pilot
source and uses its historical `*.zarr` filename convention; it receives the
raw ensemble adapter, including bounded central-interval coverage diagnostics.

Run it first with `--inventory-only` to inspect the generated manifest without
submitting. The default source root is singular `reforecast`, following the
provided example; use `--reforecast-root /net/monsoon/marchakitus/reforecasts`
if the cluster directory is plural.

Expected source/model failures are recorded in
`<results_root>/<model>/failures/<stage>/YYYY-MM.json`; their task exits cleanly so
other models and tolerant aggregation continue. The aggregation status is in
`inventory/aggregation_status.json`, and aggregate-only all-model comparison
figures are written to `_all_models/figures/`. Start with one simultaneous raw
task; raise `--max-concurrent` only after checking memory and filesystem load.

## Opening the intermediate case cache

Open all available cache slices for one model lazily, while reporting any
missing forecast-day stores and replacing those in-month gaps with NaNs:

```bash
python scripts/verification/open_reforecast_case_cache.py \
  --model-name aurora_e2s
```

The reader discovers expected month/lead coverage from
`inventory/reforecast_inventory.json` when available.  It can represent a
missing lead only when another lead exists for that month; a wholly missing
month is reported but not fabricated because its initialization times are
unknown.

## Scientific points needing confirmation

- The production compact local-day `forecast_day` numbering is zero-based;
  confirm whether figures should label this as `0–14` or presentation-facing
  `day 1–15` while retaining the stored coordinate.
- The source monthly product uses six-hour longitude-offset bands.  This
  implementation preserves that existing local-solar definition exactly.
- No compatible land mask is currently configured.  `regions.py` supports a
  land-only region when a compatible mask is supplied, but no mask is inferred
  or silently applied.
