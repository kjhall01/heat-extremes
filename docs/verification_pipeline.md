# Heat-verification pipeline

This is the short operational guide for adding and running a raw forecast
model. The longer implementation reference is
[verification.md](verification.md).

## What the pipeline does

```text
raw model Zarrs (read only)
        |
        v
metadata inventory ──> generated model YAML + partition/lead manifest
        |
        v
local-solar daily means + ERA5 matching, one model/month/lead at a time
        |
        v
canonical case cache: <results-root>/<model>/case_cache/YYYY-MM/
        |
        +──> cache-backed regional metrics / aggregate tables / figures
        |
        +──> lazy intermediate reader ──> model_scorecards.ipynb
```

The only durable high-volume product is the canonical case cache. It holds
the ensemble-mean daily temperature, ensemble event probabilities, matching
ERA5 daily temperature and event flags, and validity masks. It does **not**
copy raw forecast timesteps or member-temperature cubes. Raw forecast Zarrs
and the ERA5 stores are always opened read-only.

The raw adapter computes six-hour, longitude-band local-solar daily means and
hard-caps the project horizon at local-solar forecast days 0–14. The current
scorecard cache uses days 0–12.

## ECMWF IFS ENS

`IFS-ENS` is registered at
`/net/monsoon/marchakitus/IFS/IFS_ENS` as the normalized model name
`ifs_ens`. It uses the standard raw-reforecast adapter because its Zarr layout
matches AIFS ENS v2: source variable `2t`, ensemble dimension `number`,
`prediction_timedelta`, latitude/longitude coordinates, and date-bearing
Zarr filenames. The tracked configuration is
[`configs/verification/ifs_ens.yaml`](../configs/verification/ifs_ens.yaml).

The inventory checks store metadata and coordinates before it submits work;
it reads no temperature chunks. The registry marks IFS ENS as an ensemble
(the exact member count is not used for science; the source `number` dimension
is the runtime authority).

### 1. Inventory only

Run this from the repository root on the cluster. A model-specific manifest
keeps this IFS submission separate from the existing all-model manifest while
still writing the cache to the shared results root.

```bash
RESULT_ROOT=/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results
IFS_MANIFEST="$RESULT_ROOT/inventory/ifs_ens/reforecast_inventory.json"
IFS_CONFIGS="$RESULT_ROOT/inventory/ifs_ens/configs"

bash slurm/verification/submit_all_reforecasts_workflow.sh \
  --models "ifs_ens" \
  --manifest "$IFS_MANIFEST" \
  --config-directory "$IFS_CONFIGS" \
  --result-root "$RESULT_ROOT" \
  --years "2022 2023 2024 2025" \
  --months "6 7 8 9" \
  --max-forecast-day 12 \
  --inventory-only
```

Confirm that it reports `ifs_ens`, 16 monthly partitions, and forecast days
0–12. If it finds a different safe common lead range, use that range rather
than forcing the scorecard configuration.

### 2. One-lead smoke test

Before submitting all 16 months, build a single case-cache lead. IFS has more
members than AIFS ENS v2, so this establishes the actual memory requirement
on the cluster.

```bash
/home/kylehall/miniconda3/envs/heat-extremes/bin/python \
  scripts/verification/compute_verification_partition.py \
  --config "$IFS_CONFIGS/ifs_ens.yaml" \
  --year 2022 --month 6 --forecast-days 0 --stage case_cache --resume
```

If it succeeds, inspect its MaxRSS with `sacct` and then submit the full
dependency chain. Begin with modest concurrency until the filesystem and
memory behavior are known.

```bash
bash slurm/verification/submit_all_reforecasts_workflow.sh \
  --models "ifs_ens" \
  --manifest "$IFS_MANIFEST" \
  --config-directory "$IFS_CONFIGS" \
  --result-root "$RESULT_ROOT" \
  --years "2022 2023 2024 2025" \
  --months "6 7 8 9" \
  --max-forecast-day 12 \
  --max-concurrent 1
```

That chain is: case cache → cache-backed metrics → tolerant aggregation →
aggregate-only figures. The submission prints all job IDs and writes logs to
`$RESULT_ROOT/logs/`.

### 3. Audit and scorecard

```bash
python scripts/verification/audit_case_cache_completeness.py \
  --models ifs_ens --max-forecast-day 12
```

`model_scorecards.ipynb` now includes `ifs_ens` in `MODEL_NAMES`. Once the
case cache is complete, rerun its configuration, opening, scoring, and plot
cells. The same lazy reader opens
`<results-root>/ifs_ens/case_cache/YYYY-MM/forecast_day_*.zarr`; no raw IFS
Zarr is opened by the notebook.

## Optional: lead-dependent model q95

The scorecard currently defines hot days with the **ERA5** 1991–2020 q95, so
model-climatological q95 is not required to add IFS ENS to it. If we later
use model-native or quantile-transfer thresholds, add IFS to the separate
historical (2000–2020) model-climatology workflow after confirming source
coverage:

```bash
python scripts/verification/preflight_model_temperature_climatology.py \
  --models ifs_ens

bash slurm/verification/submit_model_temperature_q95_workflow.sh \
  --models "ifs_ens" --max-concurrent 12 --stage-max-concurrent 6
```

This workflow writes only a global q95 field plus temporary per-band daily
staging stores beneath `model_climatology/`; it does not touch raw IFS data.

## Report scorecard (Nigeria default; global available separately)

The static notebook [`model_scorecards.ipynb`](../model_scorecards.ipynb) is
useful for figure development.  For the report, submit the reproducible batch
version instead.  It first builds or resumes the canonical cases and then
writes the scorecard only if every requested model/month/lead is present.  A
missing cache slice is an error, never a quietly grey or NaN report cell.

```bash
bash slurm/verification/submit_all_reforecasts_workflow.sh \
  --models "aifs_ens_v2 ifs_ens aifs_v2 aurora_e2s graphcast_e2s" \
  --years "2022 2023 2024 2025" --months "6 7 8 9" \
  --max-forecast-day 12 --max-concurrent 1 \
  --regions "nigeria" \
  --report-scorecard
```

The final job writes these files beneath
`<result-root>/_report_scorecard/`:

- `heat_report_scorecard.csv` and `heat_report_scorecard.png`;
- `graphcast_vs_ifs_direction_check.csv`, which makes the direction of every
  GraphCast-minus-IFS difference explicit;
- `heat_report_scorecard_metadata.json`, including the scientific definitions
  and source paths.

The Nigeria PNG preserves the report's map-plus-scorecard composition, with
an ERA5 observed hot-day-incidence-change map beside absolute-value,
colour-coded metric cells. The map input is additionally saved as
`observed_hot_day_frequency_change_nigeria.nc`.

### Global T2M report figure

Use the same scorecard job with `global` explicitly selected. It writes a
full-width, map-free global metric figure to
`<result-root>/_report_scorecard_global/`, leaving the Nigeria output intact.
The global figure shares the IFS baseline, cell styling, and legend, but does
not incur an unnecessary full-world observed-frequency-map reduction. Its
first column is all-day global T2M RMSE (`rmse_all`), rather than the
hot-day-conditional RMSE used in the Nigeria heat figure.

```bash
sbatch --export=ALL,REPOSITORY_ROOT="$PWD",HEAT_VERIFICATION_RESULTS_ROOT="$RESULT_ROOT",REPORT_SCORECARD_REGIONS=global,REPORT_SCORECARD_FORECAST_DAYS="0 3 6 9" slurm/verification/submit_report_scorecard.sbatch
```

The scorecard is deliberately **raw**, with no forecast bias correction.  Its
temperature RMSE is in K; `rmse_hot` conditions on an ERA5 hot day.  Its POD
and FAR are *deterministic*: the model's deterministic/ensemble-mean T2M is
tested against ERA5's 1991--2020 local calendar-day q95.  The native hot-day
exceedance probability is evaluated separately by Brier score.  Thus a claim
that a model is "better probabilistically" must refer to the Brier column,
not to the deterministic POD/FAR columns.  `mali` and `nigeria` are explicitly
labelled rectangular reporting boxes, not country-boundary masks.

### Global Z500 + T2M scorecard

Z500 cannot be reconstructed from the heat case cache, which intentionally
stores T2M and event fields only.  The companion workflow reads the registered
raw reforecast stores in bounded model/month tasks, checks the actual source
variable and units first, and then joins exact global Z500 RMSE to the global
raw-T2M scorecard:

```bash
ERA5_Z500_STORE=/net/path/to/consolidated_era5_pressure_level.zarr \
ERA5_Z500_VARIABLE=z \
bash slurm/verification/submit_global_z500_scorecard.sh \
  --result-root /net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results
```

This requires a locally available ERA5 pressure-level Zarr; the existing ARCO
cache is surface-only and cannot supply Z500.  The preflight output
`_global_z500_scorecard/z500_preflight.json` records source-variable and unit
availability.  In the checked-in model registry, IFS ENS currently advertises
only `2t`, so its Z500 row will be reported as unavailable unless a Z500 source
is added to that registry/archive.  It is never silently omitted.  The Z500
jobs use the common initialization intersection among the Z500-capable models;
the heat scorecard's metadata separately records the all-five-model common
intersection used for T2M.

`global_model_scorecard.csv` contains `z500_rmse`, `t2m_rmse_all`, and
`t2m_rmse_hot`; it preserves the heat labels `0, 3, 6, 9, 12` and records the
paired instantaneous Z500 lead hours (`0, 72, 144, 216, 288`) in a separate
column.  This is intentionally configurable with `--lead-hours`: local-solar
daily T2M has longitude-dependent valid dates, so it has no single identical
instantaneous Z500 target time.
