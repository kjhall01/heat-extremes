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
