# heat-extremes

Tools and batch workflows for verifying heat-extreme forecasts against ERA5.
Existing notebooks and the earlier T2M/precipitation scripts are retained as
historical and exploratory records.  The production-oriented heat workflow is
documented in [docs/verification.md](docs/verification.md).

## AIFS ENS v2 verification: minimal workflow

The default configuration uses the existing compact AIFS ENS v2 monthly heat
products and verifies JJAS (June–September) initializations from 2022–2025.
It writes only compact sufficient statistics, never reconstructed forecasts or
matched ERA5 cubes.

```bash
# Inspect one bounded partition without opening a Zarr store.
python scripts/verification/compute_verification_partition.py \
  --config configs/verification/aifs_ens_v2.yaml \
  --year 2022 --month 6 --dry-run

# Small data-bearing smoke test on the cluster (one lead, global region).
python scripts/verification/compute_verification_partition.py \
  --config configs/verification/aifs_ens_v2.yaml \
  --year 2022 --month 6 --forecast-days 0 --regions global --resume

# Aggregate completed partitions and plot only from aggregate products.
python scripts/verification/aggregate_verification.py \
  --config configs/verification/aifs_ens_v2.yaml
python scripts/verification/plot_verification.py \
  --result-dirs /net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/aifs_ens_v2 \
  --output-directory /net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/aifs_ens_v2/figures
```

For the complete overnight chain on the cluster—JJAS array, aggregation, then
figures—submit one command from the repository root:

```bash
bash slurm/verification/submit_aifs_verification_workflow.sh
```

## All standard raw reforecasts

For standard stores laid out as
`/net/monsoon/marchakitus/reforecast/forecasts_<model>/init_*.zarr`, inventory
and submit each available JJAS model/month independently:

```bash
bash slurm/verification/submit_all_reforecasts_workflow.sh \
  --reforecast-root /net/monsoon/marchakitus/reforecast \
  --max-concurrent 1
```

The inventory scans only directory and store names. A source failure is
recorded under the affected model's `failures/` directory and does not stop
other models. Aggregate-only all-model figures are written under
`/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results/_all_models/figures/`.
The default registry is `Rossby Model Storage Locations - Sheet1.csv`; it
sets deterministic versus ensemble behavior and explicitly excludes Gencast.

Run the synthetic tests from a source checkout with:

```bash
PYTHONPATH=. pytest -q
```

The current compact AIFS products contain ensemble-mean temperature and event
probabilities, but not member temperatures or ensemble quantiles.  Therefore
the pipeline records interval coverage as unavailable unless the optional
selected-quantile preprocessing product is configured.  See the interval
coverage section in the verification guide before enabling it.
