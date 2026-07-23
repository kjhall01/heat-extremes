#!/bin/bash
# Submit the four T2M verification stages with afterok dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-/net/monsoon/kylehall/heat_extremes/t2m_verification}"

mkdir -p "${PIPELINE_ROOT}/logs"

forecast_job="$(sbatch --parsable "${SCRIPT_DIR}/01_build_t2m_forecast_aggregates.sbatch")"
era5_job="$(sbatch --parsable --dependency="afterok:${forecast_job}" "${SCRIPT_DIR}/02_build_t2m_era5_aggregates.sbatch")"
metrics_job="$(sbatch --parsable --dependency="afterok:${era5_job}" "${SCRIPT_DIR}/03_calculate_t2m_metrics.sbatch")"
summary_job="$(sbatch --parsable --dependency="afterok:${metrics_job}" "${SCRIPT_DIR}/04_aggregate_and_plot_t2m_metrics.sbatch")"

printf 'Submitted forecast aggregates: %s\n' "${forecast_job}"
printf 'Submitted matched ERA5 aggregates: %s\n' "${era5_job}"
printf 'Submitted case-wise metrics: %s\n' "${metrics_job}"
printf 'Submitted summaries and figures: %s\n' "${summary_job}"
