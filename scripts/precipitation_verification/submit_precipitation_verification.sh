#!/bin/bash
# Submit the four precipitation verification stages with afterok dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-/net/monsoon/kylehall/heat_extremes/precipitation_verification}"

mkdir -p "${PIPELINE_ROOT}/logs"

export_args="--export=ALL,PIPELINE_SCRIPT_DIR=${SCRIPT_DIR}"
forecast_job="$(sbatch --parsable "${export_args}" "${SCRIPT_DIR}/01_build_daily_forecast_precipitation.sbatch")"
era5_job="$(sbatch --parsable "${export_args}" --dependency="afterok:${forecast_job}" "${SCRIPT_DIR}/02_build_matched_era5_precipitation.sbatch")"
metrics_job="$(sbatch --parsable "${export_args}" --dependency="afterok:${era5_job}" "${SCRIPT_DIR}/03_calculate_precipitation_metrics.sbatch")"
summary_job="$(sbatch --parsable "${export_args}" --dependency="afterok:${metrics_job}" "${SCRIPT_DIR}/04_aggregate_and_plot_precipitation_metrics.sbatch")"

printf 'Submitted forecast precipitation: %s\n' "${forecast_job}"
printf 'Submitted matched ERA5 precipitation: %s\n' "${era5_job}"
printf 'Submitted case-wise precipitation metrics: %s\n' "${metrics_job}"
printf 'Submitted precipitation summaries and figures: %s\n' "${summary_job}"
