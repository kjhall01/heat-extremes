#!/bin/bash
# Render the report-oriented heat scorecard after the case-cache workflow.
# This job only reads completed canonical cache products (plus legacy AIFS ENS
# compact products) and the ERA5 threshold store; it never writes raw data.
#SBATCH --job-name=heat_report_scorecard
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:?Set HEAT_VERIFICATION_RESULTS_ROOT to the case-cache result root}"
OUTPUT_DIRECTORY="${REPORT_SCORECARD_OUTPUT_DIRECTORY:-${RESULT_ROOT}/_report_scorecard}"

if [[ ! -x "${PYTHON}" || ! -f "${REPOSITORY_ROOT}/scripts/verification/build_report_scorecard.py" ]]; then
    echo "Python executable or report-scorecard script is missing" >&2
    exit 2
fi

MODELS_TEXT="${REPORT_SCORECARD_MODELS:-aifs_ens_v2 ifs_ens aifs_v2 aurora_e2s graphcast_e2s}"
# Nigeria is the publishable default and avoids computing unused regional maps.
# Supply REPORT_SCORECARD_REGIONS explicitly to include Mali or global metrics.
REGIONS_TEXT="${REPORT_SCORECARD_REGIONS:-nigeria}"
FORECAST_DAYS_TEXT="${REPORT_SCORECARD_FORECAST_DAYS:-0 3 6 9 12}"
YEARS_TEXT="${REPORT_SCORECARD_YEARS:-2022 2023 2024 2025}"
MONTHS_TEXT="${REPORT_SCORECARD_MONTHS:-6 7 8 9}"

read -r -a MODELS <<< "${MODELS_TEXT}"
read -r -a REGIONS <<< "${REGIONS_TEXT}"
read -r -a FORECAST_DAYS <<< "${FORECAST_DAYS_TEXT}"
read -r -a YEARS <<< "${YEARS_TEXT}"
read -r -a MONTHS <<< "${MONTHS_TEXT}"

mkdir -p "${RESULT_ROOT}/logs" "${OUTPUT_DIRECTORY}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Models: ${MODELS_TEXT}"
echo "Regions: ${REGIONS_TEXT}"
echo "Forecast days: ${FORECAST_DAYS_TEXT}"
echo "Output: ${OUTPUT_DIRECTORY}"

srun --cpu-bind=none "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/build_report_scorecard.py" \
    --output-directory "${OUTPUT_DIRECTORY}" \
    --results-root "${RESULT_ROOT}" \
    --models "${MODELS[@]}" \
    --regions "${REGIONS[@]}" \
    --forecast-days "${FORECAST_DAYS[@]}" \
    --years "${YEARS[@]}" \
    --months "${MONTHS[@]}"

echo "Finish: $(date --iso-8601=seconds)"
