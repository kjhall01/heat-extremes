#!/bin/bash
# Submit only explicitly named missing model/month case-cache partitions.
#
# Example:
#   bash scripts/verification/submit_reforecast_cache_backfill.sh \
#     --target aifs_v2:2022-06 \
#     --target aurora_e2s:2025-09

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_reforecast_cache_backfill.sh [options]

Submit only named model/month case-cache tasks from the existing reforecast
inventory. Each target is safely rerun with OVERWRITE=1, which can replace
only that model's exact case-cache month; every other model/month is untouched.

Options:
  --target MODEL:YYYY-MM   Missing model/month to retry. Repeat as needed.
  --result-root DIRECTORY  Verification result root.
                           Default: /net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results
  --max-concurrent N       Maximum simultaneously submitted target tasks (default: 2).
  --dry-run                Print the target Slurm task indices without submitting.
  -h, --help               Show this help.

The inventory must already have been regenerated with the intended lead cap.
For the current 0--12 workflow, run the inventory launcher with
--max-forecast-day 12 before using this script.
EOF
}

RESULT_ROOT="/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results"
MAX_CONCURRENT=2
DRY_RUN=0
TARGETS=()

while (( $# )); do
    case "$1" in
        --target)
            TARGETS+=("$2")
            shift 2
            ;;
        --result-root)
            RESULT_ROOT="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if (( ${#TARGETS[@]} == 0 )); then
    echo "At least one --target MODEL:YYYY-MM is required." >&2
    usage >&2
    exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrent must be a positive integer." >&2
    exit 2
fi

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/../.." && pwd)"
PYTHON="${HEAT_EXTREMES_PYTHON:-$(command -v python || true)}"
MANIFEST="${RESULT_ROOT%/}/inventory/reforecast_inventory.json"
SBATCH_SCRIPT="${REPOSITORY_ROOT}/slurm/verification/submit_reforecast_model_task.sbatch"

if [[ ! -x "$PYTHON" || ! -f "$MANIFEST" || ! -f "$SBATCH_SCRIPT" ]]; then
    echo "Python, inventory manifest, or Slurm task script is unavailable:" >&2
    printf '  Python: %s\n  Manifest: %s\n  Slurm script: %s\n' "$PYTHON" "$MANIFEST" "$SBATCH_SCRIPT" >&2
    exit 2
fi

TASK_INDICES="$("$PYTHON" - "$MANIFEST" "${TARGETS[@]}" <<'PY'
import json
import re
import sys

manifest_path, *target_text = sys.argv[1:]
target_pattern = re.compile(r"(?P<model>[a-z0-9_]+):(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$")

targets = []
for text in target_text:
    match = target_pattern.fullmatch(text)
    if match is None:
        raise SystemExit(f"Invalid --target {text!r}; expected MODEL:YYYY-MM")
    targets.append((match["model"], int(match["year"]), int(match["month"])))
if len(set(targets)) != len(targets):
    raise SystemExit("Duplicate --target values are not allowed")

with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)

models = {item.get("model"): item for item in manifest.get("models", []) if isinstance(item, dict)}
for model, _, _ in targets:
    forecast_days = models.get(model, {}).get("forecast_days")
    if forecast_days is None or max(forecast_days, default=-1) > 12:
        raise SystemExit(
            f"{model}: manifest does not have the required 0--12 lead cap. "
            "Regenerate inventory with --max-forecast-day 12 first."
        )

task_indices = []
unmatched = set(targets)
for index, task in enumerate(manifest.get("tasks", [])):
    key = (task.get("model"), task.get("year"), task.get("month"))
    if key in unmatched:
        task_indices.append(index)
        unmatched.remove(key)

if unmatched:
    labels = ", ".join(f"{model}:{year:04d}-{month:02d}" for model, year, month in sorted(unmatched))
    raise SystemExit(f"Target(s) were not present in the manifest: {labels}")

print(
    "Submitting cache backfill for: "
    + ", ".join(f"{model}:{year:04d}-{month:02d}" for model, year, month in targets),
    file=sys.stderr,
)
print(",".join(map(str, task_indices)))
PY
)"

echo "Manifest: ${MANIFEST}"
echo "Slurm task indices: ${TASK_INDICES}"
if (( DRY_RUN )); then
    exit 0
fi

mkdir -p "${RESULT_ROOT%/}/logs"
JOB_ID="$(
    sbatch --parsable \
        --array="${TASK_INDICES}%${MAX_CONCURRENT}" \
        --export="ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT},REFORECAST_MANIFEST=${MANIFEST},HEAT_VERIFICATION_RESULTS_ROOT=${RESULT_ROOT%/},OVERWRITE=1,VERIFICATION_STAGE=case_cache" \
        --output="${RESULT_ROOT%/}/logs/reforecast_backfill_%A_%a.out" \
        --error="${RESULT_ROOT%/}/logs/reforecast_backfill_%A_%a.err" \
        "${SBATCH_SCRIPT}"
)"

echo "Submitted backfill job: ${JOB_ID}"
echo "Monitor: squeue -j ${JOB_ID}"
echo "Inspect: sacct -j ${JOB_ID} --format=JobID,State,Elapsed,MaxRSS,ExitCode"
