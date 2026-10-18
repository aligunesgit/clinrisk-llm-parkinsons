#!/usr/bin/env bash
# Full rerun of every analysis in manuscript order (used after restoring the five Ga recordings).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
step() { echo "[$(date +%H:%M:%S)] START $*"; "$@"; echo "[$(date +%H:%M:%S)] DONE  $*"; }
step python3 revised_cal_analysis.py
step python3 clinrisk_agent_analysis.py
step python3 make_revised_figures.py
step python3 make_clinrisk_figures.py
cd revision
step python3 burden_dca.py
step python3 prevalence_matched.py
step python3 external_validation.py
step python3 reviewer2_analyses.py all
step python3 paired_differences.py
step python3 policy_regeneration.py evaluate
step env POLICY_MODEL=claude-opus-5-5 python3 policy_regeneration.py evaluate
step env POLICY_MODEL=claude-haiku-5-5 python3 policy_regeneration.py evaluate
step python3 make_revision_figures.py
step python3 make_supplements.py
step python3 reviewer1_operating_sensitivity.py
echo "ALL DONE"
