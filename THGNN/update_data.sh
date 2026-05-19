#!/usr/bin/env bash
# update_data.sh — Incremental data update for THGNN (Ubuntu / remote server)
#
# Usage:
#   chmod +x update_data.sh
#   ./update_data.sh                        # uses defaults below
#   ./update_data.sh --start 2026-04-08     # override start date
#   nohup ./update_data.sh > run.log 2>&1 & # run in background, detached
#
# Progress is written to logs/update_YYYYMMDD_HHMMSS.log
# Each step is timed and logged; if any step fails the script stops immediately.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — edit these if your paths differ
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
TICKER_FILE="$DATA_DIR/valid_nifty500.txt"
MAIN_PKL="$DATA_DIR/nifty500.pkl"
NEW_PKL="$DATA_DIR/nifty500_new.pkl"
RELATION_DIR="$DATA_DIR/relation_nifty500"
GRAPH_DIR="$DATA_DIR/data_train_predict"
START_DATE="2026-04-08"
PROCESSES=4
WINDOW=20
HORIZON=3

# ---------------------------------------------------------------------------
# Argument parsing (only --start supported for now)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START_DATE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
mkdir -p "$SCRIPT_DIR/logs"
LOG="$SCRIPT_DIR/logs/update_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
step_start() { log "=== STEP $1: $2 ==="; STEP_T0=$(date +%s); }
step_done()  {
    local elapsed=$(( $(date +%s) - STEP_T0 ))
    log "=== STEP $1 done in ${elapsed}s ==="
    echo ""
}

log "Starting incremental update — start_date=$START_DATE"
log "Log file: $LOG"
log "Working directory: $SCRIPT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Download new price data
# ---------------------------------------------------------------------------
step_start 1 "Download price data from $START_DATE"

TICKERS="$(paste -sd ',' "$TICKER_FILE")"
TICKER_COUNT="$(wc -l < "$TICKER_FILE")"
log "Tickers: $TICKER_COUNT (from $TICKER_FILE)"

python "$SCRIPT_DIR/utils/download_market_data.py" \
    --tickers "$TICKERS" \
    --start "$START_DATE" \
    --output "$NEW_PKL"

log "Downloaded to $NEW_PKL"
step_done 1

# ---------------------------------------------------------------------------
# Step 1b — Merge new data into existing pickle
# ---------------------------------------------------------------------------
step_start "1b" "Merge $NEW_PKL into $MAIN_PKL"

if [[ ! -f "$MAIN_PKL" ]]; then
    log "No existing pickle found at $MAIN_PKL — using new download directly."
    mv "$NEW_PKL" "$MAIN_PKL"
else
    python - <<PYEOF
import pickle, pandas as pd, sys

old_path = "$MAIN_PKL"
new_path = "$NEW_PKL"

with open(old_path, "rb") as f:
    old = pickle.load(f)
with open(new_path, "rb") as f:
    new = pickle.load(f)

old = pd.DataFrame(old)
new = pd.DataFrame(new)
old["dt"] = pd.to_datetime(old["dt"])
new["dt"] = pd.to_datetime(new["dt"])

before = len(old)
merged = (
    pd.concat([old, new])
    .drop_duplicates(subset=["dt", "code"])
    .sort_values(["dt", "code"])
    .reset_index(drop=True)
)
added = len(merged) - before

with open(old_path, "wb") as f:
    pickle.dump(merged, f)

print(f"  Rows before : {before:,}")
print(f"  Rows after  : {len(merged):,}  (+{added:,} new rows)")
print(f"  Date range  : {merged['dt'].min().date()} -> {merged['dt'].max().date()}")
print(f"  Stocks      : {merged['code'].nunique()}")
PYEOF

    rm -f "$NEW_PKL"
    log "Merged. Removed temp file $NEW_PKL"
fi

step_done "1b"

# ---------------------------------------------------------------------------
# Step 2 — Generate relation matrices for new dates only
# ---------------------------------------------------------------------------
step_start 2 "Generate relation matrices (start-date=$START_DATE, processes=$PROCESSES)"

mkdir -p "$RELATION_DIR"
EXISTING_RELATIONS=$(ls "$RELATION_DIR"/*.csv 2>/dev/null | wc -l || echo 0)
log "Existing relation files: $EXISTING_RELATIONS"

python "$SCRIPT_DIR/utils/generate_relation.py" \
    --data-path "$MAIN_PKL" \
    --relation-dir "$RELATION_DIR" \
    --window "$WINDOW" \
    --start-date "$START_DATE" \
    --processes "$PROCESSES"

NEW_RELATIONS=$(ls "$RELATION_DIR"/*.csv 2>/dev/null | wc -l || echo 0)
log "Relation files now: $NEW_RELATIONS (added $(( NEW_RELATIONS - EXISTING_RELATIONS )))"
step_done 2

# ---------------------------------------------------------------------------
# Step 3 — Generate graph samples for new dates only
# ---------------------------------------------------------------------------
step_start 3 "Generate graph samples (start-date=$START_DATE)"

mkdir -p "$GRAPH_DIR"
EXISTING_GRAPHS=$(ls "$GRAPH_DIR"/*.pkl 2>/dev/null | wc -l || echo 0)
log "Existing graph samples: $EXISTING_GRAPHS"

python "$SCRIPT_DIR/utils/generate_data.py" \
    --data-path "$MAIN_PKL" \
    --relation-dir "$RELATION_DIR" \
    --output-dir "$GRAPH_DIR" \
    --start-date "$START_DATE" \
    --window "$WINDOW" \
    --horizon "$HORIZON"

NEW_GRAPHS=$(ls "$GRAPH_DIR"/*.pkl 2>/dev/null | wc -l || echo 0)
log "Graph samples now: $NEW_GRAPHS (added $(( NEW_GRAPHS - EXISTING_GRAPHS )))"
step_done 3

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "All steps completed successfully."
log "Main pickle : $MAIN_PKL"
log "Relations   : $RELATION_DIR  ($NEW_RELATIONS files)"
log "Graph data  : $GRAPH_DIR  ($NEW_GRAPHS files)"
log "Full log    : $LOG"
