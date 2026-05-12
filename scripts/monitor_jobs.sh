#!/bin/bash
# Stall-detection monitor. Watches the latest games.jsonl per running benchmark
# job and emits:
#   - "PROGRESS <job> <model> <n_rows>"  when a new game row is added
#   - "STALL <job> <model> no_progress_for=<min>min"  if a job has gone 30+ min
#     without a new row
# Runs forever until killed.

RUNS="C:/Users/igorc/Projects/LLM_Chess_Eval/runs"
declare -A LAST_ROWS
declare -A LAST_CHANGE_TS
declare -A STALL_REPORTED

while true; do
    NOW=$(date +%s)
    for dir in "$RUNS"/*; do
        [ -d "$dir" ] || continue
        jsonl="$dir/games.jsonl"
        [ -f "$jsonl" ] || continue
        # only watch dirs modified in the last 60 minutes (recent runs)
        mtime=$(stat -c "%Y" "$dir" 2>/dev/null)
        [ -z "$mtime" ] && continue
        age_min=$(( (NOW - mtime) / 60 ))
        # if dir hasn't been touched in 2+ hours, it's done; skip
        [ "$age_min" -gt 120 ] && continue

        rows=$(wc -l < "$jsonl" 2>/dev/null)
        rows=${rows:-0}
        key=$(basename "$dir")
        prev=${LAST_ROWS[$key]:-"-1"}

        if [ "$rows" != "$prev" ]; then
            # progress: row count changed
            model=$(echo "$key" | sed -E 's/.*games_(retry|forfeit|substitute)__([^_]+).*/\2/')
            echo "[$(date +%H:%M:%S)] PROGRESS $key model=$model rows=$rows"
            LAST_ROWS[$key]=$rows
            LAST_CHANGE_TS[$key]=$NOW
            STALL_REPORTED[$key]=""
        else
            # no change — check if stalled
            last_ts=${LAST_CHANGE_TS[$key]:-$NOW}
            stall_min=$(( (NOW - last_ts) / 60 ))
            reported=${STALL_REPORTED[$key]:-""}
            # Adjusted threshold: 60 min. Reasoning models can take 20-40 min per
            # game legitimately; 60 min without a new row is real evidence the
            # process is hung (network, API timeout, dead python). Investigate, don't wait longer.
            if [ "$stall_min" -ge 60 ] && [ -z "$reported" ]; then
                echo "[$(date +%H:%M:%S)] STALL $key no_progress_for=${stall_min}min rows=$rows"
                STALL_REPORTED[$key]="yes"
            fi
        fi
    done
    sleep 60
done
