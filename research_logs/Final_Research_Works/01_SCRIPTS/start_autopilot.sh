#!/usr/bin/env bash
# start_autopilot.sh — run the completion plan unattended in a tmux session.
#
#   ./start_autopilot.sh            # create the session and attach
#   ./start_autopilot.sh --detach   # create it and leave it running
#   ./start_autopilot.sh --stop     # stop the autopilot (running tasks survive)
#
# The session has three windows:
#   0 autopilot  the scheduler's own log
#   1 status     the task ledger, refreshed
#   2 gpus       live GPU usage across the cluster
#
# Re-attach later with:  tmux attach -t slspv
# Detach from inside:    Ctrl-b then d

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRW="$(dirname "$HERE")"
SESSION="slspv"
LOGDIR="$FRW/02_STATE"
mkdir -p "$LOGDIR"

case "${1:-}" in
  --stop)
      if tmux has-session -t "$SESSION" 2>/dev/null; then
          tmux kill-session -t "$SESSION"
          echo "autopilot session stopped."
          echo "NOTE: detached training jobs on the compute nodes keep running."
          echo "      Check with: ssh compute-node-1 nvidia-smi"
      else
          echo "no '$SESSION' session"
      fi
      exit 0 ;;
  --help|-h)
      sed -n '2,18p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already exists."
    echo "  attach:  tmux attach -t $SESSION"
    echo "  stop  :  $0 --stop"
    [[ "${1:-}" == "--detach" ]] || tmux attach -t "$SESSION"
    exit 0
fi

echo "creating tmux session '$SESSION'..."

# window 0 — the autopilot itself, teed so the log survives the session
tmux new-session -d -s "$SESSION" -n autopilot -c "$HERE" \
  "python3 autopilot.py 2>&1 | tee -a '$LOGDIR/autopilot.log'; \
   echo; echo '=== autopilot exited — window stays open ==='; exec bash"

# window 1 — the ledger, refreshed every 60 s (status probes the cluster over
# ssh, so polling it faster than that just adds load)
tmux new-window -t "$SESSION:1" -n status -c "$HERE" \
  "while true; do clear; python3 runner.py --status 2>&1 | head -70; \
   echo; echo 'refreshing every 60s — Ctrl-b 0 for the autopilot log'; \
   sleep 60; done"

# window 2 — live GPU usage across every node that answers
tmux new-window -t "$SESSION:2" -n gpus -c "$HERE" \
  "while true; do clear; date -u; \
   for n in compute-node-1 compute-node-2 compute-node-3 compute-node-4; do \
     echo \"=== \$n\"; \
     timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 \$n \
       'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null; \
        echo \"  -- procs --\"; \
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null' \
       2>/dev/null | sed 's/^/  /' || echo '  unreachable'; \
   done; sleep 30; done"

tmux select-window -t "$SESSION:0"

cat <<TXT

  tmux session '$SESSION' created.

    window 0  autopilot   the scheduler log   (also tee'd to 02_STATE/autopilot.log)
    window 1  status      the task ledger
    window 2  gpus        live GPU usage

    attach        tmux attach -t $SESSION
    switch window Ctrl-b then 0 / 1 / 2
    detach        Ctrl-b then d
    stop          $HERE/start_autopilot.sh --stop

  Per-task logs land in 03_RESULTS/<ID>/autopilot_<n>_<timestamp>.log

TXT

[[ "${1:-}" == "--detach" ]] || tmux attach -t "$SESSION"
