#!/usr/bin/env bash
# gpurun.sh — run a command on a cluster node that has a free GPU, from a node that doesn't.
#
# Why this works with zero file copying: /mnt/ricproject3 is NFS from 10.222.1.117
# and is mounted at the SAME path on every compute node, and $HOME is shared too.
# So the remote side just cd's to $PWD and everything is already there.
#
# Usage:
#   tools/gpurun.sh -- python trainSpeakerNet.py --config configs/ecapa_tdnn.yaml
#   tools/gpurun.sh -d -- python trainSpeakerNet.py --config configs/ecapa_tdnn.yaml
#   tools/gpurun.sh --status
#
# Options:
#   -d, --detach        Run detached (survives SSH/terminal death). Logs to
#                       exps/gpurun_logs/<stamp>.log and prints the log path.
#   -n, --node NODE     Force a node (default: auto-pick whichever has a free GPU).
#   -g, --gpu N         Force a GPU index (default: auto-pick the emptiest).
#   -e, --env NAME      Conda env name (default: SL_SPV).
#   -m, --min-free MB   Minimum free MiB for a GPU to count as free (default: 8000).
#   -s, --status        Show GPU state on all candidate nodes and exit.
#   -h, --help          This text.

set -euo pipefail

# Cluster GPU inventory, re-probed 2026-08-15:
#   compute-node-1 (10.222.1.119)  2x Tesla T4  15 GB  -- ADDED 2026-08-12, was idle
#   compute-node-2 (10.222.1.118)  2x Tesla T4  15 GB  -- RE-ADDED 2026-08-15. Its
#       driver was dead because Ubuntu stopped building the 535 branch after
#       kernel 5.15.0-186 and the node had rebooted into -187; migrating it to
#       the 580 branch restored both T4s. Verified computing correctly, not just
#       enumerating (exact fp64 matmul, cuDNN, P2P, full-VRAM write+verify).
#   compute-node-3 (10.222.1.120)  2x NVIDIA A10  23 GB -- DOWN as of 2026-08-15:
#       does not ping, "No route to host". Left in the list deliberately: probes
#       run in parallel and a dead node just contributes no slots, so it
#       re-joins automatically if it comes back.
#   compute-node-4 (10.222.1.121)  1x NVIDIA A40  46 GB
#   compute-node-5 (10.222.1.125)  no GPU (dev node)   head-node: no GPU
#
# NOTE the T4s are 15 GB against the A10s' 23 GB and the A40's 46 GB, and are a
# slower generation. experiments/tools/run_queue.py is memory-aware and will not
# place a large SSL model on a card that cannot hold it; a plain `gpurun.sh`
# invocation is not, so pick -n/-g yourself for the heavy models. With node-3
# down, the 20 GB SSL/hybrid jobs have exactly one home: node-4's A40.
CANDIDATE_NODES=(compute-node-1 compute-node-2 compute-node-3 compute-node-4)
CONDA_ROOT="$HOME/anaconda2025"
ENV_NAME="SL_SPV"
MIN_FREE_MB=8000
DETACH=0
FORCE_NODE=""
FORCE_GPU=""
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

die() { echo "gpurun: $*" >&2; exit 1; }

# Query one node. Emits "<node> <gpu_index> <free_mib> <util>" per usable GPU.
# Nodes with a broken/absent driver emit nothing.
# Status 2026-08-10 (re-verified): compute-node-3 (10.222.1.120) = 2x A10 23GB,
# compute-node-4 (10.222.1.121) = 1x A40 46GB. BOTH working -- node-4's driver
# fault from earlier in 2026 is resolved, and its A40 is the largest GPU here.
probe_node() {
    local node="$1"
    timeout 20 ssh "${SSH_OPTS[@]}" "$node" \
        'nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null' \
        2>/dev/null | awk -v n="$node" -F', *' 'NF>=3 && $1 ~ /^[0-9]+$/ {print n, $1, $2, $3}'
}

probe_all() {
    local node
    for node in "${CANDIDATE_NODES[@]}"; do
        probe_node "$node" &
    done
    wait
}

show_status() {
    echo "Candidate nodes: ${CANDIDATE_NODES[*]}"
    echo
    local node out
    for node in "${CANDIDATE_NODES[@]}"; do
        printf '=== %s ===\n' "$node"
        if ! out="$(timeout 20 ssh "${SSH_OPTS[@]}" "$node" 'echo REACHABLE' 2>&1)" || [[ "$out" != REACHABLE ]]; then
            echo "  SSH unreachable: $out"
            echo
            continue
        fi
        out="$(timeout 20 ssh "${SSH_OPTS[@]}" "$node" \
            'nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader 2>&1; \
             echo "-- compute apps --"; \
             nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>&1' 2>&1 || true)"
        echo "$out" | sed 's/^/  /'
        echo
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--detach)   DETACH=1; shift ;;
        -n|--node)     FORCE_NODE="${2:?}"; shift 2 ;;
        -g|--gpu)      FORCE_GPU="${2:?}"; shift 2 ;;
        -e|--env)      ENV_NAME="${2:?}"; shift 2 ;;
        -m|--min-free) MIN_FREE_MB="${2:?}"; shift 2 ;;
        -s|--status)   show_status; exit 0 ;;
        -h|--help)     sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
        --)            shift; break ;;
        *)             die "unknown option '$1' (did you forget '--' before the command?)" ;;
    esac
done

[[ $# -gt 0 ]] || die "no command given. Example: tools/gpurun.sh -- python trainSpeakerNet.py --config ..."

# $PWD must be on shared storage, or the remote side won't see the same files.
case "$PWD" in
    /mnt/ricproject3/*|"$HOME"/*|/mnt/ricproject/*|/mnt/ricproject2/*) ;;
    *) die "cwd '$PWD' is not on shared storage (/mnt/ricproject*, \$HOME); the remote node cannot see it." ;;
esac

# --- pick node + GPU ---------------------------------------------------------
if [[ -n "$FORCE_NODE" && -n "$FORCE_GPU" ]]; then
    NODE="$FORCE_NODE"; GPU="$FORCE_GPU"; FREE="(forced)"
else
    mapfile -t ROWS < <(probe_all | sort -k3 -nr)
    [[ ${#ROWS[@]} -gt 0 ]] || die "no node has a working NVIDIA driver. Try: $0 --status"

    NODE=""; GPU=""; FREE=""
    for row in "${ROWS[@]}"; do
        read -r r_node r_gpu r_free _r_util <<<"$row"
        [[ -n "$FORCE_NODE" && "$r_node" != "$FORCE_NODE" ]] && continue
        [[ -n "$FORCE_GPU"  && "$r_gpu"  != "$FORCE_GPU"  ]] && continue
        (( r_free >= MIN_FREE_MB )) || continue
        NODE="$r_node"; GPU="$r_gpu"; FREE="${r_free} MiB free"
        break
    done
    [[ -n "$NODE" ]] || die "no GPU with >= ${MIN_FREE_MB} MiB free. Current state:
$(printf '  %s\n' "${ROWS[@]}")"
fi

echo "gpurun: $NODE GPU $GPU  ($FREE)  env=$ENV_NAME  cwd=$PWD" >&2

# --- build the remote command ------------------------------------------------
ENV_BIN="$CONDA_ROOT/envs/$ENV_NAME/bin"
QUOTED_CMD="$(printf '%q ' "$@")"

# CUDA_VISIBLE_DEVICES pins the chosen GPU, which the remote process then sees
# as device 0 — so --distributed / .cuda() / device indices need no adjustment.
REMOTE_PRELUDE="set -euo pipefail
cd $(printf '%q' "$PWD")
export PATH=$(printf '%q' "$ENV_BIN"):\$PATH
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1"

if (( DETACH )); then
    LOG_DIR="$PWD/exps/gpurun_logs"
    mkdir -p "$LOG_DIR"
    STAMP="$(date +%Y%m%d-%H%M%S)"
    LOG_FILE="$LOG_DIR/${STAMP}-${NODE}-gpu${GPU}.log"
    PID_FILE="$LOG_DIR/${STAMP}-${NODE}-gpu${GPU}.pid"

    {
        echo "# gpurun detached run"
        echo "# node=$NODE gpu=$GPU env=$ENV_NAME"
        echo "# cwd=$PWD"
        echo "# started=$(date -Is)"
        echo "# cmd=$*"
        echo "# ---"
    } > "$LOG_FILE"

    # setsid detaches from the SSH session so the run survives disconnection.
    ssh "${SSH_OPTS[@]}" "$NODE" "$REMOTE_PRELUDE
setsid nohup bash -c $(printf '%q' "$QUOTED_CMD") >> $(printf '%q' "$LOG_FILE") 2>&1 < /dev/null &
echo \$! > $(printf '%q' "$PID_FILE")
disown || true"

    echo "gpurun: detached. PID $(cat "$PID_FILE" 2>/dev/null || echo '?') on $NODE" >&2
    echo "gpurun: log  -> $LOG_FILE" >&2
    echo "gpurun: tail -> tail -f $LOG_FILE" >&2
    echo "gpurun: kill -> ssh $NODE kill \$(cat $PID_FILE)" >&2
else
    # -t gives a TTY so Ctrl-C propagates and progress bars render.
    exec ssh -t "${SSH_OPTS[@]}" "$NODE" "$REMOTE_PRELUDE
$QUOTED_CMD"
fi
