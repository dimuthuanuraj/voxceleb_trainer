#!/usr/bin/env bash
# noderun.sh -- run a command in the SL_SPV env on a specific GPU node.
#
# Node map (verified 2026-08-10):
#   10.222.1.120  compute-node-3  2x A10 23GB   /mnt/ricproject3 MOUNTED   -> data-touching jobs
#   10.222.1.121  compute-node-4  1x A40 46GB   /mnt/ricproject3 ABSENT    -> $HOME-only jobs
#
# node-4 has the bigger GPU but cannot see the corpora (no NFS mount for
# /mnt/ricproject3 and no passwordless sudo to add one). $HOME *is* shared from
# the same server, so node-4 is used for embedding-space compute where the
# inputs are small .npz files staged under $HOME.
#
# Usage:
#   tools/noderun.sh 3 python -u tools/extract_embeddings.py --all
#   tools/noderun.sh 4 python -u tools/tsne_umap_compute.py --emb-dir ~/sl_spv_qc/emb
set -euo pipefail

NODE_ARG="${1:?usage: noderun.sh <3|4|host> <command...>}"; shift
case "$NODE_ARG" in
  3) HOST=10.222.1.120; WORKDIR=/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer ;;
  4) HOST=10.222.1.121; WORKDIR="\$HOME/sl_spv_qc" ;;
  *) HOST="$NODE_ARG"; WORKDIR=/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer ;;
esac

CMD="$*"
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 "$HOST" \
    "set -eo pipefail
     # NB: no 'set -u' -- conda's libblas activate hook reads unbound vars
     source \$HOME/anaconda2025/etc/profile.d/conda.sh
     conda activate SL_SPV
     cd $WORKDIR
     echo \"[noderun] \$(hostname) \$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')\"
     $CMD"
