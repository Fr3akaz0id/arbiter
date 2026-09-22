#!/bin/bash
# wait-vram.sh <gpu-index> <min-free-MiB> <timeout-s>
# VRAM admission gate for the arbiter lane on fkzllama.
# Blocks until the target GPU reports >= min-free MiB, then exits 0.
# On timeout it exits non-zero ON PURPOSE: the unit then enters failed and
# systemd's Restart=on-failure retries it. Arbiter must never squeeze onto a
# GPU a llama lane still owns -- that is how the 2x2 dual-lane config died
# (it passed every VRAM metric we recorded and was unusable in real use).
# Fail loud, never oversubscribe.
gpu="${1:?gpu index}"; need="${2:?min free MiB}"; tmo="${3:-900}"
deadline=$(( $(date +%s) + tmo ))
read_mi() {
  nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -dc '0-9\n' | head -n1
}
while :; do
  free="$(read_mi | tr -d '[:space:]')"
  [[ "$free" =~ ^[0-9]+$ ]] || free=0
  if (( free >= need )); then
    echo "gpu$gpu has ${free} MiB free (>= ${need} MiB) -- admitting arbiter"
    exit 0
  fi
  if (( $(date +%s) >= deadline )); then
    echo "FATAL: gpu$gpu stuck at ${free} MiB free (< ${need} MiB) after ${tmo}s." >&2
    echo "       something still owns the GPU; arbiter refuses to oversubscribe." >&2
    exit 1
  fi
  echo "waiting for gpu$gpu vram: ${free}/${need} MiB"
  sleep 5
done
