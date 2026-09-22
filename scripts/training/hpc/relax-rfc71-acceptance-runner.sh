#!/usr/bin/env bash
set -euo pipefail
ROOT="${RELAX_VALIDATION_ROOT:-/projects/b6ci/arazy/relax-rfc71-validation-20260922}"
OLD="${RELAX_RUNTIME_ROOT:-/projects/b6ci/arazy/relax-inference-20260917-01a0aeec}"
RUN=$ROOT/validation/run-${SLURM_JOB_ID}-acceptance
cd "$ROOT"
mkdir -p "$RUN"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
export PATH=$OLD/runtime/venv/bin:$PATH
export PYTHONPATH=$ROOT:$OLD/runtime/Megatron-LM
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 RAY_USAGE_STATS_ENABLED=0
export NCCL_SOCKET_IFNAME=hsn0 GLOO_SOCKET_IFNAME=hsn0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export RAY_TMPDIR=/tmp/relax-rfc71-${SLURM_JOB_ID}-${SLURM_PROCID}
export TMPDIR=$RAY_TMPDIR/compiler
mkdir -p "$TMPDIR"
node_ip=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
if [[ "$SLURM_PROCID" == 0 ]]; then
  heartbeat() { while :; do echo "RUNNER_HEARTBEAT $(date -u +%FT%TZ)"; sleep 10; done; }
  heartbeat & heartbeat_pid=$!
  trap 'kill -TERM "$heartbeat_pid" 2>/dev/null || true' EXIT
  port=6379; dash=8265
  export RAY_ADDRESS=$node_ip:$port
  ray start --head --node-ip-address="$node_ip" --port="$port" --dashboard-host=0.0.0.0 --dashboard-port="$dash" --num-cpus=16 --num-gpus=1 --object-store-memory=2147483648 --include-dashboard=true --disable-usage-stats --temp-dir="$RAY_TMPDIR" --block > "$RUN/ray-head.log" 2>&1 &
  ray_pid=$!
  trap 'touch "$RUN/done"; kill -TERM "$heartbeat_pid" 2>/dev/null || true; kill -TERM "$ray_pid" 2>/dev/null || true; wait "$ray_pid" || true' EXIT
  for i in $(seq 1 90); do
    grep -q 'Ray runtime started.' "$RUN/ray-head.log" && break
    kill -0 "$ray_pid" 2>/dev/null || { cat "$RUN/ray-head.log"; exit 1; }
    sleep 2
  done
  grep -q 'Ray runtime started.' "$RUN/ray-head.log"
  printf '%s\n' "$RAY_ADDRESS" > "$RUN/address"
  python - <<'PY'
import os, time, ray
ray.init(address=os.environ['RAY_ADDRESS'], log_to_driver=False)
deadline=time.monotonic()+240
while time.monotonic()<deadline:
    nodes=[n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU',0)>0]
    if len(nodes)>=4:
        print('ACCEPTANCE_CLUSTER', [(n['NodeManagerAddress'], n['Resources']['GPU']) for n in nodes], flush=True)
        break
    time.sleep(2)
else:
    raise RuntimeError('four GPU Ray nodes did not register')
ray.shutdown()
PY
  export MODEL_DIR=$OLD/runtime/Qwen3-0.6B MODEL_CONFIG_DIR=$ROOT/scripts/models
  export RAY_DASHBOARD="http://$node_ip:$dash"
  export RAY_DASHBOARD_ADDRESS="$RAY_DASHBOARD"
  export RAY_API_SERVER_ADDRESS="$RAY_DASHBOARD"
  export RUNTIME_ENV_JSON='{}'
  export INFERENCE_GPUS_PER_NODE=1 INFERENCE_ROLLOUT_GPUS=1 INFERENCE_TEACHER_GPUS=1 INFERENCE_GENRM_GPUS=1
  export INFERENCE_ACTOR_NODES=1 INFERENCE_ACTOR_GPUS_PER_NODE=1
  export INFERENCE_ROLLOUT_BATCH_SIZE=2 INFERENCE_N_SAMPLES_PER_PROMPT=2 INFERENCE_GLOBAL_BATCH_SIZE=4
  export INFERENCE_LAYOUT=decoupled INFERENCE_TRAIN_OUTPUT="$RUN/decoupled"
  export INFERENCE_WAIT_FOR_JOB=1
  echo "START_DECOUPLED $(date -u +%FT%TZ)" | tee "$RUN/acceptance.log"
  set +e
  bash scripts/entrypoint/ray-job.sh scripts/training/hpc/run-unified-inference-3role.sh 2>&1 | tee -a "$RUN/acceptance.log"
  launcher_status=${PIPESTATUS[0]}
  set -e
  echo "DECOUPLED_LAUNCHER_STATUS=$launcher_status" | tee -a "$RUN/acceptance.log"
  if [[ "$launcher_status" -ne 0 ]]; then exit "$launcher_status"; fi
  echo ACCEPTANCE_DECOUPLED_DONE | tee "$RUN/status"
else
  for i in $(seq 1 240); do [[ -s "$RUN/address" ]] && break; sleep 2; done
  export RAY_ADDRESS=$(cat "$RUN/address")
  ray start --address="$RAY_ADDRESS" --node-ip-address="$node_ip" --num-cpus=16 --num-gpus=1 --object-store-memory=2147483648 --disable-usage-stats --temp-dir="$RAY_TMPDIR" --block > "$RUN/ray-worker-${SLURM_PROCID}.log" 2>&1 &
  ray_pid=$!
  trap 'kill -TERM "$ray_pid" 2>/dev/null || true; wait "$ray_pid" || true' EXIT
  while [[ ! -f "$RUN/done" ]]; do kill -0 "$ray_pid" 2>/dev/null || exit 1; sleep 5; done
fi
