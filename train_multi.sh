 #!/usr/bin/env bash
set -euo pipefail

# Multi-run launcher for cloud training on EC2.
# Usage example:
#   bash train_multi.sh 5
#   RUNS=3 MAP=MoveToBeacon ENABLE_WANDB=true bash train_multi.sh

RUNS="${1:-${RUNS:-3}}"
MAP="${MAP:-MoveToBeacon}"
AGENT="${AGENT:-agent.DQN.DQNAgent}"
SC2PATH_VALUE="${SC2PATH_VALUE:-/home/ec2-user/StarCraftII}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-s3://csc495-rl-artifacts/csc495/checkpoints}"
REPLAY_DIR="${REPLAY_DIR:-s3://csc495-rl-artifacts/csc495/replays}"
ARTIFACT_BACKEND="${ARTIFACT_BACKEND:-s3}"
HEADLESS="${HEADLESS:-true}"
RENDER="${RENDER:-false}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-10}"
MAX_EPISODES="${MAX_EPISODES:-8000}"
ENABLE_WANDB="${ENABLE_WANDB:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-swannercjj}"
WANDB_PROJECT="${WANDB_PROJECT:-csc495}"

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
  echo "RUNS must be a positive integer. Got: $RUNS" >&2
  exit 1
fi

echo "Starting $RUNS training run(s)"
echo "Map: $MAP"
echo "Agent: $AGENT"
echo "Max episodes: $MAX_EPISODES"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Replay dir: $REPLAY_DIR"

for ((i=1; i<=RUNS; i++)); do
  run_tag="run_${i}_$(date +%Y%m%d_%H%M%S)"
  echo "========================================"
  echo "[$(date)] Launching run $i/$RUNS ($run_tag)"

  cmd=(
    python -m main
    --agent "$AGENT"
    --map "$MAP"
    --artifact_backend "$ARTIFACT_BACKEND"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --replay_dir "$REPLAY_DIR"
    --sc2path "$SC2PATH_VALUE"
    --headless "$HEADLESS"
    --render "$RENDER"
    --max_episodes "$MAX_EPISODES"
  )

  if [[ "$ENABLE_WANDB" == "true" ]]; then
    cmd+=(
      --wandb true
      --wandb_entity "$WANDB_ENTITY"
      --wandb_project "$WANDB_PROJECT"
      --wandb_run_name "$run_tag"
    )
  else
    cmd+=(--wandb false)
  fi

  "${cmd[@]}"

  echo "[$(date)] Run $i/$RUNS finished"
  if [[ "$i" -lt "$RUNS" ]]; then
    echo "Sleeping ${SLEEP_BETWEEN_RUNS}s before next run..."
    sleep "$SLEEP_BETWEEN_RUNS"
  fi
done

echo "All runs completed."
