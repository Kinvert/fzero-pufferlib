#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 CHECKPOINT.bin OUTPUT_SCRIPT.txt [CPU_EVAL_BINARY]" >&2
    exit 2
fi

checkpoint=$1
output_script=$2
cpu_eval=${3:-build/fzero-cpu}
core_lib=${FZERO_CORE_LIB:-../G-Diffuser/build/rl-headless/port/libgdiffuser_rl.so}

if [[ ! -f "$checkpoint" ]]; then
    echo "error: checkpoint not found: $checkpoint" >&2
    exit 2
fi

if [[ ! -x "$cpu_eval" ]]; then
    ./build.sh fzero "$cpu_eval" --cpu
fi

# This pins the exact canonical one-lap task that produced the validated CPU
# policy result. Policy sampling uses the standalone evaluator's explicit seed
# 1 stream; environment/race initialization uses seed 73.
"$cpu_eval" fzero "$checkpoint" \
    --headless \
    --gdx-input-script="$output_script" \
    --gdx-sampler-seed=1 \
    --base.seed=73 \
    --base.eval_episodes=1 \
    --env.core_lib_path="$core_lib" \
    --env.seed=73 \
    --env.course_index=0 \
    --env.character=0 \
    --env.machine_skin=0 \
    --env.laps=1 \
    --env.action_repeat=4 \
    --env.max_episode_frames=9000 \
    --env.stall_frames=600 \
    --env.stall_progress=100 \
    --env.engine_balance=0.5 \
    --env.cpu_oracle=0 \
    --env.steering_curriculum=1 \
    --env.reward_scale=0.03125 \
    --policy.hidden_size=128 \
    --policy.num_layers=2
