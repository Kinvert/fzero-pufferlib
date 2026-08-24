# F-Zero X

This integration has two backends with one policy contract:

- `fzero.h` is the exact headless G-Diffuser CPU oracle.
- `fzero.cu` is a batched, device-resident reduced simulator for high-throughput
  training. It loads the CPU oracle once at startup to decode the private ROM,
  exports course geometry and a canonical reset template into memory, and then
  performs reset, four-tick step, observation, reward, terminal, logging, and
  autoreset on CUDA without per-step host copies.

Neither backend opens a window, audio device, input device, or frame-rate
limiter in headless mode. The ROM remains private and is loaded at runtime;
never add it or an extracted course table to either repository.

## Interface

- Observation: 83 bounded `float32` values. The first 31 describe the racer,
  controls, and episode state; six 8-value samples describe the track 250, 500,
  1,000, 2,000, 4,000, and 8,000 distance units ahead.
  Observation 24 says the car is on a dash pad now; observations 14 and 19
  report the boost timer and any active boost. Observations 79-82 describe the
  nearest upcoming dash pad: validity, wrapped forward distance, normalized
  lateral/surface target, and normalized width.
- Action: `MultiDiscrete([11, 5, 2, 2, 2])` for steer, pitch, throttle, brake,
  and a one-tick boost tap. Steer uses neutral plus five symmetric magnitudes in
  each direction. Eleven is deliberate: ten bins cannot retain both an exact
  neutral and symmetric left/right choices. Category IDs are neutral-first:
  steer `{0, -13, +13, -25, +25, -38, +38, -50, +50, -63, +63}` and pitch
  `{0, -32, +32, -63, +63}`.
- Default task: one-lap Mute City Time Attack, Blue Falcon, one agent, four
  physics ticks per policy decision. The first curriculum exposes all 11 steer
  bins while forcing pitch neutral, throttle on, brake off, and boost off.
- The exact core retains distance progress divided by 1,000, +2 per completed
  lap, +25 for a finish, and -25 for fatal failure as raw diagnostics.
  Nonterminal learner rewards are divided by 32. The terminal learner reward
  and logged `score`/`perf` are instead `clamp(2000 / lap_frames, 0, 1)` for a
  finish, `-1` for fatal failure, and `0` for timeout/stall. Optional
  `dash_hit_reward` shaping is paid once per dash-pad contact entry, detected
  across all raw ticks inside an action repeat.

The full ABI, observation schema, termination rules, and validation plan are in
`../../../G-Diffuser/docs/PUFFERLIB_RL_SPEC.md`.

## Build the headless core

From `G-Diffuser`:

```bash
cmake -S . -B build/rl-headless -G Ninja \
  -DGDX_RL_HEADLESS=ON -DGDX_EXPANSION_KIT=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGDX_RL_TEST_ROM="$PWD/../private/baserom.us.rev0.z64"
cmake --build build/rl-headless --parallel
ctest --test-dir build/rl-headless --output-on-failure
```

The default config expects the verified US rev0 ROM at
`../private/baserom.us.rev0.z64`, relative to this PufferLib checkout. An
explicit `FZEROX_ROM=/absolute/path/to/rom.z64` overrides that path.

## Build PufferLib locally

No driver or system install is needed when CUDA is already present. If NCCL is
not available through the CUDA installation, put NVIDIA's NCCL wheel in a
repo-local uv environment:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv venv .venv
UV_CACHE_DIR="$PWD/.uv-cache" uv pip install \
  --python .venv/bin/python nvidia-nccl-cu12
source .venv/bin/activate
```

Then build from the PufferLib checkout:

```bash
CUDA_HOME=/usr/local/cuda ./build.sh fzero build/fzero-train
```

Build the CUDA environment separately (set the local GPU architecture when
`-arch=native` cannot inspect a sandboxed GPU):

```bash
CUDA_HOME=/usr/local/cuda NVCC_ARCH=sm_120 \
  ./build.sh fzero build/fzero-gpu --cu
```

The CUDA binary selects `config/fzero_gpu.ini`; the CPU binary continues to use
`config/fzero.ini`. On the current RTX 5060, an earlier observation-v1
4,096-agent, horizon-128,
replay-ratio-1 CUDA-graph run processed 5,242,880 policy decisions in 1.186 s,
or 4.4M reported end-to-end SPS with the then-current 111,360-parameter policy.
At action repeat four that is about 17.6M reduced-physics ticks/s. This clears
the throughput target but is not yet a fidelity or CPU-transfer result. The
83-input policy has 111,872 parameters; remeasure CUDA throughput before using
the v1 number for capacity planning.

The F-Zero build keeps ccache state under `build/`. When using wheel-provided
NCCL, expose its repo-local library directory at runtime:

```bash
NCCL_LIB="$(.venv/bin/python -c \
  'import nvidia.nccl, os; print(os.path.join(nvidia.nccl.__path__[0], "lib"))')"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Train and evaluate

For a narrow Mute City lap-time demonstration, set
`env.finish_time_target_frames` to a positive target. A successful terminal
transition receives and logs
`clamp(target_frames / episode_frames, 0, 1)`. Fatal outcomes receive `-1`, and
time-limit/stall truncations receive `0`. This terminal outcome replaces the
entire boundary-transition learner reward; scaled progress and dash shaping
remain on nonterminal transitions. `score` and `perf` use the same outcome,
while `episode_return` preserves the raw diagnostic return and `finish_rate`
remains binary. A zero target gives every successful finish score 1.

`dash_hit_reward` is expressed in the exact core's raw reward units and is
then multiplied by `reward_scale` on nonterminal transitions. Thus `0.1` adds
`0.003125` to the learner reward with the default `reward_scale=1/32`.

### Current 83-input booster-observation run

This fresh run trained the 111,872-parameter policy from random initialization
with exact G-Diffuser CPU physics and the GPU PPO learner. It used the four
upcoming-pad observation fields, normalized terminal pace score, and pad-entry
shaping:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build/fzero-train train --headless \
  --base.run_id=fzero-mc1-padobs-score-dash01-5min-20260824-a \
  --base.checkpoint_dir=build/fzero-learning/checkpoints \
  --base.log_dir=build/fzero-learning/logs \
  --base.eval_episodes=0 \
  --base.checkpoint_interval=256 \
  --env.course_index=0 \
  --env.laps=1 \
  --env.action_repeat=4 \
  --env.steering_curriculum=1 \
  --env.finish_time_target_frames=2000 \
  --env.dash_hit_reward=0.1 \
  --train.total_timesteps=819200 \
  --train.learning_rate=0.003 \
  --train.replay_ratio=4 \
  --train.ent_coef=0.0001
```

The run completed all 819,200 policy decisions in 293.200 seconds
(4:53.200), or 2,794 decisions/s overall, and wrote 25 checkpoints. Screening
the late checkpoints and then evaluating the shortlist on the same 64-episode
exact stochastic stream selected step 786,432:

```text
build/fzero-learning/checkpoints/fzero/fzero-mc1-padobs-score-dash01-5min-20260824-a/0000000000786432.bin
SHA-256 646e7f1116d280cf4a2c73efacb0ce9a428e306c7188da62d36f7e0ca1bac9d9
```

It finished 64/64 episodes in a mean 2,113.594 physics frames, logged mean
normalized score/perf `0.946328`, and hit 2.625 dash pads per lap. Against the
fastest historical 79-input checkpoint on the identical evaluation stream, it
is 15.312 frames (0.2552 seconds) faster and raises pad contacts from 1.390625
to 2.625 per lap. The final step-819,200 checkpoint also finished 64/64, at
2,119.219 frames, score `0.943854`, and 2.531 contacts; checkpoint selection is
therefore based on evaluated pace rather than merely taking the last weights.

Reproduce the selected checkpoint evaluation with:

```bash
build/fzero-train eval \
  build/fzero-learning/checkpoints/fzero/fzero-mc1-padobs-score-dash01-5min-20260824-a/0000000000786432.bin \
  --headless --base.eval_episodes=64 \
  --env.course_index=0 --env.laps=1 --env.action_repeat=4 \
  --env.steering_curriculum=1 --env.finish_time_target_frames=2000 \
  --env.dash_hit_reward=0.1
```

This run uses the exact CPU simulator with GPU policy inference/training. The
reduced CUDA simulator consumes the same upcoming-pad descriptors and matches
their observation values, but does not yet implement pad contact/boost physics
or `dash_hit_reward`; do not treat this checkpoint as a CUDA-simulator result.

### Historical 79-input baselines

The commands and results below through the dash-contact experiment predate the
four upcoming-pad observation fields. Their 79-input checkpoints are
intentionally incompatible with the current 83-input policy and cannot be
loaded without a first-layer migration.

Run the then-validated steering-curriculum recipe for 65,536 decisions:

```bash
build/fzero-train train --headless \
  --base.run_id=fzero-steer-65k \
  --base.eval_episodes=0 \
  --base.checkpoint_interval=64 \
  --train.total_timesteps=65536
```

The config defaults to the empirically successful `learning_rate=0.003`,
`replay_ratio=4`, and `ent_coef=0.0001`. A first run with these settings reached
52/64 finishes in fresh-process evaluation. A second 65,536-decision phase,
warm-started with `learning_rate=0.0015` and `ent_coef=0`, reached 231/256
finishes (90.23%, mean raw score 100.56):

```bash
build/fzero-train train --headless \
  --base.run_id=fzero-steer-cont65k \
  --base.load_model_path=checkpoints/fzero/fzero-steer-65k/STEP.bin \
  --base.eval_episodes=0 \
  --base.checkpoint_interval=64 \
  --train.total_timesteps=65536 \
  --train.learning_rate=0.0015 \
  --train.replay_ratio=4 \
  --train.ent_coef=0
```

The first five-minute continuation used a 2,000-frame target and the validated
checkpoint above:

```bash
build/fzero-train train --headless \
  --base.run_id=fzero-mc1-laptime-5min-cont-20260824-a \
  --base.checkpoint_dir=build/fzero-learning/checkpoints \
  --base.log_dir=build/fzero-learning/logs \
  --base.load_model_path=build/fzero-learning/checkpoints/fzero/fzero-steer-cont65k-lr15e4-r4-20260823-c/0000000000065536.bin \
  --base.eval_episodes=0 \
  --base.checkpoint_interval=256 \
  --env.course_index=0 \
  --env.laps=1 \
  --env.action_repeat=4 \
  --env.steering_curriculum=1 \
  --env.finish_time_target_frames=2000 \
  --train.total_timesteps=835584 \
  --train.learning_rate=0.0015 \
  --train.replay_ratio=4 \
  --train.ent_coef=0
```

The process was stopped after the 655,360-decision checkpoint was safely
written at five minutes. On the same fixed 64-episode exact evaluation stream,
the source checkpoint finished 61/64 with 2,204.9 mean successful frames; the
new checkpoint finished 64/64 with 2,128.9. The canonical sampler-seed-1 trace
improved from 2,241 to 2,114 physics frames.

The next 655,360-decision continuation keeps that pace target and adds `+0.1`
raw per dash-pad acquisition. The source policy already averaged 1.390625 hits
across the same 64 one-lap evaluations (89 contacts total out of 256 possible):

```bash
build/fzero-train train --headless \
  --base.run_id=fzero-mc1-laptime-dash01-5min-cont-20260824-a \
  --base.checkpoint_dir=build/fzero-learning/checkpoints \
  --base.log_dir=build/fzero-learning/logs \
  --base.load_model_path=build/fzero-learning/checkpoints/fzero/fzero-mc1-laptime-5min-cont-20260824-a/0000000000655360.bin \
  --base.eval_episodes=0 \
  --base.checkpoint_interval=256 \
  --env.course_index=0 \
  --env.laps=1 \
  --env.action_repeat=4 \
  --env.steering_curriculum=1 \
  --env.finish_time_target_frames=2000 \
  --env.dash_hit_reward=0.1 \
  --train.total_timesteps=655360 \
  --train.learning_rate=0.0015 \
  --train.replay_ratio=4 \
  --train.ent_coef=0
```

This run completed naturally in 242.268 seconds (4:02.268), or 2,705.1
decisions/s overall. Screening every saved checkpoint selected step 131,072:

```text
build/fzero-learning/checkpoints/fzero/fzero-mc1-laptime-dash01-5min-cont-20260824-a/0000000000131072.bin
SHA-256 fbde3087bb73b3225c21b32ef77b177ad98ea22892553a0456872b67e8e85e6d
```

On the same 64 exact stochastic evaluations, it retained 64/64 finishes and
raised dash contacts from 89 to 117: 1.390625 to 1.828125 per lap, a 31.5%
increase. Mean finish time moved from 2,128.906 to 2,138.656 physics frames
(+9.75 frames, about +0.163 seconds). The final step reached exactly 2.0 hits
per evaluated episode but regressed to 63/64 finishes and 2,148.0 mean finish
frames, so it is not the selected checkpoint. Keep the source pace checkpoint
for the fastest current demo; use step 131,072 when demonstrating that the
explicit pad reward changes behavior.

These results are for the one-lap steering-only curriculum. They establish that
PPO learns from the public observation and binned action contract; they are not
yet a three-lap, full-action, or multi-course result.

The policy weights are restored before the first rollout; optimizer state and
global step restart at zero. The next staged experiment is three-lap steering,
with `--env.laps=3 --env.max_episode_frames=18000`, followed by a separately
tuned full-five-head phase. That full-action schedule has not been validated
yet, so it is deliberately not presented as a default command here. In native
training, `--env.steering_curriculum=0` with no model
samples an untrained stochastic policy; it is not a no-op baseline. The
real-core adapter test below supplies the exact fixed all-zero baseline.

A short end-to-end PPO check is:

```bash
build/fzero-train train --headless \
  --base.run_id=fzero-smoke \
  --base.eval_episodes=0 \
  --base.checkpoint_interval=1 \
  --train.total_timesteps=1024
```

Reload the final checkpoint with:

```bash
build/fzero-train eval checkpoints/fzero/RUN_ID/STEP.bin \
  --headless --base.eval_episodes=256
```

Export one exact CPU-oracle episode for graphical G-Diffuser playback with:

```bash
ocean/fzero/export_render_trace.sh \
  checkpoints/fzero/RUN_ID/STEP.bin \
  build/fzero-render-eval-input.txt
```

The runner pins the canonical seed-73, one-lap Mute City steering task and an
explicit policy sampler seed of 1. It writes post-curriculum actions in
`GDX_INPUT_SCRIPT` format: each decision holds its raw pad for four game ticks,
except boost, which taps B only on the first tick. The footer records the exact
terminal score, reason, frame, distance, and semantic hash, then waits for the
finish display, requests a screenshot, and quits cleanly. The output is
published atomically only after the episode reaches a boundary.

Evaluation samples categorical actions. It is reproducible with fixed sampler
and environment seeds, but is not greedy/argmax evaluation. `base.seed` controls
model initialization and sampling while `env.seed` controls reset sequencing.
Tested solo Mute City resets are physics-invariant across environment seeds, so
different reset seeds are not evidence of task generalization.

Focused adapter tests, including worker-thread handoff, terminal autoreset,
runtime course export, deterministic reduced-core replay, initial-observation
geometry agreement, a scripted reduced-core lap, and the exact CPU no-op
baseline, run with:

```bash
bash ocean/fzero/tests/run_all.sh
FZERO_TEST_REAL_CORE=1 bash ocean/fzero/tests/run_all.sh
```

Episode logs report normalized `score`/`perf`, raw episode returns, actual
learner returns, successful finish frames, dash hits, and mean-per-episode
action-category fractions. These are the first diagnostics for detecting
steering collapse before unlocking the button heads.

## CUDA fidelity status

The CUDA backend is explicitly a reduced simulator v0, not a claim that the
7,000-line stock racer update has already been reproduced. Its first reset
observation agrees with the exact spline oracle to a measured maximum absolute
error of 0.001136 over the original 79 fields; the four dash descriptors agree
exactly in the current full-lap host parity test. Its deterministic scripted pilot
finishes the first Mute City lap in 2,412 physics frames versus 2,339 for the
CPU public-observation oracle. Those are geometry and task-shape checks, not
open-loop dynamics parity.

The next release gates are fixed-action CPU/CUDA state traces, policy-in-loop
paired outcomes, and a GPU-trained checkpoint that reaches the existing 90%
finish gate in fresh exact CPU-oracle episodes. Until those pass, GPU-only
returns must not be presented as F-Zero policy performance. Keep the exact
83-value observation, `MultiDiscrete([11,5,2,2,2])` action, reward, termination,
and action-repeat contracts stable while the solo racer/contact physics is
incrementally ported or fitted.
