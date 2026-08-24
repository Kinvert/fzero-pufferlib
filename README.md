# F-Zero X for PufferLib 5.0

A headless, trainable F-Zero X reinforcement-learning environment for
PufferLib 5.0, with an exact G-Diffuser CPU reference environment, a reduced
batched CUDA simulator, deterministic graphical policy replay, and direct
OpenGL video capture.

This repository contains only source patches and documentation. It does not
contain a ROM, extracted game assets, trained checkpoints, build products,
logs, or videos. You must provide your own legally obtained North American
revision-0 ROM.

## Current result

The current one-lap Mute City steering curriculum uses:

- `float32[83]` observations, including the nearest upcoming dash pad;
- `MultiDiscrete([11, 5, 2, 2, 2])` actions for steering, pitch, throttle,
  brake, and boost tap;
- terminal finish score `clamp(2000 / lap_frames, 0, 1)`;
- fatal outcomes scored exactly `-1`, with timeout/stall truncations scored `0`;
- an optional raw `+0.1` dash-contact shaping reward.

The 111,872-parameter policy trained for 819,200 decisions in 293.200 seconds
(4:53.200). The selected step-786,432 checkpoint finished 64/64 exact
evaluations in a mean 2,113.594 physics frames, with mean normalized score
`0.946328` and 2.625 dash contacts per lap. Checkpoint weights are deliberately
not included.

## Pinned upstream revisions

| Component | Upstream | Revision |
| --- | --- | --- |
| G-Diffuser | `https://github.com/Zorkats/G-Diffuser.git` | `c62bf97d490370b05e14cb04d149f36516937270` |
| libultraship submodule | `https://github.com/Zorkats/libultraship.git` | `a31fbffa86658b89dd04083f9bca615706665d45` |
| fzerox submodule | `https://github.com/Zorkats/fzerox.git` | `6474cfd4fcc7002bb0763ba6346b71d1715025ca` |
| PufferLib | `https://github.com/PufferAI/PufferLib.git` | `0aa034def1b3c91af9971798288f8bddd1817277` |

The patches are intentionally pinned. Applying them to another revision may
fail or silently change the tested environment contract.

## Apply the integration

Create sibling checkouts in a workspace:

```bash
export FZERO_WORKSPACE=/path/to/fzero-workspace
mkdir -p "$FZERO_WORKSPACE"

git clone --recursive https://github.com/Zorkats/G-Diffuser.git \
  "$FZERO_WORKSPACE/G-Diffuser"
git -C "$FZERO_WORKSPACE/G-Diffuser" checkout \
  c62bf97d490370b05e14cb04d149f36516937270
git -C "$FZERO_WORKSPACE/G-Diffuser" submodule update --init --recursive

git clone https://github.com/PufferAI/PufferLib.git \
  "$FZERO_WORKSPACE/PufferLib-5.0"
git -C "$FZERO_WORKSPACE/PufferLib-5.0" checkout \
  0aa034def1b3c91af9971798288f8bddd1817277
```

Then apply all four source patches from this repository:

```bash
./scripts/apply.sh "$FZERO_WORKSPACE"
```

The script first verifies every pinned revision and all four patches before
changing any source tree.

## ROM layout

The default local layout is:

```text
fzero-workspace/
├── G-Diffuser/
├── PufferLib-5.0/
└── private/
    └── baserom.us.rev0.z64
```

The ROM remains outside both Git repositories. `FZEROX_ROM` can override its
location. Never commit or redistribute the ROM or extracted assets.

## Build and verify

Build the exact headless reference from `G-Diffuser`:

```bash
cmake -S "$FZERO_WORKSPACE/G-Diffuser" \
  -B "$FZERO_WORKSPACE/G-Diffuser/build/rl-headless" -G Ninja \
  -DGDX_RL_HEADLESS=ON -DGDX_EXPANSION_KIT=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGDX_RL_TEST_ROM="$FZERO_WORKSPACE/private/baserom.us.rev0.z64"
cmake --build "$FZERO_WORKSPACE/G-Diffuser/build/rl-headless" --parallel
ctest --test-dir "$FZERO_WORKSPACE/G-Diffuser/build/rl-headless" \
  --output-on-failure
```

Build the native PufferLib trainer locally:

```bash
cd "$FZERO_WORKSPACE/PufferLib-5.0"
UV_CACHE_DIR="$PWD/.uv-cache" uv venv .venv
UV_CACHE_DIR="$PWD/.uv-cache" uv pip install \
  --python .venv/bin/python nvidia-nccl-cu12
source .venv/bin/activate
CUDA_HOME=/usr/local/cuda ./build.sh fzero build/fzero-train
```

The applied file `PufferLib-5.0/ocean/fzero/README.md` contains the exact
training, evaluation, CUDA-build, trace-export, and validation commands, along
with the complete observation/reward contract and measured results.

## Scope and fidelity

The exact G-Diffuser CPU environment is the correctness oracle and the backend
used for the reported checkpoint. Policy inference and PPO optimization run on
the GPU. The reduced CUDA simulator has exact parity for the four appended
dash-pad observation values, but does not yet reproduce dash-contact boost
physics or its shaping reward. GPU-only returns should not be presented as
exact F-Zero policy performance until the remaining dynamics parity gates pass.

The graphical replay/capture path is currently validated on Linux/WSLg with an
OpenGL backbuffer. Other graphics backends are not yet claimed.

## Legal

This is an unofficial research project and is not affiliated with or endorsed
by Nintendo. See [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES/](LICENSES/)
for bundled upstream licenses, notices, and attribution.
