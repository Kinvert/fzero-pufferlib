#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

test_dir=build/fzero-tests
mkdir -p "$test_dir"

cc -std=c11 -O2 -Wall -Wextra -fPIC -shared -pthread \
    -I../G-Diffuser/port/rl \
    ocean/fzero/tests/fake_gdiffuser_rl.c \
    -o "$test_dir/libfake_gdiffuser_rl.so"

cc -std=c11 -O2 -Wall -Wextra -D_GNU_SOURCE \
    -I. -Isrc -Iocean/fzero -Ivendor \
    -Iraylib-5.5_linux_amd64/include \
    ocean/fzero/tests/test_fzero_adapter.c \
    raylib-5.5_linux_amd64/lib/libraylib.a \
    -ldl -lGL -lm -lpthread \
    -o "$test_dir/test_fzero_adapter"

"$test_dir/test_fzero_adapter" \
    "$test_dir/libfake_gdiffuser_rl.so" \
    "$test_dir/fzero-render-eval-input.txt"

if [[ "${FZERO_TEST_REAL_CORE:-0}" == 1 ]]; then
    c++ -std=c++17 -O2 -Wall -Wextra \
        -I. -Isrc -Iocean/fzero -Ivendor \
        ocean/fzero/tests/test_fzero_gpu_core.cpp \
        ../G-Diffuser/build/rl-headless/port/libgdiffuser_rl.so \
        -Wl,-rpath,"$PWD/../G-Diffuser/build/rl-headless/port" \
        -ldl -lm -lpthread \
        -o "$test_dir/test_fzero_gpu_core"
    "$test_dir/test_fzero_gpu_core" \
        "${FZEROX_ROM:-../private/baserom.us.rev0.z64}"

    ./build.sh fzero "$test_dir/fzero-cpu" --cpu
    # Explicitly disable the forced-throttle curriculum: this is the true
    # zero-action baseline and intentionally runs without a policy checkpoint.
    "$test_dir/fzero-cpu" --headless --base.eval_episodes=1 \
        --env.steering_curriculum=0
fi
