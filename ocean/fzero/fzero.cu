#ifndef PUFFER_FZERO_GPU_CU
#define PUFFER_FZERO_GPU_CU

#define PUF_BACKEND PUF_GPU

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <float.h>
#include <link.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "pufferenv.h"

#include "fzero_gpu_core.cuh"

#define OBS_SIZE GDX_RL_OBSERVATION_SIZE
#define NUM_ATNS GDX_RL_ACTION_SIZE
#define ACT_SIZES {11, 5, 2, 2, 2}

enum {
    FZERO_GPU_ACTION_MASK_SIZE = 22,
    FZERO_GPU_BLOCK_SIZE = 256,
};

struct Log {
    float perf;
    float score;
    float episode_return;
    float learner_return;
    float episode_length;
    float episode_frames;
    float race_distance;
    float terminal_speed;
    float terminal_energy;
    float terminal_lap;
    float finish_rate;
    float terminated_rate;
    float truncated_rate;
    float spinning_out_rate;
    float falling_rate;
    float crashed_rate;
    float retired_rate;
    float time_limit_rate;
    float stalled_rate;
    float steer_fraction[GDX_RL_STEER_BINS];
    float pitch_fraction[GDX_RL_PITCH_BINS];
    float throttle_on_fraction;
    float brake_on_fraction;
    float boost_on_fraction;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    FZeroGpuState state;
};

typedef struct FZeroGpuOracleApi {
    void* handle;
    uint32_t (*abi_version)(void);
    void (*default_config)(GdxRlConfig*);
    int (*init)(const char*, const GdxRlConfig*);
    int (*reset)(uint32_t, float*);
    int (*export_course)(uint32_t, GdxRlCourseSample*, GdxRlCourseMetadata*);
    int (*export_dash_pads)(uint32_t, GdxRlDashPad*, uint32_t*);
    int (*export_reference_state)(GdxRlReferenceState*);
    const char* (*last_error)(void);
} FZeroGpuOracleApi;

static struct {
    Env* envs;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    GdxRlCourseSample* course;
    GdxRlCourseSample* host_course;
    GdxRlDashPad host_dash_pads[GDX_RL_MAX_DASH_PADS];
    GdxRlCourseMetadata course_metadata;
    GdxRlReferenceState reference;
    FZeroGpuConfig config;
    FZeroGpuOracleApi oracle;
    cudaStream_t stream;
} g_fzero_gpu;

__constant__ FZeroGpuConfig d_fzero_gpu_config;
__constant__ GdxRlDashPad d_fzero_gpu_dash_pads[GDX_RL_MAX_DASH_PADS];

static void fzero_gpu_fatal(const char* operation, const char* detail) {
    fprintf(stderr, "fzero-gpu: %s: %s\n", operation,
        detail != NULL && detail[0] != '\0' ? detail : "unknown error");
    fflush(stderr);
    exit(EXIT_FAILURE);
}

static void fzero_gpu_cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fzero_gpu_fatal(operation, cudaGetErrorString(error));
    }
}

static void fzero_gpu_load_symbol(
        const char* name, void* destination, size_t destination_size) {
    dlerror();
    void* symbol = dlsym(g_fzero_gpu.oracle.handle, name);
    const char* error = dlerror();
    if (error != NULL || symbol == NULL) {
        fzero_gpu_fatal(name, error != NULL ? error : "symbol resolved to null");
    }
    if (destination_size != sizeof(symbol)) {
        fzero_gpu_fatal(name, "unsupported function pointer representation");
    }
    memcpy(destination, &symbol, sizeof(symbol));
}

static int32_t fzero_gpu_config_i32(Dict* kwargs, const char* key) {
    double value = dict_get(kwargs, key);
    if (!isfinite(value) || value != trunc(value) ||
            value < (double)INT32_MIN || value > (double)INT32_MAX) {
        fprintf(stderr, "fzero-gpu: [env].%s must be an int32, got %.17g\n", key, value);
        exit(EXIT_FAILURE);
    }
    return (int32_t)value;
}

static uint32_t fzero_gpu_config_u32(Dict* kwargs, const char* key) {
    double value = dict_get(kwargs, key);
    if (!isfinite(value) || value != trunc(value) ||
            value < 0.0 || value > (double)UINT32_MAX) {
        fprintf(stderr, "fzero-gpu: [env].%s must be a uint32, got %.17g\n", key, value);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)value;
}

static float fzero_gpu_config_positive(Dict* kwargs, const char* key) {
    double value = dict_get(kwargs, key);
    if (!isfinite(value) || value <= 0.0 || value > (double)FLT_MAX) {
        fprintf(stderr, "fzero-gpu: [env].%s must be finite and positive, got %.17g\n", key, value);
        exit(EXIT_FAILURE);
    }
    return (float)value;
}

static float fzero_gpu_config_nonnegative(Dict* kwargs, const char* key) {
    double value = dict_get(kwargs, key);
    if (!isfinite(value) || value < 0.0 || value > (double)FLT_MAX) {
        fprintf(stderr,
            "fzero-gpu: [env].%s must be finite and non-negative, got %.17g\n",
            key, value);
        exit(EXIT_FAILURE);
    }
    return (float)value;
}

static void fzero_gpu_load_oracle(Dict* kwargs) {
    const char* library_path = dict_get_str(kwargs, "core_lib_path");
    const char* rom_path = dict_get_str(kwargs, "rom_path");
    const char* rom_override = getenv("FZEROX_ROM");
    if (rom_override != NULL && rom_override[0] != '\0') {
        rom_path = rom_override;
    }

    g_fzero_gpu.oracle.handle =
        dlmopen(LM_ID_NEWLM, library_path, RTLD_NOW | RTLD_LOCAL);
    if (g_fzero_gpu.oracle.handle == NULL) {
        fzero_gpu_fatal("dlmopen", dlerror());
    }
    fzero_gpu_load_symbol("gdx_rl_abi_version", &g_fzero_gpu.oracle.abi_version,
        sizeof(g_fzero_gpu.oracle.abi_version));
    fzero_gpu_load_symbol("gdx_rl_default_config", &g_fzero_gpu.oracle.default_config,
        sizeof(g_fzero_gpu.oracle.default_config));
    fzero_gpu_load_symbol("gdx_rl_init", &g_fzero_gpu.oracle.init,
        sizeof(g_fzero_gpu.oracle.init));
    fzero_gpu_load_symbol("gdx_rl_reset", &g_fzero_gpu.oracle.reset,
        sizeof(g_fzero_gpu.oracle.reset));
    fzero_gpu_load_symbol("gdx_rl_export_course", &g_fzero_gpu.oracle.export_course,
        sizeof(g_fzero_gpu.oracle.export_course));
    fzero_gpu_load_symbol("gdx_rl_export_dash_pads",
        &g_fzero_gpu.oracle.export_dash_pads,
        sizeof(g_fzero_gpu.oracle.export_dash_pads));
    fzero_gpu_load_symbol("gdx_rl_export_reference_state",
        &g_fzero_gpu.oracle.export_reference_state,
        sizeof(g_fzero_gpu.oracle.export_reference_state));
    fzero_gpu_load_symbol("gdx_rl_last_error", &g_fzero_gpu.oracle.last_error,
        sizeof(g_fzero_gpu.oracle.last_error));

    if (g_fzero_gpu.oracle.abi_version() != GDX_RL_ABI_VERSION) {
        fzero_gpu_fatal("oracle ABI", "G-Diffuser and PufferLib headers do not match");
    }

    GdxRlConfig oracle_config;
    g_fzero_gpu.oracle.default_config(&oracle_config);
    oracle_config.course_index = fzero_gpu_config_i32(kwargs, "course_index");
    oracle_config.character = fzero_gpu_config_i32(kwargs, "character");
    oracle_config.machine_skin = fzero_gpu_config_i32(kwargs, "machine_skin");
    oracle_config.laps = fzero_gpu_config_i32(kwargs, "laps");
    oracle_config.action_repeat = fzero_gpu_config_i32(kwargs, "action_repeat");
    oracle_config.max_episode_frames = fzero_gpu_config_i32(kwargs, "max_episode_frames");
    oracle_config.stall_frames = fzero_gpu_config_i32(kwargs, "stall_frames");
    oracle_config.stall_progress = (float)dict_get(kwargs, "stall_progress");
    oracle_config.engine_balance = (float)dict_get(kwargs, "engine_balance");
    oracle_config.cpu_oracle = 0;
    if (g_fzero_gpu.oracle.init(rom_path, &oracle_config) != 0) {
        fzero_gpu_fatal("gdx_rl_init", g_fzero_gpu.oracle.last_error());
    }

    uint32_t seed = fzero_gpu_config_u32(kwargs, "seed");
    float initial_observation[GDX_RL_OBSERVATION_SIZE];
    if (g_fzero_gpu.oracle.reset(seed, initial_observation) != 0) {
        fzero_gpu_fatal("gdx_rl_reset", g_fzero_gpu.oracle.last_error());
    }

    int32_t sample_count = fzero_gpu_config_i32(kwargs, "course_samples");
    if (sample_count < 256 || sample_count > 65536) {
        fzero_gpu_fatal("configuration", "course_samples must be in [256, 65536]");
    }
    g_fzero_gpu.host_course = (GdxRlCourseSample*)calloc(
        (size_t)sample_count, sizeof(GdxRlCourseSample));
    if (g_fzero_gpu.host_course == NULL) {
        fzero_gpu_fatal("course allocation", "out of host memory");
    }
    if (g_fzero_gpu.oracle.export_course((uint32_t)sample_count,
            g_fzero_gpu.host_course, &g_fzero_gpu.course_metadata) != 0) {
        fzero_gpu_fatal("gdx_rl_export_course", g_fzero_gpu.oracle.last_error());
    }
    if (g_fzero_gpu.course_metadata.version != GDX_RL_COURSE_EXPORT_VERSION ||
            g_fzero_gpu.course_metadata.sample_count != (uint32_t)sample_count ||
            g_fzero_gpu.course_metadata.dash_pad_count > GDX_RL_MAX_DASH_PADS ||
            !(g_fzero_gpu.course_metadata.length > 0.0f) ||
            !(g_fzero_gpu.course_metadata.sample_spacing > 0.0f)) {
        fzero_gpu_fatal("gdx_rl_export_course", "invalid course export metadata");
    }
    uint32_t dash_pad_count = 0;
    if (g_fzero_gpu.oracle.export_dash_pads(GDX_RL_MAX_DASH_PADS,
            g_fzero_gpu.host_dash_pads, &dash_pad_count) != 0) {
        fzero_gpu_fatal("gdx_rl_export_dash_pads", g_fzero_gpu.oracle.last_error());
    }
    if (dash_pad_count != g_fzero_gpu.course_metadata.dash_pad_count ||
            dash_pad_count > GDX_RL_MAX_DASH_PADS) {
        fzero_gpu_fatal("gdx_rl_export_dash_pads",
            "dash-pad count disagrees with course metadata");
    }
    for (uint32_t i = 0; i < dash_pad_count; ++i) {
        const GdxRlDashPad* pad = &g_fzero_gpu.host_dash_pads[i];
        if (!isfinite(pad->entry_distance) || pad->entry_distance < 0.0f ||
                pad->entry_distance >= g_fzero_gpu.course_metadata.length ||
                !isfinite(pad->target) || pad->target < -1.0f || pad->target > 1.0f ||
                !isfinite(pad->width) || pad->width < 0.0f || pad->width > 1.0f ||
                pad->segment_index > (uint32_t)INT32_MAX ||
                pad->segment_index >= g_fzero_gpu.course_metadata.segment_count) {
            fzero_gpu_fatal("gdx_rl_export_dash_pads", "invalid dash-pad descriptor");
        }
    }
    if (g_fzero_gpu.oracle.export_reference_state(&g_fzero_gpu.reference) != 0) {
        fzero_gpu_fatal("gdx_rl_export_reference_state", g_fzero_gpu.oracle.last_error());
    }

    FZeroGpuConfig* config = &g_fzero_gpu.config;
    memset(config, 0, sizeof(*config));
    config->sample_count = sample_count;
    config->dash_pad_count = (int32_t)dash_pad_count;
    config->laps = oracle_config.laps;
    config->action_repeat = oracle_config.action_repeat;
    config->max_episode_frames = oracle_config.max_episode_frames;
    config->stall_frames = oracle_config.stall_frames;
    config->steering_curriculum = fzero_gpu_config_i32(kwargs, "steering_curriculum");
    if (config->steering_curriculum != 0 && config->steering_curriculum != 1) {
        fzero_gpu_fatal("configuration", "steering_curriculum must be 0 or 1");
    }
    config->course_length = g_fzero_gpu.course_metadata.length;
    config->sample_spacing = g_fzero_gpu.course_metadata.sample_spacing;
    config->stall_progress = oracle_config.stall_progress;
    config->reward_scale = fzero_gpu_config_positive(kwargs, "reward_scale");
    if (config->reward_scale > 0.03125f) {
        fzero_gpu_fatal("configuration", "reward_scale must be at most 1/32");
    }
    config->finish_time_target_frames =
        fzero_gpu_config_nonnegative(kwargs, "finish_time_target_frames");
    config->engine_target_speed =
        fzero_gpu_config_positive(kwargs, "gpu_engine_target_speed");
    config->engine_acceleration =
        fzero_gpu_config_positive(kwargs, "gpu_engine_acceleration");
    config->forward_drag = fzero_gpu_config_positive(kwargs, "gpu_forward_drag");
    config->brake_drag = fzero_gpu_config_positive(kwargs, "gpu_brake_drag");
    config->steering_rate = fzero_gpu_config_positive(kwargs, "gpu_steering_rate");
    config->wall_margin = fzero_gpu_config_positive(kwargs, "gpu_wall_margin");
    if (config->forward_drag > 1.0f || config->brake_drag > 1.0f) {
        fzero_gpu_fatal("configuration", "GPU drag factors must be in (0, 1]");
    }
    config->initial_race_distance = g_fzero_gpu.reference.race_distance;
    config->initial_lap_distance = g_fzero_gpu.reference.lap_distance;
    config->initial_lateral = g_fzero_gpu.reference.lateral;
    config->initial_vertical = g_fzero_gpu.reference.vertical;
    config->initial_speed = g_fzero_gpu.reference.speed;
    config->initial_energy = g_fzero_gpu.reference.energy;
    config->max_energy = g_fzero_gpu.reference.max_energy;
    config->seed = seed;

    fprintf(stderr,
        "fzero-gpu: reduced simulator v0, course=%d samples=%d dash_pads=%u length=%.3f "
        "initial_distance=%.3f\n",
        oracle_config.course_index, sample_count, dash_pad_count, config->course_length,
        config->initial_race_distance);
}

struct FZeroGpuObservationStore {
    obs_t* values;

    __device__ __forceinline__ void operator()(int index, float value) const {
        values[index] = value;
    }
};

__device__ static void fzero_gpu_add_log(
        Env* env, const FZeroGpuStepResult* result) {
    const FZeroGpuState* state = &env->state;
    const float inv_decisions = state->episode_decisions > 0
        ? 1.0f / (float)state->episode_decisions : 0.0f;
    const float terminal_outcome = result->learner_reward;
    env->log.perf += terminal_outcome;
    env->log.score += terminal_outcome;
    env->log.episode_return += state->episode_return;
    env->log.learner_return += state->learner_return;
    env->log.episode_length += (float)state->episode_decisions;
    env->log.episode_frames += (float)state->episode_frame;
    env->log.race_distance += state->race_distance;
    env->log.terminal_speed += state->speed;
    env->log.terminal_energy += d_fzero_gpu_config.max_energy > 0.0f
        ? state->energy / d_fzero_gpu_config.max_energy : 0.0f;
    env->log.terminal_lap += (float)state->laps_completed;
    env->log.finish_rate += result->reason == FZERO_GPU_FINISHED ? 1.0f : 0.0f;
    env->log.terminated_rate += result->terminated ? 1.0f : 0.0f;
    env->log.truncated_rate += result->truncated ? 1.0f : 0.0f;
    env->log.spinning_out_rate += result->reason == FZERO_GPU_SPINNING_OUT ? 1.0f : 0.0f;
    env->log.falling_rate += result->reason == FZERO_GPU_FALLING ? 1.0f : 0.0f;
    env->log.crashed_rate += result->reason == FZERO_GPU_CRASHED ? 1.0f : 0.0f;
    env->log.retired_rate += result->reason == FZERO_GPU_RETIRED ? 1.0f : 0.0f;
    env->log.time_limit_rate += result->reason == FZERO_GPU_TIME_LIMIT ? 1.0f : 0.0f;
    env->log.stalled_rate += result->reason == FZERO_GPU_STALLED ? 1.0f : 0.0f;
    for (int i = 0; i < GDX_RL_STEER_BINS; ++i) {
        env->log.steer_fraction[i] += (float)state->steer_counts[i] * inv_decisions;
    }
    for (int i = 0; i < GDX_RL_PITCH_BINS; ++i) {
        env->log.pitch_fraction[i] += (float)state->pitch_counts[i] * inv_decisions;
    }
    env->log.throttle_on_fraction += (float)state->throttle_on_count * inv_decisions;
    env->log.brake_on_fraction += (float)state->brake_on_count * inv_decisions;
    env->log.boost_on_fraction += (float)state->boost_on_count * inv_decisions;
    env->log.n += 1.0f;
}

__global__ static void fzero_gpu_reset_kernel(Env* envs, int n,
        const GdxRlCourseSample* course, obs_t* observations,
        float* rewards, float* terminals) {
    int env_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_index >= n) {
        return;
    }
    Env* env = &envs[env_index];
    fzero_gpu_reset_state(&env->state, &d_fzero_gpu_config, (uint32_t)env_index);
    FZeroGpuObservationStore store = {
        observations + (size_t)env_index * GDX_RL_OBSERVATION_SIZE,
    };
    fzero_gpu_observe(&env->state, &d_fzero_gpu_config, course,
        d_fzero_gpu_dash_pads, store);
    rewards[env_index] = 0.0f;
    terminals[env_index] = 0.0f;
}

__global__ static void fzero_gpu_step_kernel(Env* envs, int n,
        const GdxRlCourseSample* course, obs_t* observations,
        const float* actions, float* rewards, float* terminals) {
    int env_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_index >= n) {
        return;
    }
    Env* env = &envs[env_index];
    int32_t action[GDX_RL_ACTION_SIZE];
    const int32_t action_sizes[GDX_RL_ACTION_SIZE] = {11, 5, 2, 2, 2};
    for (int head = 0; head < GDX_RL_ACTION_SIZE; ++head) {
        float raw = actions[(size_t)env_index * GDX_RL_ACTION_SIZE + head];
        int32_t category = (int32_t)raw;
        action[head] = isfinite(raw) && raw == (float)category &&
            category >= 0 && category < action_sizes[head] ? category : 0;
    }

    FZeroGpuStepResult result =
        fzero_gpu_step_state(&env->state, &d_fzero_gpu_config, course, action);
    float learner_reward = result.learner_reward;
    const bool done = result.terminated || result.truncated;
    if (done) {
        fzero_gpu_add_log(env, &result);
        fzero_gpu_reset_state(&env->state, &d_fzero_gpu_config, (uint32_t)env_index);
    }
    FZeroGpuObservationStore store = {
        observations + (size_t)env_index * GDX_RL_OBSERVATION_SIZE,
    };
    fzero_gpu_observe(&env->state, &d_fzero_gpu_config, course,
        d_fzero_gpu_dash_pads, store);
    rewards[env_index] = learner_reward;
    terminals[env_index] = done ? 1.0f : 0.0f;
}

Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals,
        unsigned char* action_mask) {
    if (n <= 0) {
        fzero_gpu_fatal("puf_vec_create", "vector size must be positive");
    }
    fzero_gpu_load_oracle(env_kwargs);
    g_fzero_gpu.n = n;
    g_fzero_gpu.observations = observations;
    g_fzero_gpu.actions = actions;
    g_fzero_gpu.rewards = rewards;
    g_fzero_gpu.terminals = terminals;
    g_fzero_gpu.action_mask = action_mask;
    g_fzero_gpu.stream = 0;

    fzero_gpu_cuda_check(cudaMalloc((void**)&g_fzero_gpu.envs,
        (size_t)n * sizeof(Env)), "cudaMalloc environments");
    fzero_gpu_cuda_check(cudaMemset(g_fzero_gpu.envs, 0,
        (size_t)n * sizeof(Env)), "cudaMemset environments");
    fzero_gpu_cuda_check(cudaMalloc((void**)&g_fzero_gpu.course,
        (size_t)g_fzero_gpu.config.sample_count * sizeof(GdxRlCourseSample)),
        "cudaMalloc course");
    fzero_gpu_cuda_check(cudaMemcpy(g_fzero_gpu.course, g_fzero_gpu.host_course,
        (size_t)g_fzero_gpu.config.sample_count * sizeof(GdxRlCourseSample),
        cudaMemcpyHostToDevice), "cudaMemcpy course");
    fzero_gpu_cuda_check(cudaMemcpyToSymbol(d_fzero_gpu_config,
        &g_fzero_gpu.config, sizeof(g_fzero_gpu.config)), "cudaMemcpyToSymbol config");
    fzero_gpu_cuda_check(cudaMemcpyToSymbol(d_fzero_gpu_dash_pads,
        g_fzero_gpu.host_dash_pads, sizeof(g_fzero_gpu.host_dash_pads)),
        "cudaMemcpyToSymbol dash pads");

    size_t mask_bytes = (size_t)n * FZERO_GPU_ACTION_MASK_SIZE;
    unsigned char* host_mask = (unsigned char*)malloc(mask_bytes);
    if (host_mask == NULL) {
        fzero_gpu_fatal("action mask allocation", "out of host memory");
    }
    memset(host_mask, g_fzero_gpu.config.steering_curriculum ? 0 : 1, mask_bytes);
    if (g_fzero_gpu.config.steering_curriculum) {
        for (int env_index = 0; env_index < n; ++env_index) {
            unsigned char* mask = host_mask + (size_t)env_index * FZERO_GPU_ACTION_MASK_SIZE;
            memset(mask, 1, GDX_RL_STEER_BINS);
            mask[11] = 1;
            mask[17] = 1;
            mask[18] = 1;
            mask[20] = 1;
        }
    }
    fzero_gpu_cuda_check(cudaMemcpy(action_mask, host_mask, mask_bytes,
        cudaMemcpyHostToDevice), "cudaMemcpy action masks");
    free(host_mask);
    return g_fzero_gpu.envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_fzero_gpu.stream = stream;
}

void puf_init(Env*, Dict*) {
}

void puf_reset(Env*) {
    int blocks = (g_fzero_gpu.n + FZERO_GPU_BLOCK_SIZE - 1) / FZERO_GPU_BLOCK_SIZE;
    fzero_gpu_reset_kernel<<<blocks, FZERO_GPU_BLOCK_SIZE, 0, g_fzero_gpu.stream>>>(
        g_fzero_gpu.envs, g_fzero_gpu.n, g_fzero_gpu.course,
        g_fzero_gpu.observations, g_fzero_gpu.rewards, g_fzero_gpu.terminals);
    fzero_gpu_cuda_check(cudaGetLastError(), "launch reset kernel");
}

void puf_step(Env*) {
    int blocks = (g_fzero_gpu.n + FZERO_GPU_BLOCK_SIZE - 1) / FZERO_GPU_BLOCK_SIZE;
    fzero_gpu_step_kernel<<<blocks, FZERO_GPU_BLOCK_SIZE, 0, g_fzero_gpu.stream>>>(
        g_fzero_gpu.envs, g_fzero_gpu.n, g_fzero_gpu.course,
        g_fzero_gpu.observations, g_fzero_gpu.actions,
        g_fzero_gpu.rewards, g_fzero_gpu.terminals);
    fzero_gpu_cuda_check(cudaGetLastError(), "launch step kernel");
}

void puf_render(Env*) {
    if (g_fzero_gpu.stream != 0) {
        fzero_gpu_cuda_check(cudaStreamSynchronize(g_fzero_gpu.stream),
            "synchronize render state");
    }
    Env host_env = {};
    fzero_gpu_cuda_check(cudaMemcpy(&host_env, g_fzero_gpu.envs,
        sizeof(host_env), cudaMemcpyDeviceToHost), "copy render state");
    if (!IsWindowReady()) {
        InitWindow(900, 300, "PufferLib F-Zero X CUDA telemetry");
    }
    BeginDrawing();
    ClearBackground((Color){12, 18, 28, 255});
    DrawText("F-Zero X reduced simulator is stepping on CUDA", 24, 24, 28, RAYWHITE);
    DrawText(TextFormat("Decision: %u   Frame: %u   Batch: %d",
        host_env.state.episode_decisions, host_env.state.episode_frame,
        g_fzero_gpu.n), 24, 78, 22, LIGHTGRAY);
    DrawText(TextFormat("Distance: %.1f   Speed: %.1f   Lateral: %.1f   Energy: %.1f",
        host_env.state.race_distance, host_env.state.speed,
        host_env.state.lateral, host_env.state.energy), 24, 118, 22, LIGHTGRAY);
    DrawText("CPU G-Diffuser remains the policy-transfer oracle.", 24, 174, 20, GRAY);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Env*) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    cudaFree(g_fzero_gpu.course);
    cudaFree(g_fzero_gpu.envs);
    g_fzero_gpu.course = NULL;
    g_fzero_gpu.envs = NULL;
    free(g_fzero_gpu.host_course);
    g_fzero_gpu.host_course = NULL;
    /* The oracle DSO owns live ucontexts and intentionally remains loaded for process life. */
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "learner_return", log->learner_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "episode_frames", log->episode_frames);
    dict_set(out, "race_distance", log->race_distance);
    dict_set(out, "terminal_speed", log->terminal_speed);
    dict_set(out, "terminal_energy", log->terminal_energy);
    dict_set(out, "terminal_lap", log->terminal_lap);
    dict_set(out, "finish_rate", log->finish_rate);
    dict_set(out, "terminated_rate", log->terminated_rate);
    dict_set(out, "truncated_rate", log->truncated_rate);
    dict_set(out, "spinning_out_rate", log->spinning_out_rate);
    dict_set(out, "falling_rate", log->falling_rate);
    dict_set(out, "crashed_rate", log->crashed_rate);
    dict_set(out, "retired_rate", log->retired_rate);
    dict_set(out, "time_limit_rate", log->time_limit_rate);
    dict_set(out, "stalled_rate", log->stalled_rate);
    for (int category = 0; category < GDX_RL_STEER_BINS; ++category) {
        char key[32];
        snprintf(key, sizeof(key), "steer_%02d_fraction", category);
        dict_set(out, key, log->steer_fraction[category]);
    }
    for (int category = 0; category < GDX_RL_PITCH_BINS; ++category) {
        char key[32];
        snprintf(key, sizeof(key), "pitch_%02d_fraction", category);
        dict_set(out, key, log->pitch_fraction[category]);
    }
    dict_set(out, "throttle_on_fraction", log->throttle_on_fraction);
    dict_set(out, "brake_on_fraction", log->brake_on_fraction);
    dict_set(out, "boost_on_fraction", log->boost_on_fraction);
    dict_set(out, "n", log->n);
}

#endif
