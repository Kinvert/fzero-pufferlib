#pragma once

#include <dlfcn.h>
#include <link.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../G-Diffuser/port/rl/gdx_rl_env.h"

typedef float obs_t;
#include "pufferenv.h"

#define OBS_SIZE GDX_RL_OBSERVATION_SIZE
#define NUM_ATNS GDX_RL_ACTION_SIZE
#define ACT_SIZES {11, 5, 2, 2, 2}

enum {
    FZERO_STEER_MASK_OFFSET = 0,
    FZERO_PITCH_MASK_OFFSET = 11,
    FZERO_THROTTLE_MASK_OFFSET = 16,
    FZERO_BRAKE_MASK_OFFSET = 18,
    FZERO_BOOST_MASK_OFFSET = 20,
    FZERO_ACTION_MASK_SIZE = 22,
};

static const int32_t FZERO_ACTION_NVECS[GDX_RL_ACTION_SIZE] = {11, 5, 2, 2, 2};
static const int8_t FZERO_STEER_VALUES[GDX_RL_STEER_BINS] = {
    0, -13, 13, -25, 25, -38, 38, -50, 50, -63, 63,
};
static const int8_t FZERO_PITCH_VALUES[GDX_RL_PITCH_BINS] = {0, -32, 32, -63, 63};

/* Standalone CPU evaluation can export the exact post-curriculum action stream
 * for deterministic playback by G-Diffuser's GDX_INPUT_SCRIPT input seam. */
#define PUF_SUPPORTS_GDX_INPUT_SCRIPT 1

typedef struct FZeroGdxInputTrace {
    FILE* file;
    char* final_path;
    char* temporary_path;
    uint32_t decisions;
    uint32_t episode_frame;
    uint32_t reset_seed;
    uint32_t sampler_seed;
    uint8_t complete;
} FZeroGdxInputTrace;

typedef struct FZeroApi {
    void* handle;
    uint32_t (*abi_version)(void);
    void (*default_config)(GdxRlConfig*);
    int (*init)(const char*, const GdxRlConfig*);
    int (*bind_thread)(void);
    int (*reset)(uint32_t, float*);
    int (*step)(const int32_t*, float*, GdxRlTransition*);
    uint64_t (*semantic_hash)(void);
    const char* (*last_error)(void);
    const int32_t* action_nvec;
} FZeroApi;

struct Log {
    float perf;
    float score;
    float episode_return;
    float learner_return;
    float episode_length;
    float episode_frames;
    float finish_frames;
    float dash_hits;
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
    int tag;
    int boundary_reached;

    int num_agents;
    unsigned int rng;
    uint32_t seed_base;
    uint32_t reset_count;
    uint32_t last_reset_seed;
    float episode_return;
    float learner_return;
    float reward_scale;
    float finish_time_target_frames;
    float dash_hit_reward;
    uint8_t steering_curriculum;
    int32_t action_repeat;
    uint32_t episode_decisions;
    uint32_t episode_dash_hits;
    uint32_t steer_counts[GDX_RL_STEER_BINS];
    uint32_t pitch_counts[GDX_RL_PITCH_BINS];
    uint32_t throttle_on_count;
    uint32_t brake_on_count;
    uint32_t boost_on_count;
    GdxRlTransition last_transition;
    FZeroGdxInputTrace* gdx_input_trace;

    FZeroApi api;
};
typedef Env FZero;

static inline void fzero_fatal(const char* operation, const char* detail) {
    fprintf(stderr, "fzero: %s: %s\n", operation,
        detail != NULL && detail[0] != '\0' ? detail : "unknown error");
    fflush(stderr);
    exit(EXIT_FAILURE);
}

static inline void fzero_core_check(FZero* env, const char* operation, int result) {
    if (result == 0) {
        return;
    }
    const char* detail = env->api.last_error != NULL
        ? env->api.last_error() : "core call failed without gdx_rl_last_error";
    fzero_fatal(operation, detail);
}

/* POSIX specifies dlsym for function symbols, but ISO C does not specify a
 * void* -> function-pointer cast. Copying the representation keeps this header
 * valid in both clang C (--cpu) and nvcc C++ (native trainer) builds. */
static inline void fzero_load_symbol(
        FZero* env, const char* name, void* destination, size_t destination_size) {
    dlerror();
    void* symbol = dlsym(env->api.handle, name);
    const char* error = dlerror();
    if (error != NULL || symbol == NULL) {
        fzero_fatal(name, error != NULL ? error : "symbol resolved to null");
    }
    if (destination_size != sizeof(symbol)) {
        fzero_fatal(name, "unsupported function/data pointer representation");
    }
    memcpy(destination, &symbol, sizeof(symbol));
}

static inline int32_t fzero_config_i32(Dict* kwargs, const char* key) {
    double value = dict_get(kwargs, key);
    if (!isfinite(value) || value != trunc(value)
            || value < (double)INT32_MIN || value > (double)INT32_MAX) {
        fprintf(stderr, "fzero: [env].%s must be an int32, got %.17g\n", key, value);
        exit(EXIT_FAILURE);
    }
    return (int32_t)value;
}

static inline uint32_t fzero_config_u32(Dict* kwargs, const char* key) {
    double value = dict_get(kwargs, key);
    if (!isfinite(value) || value != trunc(value)
            || value < 0.0 || value > (double)UINT32_MAX) {
        fprintf(stderr, "fzero: [env].%s must be a uint32, got %.17g\n", key, value);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)value;
}

/* Returns zero on success, otherwise one plus the invalid component index. */
static inline int fzero_decode_actions(
        const float raw[GDX_RL_ACTION_SIZE],
        int32_t decoded[GDX_RL_ACTION_SIZE]) {
    if (raw == NULL || decoded == NULL) {
        return 1;
    }
    for (int i = 0; i < GDX_RL_ACTION_SIZE; ++i) {
        float value = raw[i];
        if (!isfinite(value) || value != truncf(value)
                || value < 0.0f || value >= (float)FZERO_ACTION_NVECS[i]) {
            return i + 1;
        }
        decoded[i] = (int32_t)value;
    }
    return 0;
}

static inline int fzero_scale_reward(float raw_reward, float reward_scale, float* scaled_reward) {
    if (scaled_reward == NULL || !isfinite(raw_reward) || !isfinite(reward_scale)) {
        return -1;
    }
    float value = raw_reward * reward_scale;
    if (!isfinite(value) || value < -1.0f || value > 1.0f) {
        return -1;
    }
    *scaled_reward = value;
    return 0;
}

/* Normalized outcome for the one-lap pace task. A zero target is the
 * compatibility setting: every completed lap scores 1. Fatal outcomes score
 * -1, while time-limit and stall truncations score 0. This outcome replaces
 * the entire learner reward on the boundary transition; dense raw shaping is
 * used only on nonterminal transitions. */
static inline float fzero_completed_lap_score(
        float target_frames, uint32_t episode_frame) {
    if (!(target_frames > 0.0f)) {
        return 1.0f;
    }
    if (episode_frame == 0) {
        return 0.0f;
    }
    float score = target_frames / (float)episode_frame;
    if (score < 0.0f) {
        return 0.0f;
    }
    return score > 1.0f ? 1.0f : score;
}

static inline int fzero_terminal_outcome(const GdxRlTransition* transition,
        float target_frames, float* outcome) {
    if (transition == NULL || outcome == NULL) {
        return 0;
    }
    switch (transition->reason) {
        case GDX_RL_FINISHED:
            *outcome = fzero_completed_lap_score(
                target_frames, transition->episode_frame);
            return 1;
        case GDX_RL_SPINNING_OUT:
        case GDX_RL_FALLING:
        case GDX_RL_CRASHED:
        case GDX_RL_RETIRED:
            *outcome = -1.0f;
            return 1;
        case GDX_RL_TIME_LIMIT:
        case GDX_RL_STALLED:
            *outcome = 0.0f;
            return 1;
        default:
            return 0;
    }
}

static inline int fzero_compute_learner_reward(float raw_reward,
        const GdxRlTransition* transition, float reward_scale,
        float target_frames, float* learner_reward) {
    if (learner_reward == NULL || !isfinite(raw_reward)) {
        return -1;
    }
    if (fzero_terminal_outcome(transition, target_frames, learner_reward)) {
        return 0;
    }
    return fzero_scale_reward(raw_reward, reward_scale, learner_reward);
}

/* The core counts dash-surface acquisition edges inside the raw physics-tick
 * loop, so action repeat can neither miss a short pad nor pay every frame spent
 * on one. Zero leaves the exact core reward unchanged. */
static inline float fzero_shape_dash_reward(float raw_reward,
        const GdxRlTransition* transition, float reward_per_hit) {
    if (transition == NULL || !(reward_per_hit > 0.0f)) {
        return raw_reward;
    }
    return raw_reward + reward_per_hit * (float)transition->dash_pad_hits;
}

static inline void fzero_write_action_mask(FZero* env) {
    unsigned char* mask = env->agents[0].action_mask;
    if (mask == NULL) {
        return;
    }
    memset(mask, 1, FZERO_ACTION_MASK_SIZE);
    if (!env->steering_curriculum) {
        return;
    }

    memset(mask, 0, FZERO_ACTION_MASK_SIZE);
    memset(mask + FZERO_STEER_MASK_OFFSET, 1, GDX_RL_STEER_BINS);
    mask[FZERO_PITCH_MASK_OFFSET] = 1;
    mask[FZERO_THROTTLE_MASK_OFFSET + 1] = 1;
    mask[FZERO_BRAKE_MASK_OFFSET] = 1;
    mask[FZERO_BOOST_MASK_OFFSET] = 1;
}

static inline int fzero_action_is_legal(const FZero* env,
        const int32_t action[GDX_RL_ACTION_SIZE]) {
    const unsigned char* mask = env->agents[0].action_mask;
    if (mask == NULL) {
        return 1;
    }
    const int offsets[GDX_RL_ACTION_SIZE] = {
        FZERO_STEER_MASK_OFFSET,
        FZERO_PITCH_MASK_OFFSET,
        FZERO_THROTTLE_MASK_OFFSET,
        FZERO_BRAKE_MASK_OFFSET,
        FZERO_BOOST_MASK_OFFSET,
    };
    for (int head = 0; head < GDX_RL_ACTION_SIZE; ++head) {
        if (mask[offsets[head] + action[head]] == 0) {
            return 0;
        }
    }
    return 1;
}

static inline void fzero_apply_curriculum(const FZero* env,
        int32_t action[GDX_RL_ACTION_SIZE]) {
    if (!env->steering_curriculum) {
        return;
    }
    action[1] = 0;
    action[2] = 1;
    action[3] = 0;
    action[4] = 0;
}

static inline const char* fzero_gdx_buttons(
        int throttle, int brake, int boost, char output[32]) {
    size_t used = 0;
    output[0] = '\0';
    const char* names[3] = {"A", "CDOWN", "B"};
    const int enabled[3] = {throttle, brake, boost};
    for (int i = 0; i < 3; ++i) {
        if (!enabled[i]) {
            continue;
        }
        if (used != 0) {
            output[used++] = '+';
        }
        size_t length = strlen(names[i]);
        memcpy(output + used, names[i], length);
        used += length;
        output[used] = '\0';
    }
    if (used == 0) {
        memcpy(output, "-", 2);
    }
    return output;
}

/* Write one factorized action as the raw N64 pad polls consumed by the exact
 * headless step. Boost is an edge-triggered B tap on repeat zero, followed by
 * the same pad without B for the remaining repeats. */
static inline int fzero_gdx_write_action(FILE* file, uint32_t decision,
        int32_t repeat_ticks, const int32_t action[GDX_RL_ACTION_SIZE]) {
    if (file == NULL || action == NULL || repeat_ticks < 1 || repeat_ticks > 16) {
        return -1;
    }
    for (int i = 0; i < GDX_RL_ACTION_SIZE; ++i) {
        if (action[i] < 0 || action[i] >= FZERO_ACTION_NVECS[i]) {
            return -1;
        }
    }

    const int steer = FZERO_STEER_VALUES[action[0]];
    const int pitch = FZERO_PITCH_VALUES[action[1]];
    char buttons[32];
    if (fprintf(file, "# decision %u action %d %d %d %d %d\n",
            decision, action[0], action[1], action[2], action[3], action[4]) < 0) {
        return -1;
    }
    if (action[4] != 0) {
        if (fprintf(file, "INPUT %s %d %d 1\n",
                fzero_gdx_buttons(action[2], action[3], 1, buttons), steer, pitch) < 0) {
            return -1;
        }
        if (repeat_ticks > 1 && fprintf(file, "INPUT %s %d %d %d\n",
                fzero_gdx_buttons(action[2], action[3], 0, buttons), steer, pitch,
                repeat_ticks - 1) < 0) {
            return -1;
        }
    } else if (fprintf(file, "INPUT %s %d %d %d\n",
            fzero_gdx_buttons(action[2], action[3], 0, buttons), steer, pitch,
            repeat_ticks) < 0) {
        return -1;
    }
    return ferror(file) ? -1 : 0;
}

static inline void fzero_gdx_input_trace_open(FZero* env, const char* output_path,
        const char* checkpoint_path, uint32_t sampler_seed) {
    if (env == NULL || output_path == NULL || output_path[0] == '\0' ||
            env->gdx_input_trace != NULL) {
        fzero_fatal("GDX input trace", "invalid or duplicate trace request");
    }
    size_t path_size = strlen(output_path) + 32;
    FZeroGdxInputTrace* trace =
        (FZeroGdxInputTrace*)calloc(1, sizeof(FZeroGdxInputTrace));
    if (trace == NULL) {
        fzero_fatal("GDX input trace", "failed to allocate trace state");
    }
    trace->final_path = (char*)malloc(path_size);
    trace->temporary_path = (char*)malloc(path_size);
    if (trace->final_path == NULL || trace->temporary_path == NULL) {
        fzero_fatal("GDX input trace", "failed to allocate trace path");
    }
    snprintf(trace->final_path, path_size, "%s", output_path);
    snprintf(trace->temporary_path, path_size, "%s.tmp", output_path);
    trace->file = fopen(trace->temporary_path, "wb");
    if (trace->file == NULL) {
        fzero_fatal("GDX input trace", "failed to open temporary output");
    }
    trace->reset_seed = env->last_reset_seed;
    trace->sampler_seed = sampler_seed;
    env->gdx_input_trace = trace;

    fprintf(trace->file,
        "# G-Diffuser deterministic Puffer checkpoint replay\n"
        "# Generated from the exact headless F-Zero X CPU environment.\n"
        "# checkpoint %s\n"
        "# environment_seed %u\n"
        "# sampler_seed %u\n"
        "# action_repeat %d\n"
        "# action_nvec 11 5 2 2 2\n"
        "# steer_values 0 -13 13 -25 25 -38 38 -50 50 -63 63\n"
        "# pitch_values 0 -32 32 -63 63\n"
        "LOG Puffer checkpoint replay started\n",
        checkpoint_path != NULL ? checkpoint_path : "-", trace->reset_seed,
        trace->sampler_seed, env->action_repeat);
    if (ferror(trace->file)) {
        fzero_fatal("GDX input trace", "failed to write trace header");
    }
}

static inline void fzero_gdx_input_trace_record(FZero* env,
        const int32_t action[GDX_RL_ACTION_SIZE],
        const GdxRlTransition* transition) {
    FZeroGdxInputTrace* trace = env != NULL ? env->gdx_input_trace : NULL;
    if (trace == NULL || trace->complete) {
        return;
    }
    if (transition == NULL || transition->episode_frame <= trace->episode_frame ||
            transition->episode_frame - trace->episode_frame > (uint32_t)env->action_repeat) {
        fzero_fatal("GDX input trace", "transition has an invalid physics-tick delta");
    }
    uint32_t repeat_ticks = transition->episode_frame - trace->episode_frame;
    if (fzero_gdx_write_action(trace->file, trace->decisions,
            (int32_t)repeat_ticks, action) != 0) {
        fzero_fatal("GDX input trace", "failed to write action");
    }
    trace->episode_frame = transition->episode_frame;
    trace->decisions++;
}

static inline void fzero_gdx_input_trace_finish(FZero* env,
        const GdxRlTransition* transition) {
    FZeroGdxInputTrace* trace = env != NULL ? env->gdx_input_trace : NULL;
    if (trace == NULL || trace->complete || transition == NULL) {
        return;
    }
    const int finished = transition->reason == GDX_RL_FINISHED;
    fprintf(trace->file,
        "# result decisions %u raw_score %.9g reason %u terminated %u truncated %u "
        "episode_frame %u race_distance %.9g semantic_hash %016llx\n"
        "LOG Puffer policy trace exhausted; waiting for rendered terminal\n"
        "WAIT 120\n"
        "SHOT %s\n"
        "WAIT 2\n"
        "QUIT\n",
        trace->decisions, env->episode_return, (unsigned)transition->reason,
        (unsigned)transition->terminated, (unsigned)transition->truncated,
        transition->episode_frame, transition->race_distance,
        (unsigned long long)transition->semantic_hash,
        finished ? "puffer-fzero-finish" : "puffer-fzero-terminal");
    if (ferror(trace->file) || fclose(trace->file) != 0) {
        trace->file = NULL;
        fzero_fatal("GDX input trace", "failed to finalize trace output");
    }
    trace->file = NULL;
    if (rename(trace->temporary_path, trace->final_path) != 0) {
        fzero_fatal("GDX input trace", "failed to publish trace output");
    }
    trace->complete = 1;
}

static inline int fzero_gdx_input_trace_complete(const FZero* env) {
    return env != NULL && env->gdx_input_trace != NULL &&
        env->gdx_input_trace->complete != 0;
}

static inline void fzero_bind_current_thread(FZero* env) {
    /* Reaffirm the core's ucontext TLS at every quiescent API boundary. pthread ids can be reused
     * after a worker exits, so equality with a cached id is not proof that this thread is bound. */
    fzero_core_check(env, "gdx_rl_bind_thread", env->api.bind_thread());
}

static inline void fzero_load_api(FZero* env, const char* library_path) {
    env->api.handle = dlmopen(LM_ID_NEWLM, library_path, RTLD_NOW | RTLD_LOCAL);
    if (env->api.handle == NULL) {
        fzero_fatal("dlmopen", dlerror());
    }

    fzero_load_symbol(env, "gdx_rl_abi_version",
        &env->api.abi_version, sizeof(env->api.abi_version));
    fzero_load_symbol(env, "gdx_rl_default_config",
        &env->api.default_config, sizeof(env->api.default_config));
    fzero_load_symbol(env, "gdx_rl_init", &env->api.init, sizeof(env->api.init));
    fzero_load_symbol(env, "gdx_rl_bind_thread",
        &env->api.bind_thread, sizeof(env->api.bind_thread));
    fzero_load_symbol(env, "gdx_rl_reset", &env->api.reset, sizeof(env->api.reset));
    fzero_load_symbol(env, "gdx_rl_step", &env->api.step, sizeof(env->api.step));
    fzero_load_symbol(env, "gdx_rl_semantic_hash",
        &env->api.semantic_hash, sizeof(env->api.semantic_hash));
    fzero_load_symbol(env, "gdx_rl_last_error",
        &env->api.last_error, sizeof(env->api.last_error));
    fzero_load_symbol(env, "gdx_rl_action_nvec",
        &env->api.action_nvec, sizeof(env->api.action_nvec));

    uint32_t version = env->api.abi_version();
    if (version != GDX_RL_ABI_VERSION) {
        fprintf(stderr, "fzero: ABI version mismatch: adapter=%u core=%u\n",
            (unsigned)GDX_RL_ABI_VERSION, (unsigned)version);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < GDX_RL_ACTION_SIZE; ++i) {
        if (env->api.action_nvec[i] != FZERO_ACTION_NVECS[i]) {
            fprintf(stderr,
                "fzero: action nvec mismatch at %d: adapter=%d core=%d\n",
                i, FZERO_ACTION_NVECS[i], env->api.action_nvec[i]);
            exit(EXIT_FAILURE);
        }
    }
}

void puf_init(FZero* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    env->tag = 0;
    env->boundary_reached = 0;

    const char* library_path = dict_get_str(kwargs, "core_lib_path");
    const char* rom_path = dict_get_str(kwargs, "rom_path");
    const char* rom_override = getenv("FZEROX_ROM");
    if (rom_override != NULL && rom_override[0] != '\0') {
        rom_path = rom_override;
    }
    fzero_load_api(env, library_path);

    GdxRlConfig config;
    env->api.default_config(&config);
    config.course_index = fzero_config_i32(kwargs, "course_index");
    config.character = fzero_config_i32(kwargs, "character");
    config.machine_skin = fzero_config_i32(kwargs, "machine_skin");
    config.laps = fzero_config_i32(kwargs, "laps");
    config.action_repeat = fzero_config_i32(kwargs, "action_repeat");
    env->action_repeat = config.action_repeat;
    config.max_episode_frames = fzero_config_i32(kwargs, "max_episode_frames");
    config.stall_frames = fzero_config_i32(kwargs, "stall_frames");
    config.stall_progress = (float)dict_get(kwargs, "stall_progress");
    config.engine_balance = (float)dict_get(kwargs, "engine_balance");
    double reward_scale = dict_get(kwargs, "reward_scale");
    if (!isfinite(reward_scale) || reward_scale <= 0.0 || reward_scale > 0.03125) {
        fzero_fatal("configuration", "reward_scale must be finite and in (0, 1/32]");
    }
    env->reward_scale = (float)reward_scale;
    if (!isfinite(env->reward_scale) || env->reward_scale <= 0.0f) {
        fzero_fatal("configuration", "reward_scale underflows float32");
    }
    double finish_time_target_frames = dict_get(kwargs, "finish_time_target_frames");
    if (!isfinite(finish_time_target_frames) || finish_time_target_frames < 0.0) {
        fzero_fatal("configuration",
            "finish_time_target_frames must be finite and non-negative");
    }
    env->finish_time_target_frames = (float)finish_time_target_frames;
    double dash_hit_reward = dict_get(kwargs, "dash_hit_reward");
    if (!isfinite(dash_hit_reward) || dash_hit_reward < 0.0 ||
            dash_hit_reward > 0.1) {
        fzero_fatal("configuration",
            "dash_hit_reward must be finite and in [0, 0.1]");
    }
    env->dash_hit_reward = (float)dash_hit_reward;
    int32_t cpu_oracle = fzero_config_i32(kwargs, "cpu_oracle");
    if (cpu_oracle != 0 && cpu_oracle != 1) {
        fzero_fatal("configuration", "cpu_oracle must be 0 or 1");
    }
    config.cpu_oracle = (uint8_t)cpu_oracle;
    int32_t steering_curriculum = fzero_config_i32(kwargs, "steering_curriculum");
    if (steering_curriculum != 0 && steering_curriculum != 1) {
        fzero_fatal("configuration", "steering_curriculum must be 0 or 1");
    }
    env->steering_curriculum = (uint8_t)steering_curriculum;

    /* Puffer seeds Env.rng with the environment index before puf_init. Keep
     * reset streams deterministic and distinct without using global rand(). */
    uint32_t configured_seed = fzero_config_u32(kwargs, "seed");
    env->seed_base = configured_seed + (uint32_t)env->rng * UINT32_C(1000003);
    env->reset_count = 0;

    fzero_core_check(env, "gdx_rl_init", env->api.init(rom_path, &config));
}

void puf_reset(FZero* env) {
    if (env->agents[0].observations == NULL) {
        fzero_fatal("puf_reset", "observation buffer is not bound");
    }
    fzero_bind_current_thread(env);
    env->last_reset_seed = env->seed_base + env->reset_count++;
    fzero_core_check(env, "gdx_rl_reset",
        env->api.reset(env->last_reset_seed, env->agents[0].observations));
    env->episode_return = 0.0f;
    env->learner_return = 0.0f;
    env->episode_decisions = 0;
    env->episode_dash_hits = 0;
    memset(env->steer_counts, 0, sizeof(env->steer_counts));
    memset(env->pitch_counts, 0, sizeof(env->pitch_counts));
    env->throttle_on_count = 0;
    env->brake_on_count = 0;
    env->boost_on_count = 0;
    fzero_write_action_mask(env);
    if (env->agents[0].rewards != NULL) {
        env->agents[0].rewards[0] = 0.0f;
    }
    if (env->agents[0].terminals != NULL) {
        env->agents[0].terminals[0] = 0.0f;
    }
}

static inline void fzero_add_log(FZero* env, const GdxRlTransition* transition) {
    const int finished = transition->reason == GDX_RL_FINISHED;
    float outcome = 0.0f;
    (void)fzero_terminal_outcome(
        transition, env->finish_time_target_frames, &outcome);
    env->log.perf += outcome;
    env->log.score += outcome;
    env->log.episode_return += env->episode_return;
    env->log.learner_return += env->learner_return;
    env->log.episode_length += (float)env->episode_decisions;
    env->log.episode_frames += (float)transition->episode_frame;
    env->log.finish_frames += finished ? (float)transition->episode_frame : 0.0f;
    env->log.dash_hits += (float)env->episode_dash_hits;
    env->log.race_distance += transition->race_distance;
    env->log.terminal_speed += transition->speed;
    env->log.terminal_energy += transition->energy_fraction;
    env->log.terminal_lap += (float)transition->lap;
    env->log.finish_rate += finished ? 1.0f : 0.0f;
    env->log.terminated_rate += transition->terminated ? 1.0f : 0.0f;
    env->log.truncated_rate += transition->truncated ? 1.0f : 0.0f;
    env->log.spinning_out_rate += transition->reason == GDX_RL_SPINNING_OUT ? 1.0f : 0.0f;
    env->log.falling_rate += transition->reason == GDX_RL_FALLING ? 1.0f : 0.0f;
    env->log.crashed_rate += transition->reason == GDX_RL_CRASHED ? 1.0f : 0.0f;
    env->log.retired_rate += transition->reason == GDX_RL_RETIRED ? 1.0f : 0.0f;
    env->log.time_limit_rate += transition->reason == GDX_RL_TIME_LIMIT ? 1.0f : 0.0f;
    env->log.stalled_rate += transition->reason == GDX_RL_STALLED ? 1.0f : 0.0f;
    if (env->episode_decisions > 0) {
        float inv_decisions = 1.0f / (float)env->episode_decisions;
        for (int category = 0; category < GDX_RL_STEER_BINS; ++category) {
            env->log.steer_fraction[category] +=
                (float)env->steer_counts[category] * inv_decisions;
        }
        for (int category = 0; category < GDX_RL_PITCH_BINS; ++category) {
            env->log.pitch_fraction[category] +=
                (float)env->pitch_counts[category] * inv_decisions;
        }
        env->log.throttle_on_fraction += (float)env->throttle_on_count * inv_decisions;
        env->log.brake_on_fraction += (float)env->brake_on_count * inv_decisions;
        env->log.boost_on_fraction += (float)env->boost_on_count * inv_decisions;
    }
    env->log.n += 1.0f;
}

void puf_step(FZero* env) {
    Agent* agent = &env->agents[0];
    if (agent->observations == NULL || agent->actions == NULL
            || agent->rewards == NULL || agent->terminals == NULL) {
        fzero_fatal("puf_step", "agent buffers are not bound");
    }
    fzero_bind_current_thread(env);

    int32_t action[GDX_RL_ACTION_SIZE];
    int invalid = fzero_decode_actions(agent->actions, action);
    if (invalid != 0) {
        int index = invalid - 1;
        fprintf(stderr,
            "fzero: invalid action category at index %d: %.9g (expected integer in [0, %d))\n",
            index, agent->actions[index], FZERO_ACTION_NVECS[index]);
        exit(EXIT_FAILURE);
    }
    /* The mask makes PPO sample these fixed heads. Reapply them here so direct
     * CPU/manual callers receive the same curriculum semantics before a model
     * is loaded and while their action buffer is still zero-filled. */
    fzero_apply_curriculum(env, action);
    if (!fzero_action_is_legal(env, action)) {
        fzero_fatal("action mask", "policy sampled an action disabled by the current curriculum");
    }
    GdxRlTransition transition = {0};
    fzero_core_check(env, "gdx_rl_step",
        env->api.step(action, agent->observations, &transition));
    transition.reward = fzero_shape_dash_reward(
        transition.reward, &transition, env->dash_hit_reward);
    fzero_gdx_input_trace_record(env, action, &transition);
    env->last_transition = transition;
    env->episode_return += transition.reward;
    float learner_reward = 0.0f;
    if (fzero_compute_learner_reward(transition.reward, &transition,
            env->reward_scale, env->finish_time_target_frames,
            &learner_reward) != 0) {
        fzero_fatal("learner reward",
            "reward must remain finite and in [-1, 1]");
    }
    env->learner_return += learner_reward;
    env->episode_decisions += 1;
    env->episode_dash_hits += (uint32_t)transition.dash_pad_hits;
    env->steer_counts[action[0]] += 1;
    env->pitch_counts[action[1]] += 1;
    env->throttle_on_count += (uint32_t)action[2];
    env->brake_on_count += (uint32_t)action[3];
    env->boost_on_count += (uint32_t)action[4];

    const int done = transition.terminated != 0 || transition.truncated != 0;
    const float terminal_reward = learner_reward;
    agent->rewards[0] = learner_reward;
    agent->terminals[0] = done ? 1.0f : 0.0f;
    env->boundary_reached = done;

    if (done) {
        fzero_gdx_input_trace_finish(env, &transition);
        fzero_add_log(env, &transition);
        /* Puffer consumes reward/terminal from this transition alongside the
         * observation returned here. Reset immediately so that observation is
         * the next episode's first state, then restore the terminal outputs. */
        puf_reset(env);
        agent->rewards[0] = terminal_reward;
        agent->terminals[0] = 1.0f;
        env->last_transition = transition;
        env->boundary_reached = 1;
    }
}

void puf_render(FZero* env) {
    if (!IsWindowReady()) {
        InitWindow(900, 300, "PufferLib F-Zero X (headless telemetry)");
    }
    BeginDrawing();
    ClearBackground((Color){12, 18, 28, 255});
    DrawText("F-Zero X core is running headless", 24, 24, 28, RAYWHITE);
    DrawText(TextFormat("Reset seed: %u   Decision: %u   Frame: %u",
        env->last_reset_seed, env->episode_decisions,
        env->last_transition.episode_frame), 24, 78, 22, LIGHTGRAY);
    DrawText(TextFormat("Reward: %.4f   Distance: %.1f   Speed: %.1f   Energy: %.3f",
        env->agents[0].rewards != NULL ? env->agents[0].rewards[0] : 0.0f,
        env->last_transition.race_distance, env->last_transition.speed,
        env->last_transition.energy_fraction), 24, 118, 22, LIGHTGRAY);
    DrawText("Use --headless for throughput/evaluation; no video renderer is loaded.",
        24, 174, 20, GRAY);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(FZero* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (env != NULL && env->gdx_input_trace != NULL) {
        FZeroGdxInputTrace* trace = env->gdx_input_trace;
        if (trace->file != NULL) {
            fclose(trace->file);
        }
        free(trace->final_path);
        free(trace->temporary_path);
        free(trace);
        env->gdx_input_trace = NULL;
    }
    /* The game owns live ucontexts and process-lifetime caches. Its ABI has no
     * close entry point, so unloading the namespace with dlclose is unsafe. */
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "learner_return", log->learner_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "episode_frames", log->episode_frames);
    dict_set(out, "finish_frames", log->finish_rate > 0.0f
        ? log->finish_frames / log->finish_rate : 0.0f);
    dict_set(out, "dash_hits", log->dash_hits);
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
