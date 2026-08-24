#include "gdx_rl_env.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

const int32_t gdx_rl_action_nvec[GDX_RL_ACTION_SIZE] = {11, 5, 2, 2, 2};

static int initialized;
static int episode_done;
static int step_count;
static uint32_t reset_seed;
static pthread_t owner_thread;
static char last_error[256];

static int fail(const char* message) {
    snprintf(last_error, sizeof(last_error), "%s", message);
    return -1;
}

static int check_owner(void) {
    if (!initialized || !pthread_equal(owner_thread, pthread_self())) {
        return fail("fake core called from a thread that did not bind");
    }
    return 0;
}

static float reset_observation(uint32_t seed, int index) {
    return (float)(seed & UINT32_C(0xffff)) / 65535.0f + (float)index * 0.001f;
}

uint32_t gdx_rl_abi_version(void) {
    return GDX_RL_ABI_VERSION;
}

void gdx_rl_default_config(GdxRlConfig* config) {
    memset(config, 0, sizeof(*config));
    config->laps = 3;
    config->action_repeat = 4;
    config->max_episode_frames = 18000;
    config->stall_frames = 600;
    config->stall_progress = 100.0f;
    config->engine_balance = 0.5f;
}

int gdx_rl_init(const char* rom_path, const GdxRlConfig* config) {
    if (initialized) {
        return fail("fake namespace initialized twice");
    }
    if (rom_path == NULL || strcmp(rom_path, "/fake/rom.z64") != 0) {
        return fail("fake ROM path mismatch");
    }
    if (config == NULL || config->action_repeat != 4 || config->laps != 1) {
        return fail("fake config mismatch");
    }
    initialized = 1;
    owner_thread = pthread_self();
    return 0;
}

int gdx_rl_bind_thread(void) {
    if (!initialized) {
        return fail("bind before init");
    }
    owner_thread = pthread_self();
    return 0;
}

int gdx_rl_reset(uint32_t seed, float observation[GDX_RL_OBSERVATION_SIZE]) {
    if (check_owner() != 0) {
        return -1;
    }
    if (observation == NULL) {
        return fail("null reset observation");
    }
    reset_seed = seed;
    step_count = 0;
    episode_done = 0;
    for (int i = 0; i < GDX_RL_OBSERVATION_SIZE; ++i) {
        observation[i] = reset_observation(seed, i);
    }
    return 0;
}

int gdx_rl_step(const int32_t action[GDX_RL_ACTION_SIZE],
        float observation[GDX_RL_OBSERVATION_SIZE], GdxRlTransition* transition) {
    if (check_owner() != 0) {
        return -1;
    }
    if (action == NULL || observation == NULL || transition == NULL) {
        return fail("null step argument");
    }
    if (episode_done) {
        return fail("step after terminal without reset");
    }
    for (int i = 0; i < GDX_RL_ACTION_SIZE; ++i) {
        if (action[i] < 0 || action[i] >= gdx_rl_action_nvec[i]) {
            return fail("out-of-range action reached fake core");
        }
    }

    step_count += 1;
    for (int i = 0; i < GDX_RL_OBSERVATION_SIZE; ++i) {
        observation[i] = (float)(10 * step_count + i);
    }
    memset(transition, 0, sizeof(*transition));
    transition->reward = step_count == 1 ? 1.25f : 7.0f;
    transition->race_distance = 100.0f * (float)step_count;
    transition->speed = 40.0f + 5.0f * (float)step_count;
    transition->energy_fraction = 0.75f;
    transition->semantic_hash = ((uint64_t)reset_seed << 32) | (uint32_t)step_count;
    transition->episode_frame = (uint32_t)(4 * step_count);
    transition->lap = 1;
    transition->dash_pad_hits = step_count == 1 ? 1 : 0;
    if (step_count == 2) {
        transition->terminated = 1;
        transition->reason = GDX_RL_FINISHED;
        episode_done = 1;
    }
    return 0;
}

uint64_t gdx_rl_semantic_hash(void) {
    if (check_owner() != 0) {
        return 0;
    }
    return ((uint64_t)reset_seed << 32) | (uint32_t)step_count;
}

const char* gdx_rl_last_error(void) {
    return last_error;
}
