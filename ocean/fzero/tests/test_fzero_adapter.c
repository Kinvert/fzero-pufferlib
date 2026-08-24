#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../fzero.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_NEAR(actual, expected, tolerance) do { \
    float check_actual = (float)(actual); \
    float check_expected = (float)(expected); \
    if (fabsf(check_actual - check_expected) > (tolerance)) { \
        fprintf(stderr, "%s:%d: expected %.9g, got %.9g\n", \
            __FILE__, __LINE__, check_expected, check_actual); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void set_config(Dict* kwargs, const char* core_path) {
    kwargs->name = dict_strdup("env");
    dict_set_str(kwargs, "core_lib_path", core_path);
    dict_set_str(kwargs, "rom_path", "/fake/rom.z64");
    dict_set(kwargs, "seed", 100.0);
    dict_set(kwargs, "course_index", 0.0);
    dict_set(kwargs, "character", 0.0);
    dict_set(kwargs, "machine_skin", 0.0);
    dict_set(kwargs, "laps", 1.0);
    dict_set(kwargs, "action_repeat", 4.0);
    dict_set(kwargs, "max_episode_frames", 1000.0);
    dict_set(kwargs, "stall_frames", 100.0);
    dict_set(kwargs, "stall_progress", 1.0);
    dict_set(kwargs, "engine_balance", 0.5);
    dict_set(kwargs, "cpu_oracle", 0.0);
    dict_set(kwargs, "steering_curriculum", 1.0);
    dict_set(kwargs, "reward_scale", 0.03125);
    dict_set(kwargs, "finish_time_target_frames", 0.0);
    dict_set(kwargs, "dash_hit_reward", 0.1);
}

static void test_action_validation(void) {
    int action_sizes[] = ACT_SIZES;
    CHECK(sizeof(action_sizes) / sizeof(action_sizes[0]) == GDX_RL_ACTION_SIZE);
    for (int i = 0; i < GDX_RL_ACTION_SIZE; ++i) {
        CHECK(action_sizes[i] == FZERO_ACTION_NVECS[i]);
    }

    float raw[GDX_RL_ACTION_SIZE] = {10.0f, 4.0f, 1.0f, 1.0f, 1.0f};
    int32_t decoded[GDX_RL_ACTION_SIZE] = {0};
    CHECK(fzero_decode_actions(raw, decoded) == 0);
    for (int i = 0; i < GDX_RL_ACTION_SIZE; ++i) {
        CHECK(decoded[i] == (int32_t)raw[i]);
    }

    raw[0] = NAN;
    CHECK(fzero_decode_actions(raw, decoded) == 1);
    raw[0] = 0.0f;
    raw[1] = 2.5f;
    CHECK(fzero_decode_actions(raw, decoded) == 2);
    raw[1] = 0.0f;
    raw[4] = 2.0f;
    CHECK(fzero_decode_actions(raw, decoded) == 5);
    raw[4] = 0.0f;
    raw[0] = -1.0f;
    CHECK(fzero_decode_actions(raw, decoded) == 1);

    float scaled = 0.0f;
    CHECK(fzero_scale_reward(8.25f, 0.03125f, &scaled) == 0);
    CHECK_NEAR(scaled, 8.25f / 32.0f, 0.0f);
    CHECK(fzero_scale_reward(32.0f, 0.03125f, &scaled) == 0);
    CHECK_NEAR(scaled, 1.0f, 0.0f);
    CHECK(fzero_scale_reward(32.01f, 0.03125f, &scaled) != 0);
    CHECK(fzero_scale_reward(NAN, 0.03125f, &scaled) != 0);

    GdxRlTransition transition = {0};
    transition.reason = GDX_RL_FINISHED;
    transition.episode_frame = 8;
    CHECK_NEAR(fzero_completed_lap_score(0.0f, 8), 1.0f, 0.0f);
    CHECK_NEAR(fzero_completed_lap_score(2000.0f, 1000), 1.0f, 0.0f);
    CHECK_NEAR(fzero_completed_lap_score(2000.0f, 2000), 1.0f, 0.0f);
    CHECK_NEAR(fzero_completed_lap_score(2000.0f, 2241),
        2000.0f / 2241.0f, 0.0f);
    CHECK_NEAR(fzero_completed_lap_score(2000.0f, 2500), 0.8f, 1e-7f);
    CHECK(fzero_completed_lap_score(2000.0f, 2141) >
        fzero_completed_lap_score(2000.0f, 2241));

    float learner_reward = 0.0f;
    transition.episode_frame = 2241;
    CHECK(fzero_compute_learner_reward(27.5f, &transition,
        0.03125f, 2000.0f, &learner_reward) == 0);
    CHECK_NEAR(learner_reward, 2000.0f / 2241.0f, 0.0f);
    transition.episode_frame = 8;
    CHECK(fzero_compute_learner_reward(27.5f, &transition,
        0.03125f, 0.0f, &learner_reward) == 0);
    CHECK_NEAR(learner_reward, 1.0f, 0.0f);
    CHECK(fzero_compute_learner_reward(NAN, &transition,
        0.03125f, 0.0f, &learner_reward) != 0);

    const uint8_t fatal_reasons[] = {
        GDX_RL_SPINNING_OUT,
        GDX_RL_FALLING,
        GDX_RL_CRASHED,
        GDX_RL_RETIRED,
    };
    for (size_t i = 0; i < sizeof(fatal_reasons) / sizeof(fatal_reasons[0]); ++i) {
        transition.reason = fatal_reasons[i];
        CHECK(fzero_compute_learner_reward(31.0f, &transition,
            0.03125f, 2000.0f, &learner_reward) == 0);
        CHECK_NEAR(learner_reward, -1.0f, 0.0f);
    }
    transition.reason = GDX_RL_TIME_LIMIT;
    CHECK(fzero_compute_learner_reward(31.0f, &transition,
        0.03125f, 2000.0f, &learner_reward) == 0);
    CHECK_NEAR(learner_reward, 0.0f, 0.0f);
    transition.reason = GDX_RL_STALLED;
    CHECK(fzero_compute_learner_reward(-31.0f, &transition,
        0.03125f, 2000.0f, &learner_reward) == 0);
    CHECK_NEAR(learner_reward, 0.0f, 0.0f);
    transition.reason = GDX_RL_RUNNING;
    CHECK(fzero_compute_learner_reward(1.25f, &transition,
        0.03125f, 2000.0f, &learner_reward) == 0);
    CHECK_NEAR(learner_reward, 1.25f / 32.0f, 0.0f);

    transition.dash_pad_hits = 1;
    CHECK_NEAR(fzero_shape_dash_reward(1.25f, &transition, 0.1f),
        1.35f, 1e-7f);
    CHECK_NEAR(fzero_shape_dash_reward(1.25f, &transition, 0.0f),
        1.25f, 0.0f);
    transition.dash_pad_hits = 2;
    CHECK_NEAR(fzero_shape_dash_reward(1.25f, &transition, 0.1f),
        1.45f, 1e-7f);

    FZero log_env = {0};
    log_env.finish_time_target_frames = 2000.0f;
    log_env.episode_return = 123.0f;
    transition.reason = GDX_RL_FINISHED;
    transition.episode_frame = 2241;
    fzero_add_log(&log_env, &transition);
    CHECK_NEAR(log_env.log.score, 2000.0f / 2241.0f, 0.0f);
    CHECK_NEAR(log_env.log.perf, 2000.0f / 2241.0f, 0.0f);
    CHECK_NEAR(log_env.log.episode_return, 123.0f, 0.0f);

    memset(&log_env, 0, sizeof(log_env));
    log_env.finish_time_target_frames = 2000.0f;
    transition.reason = GDX_RL_CRASHED;
    fzero_add_log(&log_env, &transition);
    CHECK_NEAR(log_env.log.score, -1.0f, 0.0f);
    CHECK_NEAR(log_env.log.perf, -1.0f, 0.0f);

    memset(&log_env, 0, sizeof(log_env));
    log_env.finish_time_target_frames = 2000.0f;
    transition.reason = GDX_RL_STALLED;
    fzero_add_log(&log_env, &transition);
    CHECK_NEAR(log_env.log.score, 0.0f, 0.0f);
    CHECK_NEAR(log_env.log.perf, 0.0f, 0.0f);
}

static void* step_on_puffer_worker(void* argument) {
    puf_step((FZero*)argument);
    return NULL;
}

static float expected_reset_observation(uint32_t seed, int index) {
    return (float)(seed & UINT32_C(0xffff)) / 65535.0f + (float)index * 0.001f;
}

static char* read_stream(FILE* file) {
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    long size = ftell(file);
    CHECK(size >= 0);
    CHECK(fseek(file, 0, SEEK_SET) == 0);
    char* text = (char*)calloc((size_t)size + 1, 1);
    CHECK(text != NULL);
    CHECK(fread(text, 1, (size_t)size, file) == (size_t)size);
    return text;
}

static char* read_path(const char* path) {
    FILE* file = fopen(path, "rb");
    CHECK(file != NULL);
    char* text = read_stream(file);
    CHECK(fclose(file) == 0);
    return text;
}

static void test_gdx_action_encoding(void) {
    FILE* file = tmpfile();
    CHECK(file != NULL);
    const int32_t boost_action[GDX_RL_ACTION_SIZE] = {10, 4, 1, 1, 1};
    CHECK(fzero_gdx_write_action(file, 7, 4, boost_action) == 0);
    char* text = read_stream(file);
    CHECK(strcmp(text,
        "# decision 7 action 10 4 1 1 1\n"
        "INPUT A+CDOWN+B 63 63 1\n"
        "INPUT A+CDOWN 63 63 3\n") == 0);
    free(text);
    CHECK(fclose(file) == 0);

    file = tmpfile();
    CHECK(file != NULL);
    const int32_t neutral_buttons[GDX_RL_ACTION_SIZE] = {1, 1, 0, 0, 0};
    CHECK(fzero_gdx_write_action(file, 8, 4, neutral_buttons) == 0);
    text = read_stream(file);
    CHECK(strcmp(text,
        "# decision 8 action 1 1 0 0 0\n"
        "INPUT - -13 -32 4\n") == 0);
    free(text);
    CHECK(fclose(file) == 0);
}

static void test_thread_handoff_and_autoreset(
        const char* core_path, const char* trace_path) {
    Dict kwargs = {0};
    set_config(&kwargs, core_path);

    FZero env;
    memset(&env, 0, sizeof(env));
    env.rng = 5;
    puf_init(&env, &kwargs);

    float observations[GDX_RL_OBSERVATION_SIZE] = {0};
    float actions[GDX_RL_ACTION_SIZE] = {10.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    float reward = 0.0f;
    float terminal = 0.0f;
    unsigned char action_mask[FZERO_ACTION_MASK_SIZE] = {0};
    env.agents[0].observations = observations;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    env.agents[0].action_mask = action_mask;

    puf_reset(&env);
    for (int category = 0; category < GDX_RL_STEER_BINS; ++category) {
        CHECK(action_mask[FZERO_STEER_MASK_OFFSET + category] == 1);
    }
    CHECK(action_mask[FZERO_PITCH_MASK_OFFSET] == 1);
    CHECK(action_mask[FZERO_PITCH_MASK_OFFSET + 1] == 0);
    CHECK(action_mask[FZERO_THROTTLE_MASK_OFFSET] == 0);
    CHECK(action_mask[FZERO_THROTTLE_MASK_OFFSET + 1] == 1);
    CHECK(action_mask[FZERO_BRAKE_MASK_OFFSET] == 1);
    CHECK(action_mask[FZERO_BRAKE_MASK_OFFSET + 1] == 0);
    CHECK(action_mask[FZERO_BOOST_MASK_OFFSET] == 1);
    CHECK(action_mask[FZERO_BOOST_MASK_OFFSET + 1] == 0);
    uint32_t first_seed = UINT32_C(100) + UINT32_C(5) * UINT32_C(1000003);
    CHECK(env.last_reset_seed == first_seed);
    CHECK_NEAR(observations[0], expected_reset_observation(first_seed, 0), 1e-7f);
    fzero_gdx_input_trace_open(
        &env, trace_path, "/fake/checkpoint.bin", UINT32_C(1));

    /* This is the same setup -> Puffer worker handoff used by src/pufferl.cu. */
    pthread_t worker;
    CHECK(pthread_create(&worker, NULL, step_on_puffer_worker, &env) == 0);
    CHECK(pthread_join(worker, NULL) == 0);
    CHECK_NEAR(reward, 1.35f / 32.0f, 1e-7f);
    CHECK_NEAR(terminal, 0.0f, 0.0f);
    CHECK_NEAR(observations[0], 10.0f, 0.0f);
    CHECK(env.episode_decisions == 1);

    /* Moving back to main exercises a second bind. This step terminates; the
     * adapter must preserve reward/terminal while exposing reset observation. */
    puf_step(&env);
    uint32_t second_seed = first_seed + UINT32_C(1);
    CHECK(env.last_reset_seed == second_seed);
    CHECK_NEAR(reward, 1.0f, 0.0f);
    CHECK_NEAR(terminal, 1.0f, 0.0f);
    CHECK_NEAR(observations[0], expected_reset_observation(second_seed, 0), 1e-7f);
    CHECK(env.last_transition.terminated == 1);
    CHECK(env.last_transition.reason == GDX_RL_FINISHED);
    CHECK(env.boundary_reached == 1);
    CHECK_NEAR(env.log.n, 1.0f, 0.0f);
    CHECK_NEAR(env.log.perf, 1.0f, 0.0f);
    CHECK_NEAR(env.log.score, 1.0f, 0.0f);
    CHECK_NEAR(env.log.episode_return, 8.35f, 1e-6f);
    CHECK_NEAR(env.log.learner_return, 1.0f + 1.35f / 32.0f, 1e-7f);
    CHECK_NEAR(env.log.episode_length, 2.0f, 0.0f);
    CHECK_NEAR(env.log.episode_frames, 8.0f, 0.0f);
    CHECK_NEAR(env.log.finish_frames, 8.0f, 0.0f);
    CHECK_NEAR(env.log.dash_hits, 1.0f, 0.0f);
    CHECK_NEAR(env.log.race_distance, 200.0f, 0.0f);
    CHECK_NEAR(env.log.finish_rate, 1.0f, 0.0f);
    CHECK_NEAR(env.log.terminated_rate, 1.0f, 0.0f);
    CHECK_NEAR(env.log.truncated_rate, 0.0f, 0.0f);
    CHECK_NEAR(env.log.steer_fraction[10], 1.0f, 0.0f);
    CHECK_NEAR(env.log.pitch_fraction[0], 1.0f, 0.0f);
    CHECK_NEAR(env.log.throttle_on_fraction, 1.0f, 0.0f);
    CHECK_NEAR(env.log.brake_on_fraction, 0.0f, 0.0f);
    CHECK_NEAR(env.log.boost_on_fraction, 0.0f, 0.0f);
    CHECK(fzero_gdx_input_trace_complete(&env));
    CHECK(env.gdx_input_trace->decisions == 2);

    char* trace = read_path(trace_path);
    CHECK(strstr(trace, "# environment_seed 5000115\n") != NULL);
    CHECK(strstr(trace, "# sampler_seed 1\n") != NULL);
    CHECK(strstr(trace, "# decision 0 action 10 0 1 0 0\nINPUT A 63 0 4\n") != NULL);
    CHECK(strstr(trace, "# decision 1 action 10 0 1 0 0\nINPUT A 63 0 4\n") != NULL);
    CHECK(strstr(trace, "# result decisions 2 raw_score 8.35000038 reason 1") != NULL);
    CHECK(strstr(trace,
        "LOG Puffer policy trace exhausted; waiting for rendered terminal\n"
        "WAIT 120\nSHOT puffer-fzero-finish\nWAIT 2\nQUIT\n") != NULL);
    free(trace);

    Dict output = {0};
    puf_log(&env.log, &output);
    CHECK_NEAR(dict_get(&output, "perf"), 1.0f, 0.0f);
    CHECK_NEAR(dict_get(&output, "score"), 1.0f, 0.0f);
    CHECK_NEAR(dict_get(&output, "learner_return"),
        1.0f + 1.35f / 32.0f, 1e-7f);
    CHECK_NEAR(dict_get(&output, "finish_rate"), 1.0f, 0.0f);
    CHECK_NEAR(dict_get(&output, "episode_frames"), 8.0f, 0.0f);
    CHECK_NEAR(dict_get(&output, "finish_frames"), 8.0f, 0.0f);
    CHECK_NEAR(dict_get(&output, "dash_hits"), 1.0f, 0.0f);
    CHECK_NEAR(dict_get(&output, "steer_10_fraction"), 1.0f, 0.0f);
    CHECK_NEAR(dict_get(&output, "throttle_on_fraction"), 1.0f, 0.0f);

    dict_clear(&output);
    puf_close(&env);
    dict_clear(&kwargs);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr,
            "usage: %s <fake-libgdiffuser_rl.so> <trace-output.txt>\n", argv[0]);
        return 2;
    }
    test_action_validation();
    test_gdx_action_encoding();
    test_thread_handoff_and_autoreset(argv[1], argv[2]);
    puts("FZERO_ADAPTER_TEST_OK actions=5 worker_handoff=1 autoreset=1 gdx_trace=1");
    return 0;
}
