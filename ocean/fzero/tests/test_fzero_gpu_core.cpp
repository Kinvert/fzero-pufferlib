#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../fzero_gpu_core.cuh"

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_NEAR(actual, expected, tolerance) do { \
    float check_actual = (float)(actual); \
    float check_expected = (float)(expected); \
    if (std::fabs(check_actual - check_expected) > (tolerance)) { \
        std::fprintf(stderr, "%s:%d: expected %.9g, got %.9g\n", \
            __FILE__, __LINE__, check_expected, check_actual); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

static_assert(GDX_RL_MAX_DASH_PADS == 192, "GPU dash-pad capacity must match the core");
static_assert(GDX_RL_OBSERVATION_SIZE == 83, "GPU observation ABI must include dash-pad fields");
static_assert(sizeof(GdxRlDashPad) == 16, "dash-pad export must remain a compact POD");
static_assert(offsetof(GdxRlDashPad, entry_distance) == 0, "unexpected dash-pad layout");
static_assert(offsetof(GdxRlDashPad, target) == 4, "unexpected dash-pad layout");
static_assert(offsetof(GdxRlDashPad, width) == 8, "unexpected dash-pad layout");
static_assert(offsetof(GdxRlDashPad, segment_index) == 12, "unexpected dash-pad layout");

namespace {

struct ObservationStore {
    float* values;
    int* count;

    void operator()(int index, float value) const {
        CHECK(index == *count);
        CHECK(std::isfinite(value));
        CHECK(value >= -1.00001f && value <= 1.00001f);
        values[index] = value;
        *count = index + 1;
    }
};

void observe(const FZeroGpuState& state, const FZeroGpuConfig& config,
        const std::vector<GdxRlCourseSample>& course,
        const std::vector<GdxRlDashPad>& dash_pads,
        float values[GDX_RL_OBSERVATION_SIZE]) {
    int count = 0;
    fzero_gpu_observe(&state, &config, course.data(),
        dash_pads.empty() ? nullptr : dash_pads.data(), ObservationStore{values, &count});
    CHECK(count == GDX_RL_OBSERVATION_SIZE);
}

void test_dash_observation_boundaries(void) {
    FZeroGpuConfig config = {};
    config.course_length = 1000.0f;
    float output[4] = {};

    fzero_gpu_nearest_dash_observation(50.0f, &config, nullptr, output);
    for (float value : output) {
        CHECK_NEAR(value, 0.0f, 0.0f);
    }

    const GdxRlDashPad pads[] = {
        {100.0f, -0.5f, 0.25f, 3},
        {400.0f, 0.75f, 0.5f, 7},
    };
    config.dash_pad_count = 2;

    fzero_gpu_nearest_dash_observation(50.0f, &config, pads, output);
    CHECK_NEAR(output[0], 1.0f, 0.0f);
    CHECK_NEAR(output[1], 50.0f / 8000.0f, 1e-7f);
    CHECK_NEAR(output[2], -0.5f, 0.0f);
    CHECK_NEAR(output[3], 0.25f, 0.0f);

    fzero_gpu_nearest_dash_observation(100.0f, &config, pads, output);
    CHECK_NEAR(output[1], 0.0f, 0.0f);
    CHECK_NEAR(output[2], -0.5f, 0.0f);

    fzero_gpu_nearest_dash_observation(100.25f, &config, pads, output);
    CHECK_NEAR(output[1], 299.75f / 8000.0f, 1e-7f);
    CHECK_NEAR(output[2], 0.75f, 0.0f);
    CHECK_NEAR(output[3], 0.5f, 0.0f);

    fzero_gpu_nearest_dash_observation(400.25f, &config, pads, output);
    CHECK_NEAR(output[1], 699.75f / 8000.0f, 1e-7f);
    CHECK_NEAR(output[2], -0.5f, 0.0f);

    fzero_gpu_nearest_dash_observation(-50.0f, &config, pads, output);
    CHECK_NEAR(output[1], 150.0f / 8000.0f, 1e-7f);

    const GdxRlDashPad distant = {9000.0f, 0.0f, 1.0f, 0};
    config.course_length = 20000.0f;
    config.dash_pad_count = 1;
    fzero_gpu_nearest_dash_observation(0.0f, &config, &distant, output);
    CHECK_NEAR(output[1], 1.0f, 0.0f);
}

void test_learner_outcome_boundaries(void) {
    FZeroGpuConfig config = {};
    config.reward_scale = 0.03125f;
    config.finish_time_target_frames = 2000.0f;
    FZeroGpuStepResult result = {};
    result.raw_reward = 4.0f;
    CHECK_NEAR(fzero_gpu_learner_reward(&result, &config, 100), 0.125f, 0.0f);

    result.terminated = 1;
    result.reason = FZERO_GPU_FINISHED;
    CHECK_NEAR(fzero_gpu_learner_reward(&result, &config, 1000), 1.0f, 0.0f);
    CHECK_NEAR(fzero_gpu_learner_reward(&result, &config, 2000), 1.0f, 0.0f);
    CHECK_NEAR(fzero_gpu_learner_reward(&result, &config, 2500), 0.8f, 1e-7f);
    config.finish_time_target_frames = 0.0f;
    CHECK_NEAR(fzero_gpu_learner_reward(&result, &config, 2500), 1.0f, 0.0f);
    config.finish_time_target_frames = 2000.0f;

    const int fatal_reasons[] = {
        FZERO_GPU_SPINNING_OUT,
        FZERO_GPU_FALLING,
        FZERO_GPU_CRASHED,
        FZERO_GPU_RETIRED,
    };
    for (int reason : fatal_reasons) {
        result.reason = reason;
        CHECK_NEAR(fzero_gpu_learner_reward(&result, &config, 2500), -1.0f, 0.0f);
    }

    result.terminated = 0;
    result.truncated = 1;
    result.reason = FZERO_GPU_TIME_LIMIT;
    CHECK_NEAR(fzero_gpu_learner_reward(&result, &config, 2500), 0.0f, 0.0f);
    result.reason = FZERO_GPU_STALLED;
    CHECK_NEAR(fzero_gpu_learner_reward(&result, &config, 2500), 0.0f, 0.0f);
}

int nearest_steer_category(float desired) {
    int best_category = 0;
    float best_error = std::fabs(desired);
    for (int category = 1; category < GDX_RL_STEER_BINS; ++category) {
        float error = std::fabs(desired - (float)fzero_gpu_steer_value(category));
        if (error < best_error) {
            best_error = error;
            best_category = category;
        }
    }
    return best_category;
}

FZeroGpuConfig make_config(const GdxRlConfig& oracle,
        const GdxRlCourseMetadata& metadata, const GdxRlReferenceState& reference) {
    FZeroGpuConfig config = {};
    config.sample_count = (int32_t)metadata.sample_count;
    config.dash_pad_count = (int32_t)metadata.dash_pad_count;
    config.laps = oracle.laps;
    config.action_repeat = oracle.action_repeat;
    config.max_episode_frames = oracle.max_episode_frames;
    config.stall_frames = oracle.stall_frames;
    config.steering_curriculum = 1;
    config.course_length = metadata.length;
    config.sample_spacing = metadata.sample_spacing;
    config.stall_progress = oracle.stall_progress;
    config.reward_scale = 0.03125f;
    config.finish_time_target_frames = 2000.0f;
    config.engine_target_speed = 60.0f;
    config.engine_acceleration = 0.25f;
    config.forward_drag = 0.997f;
    config.brake_drag = 0.987f;
    config.steering_rate = 0.032f;
    config.wall_margin = 1.0f;
    config.initial_race_distance = reference.race_distance;
    config.initial_lap_distance = reference.lap_distance;
    config.initial_lateral = reference.lateral;
    config.initial_vertical = reference.vertical;
    config.initial_speed = reference.speed;
    config.initial_energy = reference.energy;
    config.max_energy = reference.max_energy;
    config.seed = 73;
    return config;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <baserom.us.rev0.z64>\n", argv[0]);
        return 2;
    }

    test_dash_observation_boundaries();
    test_learner_outcome_boundaries();

    GdxRlConfig oracle;
    gdx_rl_default_config(&oracle);
    oracle.course_index = 0;
    oracle.character = 0;
    oracle.machine_skin = 0;
    oracle.laps = 1;
    oracle.action_repeat = 4;
    oracle.max_episode_frames = 9000;
    oracle.stall_frames = 600;
    oracle.stall_progress = 100.0f;
    oracle.engine_balance = 0.5f;
    oracle.cpu_oracle = 1;
    CHECK(gdx_rl_init(argv[1], &oracle) == 0);

    float oracle_initial[GDX_RL_OBSERVATION_SIZE] = {};
    CHECK(gdx_rl_reset(73, oracle_initial) == 0);
    constexpr uint32_t kSampleCount = 4096;
    std::vector<GdxRlCourseSample> course(kSampleCount);
    GdxRlCourseMetadata metadata = {};
    GdxRlReferenceState reference = {};
    CHECK(gdx_rl_export_course(kSampleCount, course.data(), &metadata) == 0);
    CHECK(metadata.version == GDX_RL_COURSE_EXPORT_VERSION);
    CHECK(metadata.dash_pad_count <= GDX_RL_MAX_DASH_PADS);
    uint32_t queried_dash_pad_count = UINT32_MAX;
    CHECK(gdx_rl_export_dash_pads(0, nullptr, &queried_dash_pad_count) == 0);
    CHECK(queried_dash_pad_count == metadata.dash_pad_count);
    std::vector<GdxRlDashPad> dash_pads(GDX_RL_MAX_DASH_PADS);
    uint32_t dash_pad_count = 0;
    CHECK(gdx_rl_export_dash_pads(GDX_RL_MAX_DASH_PADS,
        dash_pads.data(), &dash_pad_count) == 0);
    CHECK(dash_pad_count == metadata.dash_pad_count);
    CHECK(dash_pad_count > 0);
    dash_pads.resize(dash_pad_count);
    CHECK(gdx_rl_export_reference_state(&reference) == 0);
    FZeroGpuConfig config = make_config(oracle, metadata, reference);

    FZeroGpuState first = {};
    FZeroGpuState second = {};
    fzero_gpu_reset_state(&first, &config, 17);
    fzero_gpu_reset_state(&second, &config, 17);
    CHECK(std::memcmp(&first, &second, sizeof(first)) == 0);

    float reduced_initial[GDX_RL_OBSERVATION_SIZE] = {};
    observe(first, config, course, dash_pads, reduced_initial);
    float initial_max_error = 0.0f;
    for (int i = 0; i < GDX_RL_OBSERVATION_SIZE; ++i) {
        initial_max_error = std::max(initial_max_error,
            std::fabs(reduced_initial[i] - oracle_initial[i]));
    }
    CHECK(initial_max_error < 0.01f);

    float dash_observation_max_error = 0.0f;
    float oracle_observation[GDX_RL_OBSERVATION_SIZE] = {};
    GdxRlTransition oracle_transition = {};
    const int32_t neutral_action[GDX_RL_ACTION_SIZE] = {};
    int oracle_decisions = 0;
    while (!oracle_transition.terminated && !oracle_transition.truncated &&
            oracle_decisions < 5000) {
        CHECK(gdx_rl_step(neutral_action, oracle_observation, &oracle_transition) == 0);
        GdxRlReferenceState live_reference = {};
        CHECK(gdx_rl_export_reference_state(&live_reference) == 0);
        FZeroGpuState query = first;
        query.lap_distance = live_reference.lap_distance;
        float query_observation[GDX_RL_OBSERVATION_SIZE] = {};
        observe(query, config, course, dash_pads, query_observation);
        for (int i = 79; i < 83; ++i) {
            dash_observation_max_error = std::max(dash_observation_max_error,
                std::fabs(query_observation[i] - oracle_observation[i]));
        }
        oracle_decisions++;
    }
    CHECK(oracle_transition.terminated);
    CHECK(oracle_transition.reason == GDX_RL_FINISHED);
    CHECK(dash_observation_max_error < 1e-6f);

    for (int decision = 0; decision < 100; ++decision) {
        int32_t first_action[GDX_RL_ACTION_SIZE] = {
            decision % GDX_RL_STEER_BINS,
            decision % GDX_RL_PITCH_BINS,
            decision & 1,
            (decision >> 1) & 1,
            (decision % 17) == 0,
        };
        int32_t second_action[GDX_RL_ACTION_SIZE];
        std::memcpy(second_action, first_action, sizeof(second_action));
        FZeroGpuStepResult a =
            fzero_gpu_step_state(&first, &config, course.data(), first_action);
        FZeroGpuStepResult b =
            fzero_gpu_step_state(&second, &config, course.data(), second_action);
        CHECK(std::memcmp(&a, &b, sizeof(a)) == 0);
        CHECK(std::memcmp(&first, &second, sizeof(first)) == 0);
        if (a.terminated || a.truncated) {
            break;
        }
    }

    FZeroGpuState pilot = {};
    fzero_gpu_reset_state(&pilot, &config, 0);
    FZeroGpuStepResult pilot_result = {};
    float pilot_observation[GDX_RL_OBSERVATION_SIZE] = {};
    int pilot_decisions = 0;
    while (!pilot_result.terminated && !pilot_result.truncated && pilot_decisions < 5000) {
        observe(pilot, config, course, dash_pads, pilot_observation);
        float desired_track_yaw = std::atan2(
            pilot_observation[48], std::max(0.1f, pilot_observation[47]));
        float target_yaw = desired_track_yaw - pilot.lateral * 0.0015f;
        float desired = 63.0f *
            (pilot.yaw - target_yaw) /
            (config.steering_rate * (float)config.action_repeat);
        desired = std::max(-63.0f, std::min(63.0f, desired));
        int32_t action[GDX_RL_ACTION_SIZE] = {
            nearest_steer_category(desired), 0, 1, 0, 0,
        };
        pilot_result = fzero_gpu_step_state(&pilot, &config, course.data(), action);
        pilot_decisions++;
    }
    if (!pilot_result.terminated || pilot_result.reason != FZERO_GPU_FINISHED) {
        std::fprintf(stderr,
            "reduced pilot failed: reason=%d decisions=%d frames=%u distance=%.3f "
            "lateral=%.3f yaw=%.6f speed=%.3f\n",
            pilot_result.reason, pilot_decisions, pilot.episode_frame,
            pilot.race_distance, pilot.lateral, pilot.yaw, pilot.speed);
        return 1;
    }
    CHECK_NEAR(pilot_result.learner_reward,
        fzero_gpu_terminal_outcome(FZERO_GPU_FINISHED, pilot.episode_frame,
            config.finish_time_target_frames), 0.0f);

    FZeroGpuConfig no_op_config = config;
    no_op_config.steering_curriculum = 0;
    FZeroGpuState no_op = {};
    fzero_gpu_reset_state(&no_op, &no_op_config, 0);
    FZeroGpuStepResult no_op_result = {};
    int32_t no_op_action[GDX_RL_ACTION_SIZE] = {};
    int no_op_decisions = 0;
    while (!no_op_result.terminated && !no_op_result.truncated && no_op_decisions < 1000) {
        no_op_result = fzero_gpu_step_state(
            &no_op, &no_op_config, course.data(), no_op_action);
        no_op_decisions++;
    }
    CHECK(no_op_result.truncated);
    CHECK(no_op_result.reason == FZERO_GPU_STALLED);
    CHECK(no_op.episode_frame == (uint32_t)no_op_config.stall_frames);
    CHECK_NEAR(no_op_result.learner_reward, 0.0f, 0.0f);

    std::printf("FZERO_GPU_CORE_OK initial_max_error=%.6f dash_max_error=%.9f "
                "pilot_decisions=%d "
                "pilot_frames=%u race_distance=%.3f no_op_decisions=%d\n",
        initial_max_error, dash_observation_max_error, pilot_decisions, pilot.episode_frame,
        pilot.race_distance, no_op_decisions);
    return 0;
}
