#pragma once

#include <math.h>
#include <stdint.h>

#include "../../../G-Diffuser/port/rl/gdx_rl_env.h"

#if defined(__CUDACC__)
#define FZERO_GPU_HD __host__ __device__ __forceinline__
#else
#define FZERO_GPU_HD inline
#endif

enum {
    FZERO_GPU_RUNNING = 0,
    FZERO_GPU_FINISHED = 1,
    FZERO_GPU_SPINNING_OUT = 2,
    FZERO_GPU_FALLING = 3,
    FZERO_GPU_CRASHED = 4,
    FZERO_GPU_RETIRED = 5,
    FZERO_GPU_TIME_LIMIT = 6,
    FZERO_GPU_STALLED = 7,
};

static constexpr float FZERO_GPU_MAX_SPEED = 138.9f;
typedef struct FZeroGpuVec3 {
    float x;
    float y;
    float z;
} FZeroGpuVec3;

typedef struct FZeroGpuTrackFrame {
    FZeroGpuVec3 position;
    FZeroGpuVec3 forward;
    FZeroGpuVec3 up;
    FZeroGpuVec3 side;
    float radius_left;
    float radius_right;
    uint32_t track_segment_info;
    int32_t segment_index;
} FZeroGpuTrackFrame;

typedef struct FZeroGpuConfig {
    int32_t sample_count;
    int32_t dash_pad_count;
    int32_t laps;
    int32_t action_repeat;
    int32_t max_episode_frames;
    int32_t stall_frames;
    int32_t steering_curriculum;
    float course_length;
    float sample_spacing;
    float stall_progress;
    float reward_scale;
    float finish_time_target_frames;
    float engine_target_speed;
    float engine_acceleration;
    float forward_drag;
    float brake_drag;
    float steering_rate;
    float wall_margin;
    float initial_race_distance;
    float initial_lap_distance;
    float initial_lateral;
    float initial_vertical;
    float initial_speed;
    float initial_energy;
    float max_energy;
    uint32_t seed;
} FZeroGpuConfig;

typedef struct FZeroGpuState {
    float race_distance;
    float lap_distance;
    float lateral;
    float vertical;
    float yaw;
    float speed;
    float energy;
    float best_distance;
    float progress_checkpoint;
    float episode_return;
    float learner_return;
    uint32_t episode_frame;
    uint32_t frames_without_progress;
    uint32_t episode_decisions;
    uint32_t reset_count;
    uint32_t rng;
    int32_t boost_timer;
    int32_t laps_completed;
    int32_t previous_action[GDX_RL_ACTION_SIZE];
    uint32_t steer_counts[GDX_RL_STEER_BINS];
    uint32_t pitch_counts[GDX_RL_PITCH_BINS];
    uint32_t throttle_on_count;
    uint32_t brake_on_count;
    uint32_t boost_on_count;
} FZeroGpuState;

typedef struct FZeroGpuStepResult {
    float raw_reward;
    float learner_reward;
    int32_t reason;
    int32_t terminated;
    int32_t truncated;
} FZeroGpuStepResult;

FZERO_GPU_HD float fzero_gpu_clampf(float value, float low, float high) {
    return fminf(high, fmaxf(low, value));
}

FZERO_GPU_HD float fzero_gpu_clamp_unit(float value) {
    return fzero_gpu_clampf(value, -1.0f, 1.0f);
}

FZERO_GPU_HD float fzero_gpu_clamp_zero_one(float value) {
    return fzero_gpu_clampf(value, 0.0f, 1.0f);
}

FZERO_GPU_HD FZeroGpuVec3 fzero_gpu_vec3(float x, float y, float z) {
    FZeroGpuVec3 result = {x, y, z};
    return result;
}

FZERO_GPU_HD FZeroGpuVec3 fzero_gpu_add(FZeroGpuVec3 a, FZeroGpuVec3 b) {
    return fzero_gpu_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

FZERO_GPU_HD FZeroGpuVec3 fzero_gpu_sub(FZeroGpuVec3 a, FZeroGpuVec3 b) {
    return fzero_gpu_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

FZERO_GPU_HD FZeroGpuVec3 fzero_gpu_scale(FZeroGpuVec3 value, float scale) {
    return fzero_gpu_vec3(value.x * scale, value.y * scale, value.z * scale);
}

FZERO_GPU_HD float fzero_gpu_dot(FZeroGpuVec3 a, FZeroGpuVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

FZERO_GPU_HD FZeroGpuVec3 fzero_gpu_cross(FZeroGpuVec3 a, FZeroGpuVec3 b) {
    return fzero_gpu_vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

FZERO_GPU_HD FZeroGpuVec3 fzero_gpu_normalize(FZeroGpuVec3 value) {
    float length_squared = fzero_gpu_dot(value, value);
    if (!(length_squared > 1.0e-12f)) {
        return fzero_gpu_vec3(1.0f, 0.0f, 0.0f);
    }
    return fzero_gpu_scale(value, 1.0f / sqrtf(length_squared));
}

FZERO_GPU_HD int fzero_gpu_steer_value(int category) {
    switch (category) {
        case 1: return -13;
        case 2: return 13;
        case 3: return -25;
        case 4: return 25;
        case 5: return -38;
        case 6: return 38;
        case 7: return -50;
        case 8: return 50;
        case 9: return -63;
        case 10: return 63;
        default: return 0;
    }
}

FZERO_GPU_HD int fzero_gpu_pitch_value(int category) {
    switch (category) {
        case 1: return -32;
        case 2: return 32;
        case 3: return -63;
        case 4: return 63;
        default: return 0;
    }
}

FZERO_GPU_HD float fzero_gpu_lookahead(int index) {
    switch (index) {
        case 0: return 250.0f;
        case 1: return 500.0f;
        case 2: return 1000.0f;
        case 3: return 2000.0f;
        case 4: return 4000.0f;
        default: return 8000.0f;
    }
}

FZERO_GPU_HD FZeroGpuVec3 fzero_gpu_lerp_vec3(
        const float a[3], const float b[3], float t) {
    return fzero_gpu_vec3(
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t);
}

FZERO_GPU_HD float fzero_gpu_wrap(float distance, float length) {
    float wrapped = fmodf(distance, length);
    return wrapped < 0.0f ? wrapped + length : wrapped;
}

FZERO_GPU_HD void fzero_gpu_nearest_dash_observation(
        float lap_distance, const FZeroGpuConfig* config,
        const GdxRlDashPad* dash_pads, float output[4]) {
    output[0] = 0.0f;
    output[1] = 0.0f;
    output[2] = 0.0f;
    output[3] = 0.0f;
    if (config->dash_pad_count <= 0 || dash_pads == nullptr ||
            !(config->course_length > 0.0f)) {
        return;
    }

    float best_delta = config->course_length;
    int best_index = 0;
    for (int i = 0; i < config->dash_pad_count; ++i) {
        const float delta = fzero_gpu_wrap(
            dash_pads[i].entry_distance - lap_distance, config->course_length);
        if (delta < best_delta) {
            best_delta = delta;
            best_index = i;
        }
    }

    const GdxRlDashPad* nearest = &dash_pads[best_index];
    output[0] = 1.0f;
    output[1] = fzero_gpu_clamp_zero_one(best_delta / 8000.0f);
    output[2] = fzero_gpu_clamp_unit(nearest->target);
    output[3] = fzero_gpu_clamp_zero_one(nearest->width);
}

FZERO_GPU_HD float fzero_gpu_terminal_outcome(
        int reason, uint32_t episode_frame, float finish_time_target_frames) {
    if (reason == FZERO_GPU_FINISHED) {
        if (!(finish_time_target_frames > 0.0f)) {
            return 1.0f;
        }
        if (episode_frame == 0) {
            return 0.0f;
        }
        return fzero_gpu_clamp_zero_one(
            finish_time_target_frames / (float)episode_frame);
    }
    if (reason == FZERO_GPU_TIME_LIMIT || reason == FZERO_GPU_STALLED) {
        return 0.0f;
    }
    return -1.0f;
}

FZERO_GPU_HD float fzero_gpu_learner_reward(
        const FZeroGpuStepResult* result, const FZeroGpuConfig* config,
        uint32_t episode_frame) {
    if (result->terminated || result->truncated) {
        return fzero_gpu_terminal_outcome(
            result->reason, episode_frame, config->finish_time_target_frames);
    }
    return result->raw_reward * config->reward_scale;
}

FZERO_GPU_HD float fzero_gpu_encode_radius(float radius) {
    radius = fmaxf(0.0f, radius);
    return fzero_gpu_clamp_zero_one(radius / (radius + 250.0f));
}

FZERO_GPU_HD void fzero_gpu_sample_course(const GdxRlCourseSample* samples,
        const FZeroGpuConfig* config, float distance, FZeroGpuTrackFrame* frame) {
    float wrapped = fzero_gpu_wrap(distance, config->course_length);
    float sample_position = wrapped / config->sample_spacing;
    int index0 = (int)floorf(sample_position);
    if (index0 >= config->sample_count) {
        index0 = config->sample_count - 1;
    }
    int index1 = index0 + 1;
    if (index1 == config->sample_count) {
        index1 = 0;
    }
    float t = sample_position - (float)index0;
    const GdxRlCourseSample* a = &samples[index0];
    const GdxRlCourseSample* b = &samples[index1];

    frame->position = fzero_gpu_lerp_vec3(a->position, b->position, t);
    frame->forward = fzero_gpu_normalize(fzero_gpu_lerp_vec3(a->forward, b->forward, t));
    FZeroGpuVec3 interpolated_up =
        fzero_gpu_normalize(fzero_gpu_lerp_vec3(a->up, b->up, t));
    frame->side = fzero_gpu_normalize(fzero_gpu_cross(interpolated_up, frame->forward));
    frame->up = fzero_gpu_normalize(fzero_gpu_cross(frame->forward, frame->side));
    frame->radius_left = a->radius_left + (b->radius_left - a->radius_left) * t;
    frame->radius_right = a->radius_right + (b->radius_right - a->radius_right) * t;
    const GdxRlCourseSample* nearest = t < 0.5f ? a : b;
    frame->track_segment_info = nearest->track_segment_info;
    frame->segment_index = nearest->segment_index;
}

FZERO_GPU_HD int fzero_gpu_completed_laps(float race_distance, float course_length) {
    if (race_distance < course_length) {
        return 1;
    }
    return 1 + (int)floorf(race_distance / course_length);
}

FZERO_GPU_HD void fzero_gpu_reset_state(
        FZeroGpuState* state, const FZeroGpuConfig* config, uint32_t env_index) {
    uint32_t next_reset = state->reset_count + 1u;
    uint32_t reset_seed = config->seed + env_index * 1000003u + state->reset_count;
    state->race_distance = config->initial_race_distance;
    state->lap_distance = config->initial_lap_distance;
    state->lateral = config->initial_lateral;
    state->vertical = config->initial_vertical;
    state->yaw = 0.0f;
    state->speed = config->initial_speed;
    state->energy = config->initial_energy;
    state->best_distance = state->race_distance;
    state->progress_checkpoint = state->race_distance;
    state->episode_return = 0.0f;
    state->learner_return = 0.0f;
    state->episode_frame = 0;
    state->frames_without_progress = 0;
    state->episode_decisions = 0;
    state->boost_timer = 0;
    state->laps_completed = 1;
    for (int i = 0; i < GDX_RL_ACTION_SIZE; ++i) {
        state->previous_action[i] = 0;
    }
    for (int i = 0; i < GDX_RL_STEER_BINS; ++i) {
        state->steer_counts[i] = 0;
    }
    for (int i = 0; i < GDX_RL_PITCH_BINS; ++i) {
        state->pitch_counts[i] = 0;
    }
    state->throttle_on_count = 0;
    state->brake_on_count = 0;
    state->boost_on_count = 0;
    state->reset_count = next_reset;
    state->rng = reset_seed != 0 ? reset_seed : 0xA341316Cu;
}

FZERO_GPU_HD void fzero_gpu_apply_curriculum(
        const FZeroGpuConfig* config, int32_t action[GDX_RL_ACTION_SIZE]) {
    if (!config->steering_curriculum) {
        return;
    }
    action[1] = 0;
    action[2] = 1;
    action[3] = 0;
    action[4] = 0;
}

FZERO_GPU_HD void fzero_gpu_substep(FZeroGpuState* state,
        const FZeroGpuConfig* config, const GdxRlCourseSample* samples,
        const int32_t action[GDX_RL_ACTION_SIZE], int repeat,
        FZeroGpuStepResult* result) {
    FZeroGpuTrackFrame old_frame;
    fzero_gpu_sample_course(samples, config, state->lap_distance, &old_frame);

    const float steer = (float)fzero_gpu_steer_value(action[0]) / 63.0f;
    state->yaw -= steer * config->steering_rate;

    if (action[2] != 0) {
        float headroom = fmaxf(0.0f, 1.0f - state->speed / config->engine_target_speed);
        state->speed += config->engine_acceleration * headroom;
    }
    state->speed *= config->forward_drag;
    if (action[3] != 0) {
        state->speed *= config->brake_drag;
    }

    const bool can_boost = state->race_distance >= config->course_length;
    if (action[4] != 0 && repeat == 0 && state->boost_timer == 0 &&
            can_boost && state->energy > 0.0f) {
        state->boost_timer = 100;
    }
    if (state->boost_timer > 0) {
        state->speed = fminf(FZERO_GPU_MAX_SPEED, state->speed + 0.18f);
        state->energy = fmaxf(0.0f, state->energy - 0.06f);
        state->boost_timer--;
    }
    state->speed = fzero_gpu_clampf(state->speed, 0.0f, FZERO_GPU_MAX_SPEED);

    float forward_delta = state->speed * cosf(state->yaw);
    float lateral_delta = state->speed * sinf(state->yaw);
    state->race_distance += forward_delta;
    state->lap_distance = fzero_gpu_wrap(state->lap_distance + forward_delta, config->course_length);
    state->lateral += lateral_delta;

    FZeroGpuTrackFrame new_frame;
    fzero_gpu_sample_course(samples, config, state->lap_distance, &new_frame);
    float track_turn = atan2f(
        fzero_gpu_dot(new_frame.forward, old_frame.side),
        fzero_gpu_dot(new_frame.forward, old_frame.forward));
    state->yaw -= track_turn;
    if (state->yaw > 3.14159265358979323846f) {
        state->yaw -= 6.28318530717958647692f;
    } else if (state->yaw < -3.14159265358979323846f) {
        state->yaw += 6.28318530717958647692f;
    }

    float selected_radius = state->lateral >= 0.0f
        ? new_frame.radius_left : new_frame.radius_right;
    float fall_limit = fmaxf(1.0f, selected_radius * config->wall_margin);
    if (fabsf(state->lateral) > fall_limit) {
        result->reason = FZERO_GPU_FALLING;
        result->terminated = 1;
        result->raw_reward -= 25.0f;
    }

    state->episode_frame++;
    result->raw_reward += forward_delta / 1000.0f;
    int completed_laps = fzero_gpu_completed_laps(state->race_distance, config->course_length);
    if (completed_laps > state->laps_completed) {
        result->raw_reward += 2.0f * (float)(completed_laps - state->laps_completed);
        state->laps_completed = completed_laps;
    }

    if (!result->terminated && state->race_distance >=
            config->course_length * (float)config->laps) {
        result->reason = FZERO_GPU_FINISHED;
        result->terminated = 1;
        result->raw_reward += 25.0f;
    }
    if (result->terminated) {
        return;
    }

    if (state->race_distance > state->best_distance) {
        state->best_distance = state->race_distance;
    }
    if (state->best_distance >= state->progress_checkpoint + config->stall_progress) {
        state->progress_checkpoint = state->best_distance;
        state->frames_without_progress = 0;
    } else {
        state->frames_without_progress++;
    }
    if (state->episode_frame >= (uint32_t)config->max_episode_frames) {
        result->reason = FZERO_GPU_TIME_LIMIT;
        result->truncated = 1;
    } else if (config->stall_frames > 0 &&
            state->frames_without_progress >= (uint32_t)config->stall_frames) {
        result->reason = FZERO_GPU_STALLED;
        result->truncated = 1;
    }
}

FZERO_GPU_HD FZeroGpuStepResult fzero_gpu_step_state(FZeroGpuState* state,
        const FZeroGpuConfig* config, const GdxRlCourseSample* samples,
        int32_t action[GDX_RL_ACTION_SIZE]) {
    fzero_gpu_apply_curriculum(config, action);
    FZeroGpuStepResult result = {};
    for (int repeat = 0; repeat < config->action_repeat; ++repeat) {
        fzero_gpu_substep(state, config, samples, action, repeat, &result);
        if (result.terminated || result.truncated) {
            break;
        }
    }
    for (int i = 0; i < GDX_RL_ACTION_SIZE; ++i) {
        state->previous_action[i] = action[i];
    }
    state->episode_return += result.raw_reward;
    result.learner_reward = fzero_gpu_learner_reward(&result, config, state->episode_frame);
    state->learner_return += result.learner_reward;
    state->episode_decisions++;
    state->steer_counts[action[0]]++;
    state->pitch_counts[action[1]]++;
    state->throttle_on_count += (uint32_t)action[2];
    state->brake_on_count += (uint32_t)action[3];
    state->boost_on_count += (uint32_t)action[4];
    return result;
}

template <typename Store>
FZERO_GPU_HD void fzero_gpu_observe(const FZeroGpuState* state,
        const FZeroGpuConfig* config, const GdxRlCourseSample* samples,
        const GdxRlDashPad* dash_pads, Store store) {
    FZeroGpuTrackFrame current;
    fzero_gpu_sample_course(samples, config, state->lap_distance, &current);
    const float cosine = cosf(state->yaw);
    const float sine = sinf(state->yaw);
    FZeroGpuVec3 vehicle_forward = fzero_gpu_add(
        fzero_gpu_scale(current.forward, cosine),
        fzero_gpu_scale(current.side, sine));
    FZeroGpuVec3 velocity = fzero_gpu_scale(vehicle_forward, state->speed);
    FZeroGpuVec3 vehicle_position = fzero_gpu_add(
        current.position,
        fzero_gpu_add(fzero_gpu_scale(current.side, state->lateral),
                      fzero_gpu_scale(current.up, state->vertical)));
    const float lateral_radius = fmaxf(1.0f, state->lateral >= 0.0f
        ? current.radius_left : current.radius_right);
    const float vertical_radius = fmaxf(1.0f, fmaxf(current.radius_left, current.radius_right));
    const float race_length = config->course_length * (float)config->laps;
    int index = 0;

    store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(velocity, current.forward) / FZERO_GPU_MAX_SPEED));
    store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(velocity, current.side) / FZERO_GPU_MAX_SPEED));
    store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(velocity, current.up) / FZERO_GPU_MAX_SPEED));
    store(index++, fzero_gpu_clamp_zero_one(state->speed / FZERO_GPU_MAX_SPEED));
    store(index++, fzero_gpu_clamp_unit(
        fzero_gpu_clampf(state->lateral / lateral_radius, -1.25f, 1.25f) / 1.25f));
    store(index++, fzero_gpu_clamp_unit(state->vertical / vertical_radius));
    store(index++, fzero_gpu_encode_radius(current.radius_left));
    store(index++, fzero_gpu_encode_radius(current.radius_right));

    store(index++, fzero_gpu_clamp_unit(cosine));
    store(index++, fzero_gpu_clamp_unit(sine));
    store(index++, 0.0f);
    store(index++, 1.0f);
    store(index++, 0.0f);
    store(index++, config->max_energy > 0.0f
        ? fzero_gpu_clamp_zero_one(state->energy / config->max_energy) : 0.0f);
    store(index++, fzero_gpu_clamp_zero_one((float)state->boost_timer / 100.0f));
    store(index++, fzero_gpu_clamp_zero_one(state->lap_distance / config->course_length));
    store(index++, fzero_gpu_clamp_zero_one(state->race_distance / race_length));
    store(index++, state->race_distance >= config->course_length ? 1.0f : 0.0f);
    store(index++, 0.0f);
    store(index++, state->boost_timer > 0 ? 1.0f : 0.0f);
    store(index++, 0.0f);
    store(index++, 0.0f);
    store(index++, 0.0f);
    store(index++, 0.0f);
    store(index++, 0.0f);
    store(index++, 0.0f);

    store(index++, (float)fzero_gpu_steer_value(state->previous_action[0]) / 63.0f);
    store(index++, (float)fzero_gpu_pitch_value(state->previous_action[1]) / 63.0f);
    store(index++, (float)state->previous_action[2]);
    store(index++, (float)state->previous_action[3]);
    store(index++, (float)state->previous_action[4]);

    for (int lookahead = 0; lookahead < 6; ++lookahead) {
        const float offset = fzero_gpu_lookahead(lookahead);
        FZeroGpuTrackFrame future;
        fzero_gpu_sample_course(samples, config, state->lap_distance + offset, &future);
        FZeroGpuVec3 displacement = fzero_gpu_sub(future.position, vehicle_position);
        store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(displacement, current.forward) / offset));
        store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(displacement, current.side) / offset));
        store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(displacement, current.up) / offset));
        store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(future.forward, current.forward)));
        store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(future.forward, current.side)));
        store(index++, fzero_gpu_clamp_unit(fzero_gpu_dot(future.forward, current.up)));
        store(index++, fzero_gpu_encode_radius(future.radius_left));
        store(index++, fzero_gpu_encode_radius(future.radius_right));
    }

    float nearest_dash[4];
    fzero_gpu_nearest_dash_observation(
        state->lap_distance, config, dash_pads, nearest_dash);
    for (int component = 0; component < 4; ++component) {
        store(index++, nearest_dash[component]);
    }
}

#undef FZERO_GPU_HD
