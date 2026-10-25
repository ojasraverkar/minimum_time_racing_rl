import numpy as np

from racing_rl.env import RacingEnv, make_track
from train import (
    FINAL_TEST_TRACK_SEEDS,
    TRAIN_TRACK_SEEDS,
    VALIDATION_TRACK_SEEDS,
    baseline_action,
    run_episode,
)


def test_training_validation_and_final_test_seed_sets_are_disjoint():
    assert len(VALIDATION_TRACK_SEEDS) == 12
    assert len(FINAL_TEST_TRACK_SEEDS) == 10
    assert not (set(TRAIN_TRACK_SEEDS) & set(VALIDATION_TRACK_SEEDS))
    assert not (set(TRAIN_TRACK_SEEDS) & set(FINAL_TEST_TRACK_SEEDS))
    assert not (set(VALIDATION_TRACK_SEEDS) & set(FINAL_TEST_TRACK_SEEDS))
    assert len(set(VALIDATION_TRACK_SEEDS)) == len(VALIDATION_TRACK_SEEDS)
    assert len(set(FINAL_TEST_TRACK_SEEDS)) == len(FINAL_TEST_TRACK_SEEDS)


def test_run_episode_step_diagnostics_capture_state_action_and_terminal_fields():
    env = RacingEnv(track=make_track(42), max_steps=5)
    result = run_episode(
        env,
        baseline_action,
        track_seed=42,
        capture_steps=True,
        controller_label="geometric_baseline",
    )
    rows = result["step_diagnostics"]
    assert len(rows) == result["steps"]
    assert rows
    expected = {
        "track_seed", "controller", "step", "sim_time_s",
        "pre_speed_mps", "pre_heading_error_rad", "pre_signed_lateral_offset_m",
        "action_speed_command_norm", "action_heading_command_norm",
        "post_speed_mps", "post_heading_error_rad", "post_signed_lateral_offset_m",
        "post_distance_to_centerline_m", "post_progress_fraction", "post_lap_time_s",
        "off_track", "lap_complete",
        "pre_ray_0_m", "post_ray_0_m",
        "pre_curvature_10m_norm", "post_curvature_38m_per_m",
    }
    assert expected.issubset(rows[0])
    assert all(np.isfinite(float(row["sim_time_s"])) for row in rows)
    assert [row["step"] for row in rows] == list(range(1, len(rows) + 1))
    if result["off_track"]:
        assert rows[-1]["off_track"] is True
    env.close()
