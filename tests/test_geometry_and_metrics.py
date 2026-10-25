import numpy as np

from racing_rl.env import RacingEnv, TRACK_WIDTH, Track, make_track, wrapped_progress_delta
from train import mean_trajectory_speed_mps


def _signed_area(track: Track) -> float:
    return 0.5 * float(np.sum(
        track.x * np.roll(track.y, -1) - np.roll(track.x, -1) * track.y
    ))


def test_boundary_labels_follow_counter_clockwise_orientation():
    track = make_track(42)
    assert _signed_area(track) > 0.0
    half_width = track.width / 2.0
    np.testing.assert_allclose(track.inner_x, track.x + half_width * track.nx)
    np.testing.assert_allclose(track.inner_y, track.y + half_width * track.ny)
    np.testing.assert_allclose(track.outer_x, track.x - half_width * track.nx)
    np.testing.assert_allclose(track.outer_y, track.y - half_width * track.ny)


def test_boundary_labels_follow_clockwise_orientation():
    source = make_track(42)
    track = Track(x=source.x[::-1], y=source.y[::-1], width=TRACK_WIDTH)
    assert _signed_area(track) < 0.0
    half_width = track.width / 2.0
    np.testing.assert_allclose(track.outer_x, track.x + half_width * track.nx)
    np.testing.assert_allclose(track.outer_y, track.y + half_width * track.ny)
    np.testing.assert_allclose(track.inner_x, track.x - half_width * track.nx)
    np.testing.assert_allclose(track.inner_y, track.y - half_width * track.ny)


def test_mean_speed_uses_travelled_path_distance():
    # Two 5 m segments in 2 seconds -> 5 m/s, irrespective of net displacement.
    x = np.asarray([0.0, 3.0, 6.0])
    y = np.asarray([0.0, 4.0, 8.0])
    assert mean_trajectory_speed_mps(x, y, elapsed_seconds=2.0) == 5.0


def test_mean_speed_handles_single_point_and_invalid_inputs():
    assert mean_trajectory_speed_mps([1.0], [2.0], elapsed_seconds=1.0) == 0.0
    assert np.isnan(mean_trajectory_speed_mps([1.0], [2.0], elapsed_seconds=0.0))
    try:
        mean_trajectory_speed_mps([0.0, 1.0], [0.0], elapsed_seconds=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched x/y trajectory lengths should raise ValueError")


def test_progress_wraparound_is_signed_and_shortest():
    assert wrapped_progress_delta(99.5, 0.5, 100.0) == 1.0
    assert wrapped_progress_delta(0.5, 99.5, 100.0) == -1.0
    assert wrapped_progress_delta(10.0, 12.0, 100.0) == 2.0


def test_signed_curvature_changes_sign_with_track_traversal_direction():
    # A broad ellipse avoids the degenerate offset geometry of a very tight loop.
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    x = 40.0 * np.cos(theta)
    y = 30.0 * np.sin(theta)
    counter_clockwise = Track(x=x, y=y, width=TRACK_WIDTH)
    clockwise = Track(x=x[::-1], y=y[::-1], width=TRACK_WIDTH)

    assert np.all(counter_clockwise.curvature > 0.0)
    assert np.all(clockwise.curvature < 0.0)

    ccw_env = RacingEnv(track=counter_clockwise)
    cw_env = RacingEnv(track=clockwise)
    ccw_obs, _ = ccw_env.reset(seed=3)
    cw_obs, _ = cw_env.reset(seed=3)

    assert ccw_env.observation_space.contains(ccw_obs)
    assert cw_env.observation_space.contains(cw_obs)
    assert np.all(ccw_obs[11:14] > 0.0)
    assert np.all(cw_obs[11:14] < 0.0)
    np.testing.assert_array_equal(ccw_env.observation_space.low[11:14], [-1.0, -1.0, -1.0])
    ccw_env.close()
    cw_env.close()
