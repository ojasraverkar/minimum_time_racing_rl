import numpy as np
from gymnasium.utils.env_checker import check_env
from racing_rl.env import RacingEnv, make_track


def test_reset_observation_and_stationary_noop():
    env = RacingEnv(track=make_track(42), max_steps=100)
    obs, info = env.reset(seed=42)
    assert obs.shape == (14,)
    assert env.observation_space.contains(obs)
    assert env.speed == 0.0
    x0, y0 = env.x, env.y
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert env.speed == 0.0
    assert (env.x, env.y) == (x0, y0)
    assert env.observation_space.contains(obs)
    env.close()


def test_delta_v_and_heading_actions():
    env = RacingEnv(track=make_track(101), max_steps=100)
    env.reset(seed=1)
    heading_before = env.heading
    obs, *_ = env.step(np.asarray([1.0, 1.0], dtype=np.float32))
    assert env.speed > 0.0
    assert env.heading != heading_before
    assert env.observation_space.contains(obs)
    env.close()


def test_gymnasium_contract():
    env = RacingEnv(track=make_track(202), max_steps=100)
    check_env(env, skip_render_check=True)
    env.close()
