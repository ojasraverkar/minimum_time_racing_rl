"""Run this before PPO training to validate the Gymnasium environment."""
import numpy as np
from gymnasium.utils.env_checker import check_env
from racing_rl.env import RacingEnv, make_track


def main():
    env = RacingEnv(track=make_track(123, "validation"), max_steps=80, seed=123)
    observation, info = env.reset(seed=123)
    assert env.observation_space.contains(observation), "Reset observation is outside its declared space"
    assert env.action_space.shape == (2,), "Expected [Delta-v, Delta-heading] action"
    assert observation.shape == (14,), "Expected 14 observations"
    assert env.speed == 0.0, "Vehicle should start stationary"

    # No-op action must not add speed or move a stationary vehicle.
    x0, y0 = env.x, env.y
    observation, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert env.speed == 0.0, "No-op action unexpectedly changed speed"
    assert abs(env.x - x0) < 1e-9 and abs(env.y - y0) < 1e-9, "Stationary no-op unexpectedly moved the vehicle"
    assert env.observation_space.contains(observation), "Step observation is outside its declared space"

    # Positive Delta-v should increase speed; positive Delta-heading changes heading.
    old_heading = env.heading
    observation, reward, terminated, truncated, info = env.step(np.asarray([1.0, 1.0], dtype=np.float32))
    assert env.speed > 0.0, "Positive Delta-v did not increase speed"
    assert env.heading != old_heading, "Delta-heading did not change heading"
    assert env.observation_space.contains(observation), "Observation outside space after action"

    # Use Gymnasium's own API validation.
    check_env(env, skip_render_check=True)
    env.close()
    print("ENVIRONMENT CHECK PASSED")
    print("Observation space:", (14,))
    print("Action space:", (2,), "= normalized Delta-v, normalized Delta-heading")
    print("Track width:", 12.0, "m")
    print("Ray count:", 7)


if __name__ == "__main__":
    main()
