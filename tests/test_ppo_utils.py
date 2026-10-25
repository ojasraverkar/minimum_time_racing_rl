import numpy as np
import torch

from racing_rl.ppo import PPOAgent, PPOConfig, calculate_gae


def test_gae_bootstraps_time_limit_truncation_but_not_true_termination():
    # At a truncation, bootstrap from next_value but stop the trace at the episode edge.
    advantages, returns = calculate_gae(
        rewards=np.asarray([1.0]),
        values=np.asarray([0.0]),
        next_values=np.asarray([2.0]),
        terminated=np.asarray([False]),
        episode_ends=np.asarray([True]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages, [2.98], rtol=1e-6)
    np.testing.assert_allclose(returns, [2.98], rtol=1e-6)

    # A true terminal state must not bootstrap even if a next value is supplied.
    advantages, returns = calculate_gae(
        rewards=np.asarray([1.0]),
        values=np.asarray([0.0]),
        next_values=np.asarray([99.0]),
        terminated=np.asarray([True]),
        episode_ends=np.asarray([True]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages, [1.0], rtol=1e-6)
    np.testing.assert_allclose(returns, [1.0], rtol=1e-6)


def test_ppo_checkpoint_can_be_saved_and_reloaded(tmp_path):
    torch.manual_seed(7)
    saved_config = PPOConfig(gamma=0.9985, gae_lambda=0.99)
    agent = PPOAgent(observation_dim=14, action_dim=2, config=saved_config, device="cpu")
    observation = np.zeros(14, dtype=np.float32)
    action_before, _, value_before = agent.act(observation, deterministic=True)

    checkpoint = tmp_path / "checkpoint.pt"
    agent.save(str(checkpoint), extra={"seed": 7, "timesteps": 10})

    reloaded = PPOAgent(observation_dim=14, action_dim=2, config=PPOConfig(), device="cpu")
    metadata = reloaded.load(str(checkpoint))
    action_after, _, value_after = reloaded.act(observation, deterministic=True)

    np.testing.assert_allclose(action_after, action_before, rtol=0, atol=1e-7)
    assert abs(value_after - value_before) < 1e-7
    assert metadata == {"seed": 7, "timesteps": 10}
    assert reloaded.config.gamma == 0.9985
    assert reloaded.config.gae_lambda == 0.99


def test_submission_default_ppo_config():
    config = PPOConfig()
    assert config.gamma == 0.995
    assert config.gae_lambda == 0.95


def test_ppo_config_validates_discount_and_gae_ranges():
    assert PPOConfig(gamma=0.9985, gae_lambda=0.99).gamma == 0.9985
    for kwargs in ({"gamma": 0.0}, {"gamma": 1.1}, {"gae_lambda": -0.1}, {"gae_lambda": 1.1}):
        try:
            PPOConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid PPO config was accepted: {kwargs}")


def test_ppo_update_returns_aggregated_diagnostics():
    torch.manual_seed(5)
    np.random.seed(5)
    config = PPOConfig(update_epochs=2, minibatch_size=8)
    agent = PPOAgent(observation_dim=14, action_dim=2, config=config, device="cpu")
    observations = np.random.normal(size=(16, 14)).astype(np.float32)
    actions = []
    log_probs = []
    for obs in observations:
        action, log_prob, _ = agent.act(obs)
        actions.append(action)
        log_probs.append(log_prob)
    rollout = {
        "observations": observations,
        "actions": np.asarray(actions, dtype=np.float32),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "returns": np.random.normal(size=16).astype(np.float32),
        "advantages": np.random.normal(size=16).astype(np.float32),
    }
    diagnostics = agent.update(rollout)
    for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction", "policy_std_mean"):
        assert key in diagnostics
        assert np.isfinite(diagnostics[key])
    assert 0.0 <= diagnostics["clip_fraction"] <= 1.0
    assert diagnostics["policy_std_mean"] > 0.0
