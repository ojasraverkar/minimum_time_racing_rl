"""A compact PPO actor-critic implementation in PyTorch (no external RL framework)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    update_epochs: int = 6
    minibatch_size: int = 256
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.005
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if self.value_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("loss coefficients must be nonnegative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")


class ActorCritic(nn.Module):
    """Shared MLP with a Gaussian continuous-action actor and scalar critic."""
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Linear(128, 1)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, observations: torch.Tensor):
        features = self.shared(observations)
        mean = self.actor_mean(features)  # mean is unbounded; tanh is applied to sampled actions
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return mean, std, value


def _squashed_log_prob(distribution: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # Change-of-variables correction for action = tanh(raw_action).
    return distribution.log_prob(raw_action).sum(dim=-1) - torch.log(1.0 - action.square() + 1e-6).sum(dim=-1)


class PPOAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: PPOConfig | None = None,
                 device: str | torch.device | None = None):
        self.config = config or PPOConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network = ActorCritic(observation_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.loaded_ppo_config: dict = {}
        self.loaded_checkpoint_extra: dict = {}

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std, value = self.network(obs)
        distribution = Normal(mean, std)
        raw = mean if deterministic else distribution.sample()
        action = torch.tanh(raw)
        log_prob = _squashed_log_prob(distribution, raw, action)
        return action.squeeze(0).cpu().numpy().astype(np.float32), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def value(self, observation: np.ndarray) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.network(obs)[2].item())

    def update(self, rollout: dict[str, np.ndarray]) -> dict[str, float]:
        cfg = self.config
        obs = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = len(obs)
        if n == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "policy_std_mean": float(torch.exp(self.network.log_std).mean().detach().cpu()),
            }

        indices = np.arange(n)
        # Aggregate diagnostics across every minibatch/update epoch rather than
        # returning only the final minibatch's values.
        metric_sums = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        metric_weight = 0
        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                batch = indices[start:start + cfg.minibatch_size]
                mean, std, value = self.network(obs[batch])
                distribution = Normal(mean, std)

                # Invert tanh to recover the Gaussian pre-transform action.
                clipped_actions = torch.clamp(actions[batch], -0.999999, 0.999999)
                raw_actions = 0.5 * (torch.log1p(clipped_actions) - torch.log1p(-clipped_actions))
                new_log_probs = _squashed_log_prob(distribution, raw_actions, clipped_actions)
                entropy = distribution.entropy().sum(dim=-1).mean()

                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                objective_unclipped = ratio * advantages[batch]
                objective_clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon,
                                                  1.0 + cfg.clip_epsilon) * advantages[batch]
                actor_loss = -torch.minimum(objective_unclipped, objective_clipped).mean()
                critic_loss = 0.5 * (returns[batch] - value).square().mean()
                loss = actor_loss + cfg.value_coefficient * critic_loss - cfg.entropy_coefficient * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                safe_log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
                approximate_kl = ((ratio - 1.0) - safe_log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > cfg.clip_epsilon).float().mean()
                weight = int(len(batch))
                metric_sums["actor_loss"] += float(actor_loss.detach().cpu()) * weight
                metric_sums["critic_loss"] += float(critic_loss.detach().cpu()) * weight
                metric_sums["entropy"] += float(entropy.detach().cpu()) * weight
                metric_sums["approx_kl"] += float(approximate_kl.detach().cpu()) * weight
                metric_sums["clip_fraction"] += float(clip_fraction.detach().cpu()) * weight
                metric_weight += weight

        diagnostics = {key: value / max(metric_weight, 1) for key, value in metric_sums.items()}
        diagnostics["policy_std_mean"] = float(torch.exp(self.network.log_std).mean().detach().cpu())
        return diagnostics

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "model_state_dict": self.network.state_dict(),
            "observation_dim": self.network.shared[0].in_features,
            "action_dim": self.network.actor_mean.out_features,
            "ppo_config": self.config.__dict__,
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load(self, path: str) -> dict:
        # weights_only=False is explicit because checkpoints also contain simple metadata.
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # compatibility with older Torch releases
            payload = torch.load(path, map_location=self.device)
        self.network.load_state_dict(payload["model_state_dict"])
        self.loaded_ppo_config = dict(payload.get("ppo_config", {}))
        if self.loaded_ppo_config:
            # Restore the saved experiment configuration for correct metadata if
            # this agent is used for further training as well as evaluation.
            self.config = PPOConfig(**self.loaded_ppo_config)
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.learning_rate
        self.loaded_checkpoint_extra = dict(payload.get("extra", {}))
        return dict(self.loaded_checkpoint_extra)


def calculate_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
                  terminated: np.ndarray, episode_ends: np.ndarray,
                  gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE with separate termination and episode-boundary masks.

    True terminal states do not bootstrap. Time-limit truncations bootstrap the
    final state's value but stop the GAE recursion from crossing into a reset episode.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=np.float32)
    episode_ends = np.asarray(episode_ends, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - episode_ends[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values
