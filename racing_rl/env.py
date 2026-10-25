"""Gymnasium environment for lightweight minimum-time racing research.

Aligned with the project brief:
- 2D point-mass model using vector kinematics (not a detailed car model)
- continuous action commands for speed change (Delta-v) and heading change (Delta-theta)
- geometric observations including current speed, heading/lateral position, curvature preview,
  and seven simulated raycasts to track boundaries
- progress reward plus a per-time-step penalty; no centreline-attraction reward
- off-track termination and lap completion

This module intentionally contains no PPO/training code or GUI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Simulation parameters
DT = 1.0 / 30.0
TRACK_WIDTH = 12.0
HALF_TRACK_WIDTH = TRACK_WIDTH / 2.0
N_TRACK_POINTS = 240
MAX_SPEED = 22.0
MAX_ACCELERATION = 5.0       # m/s^2, used to scale Delta-v actions
MAX_HEADING_RATE = 1.8       # rad/s, used to scale Delta-theta actions
MAX_DELTA_V = MAX_ACCELERATION * DT
MAX_DELTA_HEADING = MAX_HEADING_RATE * DT
RAY_ANGLES_DEG = (-90, -60, -30, 0, 30, 60, 90)
MAX_RAY_DISTANCE = 32.0
LOOKAHEAD_DISTANCES = (10.0, 22.0, 38.0)
CURVATURE_SCALE = 0.04
OBSERVATION_VERSION = "racing_obs_signed_curvature_v2"

# Reward scales: progress adds to 100 over one completed lap; elapsed seconds cost 1 reward/second.
PROGRESS_REWARD_SCALE = 100.0
TIME_PENALTY_PER_SECOND = 1.0
OFF_TRACK_PENALTY = 50.0
LAP_COMPLETION_BONUS = 15.0


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass
class Track:
    """Closed track centreline and boundary geometry, sampled as line segments."""
    x: np.ndarray
    y: np.ndarray
    name: str = "track"
    width: float = TRACK_WIDTH

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.x.ndim != 1 or self.y.ndim != 1 or len(self.x) != len(self.y):
            raise ValueError("Track x and y must be same-length 1D arrays")
        if len(self.x) < 40:
            raise ValueError("Track must contain at least 40 points")
        if self.width <= 0:
            raise ValueError("Track width must be positive")

        x_next = np.roll(self.x, -1)
        y_next = np.roll(self.y, -1)
        self.seg_dx = x_next - self.x
        self.seg_dy = y_next - self.y
        self.seg_len = np.hypot(self.seg_dx, self.seg_dy)
        self.seg_len2 = np.maximum(self.seg_dx**2 + self.seg_dy**2, 1e-10)
        self.length = float(self.seg_len.sum())
        self.s = np.zeros(len(self.x), dtype=np.float64)
        self.s[1:] = np.cumsum(self.seg_len[:-1])

        norm = np.maximum(self.seg_len, 1e-9)
        self.seg_tx = self.seg_dx / norm
        self.seg_ty = self.seg_dy / norm
        self.tangent = np.arctan2(self.seg_ty, self.seg_tx)
        self.nx = -self.seg_ty
        self.ny = self.seg_tx

        half = self.width / 2.0
        # The left normal points inward for counter-clockwise centreline points
        # and outward for clockwise points. Account for orientation so the names
        # remain geometrically correct for either winding direction. This is still
        # a sampled offset approximation, not a CAD-grade offset-curve algorithm.
        signed_area = 0.5 * float(np.sum(
            self.x * np.roll(self.y, -1) - np.roll(self.x, -1) * self.y
        ))
        if signed_area >= 0.0:  # counter-clockwise: left side is the interior
            self.inner_x = self.x + half * self.nx
            self.inner_y = self.y + half * self.ny
            self.outer_x = self.x - half * self.nx
            self.outer_y = self.y - half * self.ny
        else:  # clockwise: left side is the exterior
            self.outer_x = self.x + half * self.nx
            self.outer_y = self.y + half * self.ny
            self.inner_x = self.x - half * self.nx
            self.inner_y = self.y - half * self.ny

        # Discrete curvature from wrapped change in tangent angle per metre.
        tangent_next = np.roll(self.tangent, -1)
        angle_change = np.arctan2(
            np.sin(tangent_next - self.tangent),
            np.cos(tangent_next - self.tangent),
        )
        # Preserve turn direction. Positive curvature means the centreline tangent
        # turns counter-clockwise along the track's stored traversal direction;
        # negative curvature means a clockwise turn. Keeping this sign gives the
        # policy an explicit distinction between left/right bends in its preview.
        self.curvature = angle_change / np.maximum(self.seg_len, 1e-6)
        self.curvature = np.nan_to_num(self.curvature, nan=0.0, posinf=0.0, neginf=0.0)

        # Cache boundary segment arrays for fast ray/segment intersections.
        self.outer_seg_dx = np.roll(self.outer_x, -1) - self.outer_x
        self.outer_seg_dy = np.roll(self.outer_y, -1) - self.outer_y
        self.inner_seg_dx = np.roll(self.inner_x, -1) - self.inner_x
        self.inner_seg_dy = np.roll(self.inner_y, -1) - self.inner_y

    def point_at_s(self, distance: float) -> tuple[float, float, int]:
        """Return the sampled centreline point at a wrapped arc distance."""
        target = float(distance % self.length)
        index = int(np.searchsorted(self.s, target, side="right") - 1)
        index = max(0, min(index, len(self.x) - 1))
        return float(self.x[index]), float(self.y[index]), index


def make_track(seed: int = 42, name: str | None = None, n_points: int = N_TRACK_POINTS,
               width: float = TRACK_WIDTH) -> Track:
    """Generate a closed, asymmetric track using smooth Fourier harmonics.

    This is synthetic procedural geometry, not an external dataset. Use fixed
    seed sets for reproducible training and disjoint seed sets for held-out tests.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    radius = (
        47.0
        + rng.uniform(5.0, 8.0) * np.sin(2.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
        + rng.uniform(3.0, 5.5) * np.sin(3.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
        + rng.uniform(1.5, 2.8) * np.cos(4.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
    )
    x = radius * np.cos(theta)
    y = 0.74 * radius * np.sin(theta)
    # Small smooth warp breaks radial symmetry but preserves a closed layout.
    x += 1.5 * np.sin(5.0 * theta + 0.3)
    y += 1.0 * np.cos(3.0 * theta - 0.2)
    return Track(x=x, y=y, name=name or f"track-{seed}", width=width)


def project_to_track(track: Track, x: float, y: float) -> dict[str, float | int]:
    """Project a point onto every centreline segment and return the closest projection.

    Unlike nearest sampled vertex, segment projection gives smoother progress estimates.
    """
    qx = x - track.x
    qy = y - track.y
    fraction = (qx * track.seg_dx + qy * track.seg_dy) / track.seg_len2
    fraction = np.clip(fraction, 0.0, 1.0)
    px = track.x + fraction * track.seg_dx
    py = track.y + fraction * track.seg_dy
    dx = x - px
    dy = y - py
    d2 = dx * dx + dy * dy
    index = int(np.argmin(d2))
    signed_offset = dx[index] * track.nx[index] + dy[index] * track.ny[index]
    along = track.s[index] + fraction[index] * track.seg_len[index]
    along = float(along % track.length)
    return {
        "index": index,
        "distance": float(np.sqrt(d2[index])),
        "signed_offset": float(signed_offset),
        "s": along,
        "x": float(px[index]),
        "y": float(py[index]),
    }


def wrapped_progress_delta(old_s: float, new_s: float, track_length: float) -> float:
    """Signed shortest progress increment on a closed track."""
    delta = new_s - old_s
    if delta < -track_length / 2.0:
        delta += track_length
    elif delta > track_length / 2.0:
        delta -= track_length
    return float(delta)


def _ray_segment_hits(track: Track, x: float, y: float, angles: np.ndarray) -> np.ndarray:
    """Vectorized exact ray/segment intersections with both offset boundaries."""
    ray_x = np.cos(angles)[:, None]
    ray_y = np.sin(angles)[:, None]
    best = np.full(len(angles), MAX_RAY_DISTANCE, dtype=np.float64)

    for bx, by, sx, sy in (
        (track.outer_x, track.outer_y, track.outer_seg_dx, track.outer_seg_dy),
        (track.inner_x, track.inner_y, track.inner_seg_dx, track.inner_seg_dy),
    ):
        qx = bx[None, :] - x
        qy = by[None, :] - y
        sx2 = sx[None, :]
        sy2 = sy[None, :]
        denom = ray_x * sy2 - ray_y * sx2
        valid = np.abs(denom) > 1e-10
        t = np.full_like(denom, np.inf)
        u = np.full_like(denom, np.inf)
        np.divide(qx * sy2 - qy * sx2, denom, out=t, where=valid)
        np.divide(qx * ray_y - qy * ray_x, denom, out=u, where=valid)
        hit = valid & (t >= 0.0) & (t <= MAX_RAY_DISTANCE) & (u >= 0.0) & (u <= 1.0)
        if np.any(hit):
            ray_hits = np.where(hit, t, np.inf).min(axis=1)
            best = np.minimum(best, ray_hits)
    return np.asarray(np.clip(best, 0.0, MAX_RAY_DISTANCE), dtype=np.float32)


class RacingEnv(gym.Env):
    """Custom Gymnasium environment for point-mass racing-line optimization.

    Observation (14 values; version ``racing_obs_signed_curvature_v2``):
      0:7  normalized ray distances at [-90,-60,-30,0,30,60,90] degrees
      7    normalized speed
      8:10 sin/cos of local track-tangent heading error
      10   signed lateral offset normalized to half track width
      11:14 signed curvature preview at 10, 22 and 38 metres

    Checkpoint compatibility note: v2 retains the 14-value shape but changes the
    meaning of curvature preview from nonnegative magnitude to signed curvature.
    Policies trained with the previous observation definition must be retrained.

    Action (2 continuous values in [-1, 1]):
      action[0] scales a bounded speed increment Delta-v per step
      action[1] scales a bounded heading increment Delta-theta per step

    This explicitly follows the project brief's 2D point-mass/vector-motion
    approach; it is not a bicycle/tire-force vehicle model.
    """
    metadata = {"render_modes": []}

    def __init__(self, track: Track | None = None, max_steps: int = 1800,
                 seed: int = 42, randomize_tracks: bool = False):
        super().__init__()
        self.track = track if track is not None else make_track(seed)
        self.max_steps = int(max_steps)
        self.randomize_tracks = bool(randomize_tracks)
        self.base_seed = int(seed)

        # Ray distances and speed are nonnegative; heading-error sine/cosine,
        # lateral offset, and signed curvature previews can span [-1, 1].
        low = np.asarray([0.0] * 8 + [-1.0] * 6, dtype=np.float32)
        high = np.ones(14, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

        self.x = self.y = self.heading = self.speed = 0.0
        self.progress = self.sim_time = self.total_reward = 0.0
        self.steps = 0
        self.previous_s = 0.0
        self.lap_complete = self.off_track = self.finished = False
        self.trajectory_x: list[float] = []
        self.trajectory_y: list[float] = []
        self.last_rays = np.full(len(RAY_ANGLES_DEG), MAX_RAY_DISTANCE, dtype=np.float32)
        self.last_info: dict[str, Any] = {}

    def _update_rays(self) -> None:
        angles = self.heading + np.deg2rad(np.asarray(RAY_ANGLES_DEG, dtype=np.float64))
        self.last_rays = _ray_segment_hits(self.track, self.x, self.y, angles)

    def _observation(self) -> np.ndarray:
        projection = project_to_track(self.track, self.x, self.y)
        heading_error = normalize_angle(self.track.tangent[projection["index"]] - self.heading)
        lateral = float(np.clip(projection["signed_offset"] / HALF_TRACK_WIDTH, -1.0, 1.0))
        curvature = []
        for lookahead in LOOKAHEAD_DISTANCES:
            _, _, index = self.track.point_at_s(float(projection["s"]) + lookahead)
            curvature.append(float(np.clip(self.track.curvature[index] / CURVATURE_SCALE, -1.0, 1.0)))
        obs = np.concatenate([
            self.last_rays / MAX_RAY_DISTANCE,
            np.asarray([np.clip(self.speed / MAX_SPEED, 0.0, 1.0)], dtype=np.float32),
            np.asarray([np.sin(heading_error), np.cos(heading_error)], dtype=np.float32),
            np.asarray([lateral], dtype=np.float32),
            np.asarray(curvature, dtype=np.float32),
        ]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _info(self, projection: dict[str, Any]) -> dict[str, Any]:
        return {
            "track_name": self.track.name,
            "track_length": float(self.track.length),
            "progress": float(self.progress),
            "progress_fraction": float(np.clip(self.progress / self.track.length, 0.0, 1.0)),
            "lap_time": float(self.sim_time),
            "speed": float(self.speed),
            "heading": float(self.heading),
            "signed_lateral_offset": float(projection["signed_offset"]),
            "distance_to_centerline": float(projection["distance"]),
            "lap_complete": bool(self.lap_complete),
            "off_track": bool(self.off_track),
            "finished": bool(self.finished),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "track_seed" in options:
            self.track = make_track(int(options["track_seed"]))
        elif self.randomize_tracks and not (options and options.get("keep_track", False)):
            track_seed = int(self.rng.integers(0, 2**31 - 1))
            self.track = make_track(track_seed)

        self.x = float(self.track.x[0])
        self.y = float(self.track.y[0])
        self.heading = float(self.track.tangent[0])
        self.speed = 0.0
        self.progress = 0.0
        self.sim_time = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.previous_s = 0.0
        self.lap_complete = self.off_track = self.finished = False
        self.trajectory_x = [self.x]
        self.trajectory_y = [self.y]
        self._update_rays()
        projection = project_to_track(self.track, self.x, self.y)
        self.previous_s = float(projection["s"])
        info = self._info(projection)
        self.last_info = info
        return self._observation(), info

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode is finished; call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), received {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        # Point-mass controls: direct increments in speed and heading.
        delta_v = float(action[0]) * MAX_DELTA_V
        delta_heading = float(action[1]) * MAX_DELTA_HEADING
        self.speed = float(np.clip(self.speed + delta_v, 0.0, MAX_SPEED))
        self.heading = normalize_angle(self.heading + delta_heading)
        self.x += self.speed * np.cos(self.heading) * DT
        self.y += self.speed * np.sin(self.heading) * DT
        self.sim_time += DT
        self.steps += 1
        self.trajectory_x.append(self.x)
        self.trajectory_y.append(self.y)

        projection = project_to_track(self.track, self.x, self.y)
        delta_s = wrapped_progress_delta(self.previous_s, float(projection["s"]), self.track.length)
        # Ignore implausibly large projection jumps (e.g. around a sharp numerical switch).
        if abs(delta_s) > max(self.speed * DT * 3.0 + 0.25, 0.75):
            delta_s = 0.0
        self.progress += delta_s
        self.previous_s = float(projection["s"])

        # Progress shaping is normalized by lap length. Its total on a lap is ~100;
        # elapsed simulated seconds cost one reward per second. No lateral-position
        # penalty is applied, so the policy is free to explore off-centre racing lines.
        reward = PROGRESS_REWARD_SCALE * (delta_s / self.track.length) - TIME_PENALTY_PER_SECOND * DT
        terminated = False
        truncated = False
        lateral = abs(float(projection["signed_offset"]))
        if lateral > self.track.width / 2.0:
            reward -= OFF_TRACK_PENALTY
            self.off_track = True
            self.finished = True
            terminated = True
        elif self.progress >= self.track.length:
            reward += LAP_COMPLETION_BONUS
            self.lap_complete = True
            self.finished = True
            terminated = True
        elif self.steps >= self.max_steps:
            self.finished = True
            truncated = True

        self.total_reward += reward
        self._update_rays()
        info = self._info(projection)
        info.update({"reward": float(reward), "total_reward": float(self.total_reward), "delta_progress": float(delta_s)})
        self.last_info = info
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        """Rendering is deliberately provided by manual_drive.py/train.py."""
        return None

    def close(self):
        return None
