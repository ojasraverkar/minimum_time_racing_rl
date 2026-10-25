"""Small live Matplotlib dashboard for PPO training."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .env import RacingEnv, Track, RAY_ANGLES_DEG


class TrainingDashboard:
    def __init__(self, initial_track: Track, enabled: bool = True):
        self.enabled = enabled
        self.closed = False
        if not enabled:
            self.fig = None
            return
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        grid = self.fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0))
        self.ax_track = self.fig.add_subplot(grid[:, 0])
        self.ax_reward = self.fig.add_subplot(grid[0, 1])
        self.ax_lap = self.fig.add_subplot(grid[0, 2])
        self.ax_eval = self.fig.add_subplot(grid[1, 1:])

        self.track_center, = self.ax_track.plot([], [], lw=1.5, label="Centreline")
        self.track_outer, = self.ax_track.plot([], [], "--", lw=1, label="Outer boundary")
        self.track_inner, = self.ax_track.plot([], [], "--", lw=1, label="Inner boundary")
        self.current_path, = self.ax_track.plot([], [], lw=2, label="Current episode")
        self.best_path, = self.ax_track.plot([], [], lw=2, label="Best completed lap (same track)")
        self.car, = self.ax_track.plot([], [], "o", ms=7, label="Point-mass vehicle")
        self.ray_lines = [self.ax_track.plot([], [], lw=1)[0] for _ in RAY_ANGLES_DEG]
        self.hud = self.ax_track.text(0.02, 0.98, "", transform=self.ax_track.transAxes,
                                      va="top", family="monospace", fontsize=9,
                                      bbox=dict(boxstyle="round", alpha=0.85))
        self.ax_track.set_title("LIVE ENVIRONMENT / PPO")
        self.ax_track.set_xlabel("x [m]")
        self.ax_track.set_ylabel("y [m]")
        self.ax_track.axis("equal")
        self.ax_track.grid(True)
        self.ax_track.legend(loc="lower center", fontsize=7, ncol=2)

        self.reward_line, = self.ax_reward.plot([], [], lw=1, label="Episode reward")
        self.reward_avg_line, = self.ax_reward.plot([], [], "--", lw=2, label="20-episode mean")
        self.ax_reward.set_title("Training return")
        self.ax_reward.set_xlabel("Episode")
        self.ax_reward.set_ylabel("Return")
        self.ax_reward.grid(True)
        self.ax_reward.legend(fontsize=8)

        self.lap_line, = self.ax_lap.plot([], [], ".", alpha=0.65, label="Successful lap")
        self.best_lap_line, = self.ax_lap.plot([], [], lw=2, label="Best-so-far")
        self.ax_lap.set_title("Lap-time convergence")
        self.ax_lap.set_xlabel("Episode")
        self.ax_lap.set_ylabel("Lap time [s]")
        self.ax_lap.grid(True)
        self.ax_lap.legend(fontsize=8)

        self.eval_baseline_line, = self.ax_eval.plot([], [], "o-", label="Geometric baseline")
        self.eval_ppo_line, = self.ax_eval.plot([], [], "o-", label="PPO policy")
        self.ax_eval.set_title("Held-out tracks: lap time (successful runs only)")
        self.ax_eval.set_xlabel("Evaluation checkpoint")
        self.ax_eval.set_ylabel("Mean lap time [s]")
        self.ax_eval.grid(True)
        self.ax_eval.legend(fontsize=8)

        self.current_track_name = None
        self.fig.suptitle("Minimum-Time Racing — PPO Training Dashboard", fontsize=15, fontweight="bold")
        plt.tight_layout()
        self._set_track(initial_track)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def is_open(self) -> bool:
        return bool(self.enabled and not self.closed and plt.fignum_exists(self.fig.number))

    def _set_track(self, track: Track):
        self.current_track_name = track.name
        self.track_center.set_data(track.x, track.y)
        self.track_outer.set_data(track.outer_x, track.outer_y)
        self.track_inner.set_data(track.inner_x, track.inner_y)
        min_x = min(track.outer_x.min(), track.inner_x.min()) - 6
        max_x = max(track.outer_x.max(), track.inner_x.max()) + 6
        min_y = min(track.outer_y.min(), track.inner_y.min()) - 6
        max_y = max(track.outer_y.max(), track.inner_y.max()) + 6
        self.ax_track.set_xlim(min_x, max_x)
        self.ax_track.set_ylim(min_y, max_y)

    def update_live(self, env: RacingEnv, total_steps: int, episode_number: int,
                    current_return: float, best_by_track: dict[str, dict]):
        if not self.enabled:
            return
        if not self.is_open():
            self.closed = True
            return
        if self.current_track_name != env.track.name:
            self._set_track(env.track)

        self.current_path.set_data(env.trajectory_x, env.trajectory_y)
        best = best_by_track.get(env.track.name)
        if best is not None:
            self.best_path.set_data(best["x"], best["y"])
        else:
            self.best_path.set_data([], [])
        self.car.set_data([env.x], [env.y])
        for line, dist, angle_deg in zip(self.ray_lines, env.last_rays, RAY_ANGLES_DEG):
            angle = env.heading + np.deg2rad(angle_deg)
            line.set_data([env.x, env.x + float(dist) * np.cos(angle)],
                          [env.y, env.y + float(dist) * np.sin(angle)])
        frac = float(np.clip(env.progress / env.track.length, 0.0, 1.0))
        self.hud.set_text(
            f"Total steps: {total_steps}\nEpisode: {episode_number}\nTrack: {env.track.name}\n"
            f"Status: {'OFF TRACK' if env.off_track else ('LAP COMPLETE' if env.lap_complete else 'RUNNING')}\n"
            f"Speed: {env.speed:5.2f} m/s\nTime: {env.sim_time:6.2f} s\n"
            f"Progress: {100*frac:5.1f}%\nReturn so far: {current_return:7.2f}"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def update_history(self, episode_rows: list[dict], evaluation_rows: list[dict]):
        if not self.enabled:
            return
        if not self.is_open():
            self.closed = True
            return
        if episode_rows:
            episodes = np.asarray([r["episode"] for r in episode_rows])
            rewards = np.asarray([r["episode_return"] for r in episode_rows], dtype=float)
            averages = np.asarray([np.mean(rewards[max(0, i - 19):i + 1]) for i in range(len(rewards))])
            times = np.asarray([r["lap_time"] if r["lap_complete"] else np.nan for r in episode_rows], dtype=float)
            best_times = []
            best = np.inf
            for value in times:
                if np.isfinite(value):
                    best = min(best, value)
                best_times.append(best if np.isfinite(best) else np.nan)
            self.reward_line.set_data(episodes, rewards)
            self.reward_avg_line.set_data(episodes, averages)
            self.lap_line.set_data(episodes, times)
            self.best_lap_line.set_data(episodes, best_times)
            for ax in (self.ax_reward, self.ax_lap):
                ax.relim()
                ax.autoscale_view()
        if evaluation_rows:
            checks = np.asarray([r["checkpoint_steps"] for r in evaluation_rows], dtype=float)
            baseline = np.asarray([r["baseline_mean_lap_time"] for r in evaluation_rows], dtype=float)
            ppo = np.asarray([r["ppo_mean_lap_time"] for r in evaluation_rows], dtype=float)
            self.eval_baseline_line.set_data(checks, baseline)
            self.eval_ppo_line.set_data(checks, ppo)
            self.ax_eval.relim()
            self.ax_eval.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save(self, path: str):
        if self.enabled and self.fig is not None:
            self.fig.savefig(path, dpi=160, bbox_inches="tight")

    def close(self):
        if self.enabled and self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
        self.closed = True
