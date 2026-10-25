"""Train PPO on the minimum-time racing environment.

Example:
    python train.py --timesteps 200000
    python train.py --timesteps 2000 --rollout-steps 512 --no-dashboard  # smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from racing_rl.env import (
    RacingEnv, make_track, DT, MAX_SPEED, MAX_ACCELERATION,
    MAX_HEADING_RATE, MAX_DELTA_V, MAX_DELTA_HEADING,
    MAX_RAY_DISTANCE, RAY_ANGLES_DEG, HALF_TRACK_WIDTH, CURVATURE_SCALE,
    LOOKAHEAD_DISTANCES,
    PROGRESS_REWARD_SCALE, TIME_PENALTY_PER_SECOND, OFF_TRACK_PENALTY, LAP_COMPLETION_BONUS,
    OBSERVATION_VERSION, normalize_angle, project_to_track,
)
from racing_rl.ppo import PPOAgent, PPOConfig, calculate_gae
from racing_rl.dashboard import TrainingDashboard

TRAIN_TRACK_SEEDS = (101, 203, 307, 409, 503)
# Fresh validation tracks for checkpoint selection. These were not used in the
# earlier three-track validation or development evaluations.
VALIDATION_TRACK_SEEDS = (1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117)
# Fresh final-test tracks, disjoint from all training, old development, and new
# validation seeds. Do not use these to choose checkpoints or tune parameters.
# Fresh final-test tracks, deliberately reserved after prior development and smoke evaluations.
FINAL_TEST_TRACK_SEEDS = (1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381)

assert not (set(TRAIN_TRACK_SEEDS) & set(VALIDATION_TRACK_SEEDS))
assert not (set(TRAIN_TRACK_SEEDS) & set(FINAL_TEST_TRACK_SEEDS))
assert not (set(VALIDATION_TRACK_SEEDS) & set(FINAL_TEST_TRACK_SEEDS))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def baseline_action(env: RacingEnv, target_speed: float = 12.0) -> np.ndarray:
    """Simple look-ahead geometric controller for reference comparisons only."""
    projection = project_to_track(env.track, env.x, env.y)
    look_x, look_y, _ = env.track.point_at_s(float(projection["s"]) + 13.0)
    target_heading = float(np.arctan2(look_y - env.y, look_x - env.x))
    heading_error = normalize_angle(target_heading - env.heading)
    delta_heading_cmd = np.clip(heading_error / max(MAX_DELTA_HEADING, 1e-8), -1.0, 1.0)
    delta_v_cmd = np.clip((target_speed - env.speed) / max(MAX_DELTA_V, 1e-8), -1.0, 1.0)
    return np.asarray([delta_v_cmd, delta_heading_cmd], dtype=np.float32)


def _observation_error_rad(observation: np.ndarray) -> float:
    """Recover signed heading error from the observation's sin/cos pair."""
    return float(np.arctan2(float(observation[8]), float(observation[9])))


def _diagnostic_step_row(
    *, track_seed: int, controller: str, step: int,
    pre_observation: np.ndarray, post_observation: np.ndarray,
    pre_x: float, pre_y: float, pre_heading: float, pre_speed: float,
    pre_time: float, pre_progress_fraction: float, action: np.ndarray,
    env: RacingEnv, reward: float, terminated: bool, truncated: bool,
    info: dict,
) -> dict:
    """Flatten one transition into a CSV-friendly state/action diagnostic row.

    Observation-derived features are labelled ``pre_*`` because the policy chose
    the action from that state. Position/speed/outcome fields labelled ``post_*``
    describe the state reached after applying that action.
    """
    row = {
        "track_seed": int(track_seed),
        "controller": str(controller),
        "step": int(step),
        "sim_time_s": float(pre_time),
        "pre_x_m": float(pre_x),
        "pre_y_m": float(pre_y),
        "pre_speed_mps": float(pre_speed),
        "pre_heading_rad": float(pre_heading),
        "pre_heading_error_rad": _observation_error_rad(pre_observation),
        "pre_signed_lateral_offset_m": float(pre_observation[10] * HALF_TRACK_WIDTH),
        "pre_progress_fraction": float(pre_progress_fraction),
        "action_speed_command_norm": float(action[0]),
        "action_heading_command_norm": float(action[1]),
        "applied_delta_v_mps": float(action[0] * MAX_DELTA_V),
        "applied_delta_heading_rad": float(action[1] * MAX_DELTA_HEADING),
        "reward": float(reward),
        "post_x_m": float(env.x),
        "post_y_m": float(env.y),
        "post_speed_mps": float(env.speed),
        "post_heading_rad": float(env.heading),
        "post_heading_error_rad": _observation_error_rad(post_observation),
        "post_signed_lateral_offset_m": float(info.get("signed_lateral_offset", np.nan)),
        "post_distance_to_centerline_m": float(info.get("distance_to_centerline", np.nan)),
        "track_width_m": float(env.track.width),
        "post_progress_fraction": float(info.get("progress_fraction", np.nan)),
        "post_lap_time_s": float(info.get("lap_time", np.nan)),
        "delta_progress_m": float(info.get("delta_progress", np.nan)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "lap_complete": bool(info.get("lap_complete", False)),
        "off_track": bool(info.get("off_track", False)),
    }
    ray_labels = { -90: "m90", -60: "m60", -30: "m30", 0: "0", 30: "p30", 60: "p60", 90: "p90" }
    for i, angle in enumerate(RAY_ANGLES_DEG):
        label = ray_labels[int(angle)]
        row[f"pre_ray_{label}_m"] = float(pre_observation[i] * MAX_RAY_DISTANCE)
        row[f"post_ray_{label}_m"] = float(post_observation[i] * MAX_RAY_DISTANCE)
    for j, distance in enumerate(LOOKAHEAD_DISTANCES):
        key = f"{int(distance)}m"
        row[f"pre_curvature_{key}_norm"] = float(pre_observation[11 + j])
        row[f"post_curvature_{key}_norm"] = float(post_observation[11 + j])
        row[f"pre_curvature_{key}_per_m"] = float(pre_observation[11 + j] * CURVATURE_SCALE)
        row[f"post_curvature_{key}_per_m"] = float(post_observation[11 + j] * CURVATURE_SCALE)
    return row


def run_episode(env: RacingEnv, policy, track_seed: int, deterministic: bool = True,
                max_steps: int | None = None, capture_steps: bool = False,
                controller_label: str | None = None) -> dict:
    obs, _ = env.reset(options={"track_seed": int(track_seed)})
    steps = 0
    info = {}
    step_diagnostics: list[dict] = []
    limit = max_steps or env.max_steps
    label = controller_label or ("geometric_baseline" if policy == baseline_action else "ppo")
    while steps < limit:
        pre_obs = np.asarray(obs, dtype=np.float32).copy()
        pre_x, pre_y, pre_heading, pre_speed = float(env.x), float(env.y), float(env.heading), float(env.speed)
        pre_time = float(env.sim_time)
        pre_progress_fraction = float(env.progress / max(env.track.length, 1e-9))
        action = policy(env) if policy == baseline_action else policy.act(obs, deterministic=deterministic)[0]
        action = np.asarray(action, dtype=np.float32).reshape(2)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if capture_steps:
            step_diagnostics.append(_diagnostic_step_row(
                track_seed=int(track_seed), controller=label, step=steps,
                pre_observation=pre_obs, post_observation=np.asarray(obs, dtype=np.float32),
                pre_x=pre_x, pre_y=pre_y, pre_heading=pre_heading, pre_speed=pre_speed,
                pre_time=pre_time, pre_progress_fraction=pre_progress_fraction,
                action=action, env=env, reward=float(reward), terminated=bool(terminated),
                truncated=bool(truncated), info=info,
            ))
        if terminated or truncated:
            break
    result = {
        "track_seed": int(track_seed),
        "track_name": info.get("track_name", f"track-{track_seed}"),
        "lap_complete": bool(info.get("lap_complete", False)),
        "off_track": bool(info.get("off_track", False)),
        "lap_time": float(info["lap_time"]) if info.get("lap_complete", False) else float("nan"),
        "progress_fraction": float(info.get("progress_fraction", 0.0)),
        "steps": int(steps),
        "trajectory_x": env.trajectory_x.copy(),
        "trajectory_y": env.trajectory_y.copy(),
    }
    if capture_steps:
        result["step_diagnostics"] = step_diagnostics
    return result


def evaluate(agent: PPOAgent, device_env: RacingEnv, seeds: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    ppo_results = []
    baseline_results = []
    for seed in seeds:
        ppo_results.append(run_episode(device_env, agent, seed, deterministic=True))
        baseline_results.append(run_episode(device_env, baseline_action, seed))
    return ppo_results, baseline_results


def mean_successful_time(results: list[dict]) -> float:
    times = [r["lap_time"] for r in results if np.isfinite(r["lap_time"])]
    return float(np.mean(times)) if times else float("nan")


def success_rate(results: list[dict]) -> float:
    return float(np.mean([bool(r["lap_complete"]) for r in results])) if results else 0.0


def mean_trajectory_speed_mps(x: list[float] | np.ndarray,
                              y: list[float] | np.ndarray,
                              elapsed_seconds: float) -> float:
    """Return path-distance / elapsed time from the recorded vehicle trajectory.

    This is the actual mean speed along the simulated path, rather than centreline
    progress divided by time (which is a progress rate and can differ on off-centre
    or corrective trajectories).
    """
    if elapsed_seconds <= 0.0:
        return float("nan")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or len(x_arr) != len(y_arr):
        raise ValueError("Trajectory x and y must be same-length one-dimensional arrays")
    if len(x_arr) < 2:
        return 0.0
    path_distance = float(np.hypot(np.diff(x_arr), np.diff(y_arr)).sum())
    return path_distance / float(elapsed_seconds)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Use stable field order and omit trajectory arrays from logged evaluation summaries.
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("trajectory_")} for row in rows]
    keys = list(clean_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_rows)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for minimum-time racing-line optimization")
    parser.add_argument("--timesteps", type=int, default=150_000, help="Total environment interaction steps")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Steps collected per PPO update")
    parser.add_argument("--max-episode-steps", type=int, default=1800, help="Per-episode time limit")
    parser.add_argument("--eval-every", type=int, default=20_000, help="Evaluate on held-out tracks after this many steps")
    parser.add_argument("--render-every", type=int, default=30, help="Refresh live dashboard every N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="PPO discount factor; recommended submission configuration is 0.995")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda in [0, 1]; override for controlled advantage-estimation experiments")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-dashboard", action="store_true", help="Train without live Matplotlib dashboard")
    parser.add_argument("--device", default=None, help="Optional torch device: cpu or cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.timesteps <= 0 or args.rollout_steps <= 0 or args.max_episode_steps <= 0:
        raise SystemExit("timesteps, rollout-steps and max-episode-steps must be positive")
    if args.eval_every <= 0 or args.render_every <= 0:
        raise SystemExit("eval-every and render-every must be positive")
    if not 0.0 < args.gamma <= 1.0:
        raise SystemExit("gamma must be in (0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise SystemExit("gae-lambda must be in [0, 1]")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    first_track = make_track(TRAIN_TRACK_SEEDS[0], name=f"train-{TRAIN_TRACK_SEEDS[0]}")
    env = RacingEnv(track=first_track, max_steps=args.max_episode_steps, seed=args.seed)
    ppo_config = PPOConfig(gamma=args.gamma, gae_lambda=args.gae_lambda)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], ppo_config, device=device)
    dashboard = TrainingDashboard(first_track, enabled=not args.no_dashboard)
    # Evaluation uses a separate environment so it never interrupts the live training episode.
    eval_env = RacingEnv(track=make_track(VALIDATION_TRACK_SEEDS[0], name=f"validation-{VALIDATION_TRACK_SEEDS[0]}"),
                         max_steps=args.max_episode_steps, seed=args.seed + 99)

    train_rng = np.random.default_rng(args.seed + 17)
    current_track_seed = int(train_rng.choice(TRAIN_TRACK_SEEDS))
    obs, _ = env.reset(seed=args.seed, options={"track_seed": current_track_seed})
    episode_return = 0.0
    episode_steps = 0
    episode_number = 0
    total_steps = 0
    next_eval = args.eval_every
    episode_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    validation_track_rows: list[dict] = []
    update_rows: list[dict] = []
    best_by_track: dict[str, dict] = {}
    best_validation_score: tuple[float, float, float] | None = None
    best_validation_summary: dict | None = None
    update_number = 0
    start_wall = time.time()

    # PPO rollout arrays. Truncations bootstrap the critic, but GAE never leaks
    # across a reset episode; true terminal states do not bootstrap.
    observations, actions, log_probs = [], [], []
    rewards, values, next_values = [], [], []
    terminateds, episode_ends = [], []

    print("=" * 76)
    print("MINIMUM-TIME RACING — PPO TRAINING")
    print(f"Device: {device} | Timesteps: {args.timesteps} | dt: {DT:.4f} s")
    print(f"Training track seeds: {TRAIN_TRACK_SEEDS}")
    print(f"Expanded validation seeds for checkpoint selection ({len(VALIDATION_TRACK_SEEDS)} tracks): {VALIDATION_TRACK_SEEDS}")
    print(f"Reserved final-test seeds ({len(FINAL_TEST_TRACK_SEEDS)} tracks): {FINAL_TEST_TRACK_SEEDS}")
    print(f"PPO config: gamma={args.gamma}, gae_lambda={args.gae_lambda}")
    print(f"Observation version: {OBSERVATION_VERSION}")
    print("Action: bounded Delta-v and Delta-heading; no detailed tire/vehicle physics.")
    print("=" * 76)

    while total_steps < args.timesteps:
        chunk_size = min(args.rollout_steps, args.timesteps - total_steps)
        for _ in range(chunk_size):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_value = 0.0 if terminated else agent.value(next_obs)

            observations.append(obs.copy())
            actions.append(action.copy())
            log_probs.append(log_prob)
            rewards.append(float(reward))
            values.append(value)
            next_values.append(next_value)
            terminateds.append(bool(terminated))
            episode_ends.append(bool(terminated or truncated))

            obs = next_obs
            episode_return += float(reward)
            episode_steps += 1
            total_steps += 1

            if total_steps % args.render_every == 0:
                dashboard.update_live(env, total_steps, episode_number + 1,
                                      episode_return, best_by_track)
                if dashboard.enabled and dashboard.closed:
                    print("Dashboard closed; continue in headless mode.")
                    dashboard.enabled = False

            if terminated or truncated:
                episode_number += 1
                row = {
                    "episode": episode_number,
                    "total_steps": total_steps,
                    "track_name": env.track.name,
                    "episode_return": float(episode_return),
                    "episode_steps": int(episode_steps),
                    "episode_duration_s": float(episode_steps * DT),
                    "lap_complete": bool(info["lap_complete"]),
                    "off_track": bool(info["off_track"]),
                    "lap_time": float(info["lap_time"]) if info["lap_complete"] else "",
                    "progress_fraction": float(info["progress_fraction"]),
                    "mean_speed_mps": mean_trajectory_speed_mps(
                        env.trajectory_x, env.trajectory_y, episode_steps * DT
                    ),
                }
                episode_rows.append(row)
                if info["lap_complete"]:
                    previous = best_by_track.get(env.track.name)
                    if previous is None or info["lap_time"] < previous["lap_time"]:
                        best_by_track[env.track.name] = {
                            "lap_time": float(info["lap_time"]),
                            "x": env.trajectory_x.copy(),
                            "y": env.trajectory_y.copy(),
                        }
                print(
                    f"Ep {episode_number:04d} | step {total_steps:07d} | {env.track.name:12s} | "
                    f"return {episode_return:8.2f} | progress {100*info['progress_fraction']:6.1f}% | "
                    f"lap {str(info['lap_complete']):5s} | off-track {str(info['off_track']):5s}"
                )
                # Save live progress in case the user stops a long training run.
                write_csv(args.output_dir / "episode_metrics.csv", episode_rows)
                dashboard.update_history(episode_rows, [{
                    "checkpoint_steps": r["checkpoint_steps"],
                    "baseline_mean_lap_time": float(r["baseline_mean_lap_time"]) if r["baseline_mean_lap_time"] != "" else np.nan,
                    "ppo_mean_lap_time": float(r["ppo_mean_lap_time"]) if r["ppo_mean_lap_time"] != "" else np.nan,
                } for r in evaluation_rows])
                obs, _ = env.reset(options={"track_seed": int(train_rng.choice(TRAIN_TRACK_SEEDS))})
                current_track_seed = int(env.track.name.split("-")[-1]) if env.track.name.split("-")[-1].isdigit() else current_track_seed
                episode_return = 0.0
                episode_steps = 0

            if total_steps >= args.timesteps:
                break

        # Update PPO from this on-policy batch.
        adv, returns = calculate_gae(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            np.asarray(next_values, dtype=np.float32),
            np.asarray(terminateds, dtype=bool),
            np.asarray(episode_ends, dtype=bool),
            agent.config.gamma,
            agent.config.gae_lambda,
        )
        losses = agent.update({
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "returns": returns.astype(np.float32),
            "advantages": adv.astype(np.float32),
        })

        observations.clear(); actions.clear(); log_probs.clear()
        rewards.clear(); values.clear(); next_values.clear()
        terminateds.clear(); episode_ends.clear()

        update_number += 1
        update_row = {
            "update": update_number,
            "total_steps": total_steps,
            **{key: float(value) for key, value in losses.items()},
            "gamma": float(agent.config.gamma),
            "gae_lambda": float(agent.config.gae_lambda),
        }
        update_rows.append(update_row)
        write_csv(args.output_dir / "train_update_metrics.csv", update_rows)
        print(
            f"  PPO update | actor {losses['actor_loss']:.4f} | critic {losses['critic_loss']:.4f} "
            f"| entropy {losses['entropy']:.4f} | KL {losses['approx_kl']:.5f} "
            f"| clip frac {losses['clip_fraction']:.3f}"
        )

        if total_steps >= next_eval or total_steps == args.timesteps:
            ppo_results, baseline_results = evaluate(agent, eval_env, VALIDATION_TRACK_SEEDS)
            ppo_mean_time = mean_successful_time(ppo_results)
            baseline_mean_time = mean_successful_time(baseline_results)
            eval_row = {
                "checkpoint_steps": total_steps,
                "ppo_success_rate": success_rate(ppo_results),
                "baseline_success_rate": success_rate(baseline_results),
                "ppo_successful_laps": int(sum(bool(r["lap_complete"]) for r in ppo_results)),
                "baseline_successful_laps": int(sum(bool(r["lap_complete"]) for r in baseline_results)),
                "ppo_mean_lap_time": ppo_mean_time if np.isfinite(ppo_mean_time) else "",
                "baseline_mean_lap_time": baseline_mean_time if np.isfinite(baseline_mean_time) else "",
                "ppo_mean_progress_fraction": float(np.mean([r["progress_fraction"] for r in ppo_results])),
                "baseline_mean_progress_fraction": float(np.mean([r["progress_fraction"] for r in baseline_results])),
            }
            evaluation_rows.append(eval_row)
            write_csv(args.output_dir / "evaluation_metrics.csv", evaluation_rows)
            for ppo_result, baseline_result in zip(ppo_results, baseline_results):
                validation_track_rows.append({
                    "checkpoint_steps": total_steps,
                    "track_seed": int(ppo_result["track_seed"]),
                    "ppo_lap_complete": bool(ppo_result["lap_complete"]),
                    "ppo_off_track": bool(ppo_result["off_track"]),
                    "ppo_lap_time_s": float(ppo_result["lap_time"]) if np.isfinite(ppo_result["lap_time"]) else "",
                    "ppo_progress_fraction": float(ppo_result["progress_fraction"]),
                    "baseline_lap_complete": bool(baseline_result["lap_complete"]),
                    "baseline_off_track": bool(baseline_result["off_track"]),
                    "baseline_lap_time_s": float(baseline_result["lap_time"]) if np.isfinite(baseline_result["lap_time"]) else "",
                    "baseline_progress_fraction": float(baseline_result["progress_fraction"]),
                })
            write_csv(args.output_dir / "validation_track_metrics.csv", validation_track_rows)

            checkpoint_extra = {
                "timesteps": total_steps,
                "seed": args.seed,
                "observation_version": OBSERVATION_VERSION,
                "checkpoint_role": "periodic_validation",
                "validation_track_seeds": list(VALIDATION_TRACK_SEEDS),
            }
            agent.save(str(args.output_dir / "checkpoint_latest.pt"), extra=checkpoint_extra)
            step_checkpoint = args.output_dir / f"checkpoint_step_{total_steps:09d}.pt"
            agent.save(str(step_checkpoint), extra=checkpoint_extra)

            # Select best checkpoint lexicographically: highest completion rate,
            # then lower mean time among completed laps, then greater mean progress.
            # This uses validation tracks only; final test seeds are disjoint.
            score = (
                float(eval_row["ppo_success_rate"]),
                -float(ppo_mean_time) if np.isfinite(ppo_mean_time) else -1.0e9,
                float(eval_row["ppo_mean_progress_fraction"]),
            )
            if best_validation_score is None or score > best_validation_score:
                best_validation_score = score
                best_validation_summary = {
                    "checkpoint_steps": total_steps,
                    "validation_score": list(score),
                    "validation_metrics": dict(eval_row),
                }
                agent.save(
                    str(args.output_dir / "ppo_best_validation.pt"),
                    extra={**checkpoint_extra, "checkpoint_role": "best_validation", "validation_score": list(score)},
                )
                save_json(args.output_dir / "best_validation_summary.json", best_validation_summary)

            dashboard.update_history(episode_rows, [
                {
                    "checkpoint_steps": r["checkpoint_steps"],
                    "baseline_mean_lap_time": float(r["baseline_mean_lap_time"]) if r["baseline_mean_lap_time"] != "" else np.nan,
                    "ppo_mean_lap_time": float(r["ppo_mean_lap_time"]) if r["ppo_mean_lap_time"] != "" else np.nan,
                } for r in evaluation_rows
            ])
            print(
                f"  Held-out eval @ {total_steps}: PPO completion={eval_row['ppo_success_rate']:.0%}, "
                f"baseline completion={eval_row['baseline_success_rate']:.0%}, "
                f"PPO mean successful lap={eval_row['ppo_mean_lap_time'] or 'N/A'}, "
                f"baseline mean successful lap={eval_row['baseline_mean_lap_time'] or 'N/A'}"
            )
            # This is a validation trajectory used for checkpoint selection, not final-test data.
            np.savez(args.output_dir / "last_validation_trajectory.npz",
                     x=np.asarray(ppo_results[-1]["trajectory_x"]),
                     y=np.asarray(ppo_results[-1]["trajectory_y"]),
                     track_seed=VALIDATION_TRACK_SEEDS[-1])
            next_eval += args.eval_every

    # Final artefacts.
    agent.save(str(args.output_dir / "ppo_final.pt"), extra={
        "timesteps": total_steps,
        "seed": args.seed,
        "observation_version": OBSERVATION_VERSION,
        "checkpoint_role": "final_training_step",
        "train_track_seeds": list(TRAIN_TRACK_SEEDS),
        "validation_track_seeds": list(VALIDATION_TRACK_SEEDS),
    })
    dashboard.save(str(args.output_dir / "training_dashboard.png"))
    write_csv(args.output_dir / "episode_metrics.csv", episode_rows)
    write_csv(args.output_dir / "evaluation_metrics.csv", evaluation_rows)
    summary = {
        "project": "Reinforcement Learning for Optimal Control in Minimum-Time Lap Simulation",
        "timesteps_completed": total_steps,
        "episodes_completed": episode_number,
        "random_seed": args.seed,
        "observation_version": OBSERVATION_VERSION,
        "rollout_steps": args.rollout_steps,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "max_episode_steps": args.max_episode_steps,
        "evaluation_interval_steps": args.eval_every,
        "render_interval_steps": args.render_every,
        "training_track_seeds": list(TRAIN_TRACK_SEEDS),
        "validation_track_seeds_for_checkpoint_selection": list(VALIDATION_TRACK_SEEDS),
        "validation_track_count": len(VALIDATION_TRACK_SEEDS),
        "final_test_track_seeds": list(FINAL_TEST_TRACK_SEEDS),
        "final_test_track_count": len(FINAL_TEST_TRACK_SEEDS),
        "ppo_config": agent.config.__dict__,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "device": str(agent.device),
        "dt_seconds": DT,
        "track_width_m": env.track.width,
        "max_speed_mps": MAX_SPEED,
        "observation_dim": int(env.observation_space.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
        "action_definition": ["normalized Delta-v per step", "normalized Delta-heading per step"],
        "reward_definition": {
            "progress": f"+{PROGRESS_REWARD_SCALE} * delta_progress / track_length",
            "time_penalty": f"-{TIME_PENALTY_PER_SECOND} * dt per step",
            "off_track": -OFF_TRACK_PENALTY,
            "lap_completion_bonus": LAP_COMPLETION_BONUS,
            "lateral_penalty": 0,
        },
        "last_validation_evaluation": evaluation_rows[-1] if evaluation_rows else None,
        "best_validation_checkpoint": best_validation_summary,
        "number_of_ppo_updates": update_number,
        "elapsed_wall_seconds": time.time() - start_wall,
        "caveat": "A finite experiment does not prove global optimality; report measured outcomes only.",
    }
    save_json(args.output_dir / "summary.json", summary)
    print("\nTraining finished. Outputs saved to:", args.output_dir.resolve())
    print(
        "Artifacts: ppo_final.pt, ppo_best_validation.pt, checkpoint_step_*.pt, "
        "episode_metrics.csv, train_update_metrics.csv, evaluation_metrics.csv, "
        "validation_track_metrics.csv, summary.json"
    )
    if dashboard.enabled and dashboard.is_open():
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
