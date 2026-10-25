"""Evaluate a saved PPO checkpoint against a simple geometric controller on held-out tracks."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from racing_rl.env import OBSERVATION_VERSION, RacingEnv, make_track
from racing_rl.ppo import PPOAgent, PPOConfig
from train import FINAL_TEST_TRACK_SEEDS, run_episode, baseline_action

# Final-test tracks are disjoint from both training and checkpoint-selection seeds.
TEST_TRACK_SEEDS = FINAL_TEST_TRACK_SEEDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("outputs/ppo_best_validation.pt"),
                        help="Checkpoint selected on validation tracks (default: ppo_best_validation.pt)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--max-episode-steps", type=int, default=1800)
    args = parser.parse_args()
    if args.max_episode_steps <= 0:
        raise SystemExit("--max-episode-steps must be positive")
    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}. Train first with python train.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = RacingEnv(track=make_track(TEST_TRACK_SEEDS[0]), max_steps=args.max_episode_steps)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], PPOConfig(), device=device)
    checkpoint_extra = agent.load(str(args.model))
    if checkpoint_extra.get("observation_version") != OBSERVATION_VERSION:
        env.close()
        raise SystemExit(
            "Incompatible checkpoint observation definition. This revision uses signed curvature "
            f"({OBSERVATION_VERSION}); retrain with this source before evaluation. "
            "Older checkpoints used nonnegative curvature magnitude despite having the same input shape."
        )

    rows = []
    for seed in TEST_TRACK_SEEDS:
        ppo = run_episode(env, agent, seed, deterministic=True)
        baseline = run_episode(env, baseline_action, seed)
        rows.append({
            "track_seed": seed,
            "ppo_lap_complete": ppo["lap_complete"],
            "ppo_off_track": ppo["off_track"],
            "ppo_lap_time_s": ppo["lap_time"] if np.isfinite(ppo["lap_time"]) else "",
            "ppo_progress_fraction": ppo["progress_fraction"],
            "baseline_lap_complete": baseline["lap_complete"],
            "baseline_off_track": baseline["off_track"],
            "baseline_lap_time_s": baseline["lap_time"] if np.isfinite(baseline["lap_time"]) else "",
            "baseline_progress_fraction": baseline["progress_fraction"],
        })
        track = env.track
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(track.outer_x, track.outer_y, "--", lw=1, label="Outer boundary")
        ax.plot(track.x, track.y, lw=1.5, label="Centreline")
        ax.plot(track.inner_x, track.inner_y, "--", lw=1, label="Inner boundary")
        ax.plot(baseline["trajectory_x"], baseline["trajectory_y"], lw=1.5, label="Geometric baseline")
        ax.plot(ppo["trajectory_x"], ppo["trajectory_y"], lw=1.7, label="PPO policy")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Held-out track {seed} — PPO vs baseline")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"track_{seed}_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Track {seed}: PPO complete={ppo['lap_complete']} time={ppo['lap_time']}; "
              f"baseline complete={baseline['lap_complete']} time={baseline['lap_time']}")

    with (args.output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    ppo_success = float(np.mean([r["ppo_lap_complete"] for r in rows]))
    base_success = float(np.mean([r["baseline_lap_complete"] for r in rows]))
    ppo_times = [float(r["ppo_lap_time_s"]) for r in rows if r["ppo_lap_time_s"] != ""]
    baseline_times = [float(r["baseline_lap_time_s"]) for r in rows if r["baseline_lap_time_s"] != ""]
    matched_times = [
        (float(r["ppo_lap_time_s"]), float(r["baseline_lap_time_s"]))
        for r in rows if r["ppo_lap_time_s"] != "" and r["baseline_lap_time_s"] != ""
    ]
    summary = {
        "evaluation_type": "standalone_held_out_evaluation",
        "model_path": str(args.model),
        "observation_version": OBSERVATION_VERSION,
        "checkpoint_role": checkpoint_extra.get("checkpoint_role", "unspecified"),
        "checkpoint_timesteps": checkpoint_extra.get("timesteps"),
        "ppo_config": agent.loaded_ppo_config,
        "track_seeds": list(TEST_TRACK_SEEDS),
        "max_episode_steps": args.max_episode_steps,
        "number_of_tracks": len(rows),
        "ppo_completed_tracks": int(sum(bool(r["ppo_lap_complete"]) for r in rows)),
        "ppo_completion_rate": ppo_success,
        "ppo_successful_laps": len(ppo_times),
        "ppo_mean_successful_lap_time_s": float(np.mean(ppo_times)) if ppo_times else None,
        "baseline_completed_tracks": int(sum(bool(r["baseline_lap_complete"]) for r in rows)),
        "baseline_completion_rate": base_success,
        "baseline_successful_laps": len(baseline_times),
        "baseline_mean_successful_lap_time_s": float(np.mean(baseline_times)) if baseline_times else None,
        "matched_successful_track_count": len(matched_times),
        "ppo_mean_time_on_matched_tracks_s": float(np.mean([pair[0] for pair in matched_times])) if matched_times else None,
        "baseline_mean_time_on_matched_tracks_s": float(np.mean([pair[1] for pair in matched_times])) if matched_times else None,
        "mean_paired_time_difference_s_ppo_minus_baseline": float(np.mean([pair[0] - pair[1] for pair in matched_times])) if matched_times else None,
        "device": str(device),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"PPO held-out completion rate: {ppo_success:.0%} ({len(ppo_times)}/{len(rows)} successful laps)")
    print(f"Baseline held-out completion rate: {base_success:.0%} ({len(baseline_times)}/{len(rows)} successful laps)")
    print(f"PPO mean successful lap time: {summary['ppo_mean_successful_lap_time_s'] if ppo_times else 'N/A'} s")
    print(f"Baseline mean successful lap time: {summary['baseline_mean_successful_lap_time_s'] if baseline_times else 'N/A'} s")
    print(f"Matched successful tracks: {len(matched_times)}")
    if matched_times:
        print(f"PPO vs baseline mean time on matched tracks: "
              f"{summary['ppo_mean_time_on_matched_tracks_s']:.2f} s vs "
              f"{summary['baseline_mean_time_on_matched_tracks_s']:.2f} s")
    print(f"Detailed results and evaluation_summary.json saved to {args.output_dir.resolve()}")
    env.close()


if __name__ == "__main__":
    main()
