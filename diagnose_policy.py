"""Export per-step policy/control diagnostics on development tracks.

This script is intended for diagnosis, not final-test selection. It records the
observation used to choose each action, the action itself, the resulting state,
raycast distances, curvature preview, and terminal reason. Use development tracks
whose results have already been inspected (for example 1039 and 1013).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from racing_rl.env import DT, OBSERVATION_VERSION, RacingEnv, make_track
from racing_rl.ppo import PPOAgent, PPOConfig
from train import baseline_action, run_episode


def parse_seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("Provide at least one seed")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seed list must not contain duplicates")
    return seeds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _select_rows(rows: list[dict[str, Any]], *, model_label: str, seed: int, controller: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["model_label"] == model_label and
            int(row["track_seed"]) == seed and row["controller"] == controller]


def _plot_diagnostic(track_seed: int, model_label: str, model_path: Path,
                     step_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]],
                     out_path: Path) -> None:
    ppo_rows = _select_rows(step_rows, model_label=model_label, seed=track_seed, controller="ppo")
    base_rows = baseline_rows
    if not ppo_rows:
        return
    track = make_track(track_seed)
    ppo_t = np.asarray([float(row["sim_time_s"]) + DT for row in ppo_rows])
    base_t = np.asarray([float(row["sim_time_s"]) + DT for row in base_rows])

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(ppo_t, [float(r["post_speed_mps"]) for r in ppo_rows], label=f"PPO ({model_label})")
    if base_rows:
        axes[0].plot(base_t, [float(r["post_speed_mps"]) for r in base_rows], label="Geometric baseline", alpha=0.8)
    axes[0].set_ylabel("Speed [m/s]")

    axes[1].plot(ppo_t, np.rad2deg([float(r["post_heading_error_rad"]) for r in ppo_rows]), label="PPO")
    if base_rows:
        axes[1].plot(base_t, np.rad2deg([float(r["post_heading_error_rad"]) for r in base_rows]), label="Baseline", alpha=0.8)
    axes[1].set_ylabel("Heading error [deg]")

    axes[2].plot(ppo_t, [float(r["post_signed_lateral_offset_m"]) for r in ppo_rows], label="PPO")
    axes[2].axhline(track.width / 2, color="black", linestyle="--", linewidth=1, label="Track edge")
    axes[2].axhline(-track.width / 2, color="black", linestyle="--", linewidth=1)
    if base_rows:
        axes[2].plot(base_t, [float(r["post_signed_lateral_offset_m"]) for r in base_rows], label="Baseline", alpha=0.8)
    axes[2].set_ylabel("Lateral offset [m]")

    for prefix, times, subset, label in (("PPO", ppo_t, ppo_rows, "PPO"),
                                         ("Base", base_t, base_rows, "Baseline")):
        if not subset:
            continue
        for ray, legend in (("0", "front"), ("m30", "−30°"), ("p30", "+30°")):
            axes[3].plot(times, [float(r[f"post_ray_{ray}_m"]) for r in subset],
                         label=f"{label} {legend}", alpha=0.85)
    axes[3].set_ylabel("Ray distance [m]")

    axes[4].plot(ppo_t, [float(r["action_speed_command_norm"]) for r in ppo_rows], label="PPO speed cmd")
    axes[4].plot(ppo_t, [float(r["action_heading_command_norm"]) for r in ppo_rows], label="PPO heading cmd", alpha=0.8)
    if base_rows:
        axes[4].plot(base_t, [float(r["action_speed_command_norm"]) for r in base_rows], linestyle="--", label="Baseline speed cmd", alpha=0.7)
        axes[4].plot(base_t, [float(r["action_heading_command_norm"]) for r in base_rows], linestyle="--", label="Baseline heading cmd", alpha=0.7)
    axes[4].set_ylabel("Action [-1, 1]")
    axes[4].set_xlabel("Simulation time [s]")

    # A marker makes the last recorded event easy to locate; the CSV keeps the
    # exact terminal row and all preceding transitions for closer inspection.
    if any(bool(r["off_track"]) for r in ppo_rows):
        failure_time = float(ppo_rows[-1]["sim_time_s"]) + DT
        for ax in axes:
            ax.axvline(failure_time, color="red", linestyle=":", linewidth=1.4)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=3, loc="best")
    fig.suptitle(f"Step diagnostics — track {track_seed} — {model_label}\n{model_path.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-step racing diagnostics on already-inspected development tracks")
    parser.add_argument("--models", nargs="+", required=True, type=Path,
                        help="One or more compatible PPO checkpoints; e.g. three ppo_best_validation.pt files")
    parser.add_argument("--track-seeds", type=parse_seed_list, default=parse_seed_list("1039,1013"),
                        help="Comma-separated development track seeds (default: 1039,1013)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics"))
    parser.add_argument("--max-episode-steps", type=int, default=1800)
    args = parser.parse_args()
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    for path in args.models:
        if not path.exists():
            parser.error(f"Model file not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = RacingEnv(track=make_track(args.track_seeds[0]), max_steps=args.max_episode_steps)
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    baseline_by_seed: dict[int, dict] = {}
    baseline_step_by_seed: dict[int, list[dict]] = {}

    # Baseline diagnostics are run once per development track and then overlaid
    # with every PPO policy's trace.
    for seed in args.track_seeds:
        result = run_episode(env, baseline_action, seed, capture_steps=True,
                             controller_label="geometric_baseline")
        baseline_by_seed[seed] = result
        rows = result.get("step_diagnostics", [])
        for row in rows:
            row["model_label"] = "geometric_baseline"
        baseline_step_by_seed[seed] = rows
        step_rows.extend(rows)
        episode_rows.append({
            "model_label": "geometric_baseline", "model_path": "",
            "track_seed": seed, "controller": "geometric_baseline",
            "lap_complete": bool(result["lap_complete"]), "off_track": bool(result["off_track"]),
            "lap_time_s": float(result["lap_time"]) if np.isfinite(result["lap_time"]) else "",
            "progress_fraction": float(result["progress_fraction"]), "steps": int(result["steps"]),
        })

    for model_path in args.models:
        label = model_path.parent.name or model_path.stem
        # Use stem too when multiple checkpoints share the same parent folder.
        if any(r["model_label"] == label for r in episode_rows if r["controller"] == "ppo"):
            label = f"{label}_{model_path.stem}"
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], PPOConfig(), device=device)
        extra = agent.load(str(model_path))
        if extra.get("observation_version") != OBSERVATION_VERSION:
            raise SystemExit(
                f"Incompatible observation version in {model_path}; expected {OBSERVATION_VERSION}. "
                "Retrain policies made with the earlier curvature-magnitude observation."
            )
        for seed in args.track_seeds:
            result = run_episode(env, agent, seed, deterministic=True, capture_steps=True, controller_label="ppo")
            for row in result.get("step_diagnostics", []):
                row["model_label"] = label
                row["model_path"] = str(model_path)
            step_rows.extend(result.get("step_diagnostics", []))
            episode_rows.append({
                "model_label": label, "model_path": str(model_path),
                "track_seed": seed, "controller": "ppo",
                "lap_complete": bool(result["lap_complete"]), "off_track": bool(result["off_track"]),
                "lap_time_s": float(result["lap_time"]) if np.isfinite(result["lap_time"]) else "",
                "progress_fraction": float(result["progress_fraction"]), "steps": int(result["steps"]),
            })
            _plot_diagnostic(seed, label, model_path, step_rows,
                             baseline_step_by_seed[seed],
                             args.output_dir / f"diagnostic_{label}_track_{seed}.png")
        print(f"Exported diagnostics for {label} from {model_path}")

    write_csv(args.output_dir / "step_diagnostics.csv", step_rows)
    write_csv(args.output_dir / "episode_summary.csv", episode_rows)
    metadata = {
        "purpose": "development failure diagnosis only; not for final-test checkpoint selection",
        "observation_version": OBSERVATION_VERSION,
        "model_paths": [str(p) for p in args.models],
        "track_seeds": list(args.track_seeds),
        "logged_fields": [
            "pre-action observation features, normalized raycasts, signed curvature preview",
            "normalized action commands and applied delta-v/delta-heading",
            "post-action position, speed, heading error, lateral offset, progress and terminal flags",
        ],
        "notes": [
            "sim_time_s is pre-action time; post-state is reached one DT later",
            "curvature_per_m fields are reconstructed from normalized observation values and CURVATURE_SCALE",
            "use rows immediately preceding the terminal off_track row to diagnose the departure",
        ],
    }
    (args.output_dir / "diagnostics_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env.close()
    print(f"Step diagnostics, episode summary and plots saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
