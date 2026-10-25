"""Re-score saved PPO checkpoints on the expanded validation track set.

No training is performed. This allows existing compatible checkpoints to be
compared using the fresh 12-track validation set before a separate final test.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from racing_rl.env import OBSERVATION_VERSION, RacingEnv, make_track
from racing_rl.ppo import PPOAgent, PPOConfig
from train import (
    FINAL_TEST_TRACK_SEEDS, TRAIN_TRACK_SEEDS, VALIDATION_TRACK_SEEDS,
    baseline_action, run_episode, success_rate,
)


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


def parse_checkpoint_steps(path: Path) -> int | None:
    match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def checkpoint_metadata(path: Path, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        agent_payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older Torch versions
        agent_payload = torch.load(path, map_location=device)
    extra = dict(agent_payload.get("extra", {}))
    ppo_config = dict(agent_payload.get("ppo_config", {}))
    step = extra.get("timesteps")
    if step is None:
        step = parse_checkpoint_steps(path)
    return extra, {"timesteps": int(step) if step is not None else None, **ppo_config}


def discover_checkpoints(run_dir: Path, device: str) -> list[tuple[Path, dict, dict]]:
    step_files = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if not step_files:
        step_files = [p for p in (run_dir / "ppo_best_validation.pt", run_dir / "ppo_final.pt", run_dir / "checkpoint_latest.pt") if p.exists()]
    # Deduplicate by saved training step. Prefer the previous selected checkpoint
    # at its step and the actual final checkpoint at the last step; weights at the
    # same saved step are equivalent, while the metadata labels remain informative.
    by_step: dict[int | str, tuple[Path, dict, dict]] = {}
    for path in step_files:
        extra, meta = checkpoint_metadata(path, device)
        if extra.get("observation_version") != OBSERVATION_VERSION:
            raise ValueError(
                f"Checkpoint {path} has observation version {extra.get('observation_version')!r}; "
                f"expected {OBSERVATION_VERSION!r}."
            )
        key: int | str = meta["timesteps"] if meta["timesteps"] is not None else path.name
        by_step[key] = (path, extra, meta)

    for preferred_name in ("ppo_best_validation.pt", "ppo_final.pt"):
        preferred = run_dir / preferred_name
        if not preferred.exists():
            continue
        extra, meta = checkpoint_metadata(preferred, device)
        if extra.get("observation_version") != OBSERVATION_VERSION:
            raise ValueError(f"Incompatible observation definition in {preferred}")
        key = meta["timesteps"] if meta["timesteps"] is not None else preferred.name
        # Only replace the same saved step; do not add duplicate weights from
        # aliases such as ppo_final.pt and checkpoint_step_000150000.pt.
        if key in by_step or not step_files:
            by_step[key] = (preferred, extra, meta)
    return sorted(by_step.values(), key=lambda item: (item[2]["timesteps"] is None, item[2]["timesteps"] or 0))


def choose_checkpoints(
    run_dir: Path,
    candidates: list[tuple[Path, dict, dict]],
    max_count: int | None,
    all_checkpoints: bool,
) -> list[tuple[Path, dict, dict]]:
    """Default to the prior best-validation snapshot; optionally broaden."""
    if all_checkpoints or max_count is None or len(candidates) <= max_count:
        return candidates
    by_step = {item[2]["timesteps"]: item for item in candidates}
    prioritized: list[tuple[Path, dict, dict]] = []
    for filename in ("ppo_best_validation.pt", "ppo_final.pt"):
        path = run_dir / filename
        if not path.exists():
            continue
        _extra, metadata = checkpoint_metadata(path, "cpu")
        step = metadata["timesteps"]
        chosen = by_step.get(step)
        if chosen is not None and all(chosen[2]["timesteps"] != item[2]["timesteps"] for item in prioritized):
            prioritized.append(chosen)
        if len(prioritized) >= max_count:
            return sorted(prioritized, key=lambda item: item[2]["timesteps"] or 0)
    # The default two-checkpoint screen is strictly prior-best versus final;
    # if they are the same saved step, evaluate it once rather than silently
    # adding another candidate and making the default much slower.
    if max_count <= 2:
        return sorted(prioritized, key=lambda item: item[2]["timesteps"] or 0)
    remaining = [item for item in candidates if all(item[2]["timesteps"] != p[2]["timesteps"] for p in prioritized)]
    slots = max_count - len(prioritized)
    if slots > 0 and remaining:
        indices = np.linspace(0, len(remaining) - 1, min(slots, len(remaining))).round().astype(int)
        for index in sorted(set(indices.tolist())):
            prioritized.append(remaining[index])
    return sorted(prioritized, key=lambda item: item[2]["timesteps"] or 0)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_checkpoint(path: Path, run_name: str, env: RacingEnv, seeds: tuple[int, ...], device: str,
                        baseline_results_by_seed: dict[int, dict]) -> tuple[dict, list[dict]]:
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], PPOConfig(), device=device)
    extra = agent.load(str(path))
    if extra.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError(f"Incompatible observation definition in {path}")

    ppo_results = [run_episode(env, agent, seed, deterministic=True) for seed in seeds]
    # The baseline is deterministic, so compute each baseline track once rather
    # than rerunning it for every periodic checkpoint.
    baseline_results = [baseline_results_by_seed[seed] for seed in seeds]
    ppo_times = [float(r["lap_time"]) for r in ppo_results if np.isfinite(r["lap_time"])]
    base_times = [float(r["lap_time"]) for r in baseline_results if np.isfinite(r["lap_time"])]
    matched = [(float(p["lap_time"]), float(b["lap_time"])) for p, b in zip(ppo_results, baseline_results)
               if np.isfinite(p["lap_time"]) and np.isfinite(b["lap_time"])]
    extra_meta, config = checkpoint_metadata(path, device)
    row = {
        "run_name": run_name,
        "checkpoint_path": str(path),
        "checkpoint_timesteps": extra_meta.get("timesteps", config.get("timesteps")),
        "training_seed": extra_meta.get("seed"),
        "gamma": config.get("gamma"),
        "gae_lambda": config.get("gae_lambda"),
        "validation_track_count": len(seeds),
        "ppo_completed_tracks": int(sum(bool(r["lap_complete"]) for r in ppo_results)),
        "ppo_completion_rate": success_rate(ppo_results),
        "ppo_mean_successful_lap_time_s": mean_or_none(ppo_times),
        "ppo_mean_progress_fraction": float(np.mean([r["progress_fraction"] for r in ppo_results])),
        "baseline_completed_tracks": int(sum(bool(r["lap_complete"]) for r in baseline_results)),
        "baseline_completion_rate": success_rate(baseline_results),
        "baseline_mean_successful_lap_time_s": mean_or_none(base_times),
        "matched_successful_track_count": len(matched),
        "ppo_mean_time_on_matched_tracks_s": mean_or_none([x[0] for x in matched]),
        "baseline_mean_time_on_matched_tracks_s": mean_or_none([x[1] for x in matched]),
        "mean_paired_time_difference_s_ppo_minus_baseline": mean_or_none([x[0] - x[1] for x in matched]),
    }
    track_rows: list[dict] = []
    for ppo, baseline in zip(ppo_results, baseline_results):
        track_rows.append({
            "run_name": run_name,
            "checkpoint_path": str(path),
            "checkpoint_timesteps": row["checkpoint_timesteps"],
            "track_seed": int(ppo["track_seed"]),
            "ppo_lap_complete": bool(ppo["lap_complete"]),
            "ppo_off_track": bool(ppo["off_track"]),
            "ppo_lap_time_s": float(ppo["lap_time"]) if np.isfinite(ppo["lap_time"]) else "",
            "ppo_progress_fraction": float(ppo["progress_fraction"]),
            "baseline_lap_complete": bool(baseline["lap_complete"]),
            "baseline_off_track": bool(baseline["off_track"]),
            "baseline_lap_time_s": float(baseline["lap_time"]) if np.isfinite(baseline["lap_time"]) else "",
            "baseline_progress_fraction": float(baseline["progress_fraction"]),
        })
    return row, track_rows


def selection_score(row: dict[str, Any]) -> tuple[float, float, float]:
    completion = float(row["ppo_completion_rate"])
    mean_time = row["ppo_mean_successful_lap_time_s"]
    neg_time = -float(mean_time) if mean_time is not None else -1.0e9
    progress = float(row["ppo_mean_progress_fraction"])
    return completion, neg_time, progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate saved checkpoints on 12 fresh validation tracks; no training")
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="Run directories containing checkpoint_step_*.pt files")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint_revalidation"))
    parser.add_argument("--max-episode-steps", type=int, default=1800)
    parser.add_argument("--max-checkpoints-per-run", type=int, default=1,
                        help="Checkpoint cap per run (default 1: prior best-validation checkpoint; use 2 to also test final checkpoint)")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate every unique saved checkpoint step; may be computationally expensive")
    args = parser.parse_args()
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if args.max_checkpoints_per_run is not None and args.max_checkpoints_per_run <= 0:
        parser.error("--max-checkpoints-per-run must be positive")
    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            parser.error(f"Run directory not found: {run_dir}")

    # Fail early if the seed partition has been accidentally changed to overlap.
    if set(TRAIN_TRACK_SEEDS) & set(VALIDATION_TRACK_SEEDS):
        raise SystemExit("Training and expanded-validation seeds overlap")
    if set(VALIDATION_TRACK_SEEDS) & set(FINAL_TEST_TRACK_SEEDS):
        raise SystemExit("Expanded-validation and reserved final-test seeds overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = RacingEnv(track=make_track(VALIDATION_TRACK_SEEDS[0]), max_steps=args.max_episode_steps)
    all_checkpoint_rows: list[dict] = []
    all_track_rows: list[dict] = []
    selections: list[dict] = []
    print(f"Computing baseline once on {len(VALIDATION_TRACK_SEEDS)} expanded-validation tracks...")
    baseline_results_by_seed = {
        seed: run_episode(env, baseline_action, seed) for seed in VALIDATION_TRACK_SEEDS
    }

    for run_dir in args.run_dirs:
        run_name = run_dir.name
        checkpoints = discover_checkpoints(run_dir, device)
        if not checkpoints:
            print(f"WARNING: no checkpoints found in {run_dir}; skipping")
            continue
        checkpoints = choose_checkpoints(
            run_dir, checkpoints, args.max_checkpoints_per_run, args.all_checkpoints
        )

        run_rows: list[dict] = []
        print(f"\nRevalidating {run_name}: {len(checkpoints)} checkpoint(s) on {len(VALIDATION_TRACK_SEEDS)} validation tracks")
        for i, (path, _extra, meta) in enumerate(checkpoints, start=1):
            row, track_rows = evaluate_checkpoint(
                path, run_name, env, VALIDATION_TRACK_SEEDS, device, baseline_results_by_seed
            )
            run_rows.append(row)
            all_checkpoint_rows.append(row)
            all_track_rows.extend(track_rows)
            print(f"  [{i}/{len(checkpoints)}] step={row['checkpoint_timesteps']} "
                  f"completion={row['ppo_completed_tracks']}/{len(VALIDATION_TRACK_SEEDS)} "
                  f"mean_time={row['ppo_mean_successful_lap_time_s']}")

        best = max(run_rows, key=selection_score)
        output_run_dir = args.output_dir / run_name
        output_run_dir.mkdir(parents=True, exist_ok=True)
        selected_source = Path(best["checkpoint_path"])
        selected_copy = output_run_dir / "ppo_best_expanded_validation.pt"
        # Preserve the learned weights/config while updating checkpoint metadata
        # so downstream reports identify the expanded-validation selection.
        try:
            payload = torch.load(selected_source, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(selected_source, map_location="cpu")
        selected_extra = dict(payload.get("extra", {}))
        selected_extra.update({
            "checkpoint_role": "best_expanded_validation",
            "expanded_validation_track_seeds": list(VALIDATION_TRACK_SEEDS),
            "expanded_validation_selection": {
                "completion_rate": best["ppo_completion_rate"],
                "completed_tracks": best["ppo_completed_tracks"],
                "mean_successful_lap_time_s": best["ppo_mean_successful_lap_time_s"],
                "mean_progress_fraction": best["ppo_mean_progress_fraction"],
                "selection_rule": "completion rate, then mean successful-lap time, then progress",
            },
        })
        payload["extra"] = selected_extra
        torch.save(payload, selected_copy)
        selection = {
            "run_name": run_name,
            "selected_source_checkpoint": str(selected_source),
            "selected_checkpoint_copy": str(selected_copy),
            "selected_checkpoint_timesteps": best["checkpoint_timesteps"],
            "selection_rule": "highest expanded-validation completion rate, then lowest mean successful-lap time, then highest mean progress",
            "validation_track_seeds": list(VALIDATION_TRACK_SEEDS),
            "final_test_track_seeds_reserved": list(FINAL_TEST_TRACK_SEEDS),
            "selected_metrics": best,
            "all_evaluated_checkpoints": len(run_rows),
        }
        (output_run_dir / "selected_checkpoint.json").write_text(json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8")
        selections.append({
            "run_name": run_name,
            "selected_checkpoint_path": str(selected_copy),
            "source_checkpoint_path": str(selected_source),
            "selected_checkpoint_timesteps": best["checkpoint_timesteps"],
            "validation_completion_rate": best["ppo_completion_rate"],
            "validation_completed_tracks": best["ppo_completed_tracks"],
            "validation_mean_successful_lap_time_s": best["ppo_mean_successful_lap_time_s"],
            "validation_mean_progress_fraction": best["ppo_mean_progress_fraction"],
            "baseline_completion_rate": best["baseline_completion_rate"],
            "matched_successful_track_count": best["matched_successful_track_count"],
            "ppo_mean_time_on_matched_tracks_s": best["ppo_mean_time_on_matched_tracks_s"],
            "baseline_mean_time_on_matched_tracks_s": best["baseline_mean_time_on_matched_tracks_s"],
        })

    write_csv(args.output_dir / "checkpoint_metrics.csv", all_checkpoint_rows)
    write_csv(args.output_dir / "checkpoint_track_metrics.csv", all_track_rows)
    write_csv(args.output_dir / "selected_checkpoints.csv", selections)
    summary = {
        "purpose": "checkpoint selection using a newly expanded validation set; not final-test evaluation",
        "observation_version": OBSERVATION_VERSION,
        "training_track_seeds": list(TRAIN_TRACK_SEEDS),
        "expanded_validation_track_seeds": list(VALIDATION_TRACK_SEEDS),
        "final_test_track_seeds_reserved": list(FINAL_TEST_TRACK_SEEDS),
        "number_of_validation_tracks": len(VALIDATION_TRACK_SEEDS),
        "rule": "highest completion rate; then lowest mean successful-lap time; then highest progress",
        "runs_evaluated": [str(p) for p in args.run_dirs],
        "checkpoint_rows_written": len(all_checkpoint_rows),
        "selected_runs": selections,
        "caveat": "This revalidation can select among saved checkpoints but cannot undo model/parameter choices already influenced by previously inspected results. Final-test seeds remain reserved.",
    }
    (args.output_dir / "revalidation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    env.close()
    print(f"\nExpanded-validation outputs written to {args.output_dir.resolve()}")
    print("Use ppo_best_expanded_validation.pt for diagnostics and only run the reserved final test after selecting a candidate.")


if __name__ == "__main__":
    main()
