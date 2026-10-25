# Methodology and Design Rationale

## 1. Problem formulation

The task is cast as a continuous-action Markov decision process. At time step `t`, the agent observes local geometric/speed information, selects a bounded change in speed and heading, receives a scalar reward, and sees the next observation. No offline dataset is required; trajectories are generated through online environment interactions.

## 2. 2D point-mass state

The vehicle state is `(x, y, heading, speed)`. With fixed time step `dt`, the control vector is `a = (a_v, a_theta)`, with each component in `[-1, 1]`:

```text
Delta-v       = a_v     * MAX_ACCELERATION * dt
Delta-heading = a_theta * MAX_HEADING_RATE * dt
speed         = clip(speed + Delta-v, 0, MAX_SPEED)
heading       = wrap(heading + Delta-heading)
x             = x + speed * cos(heading) * dt
y             = y + speed * sin(heading) * dt
```

This follows the brief's intention to focus on geometric pathfinding rather than complex vehicle kinematics. There are no tire-force, sideslip, suspension, or powertrain models.

## 3. Track geometry and sensing

Track centreline points are generated using a seeded Fourier/radial construction. Segment lengths define cumulative arc distance. Offset boundary polylines define the allowed track surface. The seven ray directions are relative to the vehicle heading: `[-90, -60, -30, 0, 30, 60, 90]` degrees. Rays are calculated using vectorized ray/segment intersections with the inner and outer boundary segments and clipped to a maximum sensing distance.

The observation includes:

1. Seven normalized ray distances;
2. Normalized current speed;
3. Sine and cosine of heading error relative to the local track tangent;
4. Signed lateral position normalized by half track width;
5. Three signed curvature-preview values ahead of the vehicle, normalised to `[-1, 1]`. Positive and negative values distinguish the direction of the upcoming tangent turn.

The centreline helps estimate progress and local geometry, but the reward has no centreline-attraction term.

## 4. Progress and reward

The vehicle is projected onto the nearest centreline segment. A fractional segment projection produces a smoother arc-length estimate than selecting only the nearest vertex. The difference in arc distance is wrapped at the start/finish line to avoid a false negative jump.

For a normal transition:

```text
r_t = 100 * delta_s / track_length - dt
```

A completed lap adds `+15`; leaving the permitted track adds `-50` and terminates the episode. The progress term provides a learning signal along the route, while elapsed simulation time has a negative cost. Faster completion of the same length therefore accumulates less time penalty. There is deliberately no penalty for being away from the centreline, so valid apex-clipping paths are allowed.

This shaping is an engineering choice, not a mathematical proof of global optimality. Results must be compared empirically and may require reward/scale sensitivity tests.

## 5. PPO

The actor-critic network shares a two-layer MLP trunk. The actor parameterizes a diagonal Gaussian distribution; `tanh` maps sampled actions into the valid `[-1, 1]` control range. The action log probability includes the corresponding `tanh` change-of-variables correction. The critic estimates the state value.

Training uses generalized advantage estimation, normalized advantages, the clipped PPO policy objective, a critic loss, entropy regularization, shuffled minibatches, and gradient clipping. Time-limit truncation bootstraps the critic, while GAE does not flow across episode resets.

## 6. PPO hyperparameters

The default run uses learning rate `3e-4`, discount factor `0.995`, GAE parameter `0.95`, PPO clip parameter `0.2`, six update epochs per rollout, minibatches of `256`, value coefficient `0.5`, entropy coefficient `0.005`, and gradient-norm clipping at `0.5`. The shared actor-critic trunk has two hidden layers of 128 units with `Tanh` activations. Discount factor and GAE lambda can be varied with `--gamma` and `--gae-lambda`. The version-2 observation keeps the 14-value input dimension but makes the last three features signed curvature; old checkpoints must not be evaluated as if these feature meanings were unchanged. The submission default uses gamma `0.995` because it was the most promising configuration in the prior controlled comparisons; variation across independent training seeds remains a limitation.

## 7. Training and unseen-track evaluation

Training episodes sample from fixed procedural training seeds `(101, 203, 307, 409, 503)`. The revised protocol selects checkpoints on a fresh 12-track validation set `(1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117)`. A separate 10-track final test `(1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381)` is held back until a candidate is selected. Previously inspected development tracks, including 901–937 and 1009–1039, are reserved for diagnosis rather than final-test claims. The selected checkpoint maximizes validation completion rate first, then minimizes mean time over successful laps, then maximizes mean progress. A simple look-ahead geometric controller provides a reference baseline. Metrics include completion rate, successful-lap time, progress fraction, episode return, off-track events, per-update approximate KL, clipping fraction, entropy and policy standard deviation. Checkpoints are saved at evaluation steps and the training run's best validation checkpoint is retained.

For checkpoint-reselection of existing runs, `revalidate_checkpoints.py` evaluates saved checkpoints on the expanded validation set without additional training. By default it evaluates each run's prior best-validation checkpoint; optional flags include the final checkpoint or a broader/exhaustive snapshot sweep. Selection uses completion rate first, then successful-lap time, then progress. For failure diagnosis, `diagnose_policy.py` logs the policy's pre-action observation, action, post-action state, seven ray distances, signed curvature previews, reward, and terminal flags on previously inspected development tracks. These diagnostic tracks must not be reused as final-test evidence. For a defensible result, run multiple independent training seeds and report both completion rate and mean lap time over successful runs. Compare PPO and baseline on the exact same final-test tracks. Do not use final-test performance to select a checkpoint or tune hyperparameters, and do not report a mean lap time without its number of completed laps. `train_update_metrics.csv` records optimisation diagnostics across minibatches and epochs; these help diagnose training dynamics but are not themselves measures of driving performance.

## 8. Limitations

- Procedural tracks are smooth Fourier-based closed paths, not arbitrary race circuits or imported CAD geometry.
- Offset boundary curves and nearest-segment projection are simplified geometric methods; very tight/self-intersecting tracks need stronger geometry checks.
- The model is a point mass. It cannot represent tire limits, understeer/oversteer, lateral acceleration constraints, braking distance physics, or realistic car-body collisions.
- The finite held-out set tests generalization but does not prove broad generalization or global optimality.
