# Minimum-Time Racing with PPO

A small racing simulation for training a Proximal Policy Optimization (PPO) agent to complete procedurally generated tracks quickly. The car is modeled as a point mass, and the policy controls acceleration and heading rate. A look-ahead geometric controller is included as a baseline.

## Results

The saved run used 150,000 environment steps and one training seed (42). The selected checkpoint completed all 10 held-out test tracks, with a mean successful lap time of 16.16 s. The baseline also completed all 10, with a mean of 22.33 s. These figures describe this finite procedural-track test set; they do not establish performance on real circuits or across independent training runs.

The evaluation table, summary, and track maps are in [`outputs/evaluation`](outputs/evaluation/). The selected policy is `outputs/ppo_best_validation.pt`. A compact, checkpoint-free results bundle is provided as `dashboard_results.zip` for the dashboard deployment; it contains the saved metrics and held-out track maps.

## Run locally

Python 3.10 or newer is recommended. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python check_environment.py
python -m pytest -q
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

Train a policy (default: 150,000 steps):

```bash
python train.py --timesteps 150000 --seed 42 --output-dir outputs
```

The trainer writes checkpoints and CSV/JSON metrics to the output directory. For a short execution check, use a separate directory:

```bash
python train.py --timesteps 3000 --rollout-steps 512 --eval-every 1500 --no-dashboard --output-dir outputs_smoke
```

Evaluate the validation-selected checkpoint on the held-out test tracks:

```bash
python evaluate.py --model outputs/ppo_best_validation.pt --output-dir outputs/evaluation
```

Use a fresh output directory for a new experiment so results from separate runs are not mixed. See [`METHODOLOGY.md`](METHODOLOGY.md) for the environment, reward, PPO setup, and evaluation protocol.

## Dashboard

The dashboard is a read-only Streamlit app. It displays the bundled run or lets a viewer upload another results ZIP. It does not train the agent or load its checkpoint.

Run locally:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

To deploy with Streamlit Community Cloud:

1. Push this project to a GitHub repository, including `streamlit_dashboard.py`, `requirements.txt`, and `dashboard_results.zip`.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app** and select the repository and branch.
3. Set the app file to `streamlit_dashboard.py`, then deploy.

The archive contains recorded CSV/JSON metrics and held-out track maps from `outputs/`, not console-log reconstructions or the policy checkpoint. To publish another run, replace the bundle with a ZIP containing that run's result files and `track_*_comparison.png` images, then commit the change. Keep `requirements.txt` focused on the dashboard; training dependencies are in `requirements-dev.txt`.

## Repository contents

- `racing_rl/`: environment, PPO implementation, training plots, and manual driving interface.
- `train.py`, `evaluate.py`: training and held-out evaluation entry points.
- `revalidate_checkpoints.py`, `diagnose_policy.py`: checkpoint analysis and policy diagnostics.
- `tests/`: environment, geometry, PPO utility, and diagnostic tests.
- `outputs/`: saved run results and selected checkpoint.
- `streamlit_dashboard.py`: results dashboard for local use or cloud deployment.

For a concise checklist covering validation, the saved experiment, and packaging, see [`SUBMISSION_RUNBOOK.md`](SUBMISSION_RUNBOOK.md).
