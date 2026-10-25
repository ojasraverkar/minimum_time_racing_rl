# Submission checklist

## 1. Prepare the repository

Review the README and methodology, then confirm the project files and results you intend to submit are present. The `.gitignore` keeps caches, local logs, smoke-test outputs, and intermediate checkpoints out of Git while retaining the selected model and recorded result tables.

If this folder is not already a Git repository, initialize one and make the first commit:

```bash
git init
git add .
git status --short
git commit -m "Prepare racing reinforcement learning project"
```

If a Git repository already exists, skip `git init`. Review `git status` before committing; do not add `.venv`, `outputs_smoke`, or local log files. The supplied folder may be a downloaded archive without `.git` history.

Create an empty GitHub repository, then connect and push (replace the URL with your repository's URL):

```bash
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

If `origin` is already configured, inspect it with `git remote -v` and push to the intended repository rather than adding a duplicate remote.

## 2. Reproduce the checks

Use Python 3.10 or newer. Install the development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python check_environment.py
python -m pytest -q
```

The submitted run's recorded evaluation is under `outputs/evaluation/`. Re-running training or evaluation overwrites files in the selected output directory, so use a new directory for a new experiment.

## 3. Deploy the results dashboard

The Streamlit app is an optional presentation view of the experiment data. It is read-only and does not run the PPO agent. The deployment files are at the repository root: `streamlit_dashboard.py`, `requirements.txt`, and `dashboard_results.zip`.

For a local preview:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

For Streamlit Community Cloud:

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) and choose **Create app**.
2. Choose the GitHub repository and branch.
3. Set **Main file path** to `streamlit_dashboard.py` and deploy.

The dashboard reads the bundled metrics archive automatically. To publish results from another experiment, create a ZIP containing its CSV and JSON result files and replace `dashboard_results.zip` in the repository. Do not include model checkpoints in the dashboard bundle unless there is a separate reason to distribute them.

## 4. Final review

- Confirm the GitHub repository contains source code, tests, documentation, the selected checkpoint, and the committed result tables.
- Open the deployed dashboard and check that the held-out evaluation and summary load.
- Add the repository and dashboard URLs to the submission form or report as required by the course.
