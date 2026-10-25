"""Streamlit dashboard for the Minimum-Time Racing RL project.

Run from the project folder:
    pip install streamlit plotly
    streamlit run streamlit_dashboard.py

By default the app reads the local ./outputs folder. You can instead upload a ZIP
containing the outputs folder or individual CSV/JSON result files in the sidebar.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Page setup and visual styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Minimum-Time Racing RL | Experiment Dashboard",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1500px;}
    .hero {
        padding: 1.35rem 1.6rem; border-radius: 18px;
        background: linear-gradient(115deg, rgba(25,38,61,.98), rgba(32,79,94,.96));
        color: #f8fafc; margin-bottom: 1.1rem;
        border: 1px solid rgba(255,255,255,.12);
    }
    .hero h1 {font-size: 2rem; margin: 0 0 .35rem 0; letter-spacing: -.02em;}
    .hero p {font-size: 1rem; margin: 0; color: #dbeafe;}
    .small-muted {font-size: .85rem; color: #64748b;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.20); border-radius: 14px;
        padding: 12px 14px; background: rgba(128,128,128,.035);
    }
    div[data-testid="stMetricLabel"] p {font-size: .86rem;}
    [data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🏁 Minimum-Time Racing RL</h1>
      <p>Experiment dashboard · PPO policy, geometric baseline, training dynamics and held-out evaluation</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data source handling: local outputs directory OR uploaded ZIP / result files
# -----------------------------------------------------------------------------
SUPPORTED_SUFFIXES = {".csv", ".json", ".png", ".jpg", ".jpeg"}


def _normalise_member_name(name: str) -> str:
    """Normalise zip paths and prevent path-related surprises."""
    path = PurePosixPath(name.replace("\\", "/"))
    return str(path)


def _read_zip_members(uploaded_zip) -> dict[str, bytes]:
    """Read only result artifacts from a small ZIP; never extract it to disk."""
    if isinstance(uploaded_zip, (bytes, bytearray)):
        content = bytes(uploaded_zip)
    elif isinstance(uploaded_zip, Path):
        content = uploaded_zip.read_bytes()
    else:
        content = uploaded_zip.getvalue()
    if len(content) > 100 * 1024 * 1024:
        raise ValueError("ZIP is larger than the 100 MB dashboard upload limit.")
    members: dict[str, bytes] = {}
    total_uncompressed = 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        if len(infos) > 500:
            raise ValueError("ZIP contains too many files (maximum 500).")
        for info in infos:
            name = _normalise_member_name(info.filename)
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            if info.file_size > 40 * 1024 * 1024:
                continue
            total_uncompressed += info.file_size
            if total_uncompressed > 160 * 1024 * 1024:
                raise ValueError("Uncompressed result data exceeds the 160 MB dashboard limit.")
            members[name] = archive.read(info.filename)
    return members


def _find_member(members: dict[str, bytes], basename: str) -> tuple[str, bytes] | None:
    matches = [(name, data) for name, data in members.items()
               if Path(name).name.lower() == basename.lower()]
    if not matches:
        return None
    # Prefer files in evaluation/ for evaluation.csv and in the root for summary.json.
    if basename == "evaluation.csv":
        matches.sort(key=lambda pair: ("evaluation/" not in pair[0].lower(), len(pair[0])))
    elif basename == "summary.json":
        matches.sort(key=lambda pair: (Path(pair[0]).parent.name != "outputs", len(pair[0])))
    else:
        matches.sort(key=lambda pair: (len(pair[0]), pair[0]))
    return matches[0]


def _find_local_file(root: Path, basename: str) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    matches = list(root.rglob(basename))
    if not matches:
        return None
    if basename == "evaluation.csv":
        matches.sort(key=lambda p: ("evaluation" not in str(p.parent).lower(), len(str(p))))
    else:
        matches.sort(key=lambda p: (len(str(p)), str(p)))
    return matches[0]


@st.cache_data(show_spinner=False)
def _parse_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def _parse_json(data: bytes) -> dict[str, Any]:
    obj = json.loads(data.decode("utf-8-sig"))
    return obj if isinstance(obj, dict) else {"data": obj}


BUNDLED_ARCHIVE = Path(__file__).resolve().parent / "dashboard_results.zip"
source_options = (["Bundled final results"] if BUNDLED_ARCHIVE.is_file() else []) + [
    "Upload results ZIP", "Local outputs folder"
]
with st.sidebar:
    st.header("Data source")
    data_mode = st.radio("Load results from", source_options, index=0,
                         help="Bundled final results load automatically on deployment. Upload a ZIP to inspect another run.")
    local_root_text = "outputs"
    uploaded_archive = None
    uploaded_files = None
    root_dir = Path("outputs")
    if data_mode == "Local outputs folder":
        local_root_text = st.text_input("Outputs folder path", value="outputs",
                                        help="Path relative to the folder where Streamlit is launched, or an absolute path.")
        root_dir = Path(local_root_text).expanduser()
        if not root_dir.is_absolute():
            root_dir = Path.cwd() / root_dir
    elif data_mode == "Upload results ZIP":
        uploaded_archive = st.file_uploader("Upload outputs ZIP", type=["zip"], max_upload_size=100)
        uploaded_files = st.file_uploader(
            "Or upload result files individually",
            type=["csv", "json", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

    st.divider()
    st.caption("Expected artifacts")
    st.code("episode_metrics.csv\nevaluation/evaluation.csv\nevaluation_summary.json\nsummary.json", language="text")
    st.caption("Missing files are shown as unavailable. The bundled training and evaluation charts use the saved CSV results.")

members: dict[str, bytes] = {}
if data_mode == "Bundled final results":
    try:
        members.update(_read_zip_members(BUNDLED_ARCHIVE))
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        st.error(f"Could not read bundled results: {exc}")
if uploaded_archive is not None:
    try:
        members.update(_read_zip_members(uploaded_archive))
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        st.error(f"Could not read ZIP: {exc}")
if uploaded_files:
    for file in uploaded_files:
        members[file.name] = file.getvalue()


def load_bytes(basename: str) -> tuple[bytes | None, str | None]:
    if data_mode == "Local outputs folder":
        path = _find_local_file(root_dir, basename)
        if path is None:
            return None, None
        try:
            return path.read_bytes(), str(path)
        except OSError:
            return None, None
    item = _find_member(members, basename)
    if item is None:
        return None, None
    return item[1], item[0]


def load_csv(basename: str) -> tuple[pd.DataFrame | None, str | None]:
    payload, origin = load_bytes(basename)
    if payload is None:
        return None, origin
    try:
        return _parse_csv(payload), origin
    except Exception as exc:  # tolerate partially written/corrupt optional result files
        st.warning(f"Couldn't parse {basename} ({origin}): {exc}")
        return None, origin


def load_json(basename: str) -> tuple[dict[str, Any] | None, str | None]:
    payload, origin = load_bytes(basename)
    if payload is None:
        return None, origin
    try:
        return _parse_json(payload), origin
    except Exception as exc:
        st.warning(f"Couldn't parse {basename} ({origin}): {exc}")
        return None, origin


episodes, episodes_origin = load_csv("episode_metrics.csv")
evaluation, evaluation_origin = load_csv("evaluation.csv")
evaluation_summary, evaluation_summary_origin = load_json("evaluation_summary.json")
summary, summary_origin = load_json("summary.json")
eval_checkpoints, eval_checkpoints_origin = load_csv("evaluation_metrics.csv")
updates, updates_origin = load_csv("train_update_metrics.csv")
validation, validation_origin = load_csv("validation_track_metrics.csv")

# Clean CSV data types and standardise common boolean strings.
def _bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _numeric(df: pd.DataFrame | None, col: str) -> pd.Series | None:
    if df is None or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _metric_float(obj: dict[str, Any] | None, key: str, fallback: float | None = None) -> float | None:
    if obj and key in obj and obj[key] is not None:
        try:
            val = float(obj[key])
            return val if math.isfinite(val) else fallback
        except (TypeError, ValueError):
            pass
    return fallback


def _format_seconds(value: float | None) -> str:
    return f"{value:.2f} s" if value is not None and math.isfinite(value) else "N/A"


def _completion_count(df: pd.DataFrame | None, col: str) -> tuple[int, int, float | None]:
    if df is None or col not in df.columns or df.empty:
        return 0, 0, None
    completed = int(_bool_series(df[col]).sum())
    total = int(len(df))
    return completed, total, completed / total if total else None


# -----------------------------------------------------------------------------
# Summary metrics
# -----------------------------------------------------------------------------
if summary is None:
    summary = {}
if evaluation_summary is None:
    evaluation_summary = {}

# Evaluation CSV provides per-track results; use summary JSON if it is present,
# otherwise calculate directly from rows.
ppo_completed, eval_n, ppo_rate = _completion_count(evaluation, "ppo_lap_complete")
base_completed, base_n, base_rate = _completion_count(evaluation, "baseline_lap_complete")

ppo_mean = _metric_float(evaluation_summary, "ppo_mean_successful_lap_time_s")
base_mean = _metric_float(evaluation_summary, "baseline_mean_successful_lap_time_s")
matched_ppo_mean = _metric_float(evaluation_summary, "ppo_mean_time_on_matched_tracks_s")
matched_base_mean = _metric_float(evaluation_summary, "baseline_mean_time_on_matched_tracks_s")
matched_count = evaluation_summary.get("matched_successful_track_count")

if evaluation is not None:
    if ppo_mean is None and "ppo_lap_time_s" in evaluation.columns:
        vals = pd.to_numeric(evaluation["ppo_lap_time_s"], errors="coerce").dropna()
        ppo_mean = float(vals.mean()) if len(vals) else None
    if base_mean is None and "baseline_lap_time_s" in evaluation.columns:
        vals = pd.to_numeric(evaluation["baseline_lap_time_s"], errors="coerce").dropna()
        base_mean = float(vals.mean()) if len(vals) else None
    if "ppo_lap_time_s" in evaluation.columns and "baseline_lap_time_s" in evaluation.columns:
        matched = evaluation.dropna(subset=["ppo_lap_time_s", "baseline_lap_time_s"])
        if len(matched):
            if matched_ppo_mean is None:
                matched_ppo_mean = float(pd.to_numeric(matched["ppo_lap_time_s"], errors="coerce").mean())
            if matched_base_mean is None:
                matched_base_mean = float(pd.to_numeric(matched["baseline_lap_time_s"], errors="coerce").mean())
            if matched_count is None:
                matched_count = len(matched)

pct_faster = None
if matched_ppo_mean is not None and matched_base_mean not in (None, 0):
    pct_faster = (matched_base_mean - matched_ppo_mean) / matched_base_mean * 100.0

n_steps = summary.get("timesteps_completed")
if n_steps is None and summary.get("total_timesteps") is not None:
    n_steps = summary.get("total_timesteps")
config = summary.get("ppo_config", {}) if isinstance(summary.get("ppo_config", {}), dict) else {}
if evaluation_summary.get("ppo_config"):
    config = evaluation_summary["ppo_config"]

# -----------------------------------------------------------------------------
# Headline KPIs
# -----------------------------------------------------------------------------
if data_mode == "Local outputs folder":
    st.caption(f"Data source: `{root_dir}`")
elif data_mode == "Bundled final results":
    st.caption("Data source: bundled final submission run. The dashboard reads saved results and does not retrain the policy.")
else:
    st.caption("Data source: uploaded results. Files are processed in this session only.")

if not any([episodes is not None, evaluation is not None, summary, evaluation_summary]):
    st.warning("No readable results found yet. Upload your outputs ZIP, or point the app at the folder produced by `train.py` and `evaluate.py`.")
    st.info("The most useful files are `episode_metrics.csv`, `summary.json`, `evaluation/evaluation.csv`, and `evaluation/evaluation_summary.json`.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Training steps", f"{int(n_steps):,}" if n_steps is not None else "N/A")
k2.metric("Training episodes", f"{len(episodes):,}" if episodes is not None else "N/A")
k3.metric("PPO completion", f"{ppo_completed}/{eval_n}" if eval_n else "N/A",
          f"{ppo_rate:.0%}" if ppo_rate is not None else None)
k4.metric("Baseline completion", f"{base_completed}/{base_n}" if base_n else "N/A",
          f"{base_rate:.0%}" if base_rate is not None else None)
k5.metric("PPO vs baseline", f"{pct_faster:+.1f}%" if pct_faster is not None else "N/A",
          help="Positive means PPO is faster on tracks where both controllers completed the lap.")

if pct_faster is not None:
    st.caption("Lap-time comparison uses only tracks completed by both methods. Positive percentage means PPO's matched-track mean lap time is lower.")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_training, tab_evaluation, tab_diagnostics, tab_files = st.tabs(
    ["Overview", "Training", "PPO vs baseline", "PPO diagnostics", "Data & configuration"]
)

with tab_overview:
    st.subheader("Experiment overview")
    if summary:
        meta_cols = st.columns(4)
        meta_cols[0].metric("Training seed", str(summary.get("random_seed", "N/A")))
        meta_cols[1].metric("Rollout steps", str(summary.get("rollout_steps", "N/A")))
        try:
            dt_value = float(summary.get("dt_seconds"))
            dt_label = f"{dt_value:.4f} s" if math.isfinite(dt_value) and dt_value > 0 else "N/A"
        except (TypeError, ValueError):
            dt_label = "N/A"
        meta_cols[2].metric("Simulation step", dt_label)
        meta_cols[3].metric("Max episode steps", str(summary.get("max_episode_steps", "N/A")))
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Evaluation summary")
        if evaluation is not None and not evaluation.empty:
            show_cols = [c for c in [
                "track_seed", "ppo_lap_complete", "ppo_lap_time_s", "ppo_progress_fraction",
                "baseline_lap_complete", "baseline_lap_time_s", "baseline_progress_fraction",
            ] if c in evaluation.columns]
            display_eval = evaluation[show_cols].copy()
            rename = {
                "track_seed": "Track seed", "ppo_lap_complete": "PPO completed",
                "ppo_lap_time_s": "PPO time (s)", "ppo_progress_fraction": "PPO progress",
                "baseline_lap_complete": "Baseline completed", "baseline_lap_time_s": "Baseline time (s)",
                "baseline_progress_fraction": "Baseline progress",
            }
            display_eval.rename(columns=rename, inplace=True)
            for col in ["PPO completed", "Baseline completed"]:
                if col in display_eval:
                    display_eval[col] = _bool_series(display_eval[col]).map({True: "Yes", False: "No"})
            for col in ["PPO progress", "Baseline progress"]:
                if col in display_eval:
                    display_eval[col] = pd.to_numeric(display_eval[col], errors="coerce") * 100
            st.dataframe(display_eval, use_container_width=True, hide_index=True)
        else:
            st.info("Evaluation per-track CSV not found. Upload evaluation.csv to populate this table.")
    with right:
        st.markdown("#### Mean successful lap time")
        time_data = []
        if ppo_mean is not None:
            time_data.append({"Controller": "PPO", "Mean lap time (s)": ppo_mean})
        if base_mean is not None:
            time_data.append({"Controller": "Geometric baseline", "Mean lap time (s)": base_mean})
        if time_data:
            fig = px.bar(pd.DataFrame(time_data), x="Controller", y="Mean lap time (s)",
                          color="Controller", text_auto=".2f", title="Successful laps only")
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Lap-time summary is unavailable until evaluation data is supplied.")

    st.markdown("#### How to interpret the headline comparison")
    st.write(
        "Completion rate measures reliability. Mean lap time measures speed only among successful laps. "
        "The matched-track comparison uses only circuits that both PPO and the geometric baseline completed, "
        "so a failed lap is never treated as a zero-second lap."
    )

with tab_training:
    st.subheader("Training dynamics")
    if episodes is None or episodes.empty:
        st.info("`episode_metrics.csv` was not found. Upload it or set the local outputs folder correctly.")
    else:
        ep = episodes.copy()
        if "episode" not in ep.columns:
            ep["episode"] = range(1, len(ep) + 1)
        ep["episode"] = pd.to_numeric(ep["episode"], errors="coerce")
        ep.sort_values("episode", inplace=True)
        reward_col = "episode_return" if "episode_return" in ep.columns else None
        progress_col = "progress_fraction" if "progress_fraction" in ep.columns else None
        lap_col = "lap_time" if "lap_time" in ep.columns else None

        if reward_col:
            ep[reward_col] = pd.to_numeric(ep[reward_col], errors="coerce")
            window = st.slider("Reward moving-average window", min_value=5, max_value=50, value=20, step=5)
            ep["reward_moving_average"] = ep[reward_col].rolling(window, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ep["episode"], y=ep[reward_col], mode="lines",
                                     name="Episode return", opacity=0.32))
            fig.add_trace(go.Scatter(x=ep["episode"], y=ep["reward_moving_average"], mode="lines",
                                     name=f"{window}-episode moving average", line=dict(width=3)))
            fig.update_layout(title="Episode return over training", xaxis_title="Episode",
                              yaxis_title="Return", hovermode="x unified", height=390)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if progress_col:
                ep[progress_col] = pd.to_numeric(ep[progress_col], errors="coerce")
                fig = px.scatter(ep, x="episode", y=progress_col, color="lap_complete" if "lap_complete" in ep else None,
                                 title="Progress fraction per episode", labels={progress_col: "Progress fraction"})
                fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
                fig.update_layout(height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No `progress_fraction` column available.")
        with c2:
            if lap_col:
                ep[lap_col] = pd.to_numeric(ep[lap_col], errors="coerce")
                successful = ep.dropna(subset=[lap_col])
                if not successful.empty:
                    fig = px.scatter(successful, x="episode", y=lap_col, title="Successful lap times",
                                     labels={lap_col: "Lap time (s)"})
                    fig.update_layout(height=340)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No successful lap times are present in this episode CSV.")
            else:
                st.info("No `lap_time` column available.")

        if "lap_complete" in ep.columns or "off_track" in ep.columns:
            c1, c2, c3 = st.columns(3)
            complete_count = int(_bool_series(ep["lap_complete"]).sum()) if "lap_complete" in ep.columns else 0
            offtrack_count = int(_bool_series(ep["off_track"]).sum()) if "off_track" in ep.columns else 0
            truncated_count = int(_bool_series(ep["truncated"]).sum()) if "truncated" in ep.columns else None
            c1.metric("Completed training laps", f"{complete_count}/{len(ep)}")
            c2.metric("Off-track episodes", f"{offtrack_count}")
            c3.metric("Truncated episodes", str(truncated_count) if truncated_count is not None else "Not logged")

with tab_evaluation:
    st.subheader("PPO versus geometric baseline")
    if evaluation is None or evaluation.empty:
        st.info("`evaluation.csv` not found. Run `python evaluate.py --model outputs/ppo_best_validation.pt --output-dir outputs/evaluation` or upload the evaluation results.")
    else:
        ev = evaluation.copy()
        if "track_seed" not in ev.columns:
            ev["track_seed"] = range(1, len(ev) + 1)
        for col in ["ppo_lap_time_s", "baseline_lap_time_s", "ppo_progress_fraction", "baseline_progress_fraction"]:
            if col in ev.columns:
                ev[col] = pd.to_numeric(ev[col], errors="coerce")
        time_rows = []
        for _, row in ev.iterrows():
            track = str(row["track_seed"])
            if pd.notna(row.get("ppo_lap_time_s")):
                time_rows.append({"Track": track, "Controller": "PPO", "Lap time (s)": row["ppo_lap_time_s"]})
            if pd.notna(row.get("baseline_lap_time_s")):
                time_rows.append({"Track": track, "Controller": "Geometric baseline", "Lap time (s)": row["baseline_lap_time_s"]})
        if time_rows:
            times_df = pd.DataFrame(time_rows)
            fig = px.bar(times_df, x="Track", y="Lap time (s)", color="Controller", barmode="group",
                         title="Lap time by track — successful laps only", text_auto=".2f")
            fig.update_layout(height=430, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Neither controller has recorded successful lap times in this evaluation CSV.")

        progress_rows = []
        for _, row in ev.iterrows():
            track = str(row["track_seed"])
            if pd.notna(row.get("ppo_progress_fraction")):
                progress_rows.append({"Track": track, "Controller": "PPO", "Progress": row["ppo_progress_fraction"]})
            if pd.notna(row.get("baseline_progress_fraction")):
                progress_rows.append({"Track": track, "Controller": "Geometric baseline", "Progress": row["baseline_progress_fraction"]})
        if progress_rows:
            fig = px.bar(pd.DataFrame(progress_rows), x="Track", y="Progress", color="Controller", barmode="group",
                         title="Track progress when episode ended")
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

        if matched_count is not None:
            st.caption(f"Matched successful tracks: {matched_count}. Matched means: PPO {_format_seconds(matched_ppo_mean)}; baseline {_format_seconds(matched_base_mean)}.")

        # Show the held-out circuit layouts and controller paths as a compact gallery.
        st.markdown("#### Held-out track maps")
        track_images: list[tuple[str, bytes | Path]] = []
        if data_mode == "Local outputs folder":
            search_root = root_dir
            if search_root.exists():
                for p in search_root.rglob("track_*_comparison.png"):
                    track_images.append((p.name, p))
        else:
            for name, data in members.items():
                if Path(name).name.startswith("track_") and Path(name).name.endswith("_comparison.png"):
                    track_images.append((Path(name).name, data))
        track_images.sort(key=lambda x: x[0])
        if track_images:
            st.caption("Each map shows the track boundaries and the path taken by PPO and the geometric baseline.")
            result_by_seed = {}
            if "track_seed" in ev.columns:
                result_by_seed = {str(int(row["track_seed"])): row for _, row in ev.iterrows()}
            for start in range(0, len(track_images), 2):
                gallery_columns = st.columns(2)
                for column, (image_name, image_data) in zip(gallery_columns, track_images[start:start + 2]):
                    track_seed = Path(image_name).stem.removeprefix("track_").removesuffix("_comparison")
                    result = result_by_seed.get(track_seed)
                    with column.container(border=True):
                        st.markdown(f"**Track {track_seed}**")
                        image_source = str(image_data) if isinstance(image_data, Path) else image_data
                        st.image(
                            image_source,
                            alt=f"Track {track_seed} map comparing PPO and geometric baseline paths",
                            width="stretch",
                        )
                        if result is not None:
                            ppo_time = result.get("ppo_lap_time_s")
                            baseline_time = result.get("baseline_lap_time_s")
                            ppo_label = f"{float(ppo_time):.2f} s" if pd.notna(ppo_time) else "did not finish"
                            baseline_label = f"{float(baseline_time):.2f} s" if pd.notna(baseline_time) else "did not finish"
                            st.caption(f"PPO: {ppo_label} · Baseline: {baseline_label}")
        else:
            st.info("No track maps were found. Include the `track_*_comparison.png` files from the evaluation folder in the results ZIP to show the gallery.")

with tab_diagnostics:
    st.subheader("PPO optimisation diagnostics")
    if updates is None or updates.empty:
        st.info("`train_update_metrics.csv` not found. This tab populates when your training run exports update diagnostics.")
    else:
        diag = updates.copy()
        # Find an x-axis column that is useful for the training update trace.
        x_candidates = ["total_steps", "timesteps", "checkpoint_steps", "update", "update_index", "iteration"]
        x_col = next((col for col in x_candidates if col in diag.columns), None)
        if x_col is None:
            diag["update_index"] = range(1, len(diag) + 1)
            x_col = "update_index"
        numeric_cols = [c for c in diag.columns if c != x_col and pd.api.types.is_numeric_dtype(diag[c])]
        preferred = [c for c in ["approx_kl", "clip_fraction", "entropy", "policy_loss", "value_loss", "actor_loss", "critic_loss", "policy_std"] if c in numeric_cols]
        selected_metrics = st.multiselect("Metrics to plot", options=preferred or numeric_cols,
                                          default=(preferred[:3] if preferred else numeric_cols[:3]))
        if selected_metrics:
            chart_data = diag[[x_col] + selected_metrics].melt(id_vars=x_col, var_name="Metric", value_name="Value")
            fig = px.line(chart_data, x=x_col, y="Value", color="Metric", markers=True,
                          title="PPO training diagnostics")
            fig.update_layout(height=420, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(diag.tail(15), use_container_width=True, hide_index=True)

with tab_files:
    st.subheader("Configuration and available artifacts")
    if config:
        st.markdown("#### PPO configuration")
        config_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in config.items()])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
    if summary:
        st.markdown("#### Training metadata")
        exclude = {"ppo_config", "software_versions"}
        metadata = {k: v for k, v in summary.items() if k not in exclude}
        st.json(metadata, expanded=False)
        if summary.get("software_versions"):
            st.markdown("#### Software versions")
            st.json(summary["software_versions"], expanded=False)
    if evaluation_summary:
        st.markdown("#### Evaluation summary JSON")
        st.json(evaluation_summary, expanded=False)
    available_files = []
    for basename, origin in [
        ("episode_metrics.csv", episodes_origin), ("evaluation.csv", evaluation_origin),
        ("evaluation_summary.json", evaluation_summary_origin), ("summary.json", summary_origin),
        ("evaluation_metrics.csv", eval_checkpoints_origin),
        ("train_update_metrics.csv", updates_origin), ("validation_track_metrics.csv", validation_origin),
    ]:
        available_files.append({"Artifact": basename, "Status": "Loaded" if origin else "Not found", "Location": origin or "—"})
    st.markdown("#### Artifact status")
    st.dataframe(pd.DataFrame(available_files), use_container_width=True, hide_index=True)

st.divider()
st.caption("Minimum-Time Racing RL · Dashboard uses recorded experiment artifacts only. It does not retrain the agent or infer missing metrics.")
