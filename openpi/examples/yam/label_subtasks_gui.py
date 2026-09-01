"""Streamlit GUI for labeling per-frame subtasks of the raw YAM demos.

Shows one demo's top + wrist cameras (decoded on demand from the raw mp4s) and lets you tile the
episode with contiguous [start, end] subtask segments: each segment auto-starts right after the
previous one ends, so you only ever pick the *end* frame. Labels are saved as
`<demo>/subtask_labels.json` next to the raw data, which
`examples/yam/convert_yam_data_to_lerobot.py` reads when building the LeRobot dataset.

Hotkeys (native Streamlit button shortcuts, needs streamlit >= 1.45):
    a / d = -/+1 frame, q / e = -/+10 frames, 1..9 = pick subtask, m = mark segment end, z = undo

Usage:
    uv run --with streamlit streamlit run examples/yam/label_subtasks_gui.py
"""

import json
import pathlib
import re

import cv2
import numpy as np
import streamlit as st

DATA_ROOT = pathlib.Path("/iris/u/kewalk/memory_project/data/bin_memory_banana")
SUBTASKS = [  # 0816-training vocabulary (hotkeys 1..5). First-round labels come from
    # examples/yam/autolabel_banana0630_subtasks.py; originals in subtask_labels_3task_backup.json.
    "open both lids",
    "inspect both bins",
    "close both lids and reset arms",
    "open left bin",
    "open right bin",
]
LABEL_FILE = "subtask_labels.json"
TOP_MP4, LEFT_MP4, RIGHT_MP4 = "top_camera_rgb.mp4", "left_camera_rgb.mp4", "right_camera_rgb.mp4"

st.set_page_config(layout="wide", page_title="YAM Subtask Labeler")


def _natural_demo_key(p: pathlib.Path) -> int:
    m = re.search(r"(\d+)$", p.name)
    return int(m.group(1)) if m else 0


@st.cache_resource
def _capture(path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(path)


@st.cache_data
def _num_frames(path: str) -> int:
    return int(_capture(path).get(cv2.CAP_PROP_FRAME_COUNT))


@st.cache_data(max_entries=256)
def _frame(path: str, idx: int) -> np.ndarray:
    cap = _capture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return np.zeros((480, 640, 3), np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


demos = sorted(
    (p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("demo")),
    key=_natural_demo_key,
)
if not demos:
    st.error(f"No demo folders found in {DATA_ROOT}")
    st.stop()

demo = st.sidebar.selectbox("Demo", demos, format_func=lambda p: p.name)
label_path = demo / LABEL_FILE
final = _num_frames(str(demo / TOP_MP4)) - 1
if final < 1:
    st.error(f"Could not read frames from {demo / TOP_MP4}")
    st.stop()

# On demo switch: load existing labels and jump to the first unlabeled frame.
if st.session_state.get("demo") != demo.name:
    st.session_state.demo = demo.name
    st.session_state.segments = json.loads(label_path.read_text()) if label_path.exists() else []
    st.session_state.goto = st.session_state.segments[-1]["end"] + 1 if st.session_state.segments else 0
st.session_state.setdefault("task_idx", 0)

segments = st.session_state.segments
start = segments[-1]["end"] + 1 if segments else 0  # fixed start frame of the next segment
done = start > final

st.title(f"{demo.name} - YAM Subtask Labeler")

# Frame navigation. These render before the slider so their state change applies in the same run.
cur = int(st.session_state.get("frame", 0))
nav = st.columns([1, 1, 1, 1, 6])
if nav[0].button("-10 (q)", shortcut="Q", width="stretch"):
    st.session_state.goto = cur - 10
if nav[1].button("-1 (a)", shortcut="A", width="stretch"):
    st.session_state.goto = cur - 1
if nav[2].button("+1 (d)", shortcut="D", width="stretch"):
    st.session_state.goto = cur + 1
if nav[3].button("+10 (e)", shortcut="E", width="stretch"):
    st.session_state.goto = cur + 10

if (goto := st.session_state.pop("goto", None)) is not None:
    st.session_state.frame = int(np.clip(goto, 0, final))
st.session_state.frame = min(int(st.session_state.get("frame", 0)), final)
frame_idx = st.slider("frame", 0, final, key="frame")

img_col, ctl_col = st.columns([2, 1])
with img_col:
    st.image(
        _frame(str(demo / TOP_MP4), frame_idx),
        width="stretch",
        caption=f"top camera - frame {frame_idx} / {final}",
    )
    wl, wr = st.columns(2)
    wl.image(_frame(str(demo / LEFT_MP4), frame_idx), width="stretch", caption="left wrist")
    wr.image(_frame(str(demo / RIGHT_MP4), frame_idx), width="stretch", caption="right wrist")

with ctl_col:
    if done:
        st.success("All frames labeled (autosaved).")
    else:
        st.caption(f"Subtask (keys 1-{len(SUBTASKS)}):")
        for i, name in enumerate(SUBTASKS):
            active = i == st.session_state.task_idx
            if st.button(
                name,
                key=f"task{i}",
                shortcut=str(i + 1),
                type="primary" if active else "secondary",
                width="stretch",
            ):
                st.session_state.task_idx = i
                st.rerun()
        st.info(f"Segment start: {start}  |  end: {frame_idx}")
        if st.button("Mark segment end (m)", shortcut="M", type="primary", width="stretch"):
            if frame_idx >= start:
                segments.append(
                    {"task": SUBTASKS[st.session_state.task_idx], "start": start, "end": frame_idx}
                )
                if frame_idx == final:
                    label_path.write_text(json.dumps(segments, indent=4))
                    st.toast("Demo fully labeled - saved.")
                st.session_state.goto = frame_idx + 1
                st.rerun()
            else:
                st.error(f"End frame must be >= the segment start ({start}).")

    st.divider()
    st.subheader("Segments")
    for i, seg in enumerate(segments):
        st.write(f"{i + 1}. **{seg['task']}**  ({seg['start']} - {seg['end']})")
    if segments and st.button("Undo last segment (z)", shortcut="Z"):
        st.session_state.goto = segments.pop()["start"]
        st.rerun()

    st.divider()
    if st.button("Save progress", width="stretch"):
        label_path.write_text(json.dumps(segments, indent=4))
        st.toast("Saved (complete)." if done else "Saved partial progress.")

st.sidebar.divider()
st.sidebar.markdown(
    f"""### Hotkeys
* **a / d**: -/+1 frame
* **q / e**: -/+10 frames
* **1-{len(SUBTASKS)}**: pick subtask
* **m**: mark segment end
* **z**: undo last segment
"""
)
