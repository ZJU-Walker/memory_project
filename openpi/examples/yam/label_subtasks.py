"""Browser-based subtask labeler for raw YAM demos (no Streamlit; stdlib server only).

Tiles each episode with contiguous ``[start, end]`` subtask segments and writes
``<demo>/subtask_labels.json``, the file ``convert_yam_data_to_lerobot.py`` reads when
building the LeRobot dataset. Segments always start where the previous one ended, so you
only ever choose an *end* frame.

The recorded mp4s are already H.264, so they are streamed to a ``<video>`` element and the
browser does the decoding: scrubbing and playback are instant and no frame is ever decoded
server-side. Frame indices map to timestamps with the video's exact rational frame rate
(the YAM rigs record ~29.63 fps, not 30 -- assuming 30 drifts by ~12 frames per episode).

For the bin-memory task, ``--memory-task`` selects the five-phase schema (open lids ->
inspect -> close and reset -> wait -> execute). Its wait and execute phases are sided, so
each episode is labeled left or right and the two must agree: a `wait; target bin is left`
followed by `open right bin` is rejected, because the waiting segment *is* the memory
supervision and a contradiction there silently corrupts training.

Usage::

    uv run examples/yam/label_subtasks.py --data-dir /path/to/0816_banana --memory-task

    # or an arbitrary flat vocabulary (up to 9 labels):
    uv run examples/yam/label_subtasks.py --data-dir /path/to/demos \\
        --subtasks "observe bins" "open left bin" "open right bin"

then open the printed URL. Over SSH, forward the port::

    ssh -L 8000:localhost:8000 <host>

Keys: J/L step 1 frame, H/K step 10, arrows seek, Space play/pause, 1-9 pick subtask,
Enter mark segment end, U undo, S save, [ / ] previous / next demo, , / . set side.
"""

import argparse
import dataclasses
import functools
import http.server
import json
import pathlib
import re
import socketserver
import subprocess
import sys
import threading
import urllib.parse
import webbrowser

TOP_MP4 = "top_camera_rgb.mp4"
LEFT_MP4 = "left_camera_rgb.mp4"
RIGHT_MP4 = "right_camera_rgb.mp4"
LABEL_FILE = "subtask_labels.json"
CAMERAS = {"top": TOP_MP4, "left": LEFT_MP4, "right": RIGHT_MP4}

DEFAULT_SUBTASKS = ("observe bins", "open left bin", "open right bin")

# The bin-memory task's five phases, in the order they must occur. Phases 4 and 5 have a
# left/right variant; the labeler exposes the pair belonging to the episode's chosen side.
# `boundary` is shown in the UI as the rule for where the segment ends.
MEMORY_PHASES: tuple[dict, ...] = (
    {
        "key": "open_lids",
        "label": "open both lids",
        "boundary": "episode start until both lids are fully open",
    },
    {
        "key": "inspect",
        "label": "inspect both bins",
        "boundary": "both objects clearly visible until the first closing motion",
    },
    {
        "key": "close_reset",
        "label": "close both lids and reset arms",
        "boundary": "first closing motion until both lids closed and both arms neutral",
    },
    {
        "key": "wait",
        "label": "wait; target bin is {side}",
        "boundary": "reset complete until the first side-specific arm movement",
    },
    {
        "key": "execute",
        "label": "open {side} bin",
        "boundary": "first side-specific motion until the target lid is open",
    },
)

# Label strings that vary by side, keyed by phase; used to infer a segment's side.
_SIDED_KEYS = {phase["key"] for phase in MEMORY_PHASES if "{side}" in phase["label"]}


def memory_subtasks(side: str) -> list[str]:
    """The five phase labels for one episode, with sided phases bound to ``side``."""
    return [phase["label"].format(side=side) for phase in MEMORY_PHASES]


def all_memory_subtasks() -> list[str]:
    """Every label the memory task can produce (6: three shared + two per side)."""
    labels = []
    for phase in MEMORY_PHASES:
        if "{side}" in phase["label"]:
            labels.extend(phase["label"].format(side=s) for s in ("left", "right"))
        else:
            labels.append(phase["label"])
    return labels


def _phase_of(task: str) -> dict | None:
    """The phase a label belongs to, or None if it matches no known phase."""
    for phase in MEMORY_PHASES:
        if "{side}" in phase["label"]:
            if any(task == phase["label"].format(side=s) for s in ("left", "right")):
                return phase
        elif task == phase["label"]:
            return phase
    return None


def _side_of(task: str) -> str | None:
    """The side a sided label refers to, or None for shared labels."""
    phase = _phase_of(task)
    if phase is None or phase["key"] not in _SIDED_KEYS:
        return None
    return next(s for s in ("left", "right") if task == phase["label"].format(side=s))


def validate_phase_schema(segments: list[dict]) -> list[str]:
    """Check memory-task structure: phases in order, and one consistent side per episode.

    A `wait; target bin is left` followed by `open right bin` is the most damaging possible
    labeling error -- the waiting segment is precisely the memory supervision -- so it is
    caught here rather than surfacing as degraded training much later.
    """
    problems: list[str] = []
    if not segments or any(_phase_of(seg["task"]) is None for seg in segments):
        return problems  # a non-memory vocabulary; phase rules do not apply

    order = [phase["key"] for phase in MEMORY_PHASES]
    seen = [order.index(_phase_of(seg["task"])["key"]) for seg in segments]
    problems.extend(
        f"segment {i + 1} ({segments[i]['task']!r}) goes backwards in phase order after {segments[i - 1]['task']!r}"
        for i in range(1, len(seen))
        if seen[i] < seen[i - 1]
    )

    sides = {s for seg in segments if (s := _side_of(seg["task"])) is not None}
    if len(sides) > 1:
        sided = [seg["task"] for seg in segments if _side_of(seg["task"])]
        problems.append(f"episode mixes sides: {sided} -- wait and execute must agree")
    return problems


def _natural_demo_key(path: pathlib.Path) -> int:
    """Sort demo folders numerically (demo1, demo2, ..., demo10), not lexically."""
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def find_demos(data_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        (p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("demo")),
        key=_natural_demo_key,
    )


@functools.lru_cache(maxsize=512)
def probe_video(path: str) -> tuple[int, float]:
    """Return ``(num_frames, fps)`` for a video, using its exact rational frame rate.

    ``nb_frames`` is trusted when present; otherwise frames are counted by decoding
    packets. Falling back to ``duration * fps`` would be off by one on many files.
    """
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    numerator, _, denominator = stream["avg_frame_rate"].partition("/")
    fps = float(numerator) / float(denominator or 1)

    num_frames = int(stream.get("nb_frames") or 0)
    if num_frames <= 0:
        counted = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        num_frames = int(counted.stdout.strip() or 0)
    return num_frames, fps


def _metadata_steps(demo: pathlib.Path) -> int:
    """Recorded step count from ``metadata.json`` -- cheap, no ffprobe.

    Used only for the sidebar's done/not-done dot. It can exceed the true labelable length
    when a video is truncated, so the authoritative bound stays :func:`episode_length`.
    """
    path = demo / "metadata.json"
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text()).get("num_steps") or 0)
    except (json.JSONDecodeError, ValueError):
        return 0


def episode_length(demo: pathlib.Path) -> int:
    """Frames available for labeling: the minimum over proprio steps and all cameras.

    Mirrors the converter, which truncates to the shortest stream, so a label file can
    never run past what the dataset will contain.
    """
    lengths = [probe_video(str(demo / name))[0] for name in CAMERAS.values()]
    metadata_path = demo / "metadata.json"
    if metadata_path.exists():
        steps = json.loads(metadata_path.read_text()).get("num_steps")
        if steps:
            lengths.append(int(steps))
    return min(lengths)


def load_segments(demo: pathlib.Path, label_file: str = LABEL_FILE) -> list[dict]:
    path = demo / label_file
    return json.loads(path.read_text()) if path.exists() else []


def validate_segments(segments: list[dict], num_frames: int, subtasks: list[str]) -> list[str]:
    """Return human-readable problems that would make the converter skip this demo."""
    problems = []
    expected_start = 0
    for i, seg in enumerate(segments):
        if seg["start"] != expected_start:
            problems.append(
                f"segment {i + 1} starts at {seg['start']}, expected {expected_start} "
                "(segments must tile the episode contiguously from frame 0)"
            )
        if seg["end"] < seg["start"]:
            problems.append(f"segment {i + 1} ends ({seg['end']}) before it starts ({seg['start']})")
        if seg["task"] not in subtasks:
            problems.append(f"segment {i + 1} has unknown subtask {seg['task']!r}")
        expected_start = seg["end"] + 1
    if segments and expected_start < num_frames:
        problems.append(f"episode not covered: labeled {expected_start} of {num_frames} frames")
    return problems + validate_phase_schema(segments)


def save_segments(
    demo: pathlib.Path,
    segments: list[dict],
    num_frames: int,
    subtasks: list[str],
    label_file: str = LABEL_FILE,
) -> dict:
    """Write the label file atomically; report completeness without blocking partial saves."""
    problems = validate_segments(segments, num_frames, subtasks)
    path = demo / label_file
    if not segments:
        path.unlink(missing_ok=True)
        return {"ok": True, "complete": False, "problems": problems}

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(segments, indent=4))
    tmp.replace(path)
    covered = segments[-1]["end"] + 1
    return {"ok": True, "complete": covered >= num_frames and not problems, "problems": problems}


@dataclasses.dataclass
class LabelerState:
    data_dir: pathlib.Path
    subtasks: list[str]
    memory_mode: bool = False
    label_file: str = LABEL_FILE
    excluded_demos: frozenset[str] = frozenset()

    def subtasks_for(self, side: str) -> list[str]:
        """Labels offered for a demo; in memory mode the sided phases follow ``side``."""
        return memory_subtasks(side) if self.memory_mode else self.subtasks

    @staticmethod
    def side_of_segments(segments: list[dict]) -> str | None:
        for seg in segments:
            if (side := _side_of(seg["task"])) is not None:
                return side
        return None

    def demo_summaries(self) -> list[dict]:
        """Listing for the sidebar.

        ``num_frames`` is intentionally *not* probed here: ffprobing three videos per demo
        costs ~90 subprocesses and seconds of latency on a shared filesystem, which would
        leave the page blank on startup. The listing only needs labeling progress; exact
        frame counts are probed per demo in :meth:`demo_detail`.
        """
        summaries = []
        for demo in find_demos(self.data_dir):
            if demo.name in self.excluded_demos:
                continue
            segments = load_segments(demo, self.label_file)
            covered = segments[-1]["end"] + 1 if segments else 0
            summaries.append(
                {
                    "name": demo.name,
                    "num_segments": len(segments),
                    "covered": covered,
                    "complete": bool(segments) and covered >= _metadata_steps(demo),
                    "side": self.side_of_segments(segments),
                }
            )
        return summaries

    def demo_detail(self, name: str, side: str = "left") -> dict:
        demo = self.data_dir / name
        num_frames = episode_length(demo)
        _, fps = probe_video(str(demo / TOP_MP4))
        segments = load_segments(demo, self.label_file)
        side = self.side_of_segments(segments) or side
        return {
            "name": demo.name,
            "num_frames": num_frames,
            "fps": fps,
            "segments": segments,
            "subtasks": self.subtasks_for(side),
            "boundaries": [p["boundary"] for p in MEMORY_PHASES] if self.memory_mode else [],
            "memory_mode": self.memory_mode,
            "side": side,
        }


class Handler(http.server.SimpleHTTPRequestHandler):
    state: LabelerState

    def log_message(self, *args):
        """Silence the per-request console spam; video seeks generate hundreds."""

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        url = urllib.parse.urlparse(self.path)
        route = url.path
        query = urllib.parse.parse_qs(url.query)

        if route == "/":
            body = PAGE_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route == "/api/demos":
            self._send_json({"data_dir": str(self.state.data_dir), "demos": self.state.demo_summaries()})
        elif route == "/api/demo":
            self._send_json(self.state.demo_detail(query["name"][0], query.get("side", ["left"])[0]))
        elif route == "/video":
            self._serve_video(query["demo"][0], query["cam"][0])
        else:
            self.send_error(404)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        """Answer the probes some browsers send before streaming a video."""
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        if url.path != "/video":
            self.send_error(404)
            return
        camera = query.get("cam", [""])[0]
        path = self.state.data_dir / query.get("demo", [""])[0] / CAMERAS.get(camera, "")
        if camera not in CAMERAS or not path.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if urllib.parse.urlparse(self.path).path != "/api/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        demo = self.state.data_dir / payload["demo"]
        # Validate against every allowed label, not just the current side's five, so a
        # mixed-side episode is reported by the phase check rather than as "unknown subtask".
        vocabulary = all_memory_subtasks() if self.state.memory_mode else self.state.subtasks
        result = save_segments(
            demo,
            payload["segments"],
            episode_length(demo),
            vocabulary,
            self.state.label_file,
        )
        self._send_json(result)

    def _serve_video(self, demo_name: str, camera: str) -> None:
        """Stream an mp4 with HTTP range support so the browser can seek."""
        if camera not in CAMERAS:
            self.send_error(404)
            return
        path = self.state.data_dir / demo_name / CAMERAS[camera]
        if not path.is_file():
            self.send_error(404)
            return

        size = path.stat().st_size
        start, end = 0, size - 1
        status = 200
        if range_header := self.headers.get("Range"):
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                start = int(match.group(1) or 0)
                end = int(match.group(2) or size - 1)
                status = 206

        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        # Send exactly the requested range: shutil.copyfileobj's `length` is a buffer size,
        # not a limit, so using it here would stream to EOF and contradict Content-Length.
        # A seeking browser abandons in-flight ranges constantly; that shows up as
        # BrokenPipe/ConnectionReset and is normal, not an error worth reporting.
        remaining = length
        try:
            with path.open("rb") as f:
                f.seek(start)
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


PAGE_HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>YAM Subtask Labeler</title>
<style>
  :root {
    --bg:#12141a; --panel:#1b1e26; --line:#2c3040; --text:#e6e8ef; --muted:#98a0b5;
    --accent:#5aa2ff; --good:#43c08a; --warn:#e6b455; --bad:#e0655f;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  #app { display:grid; grid-template-columns:190px 1fr 310px; height:100vh; }
  .col { overflow-y:auto; padding:12px; }
  #demos { border-right:1px solid var(--line); background:var(--panel); }
  #side { border-left:1px solid var(--line); background:var(--panel); }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.09em;
       color:var(--muted); margin:16px 0 8px; font-weight:600; }
  h2:first-child { margin-top:0; }
  .demo { padding:6px 9px; border-radius:6px; cursor:pointer; display:flex;
          justify-content:space-between; gap:6px; align-items:center; }
  .demo:hover { background:#232733; }
  .demo.active { background:var(--accent); color:#08101f; font-weight:600; }
  .dot { width:8px; height:8px; border-radius:50%; background:#4a5064; flex:none; }
  .dot.partial { background:var(--warn); } .dot.complete { background:var(--good); }
  #stage { display:flex; flex-direction:column; padding:12px; gap:10px; min-width:0; }
  #videos { display:grid; grid-template-columns:2fr 1fr; gap:8px; min-height:0; flex:1; }
  #topwrap { position:relative; background:#000; border-radius:8px; overflow:hidden;
             display:flex; align-items:center; justify-content:center; }
  #wrists { display:grid; grid-template-rows:1fr 1fr; gap:8px; min-height:0; }
  .wristwrap { position:relative; background:#000; border-radius:8px; overflow:hidden;
               display:flex; align-items:center; justify-content:center; }
  video { width:100%; height:100%; object-fit:contain; display:block; }
  .tag { position:absolute; top:6px; left:8px; background:#000a; padding:2px 7px;
         border-radius:4px; font-size:11px; color:#cfd6e6; }
  #timeline { height:38px; width:100%; border-radius:6px; background:#0d0f14;
              border:1px solid var(--line); cursor:pointer; display:block; }
  #scrub { width:100%; accent-color:var(--accent); }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  button { background:#262b38; color:var(--text); border:1px solid var(--line);
           border-radius:6px; padding:6px 11px; cursor:pointer; font-size:13px; }
  button:hover:not(:disabled) { background:#303646; }
  button:disabled { opacity:.4; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#08101f; font-weight:600; }
  button.task { width:100%; text-align:left; margin-bottom:5px; }
  button.task.active { background:var(--accent); border-color:var(--accent);
                       color:#08101f; font-weight:600; }
  .hint { font-size:11px; font-weight:400; opacity:.72; margin-top:3px; white-space:normal; }
  .sidetag { font-size:10px; font-weight:700; background:#33405c; color:#a8c4ff;
             border-radius:4px; padding:1px 5px; }
  .seg { display:flex; justify-content:space-between; gap:6px; padding:5px 8px;
         border-radius:5px; background:#232733; margin-bottom:4px; font-size:13px; }
  .seg .rng { color:var(--muted); font-variant-numeric:tabular-nums; }
  #counter { font-variant-numeric:tabular-nums; font-size:15px; font-weight:600; }
  #status { font-size:12px; color:var(--muted); min-height:16px; }
  .pill { padding:2px 8px; border-radius:99px; font-size:11px; background:#262b38; color:var(--muted); }
  .pill.good { background:#173a2c; color:var(--good); }
  .pill.warn { background:#3b3018; color:var(--warn); }
  kbd { background:#262b38; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:4px; padding:0 5px; font-size:11px; font-family:inherit; }
  #help { color:var(--muted); font-size:12px; line-height:1.9; }
</style>

<div id="app">
  <div class="col" id="demos"><h2>Demos</h2><div id="demolist"></div></div>

  <div id="stage">
    <div class="row" style="justify-content:space-between">
      <div class="row">
        <span id="counter">-</span>
        <span class="pill" id="timepill">-</span>
        <span class="pill" id="progresspill">-</span>
      </div>
      <div class="row">
        <button id="prevdemo" title="Previous demo">[</button>
        <button id="nextdemo" title="Next demo">]</button>
      </div>
    </div>

    <div id="videos">
      <div id="topwrap"><video id="vtop" preload="auto" muted playsinline></video>
        <span class="tag">top</span></div>
      <div id="wrists">
        <div class="wristwrap"><video id="vleft" preload="auto" muted playsinline></video>
          <span class="tag">left wrist</span></div>
        <div class="wristwrap"><video id="vright" preload="auto" muted playsinline></video>
          <span class="tag">right wrist</span></div>
      </div>
    </div>

    <canvas id="timeline" height="38"></canvas>
    <input type="range" id="scrub" min="0" max="0" value="0" step="1">
    <div class="row">
      <button data-step="-10">-10 <kbd>H</kbd></button>
      <button data-step="-1">-1 <kbd>J</kbd></button>
      <button id="play">Play <kbd>space</kbd></button>
      <button data-step="1">+1 <kbd>L</kbd></button>
      <button data-step="10">+10 <kbd>K</kbd></button>
      <span id="status"></span>
    </div>
  </div>

  <div class="col" id="side">
    <div id="sidebox" style="display:none">
      <h2>Target bin (this episode)</h2>
      <div class="row">
        <button id="sideleft" style="flex:1">Left <kbd>,</kbd></button>
        <button id="sideright" style="flex:1">Right <kbd>.</kbd></button>
      </div>
    </div>
    <h2>Subtask</h2><div id="tasks"></div>
    <h2>Pending segment</h2>
    <div id="pending" style="color:var(--muted)"></div>
    <button id="mark" class="primary" style="width:100%;margin-top:8px">
      Mark segment end <kbd>enter</kbd></button>
    <h2>Segments</h2><div id="segments"></div>
    <div class="row" style="margin-top:8px">
      <button id="undo">Undo <kbd>U</kbd></button>
      <button id="save">Save <kbd>S</kbd></button>
    </div>
    <h2>Keys</h2>
    <div id="help">
      <kbd>J</kbd>/<kbd>L</kbd> step 1 &nbsp; <kbd>H</kbd>/<kbd>K</kbd> step 10<br>
      <kbd>space</kbd> play/pause &nbsp; <kbd>1</kbd>-<kbd>9</kbd> subtask<br>
      <kbd>enter</kbd> mark end &nbsp; <kbd>U</kbd> undo &nbsp; <kbd>S</kbd> save<br>
      <kbd>[</kbd>/<kbd>]</kbd> prev/next demo
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const vids = [$("vtop"), $("vleft"), $("vright")];
let demos = [], demo = null, frame = 0, taskIdx = 0, segments = [], dirty = false;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
// Seek to the middle of a frame's display interval: the browser renders the frame whose
// interval contains currentTime, so frame/fps alone can land on the boundary and show
// the neighbouring frame.
const frameToTime = f => (f + 0.5) / demo.fps;
const timeToFrame = t => clamp(Math.floor(t * demo.fps), 0, demo.num_frames - 1);
const segStart = () => segments.length ? segments[segments.length - 1].end + 1 : 0;

// Surface failures in the page: a silent throw here leaves an empty shell with black
// video panes and no clue why.
function fail(what, err) {
  console.error(what, err);
  $("status").style.color = "var(--bad)";
  $("status").textContent = `${what}: ${err}`;
  $("demolist").innerHTML =
    `<div style="color:var(--bad);font-size:12px">${what}<br>${err}</div>`;
}
window.addEventListener("error", e => fail("script error", e.message));
window.addEventListener("unhandledrejection", e => fail("request failed", e.reason));

async function loadDemos() {
  $("demolist").innerHTML = `<div style="color:var(--muted);font-size:12px">loading...</div>`;
  try {
    const r = await fetch("/api/demos").then(r => r.json());
    demos = r.demos;
  } catch (e) { fail("could not load demo list", e); return; }
  renderDemos();
  if (!demo) {
    const next = demos.find(d => !d.complete) || demos[0];
    if (next) openDemo(next.name);
  }
}

function renderDemos() {
  $("demolist").innerHTML = "";
  for (const d of demos) {
    const el = document.createElement("div");
    el.className = "demo" + (demo && d.name === demo.name ? " active" : "");
    const cls = d.complete ? "complete" : (d.num_segments ? "partial" : "");
    const side = d.side ? `<span class="sidetag">${d.side[0].toUpperCase()}</span>` : "";
    el.innerHTML = `<span>${d.name}</span><span class="row" style="gap:4px">${side}` +
                   `<span class="dot ${cls}"></span></span>`;
    el.onclick = () => openDemo(d.name);
    $("demolist").appendChild(el);
  }
}

async function openDemo(name) {
  if (dirty && !confirm("Unsaved changes will be lost. Switch demo?")) return;
  try {
    demo = await fetch("/api/demo?name=" + encodeURIComponent(name)).then(r => r.json());
  } catch (e) { fail(`could not open ${name}`, e); return; }
  segments = demo.segments; taskIdx = 0; dirty = false;
  for (const [v, cam] of [[vids[0], "top"], [vids[1], "left"], [vids[2], "right"]])
    v.src = `/video?demo=${encodeURIComponent(name)}&cam=${cam}`;
  $("scrub").max = demo.num_frames - 1;
  renderDemos(); renderTasks(); renderSegments();
  // Resume at the first unlabeled frame.
  vids[0].addEventListener("loadeddata", () => setFrame(segStart()), { once: true });
  setFrame(segStart());
}

function setFrame(f, seek = true) {
  frame = clamp(Math.round(f), 0, demo.num_frames - 1);
  if (seek) { const t = frameToTime(frame); for (const v of vids) v.currentTime = t; }
  $("scrub").value = frame;
  $("counter").textContent = `frame ${frame} / ${demo.num_frames - 1}`;
  $("timepill").textContent = (frame / demo.fps).toFixed(2) + "s";
  const covered = segments.length ? segments[segments.length - 1].end + 1 : 0;
  const pct = Math.round(100 * covered / demo.num_frames);
  const pill = $("progresspill");
  pill.textContent = `${covered}/${demo.num_frames} labeled (${pct}%)`;
  pill.className = "pill " + (covered >= demo.num_frames ? "good" : covered ? "warn" : "");
  renderPending(); drawTimeline();
}

function renderTasks() {
  $("tasks").innerHTML = "";
  demo.subtasks.forEach((name, i) => {
    const b = document.createElement("button");
    b.className = "task" + (i === taskIdx ? " active" : "");
    const hint = demo.boundaries && demo.boundaries[i]
      ? `<div class="hint">${demo.boundaries[i]}</div>` : "";
    b.innerHTML = `<kbd>${i + 1}</kbd> ${name}${hint}`;
    b.onclick = () => { taskIdx = i; renderTasks(); renderPending(); };
    $("tasks").appendChild(b);
  });
  if (demo.memory_mode) {
    $("sidebox").style.display = "";
    $("sideleft").className = demo.side === "left" ? "primary" : "";
    $("sideright").className = demo.side === "right" ? "primary" : "";
    // Side is locked once a sided segment exists; undo it to change the side.
    const locked = segments.some(s => /target bin is|open (left|right) bin/.test(s.task));
    $("sideleft").disabled = $("sideright").disabled = locked;
  }
}

// Re-fetch the label set for the other side. Only allowed before any sided segment exists.
async function setSide(side) {
  if (!demo || demo.side === side) return;
  const keep = segments;
  demo = await fetch(`/api/demo?name=${encodeURIComponent(demo.name)}&side=${side}`)
    .then(r => r.json());
  demo.side = side; segments = keep; demo.segments = keep;
  renderTasks(); renderPending(); drawTimeline();
}

function renderPending() {
  const s = segStart();
  const done = s >= demo.num_frames;
  $("mark").disabled = done || frame < s;
  $("pending").innerHTML = done
    ? `<span style="color:var(--good)">Episode fully covered.</span>`
    : `<b>${demo.subtasks[taskIdx]}</b><br><span class="rng">frames ${s} &rarr; ${frame}</span>` +
      (frame < s ? `<br><span style="color:var(--bad)">end must be &ge; ${s}</span>` : "");
}

function renderSegments() {
  $("segments").innerHTML = "";
  segments.forEach((sg, i) => {
    const el = document.createElement("div");
    el.className = "seg";
    el.innerHTML = `<span>${i + 1}. ${sg.task}</span><span class="rng">${sg.start}-${sg.end}</span>`;
    el.onclick = () => setFrame(sg.start);
    $("segments").appendChild(el);
  });
  $("undo").disabled = !segments.length;
}

function drawTimeline() {
  const c = $("timeline"), ctx = c.getContext("2d");
  c.width = c.clientWidth;
  const w = c.width, h = c.height, n = demo.num_frames;
  ctx.clearRect(0, 0, w, h);
  const colors = ["#5aa2ff", "#43c08a", "#e6b455", "#c07ae0", "#e0655f", "#4fd0d0"];
  segments.forEach((sg, i) => {
    ctx.fillStyle = colors[demo.subtasks.indexOf(sg.task) % colors.length] || "#666";
    const x0 = w * sg.start / n, x1 = w * (sg.end + 1) / n;
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    if (x1 - x0 > 46) {
      ctx.fillStyle = "#0b1220"; ctx.font = "11px ui-sans-serif";
      ctx.fillText(sg.task, x0 + 5, h / 2 + 4, x1 - x0 - 10);
    }
  });
  const s = segStart();
  if (s < n && frame >= s) {  // pending segment preview
    ctx.fillStyle = "#ffffff22";
    ctx.fillRect(w * s / n, 0, w * (frame + 1 - s) / n, h);
  }
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath();
  const x = w * (frame + 0.5) / n;
  ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
}

function mark() {
  const s = segStart();
  if (s >= demo.num_frames || frame < s) return;
  segments.push({ task: demo.subtasks[taskIdx], start: s, end: frame });
  dirty = true;
  renderSegments();
  setFrame(Math.min(frame + 1, demo.num_frames - 1));
  if (segStart() >= demo.num_frames) save();   // autosave on completion
}

function undo() {
  if (!segments.length) return;
  const sg = segments.pop(); dirty = true;
  renderSegments(); setFrame(sg.start);
}

async function save() {
  const r = await fetch("/api/save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ demo: demo.name, segments }),
  }).then(r => r.json());
  dirty = false;
  $("status").textContent = r.problems.length
    ? "saved - " + r.problems.join("; ")
    : (r.complete ? "saved - demo complete" : "saved - partial progress");
  $("status").style.color = r.problems.length ? "var(--bad)" : "var(--muted)";
  const d = demos.find(d => d.name === demo.name);
  if (d) { d.complete = r.complete; d.num_segments = segments.length; renderDemos(); }
}

function stepDemo(delta) {
  const i = demos.findIndex(d => d.name === demo.name);
  const next = demos[i + delta];
  if (next) openDemo(next.name);
}

// Keep the three cameras in lockstep during playback and update the frame readout.
vids[0].addEventListener("timeupdate", () => {
  if (!vids[0].paused) setFrame(timeToFrame(vids[0].currentTime), false);
});
vids[0].addEventListener("play", () => vids.slice(1).forEach(v => {
  v.currentTime = vids[0].currentTime; v.play();
}));
vids[0].addEventListener("pause", () => {
  vids.slice(1).forEach(v => v.pause());
  setFrame(timeToFrame(vids[0].currentTime));
});

$("scrub").oninput = e => { vids[0].pause(); setFrame(+e.target.value); };
$("timeline").onclick = e => {
  vids[0].pause();
  const r = e.target.getBoundingClientRect();
  setFrame(Math.floor(demo.num_frames * (e.clientX - r.left) / r.width));
};
document.querySelectorAll("[data-step]").forEach(b =>
  b.onclick = () => { vids[0].pause(); setFrame(frame + +b.dataset.step); });
$("play").onclick = () => vids[0].paused ? vids[0].play() : vids[0].pause();
$("mark").onclick = mark; $("undo").onclick = undo; $("save").onclick = save;
$("sideleft").onclick = () => setSide("left"); $("sideright").onclick = () => setSide("right");
$("prevdemo").onclick = () => stepDemo(-1); $("nextdemo").onclick = () => stepDemo(1);
window.addEventListener("resize", () => demo && drawTimeline());
window.addEventListener("beforeunload", e => { if (dirty) e.preventDefault(); });

document.addEventListener("keydown", e => {
  if (!demo || e.target.tagName === "INPUT") return;
  const step = d => { vids[0].pause(); setFrame(frame + d); };
  const k = e.key.toLowerCase();
  if (k === "j" || e.key === "ArrowLeft") step(-1);
  else if (k === "l" || e.key === "ArrowRight") step(1);
  else if (k === "h" || e.key === "ArrowDown") step(-10);
  else if (k === "k" || e.key === "ArrowUp") step(10);
  else if (e.key === " ") { $("play").click(); }
  else if (e.key === "Enter") mark();
  else if (k === "u") undo();
  else if (k === "s") save();
  else if (e.key === "[") stepDemo(-1);
  else if (e.key === "]") stepDemo(1);
  else if (e.key === ",") setSide("left");
  else if (e.key === ".") setSide("right");
  else if (/^[1-9]$/.test(e.key) && +e.key <= demo.subtasks.length) {
    taskIdx = +e.key - 1; renderTasks(); renderPending();
  } else return;
  e.preventDefault();
});

loadDemos();
</script>
"""


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        """Keep the console clean when a seeking browser drops a video connection."""
        if not isinstance(sys.exc_info()[1], BrokenPipeError | ConnectionResetError):
            super().handle_error(request, client_address)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--memory-task",
        action="store_true",
        help="use the five-phase bin-memory schema with a per-episode left/right side",
    )
    parser.add_argument("--subtasks", nargs="+", default=list(DEFAULT_SUBTASKS))
    parser.add_argument(
        "--label-file",
        default=LABEL_FILE,
        help="label filename inside each demo (use a versioned overlay without replacing canonical labels)",
    )
    parser.add_argument(
        "--exclude-demos",
        nargs="*",
        default=[],
        help="demo directory names to hide from this review session",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error(f"no such directory: {args.data_dir}")
    excluded_demos = frozenset(args.exclude_demos)
    demos = [demo for demo in find_demos(args.data_dir) if demo.name not in excluded_demos]
    if not demos:
        parser.error(f"no demo* folders in {args.data_dir}")
    if len(args.subtasks) > 9:
        parser.error("at most 9 subtasks (hotkeys 1-9)")
    if pathlib.Path(args.label_file).name != args.label_file:
        parser.error("--label-file must be a filename, not a path")

    Handler.state = LabelerState(
        data_dir=args.data_dir,
        subtasks=args.subtasks,
        memory_mode=args.memory_task,
        label_file=args.label_file,
        excluded_demos=excluded_demos,
    )
    labeled = sum(1 for d in demos if (d / args.label_file).exists())
    print(f"{len(demos)} demos in {args.data_dir} ({labeled} already have {args.label_file})")
    if excluded_demos:
        print(f"excluded from this review session: {', '.join(sorted(excluded_demos))}")
    if args.memory_task:
        print("memory-task schema (pick the episode's side with , / . or the buttons):")
        for i, phase in enumerate(MEMORY_PHASES):
            print(f"  [{i + 1}] {phase['label']:32s} {phase['boundary']}")
    else:
        print(f"subtasks: {', '.join(f'[{i + 1}] {s}' for i, s in enumerate(args.subtasks))}")
    print(f"\n  http://localhost:{args.port}\n")
    print(f"remote? forward the port:  ssh -L {args.port}:localhost:{args.port} $(hostname)\n")

    with _Server(("127.0.0.1", args.port), Handler) as httpd:
        if args.open_browser:
            threading.Timer(0.5, webbrowser.open, [f"http://localhost:{args.port}"]).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
