# Plate Memory Task — Base Model Training Plan

## Purpose

This document describes the **base / no-memory baseline** for the plate memory task.

The goal is to train a standard pi0.5 LoRA policy on the collected plate-memory episodes **without any memory image, memory bank, or attention retrieval module**.

This base model is important because it tells us how well a normal VLA policy can do when it only receives:

1. current camera observations
2. current robot state
3. current language instruction

It should be compared later against the memory-conditioned model.

---

## Task Summary

Physical setup:

Objects:
- gray paper box
- red chili
- yellow cheese
- green block

Plates:
- green plate
- light blue plate
- orange plate
- purple plate

Other:
- plastic bin

Each episode contains multiple segments:

1. **observe/reference**
   - Instruction:
     > Observe and remember which object is on each colored plate.

2. **mix**
   - Instruction:
     > Move all objects into the plastic bin.

3. **query segments**
   - Example instruction:
     > Place the object originally on the orange plate onto the green plate.

The key point is that during query segments, the current image only shows the mixed objects and plates. The original object-to-plate mapping is no longer directly visible.

---

## What This Base Model Should Train On

The base model should use the original collected episode data.

For each frame, the training sample should contain:

- top camera image
- left camera image
- right camera image
- robot state
- robot action
- segment-level language instruction

The base model should **not** receive:

- memory image
- reference crop
- memory bank
- oracle mapping
- resolved object-name instruction

In other words, for query segments, keep the original memory-dependent instruction.

Example:

Use this:

> Place the object originally on the orange plate onto the green plate.

Do **not** convert it to:

> Pick up the yellow cheese and place it onto the green plate.

This is intentional. The base model is the no-memory baseline.

---

## Data Structure to Inspect

Claude Code should first inspect the collected dataset structure.

Expected raw episode structure:

```text
dataset/PLATE_MEMORY_TASK/
  episode_0001/
    top_camera_rgb.mp4
    left_camera_rgb.mp4
    right_camera_rgb.mp4
    left_joint_positions.npy
    left_control.npy
    metadata.json
    events.json
    write_complete.flag
```

The most important file is `events.json`.

It should contain segment boundaries and instructions, such as:

```text
observe: start_frame, end_frame, instruction
mix: start_frame, end_frame, instruction
query_1: start_frame, end_frame, instruction, source_plate, target_plate, correct_object
query_2: ...
```

The converter should use these segment boundaries to assign the correct language instruction to every frame.

---

## Conversion Goal

Create a LeRobot/openpi-compatible dataset for the base model.

Suggested repo id:

```text
kewalk/plate_memory_base
```

For every frame:

1. load the three camera images
2. load robot state from `left_joint_positions.npy`
3. load action from `left_control.npy`
4. determine which segment the frame belongs to using `events.json`
5. assign the segment instruction as the `task` string

Example:

```text
observe frames:
task = "Observe and remember which object is on each colored plate."

mix frames:
task = "Move all objects into the plastic bin."

query_1 frames:
task = "Place the object originally on the orange plate onto the green plate."
```

Important: unlike the earlier cup task, this dataset has **different task strings inside the same episode**.

---

## Base Policy Definition

Train a normal pi0.5 LoRA policy with the same observation structure as the previous YAM/cup policy:

Inputs:
- 3 camera images
- robot state
- language instruction

Output:
- robot action

No additional memory image input should be added for this base model.

This should be the clean baseline architecture.

---

## Training Steps

High-level steps:

1. Validate raw episodes.
2. Convert raw episodes into LeRobot/openpi format.
3. Register a new openpi training config for the base model.
4. Compute normalization statistics.
5. Run a short smoke training job.
6. Run the full LoRA training job.
7. Save checkpoints under a clear experiment name.

Suggested config / experiment names:

```text
config name: pi05_plate_memory_base_lora
experiment name: plate_memory_base_v1
```

The training setup can follow the previous pi0.5 YAM/cup LoRA pipeline, but with the new dataset and segment-level task strings.

---

## What We Expect From This Base Model

This model may learn the manipulation skills:

- move objects into the bin
- pick objects from the bin
- place objects onto colored plates

But it is **not expected to reliably solve the memory-dependent query**, because it does not receive the reference memory during query time.

That is okay. This is the baseline.

The expected behavior is:

- good or reasonable performance on simple visible manipulation
- weak performance on queries that require knowing the original object-to-plate mapping

---

## Evaluation Plan

Evaluate the base model in two stages.

### 1. Offline Evaluation

Replay held-out recorded episodes frame by frame.

For each frame:
- use the current recorded images
- use the current recorded state
- use the segment instruction from `events.json`
- run policy inference
- compare predicted action to recorded action

Report:
- action MAE / RMSE
- per-segment action error
- especially query-segment error

The important comparison is query-segment performance.

---

### 2. Real Robot / Closed-Loop Evaluation

Use a simple high-level stage controller.

The policy does not need to know when a stage is done.

The evaluation script can manually or scriptedly switch stages:

```text
OBSERVE:
prompt = "Observe and remember which object is on each colored plate."

MIX:
prompt = "Move all objects into the plastic bin."

QUERY_1:
prompt = "Place the object originally on the orange plate onto the green plate."

QUERY_2:
prompt = ...
```

During each stage, repeatedly call the policy with the current prompt.

For early experiments, stage switching can be manual:

```text
press "n" to move to the next stage
```

Automatic done detection is not required for this baseline.

---

## Base Model Metrics

Report:

1. **Mix success**
   - Did the robot move all objects into the bin?

2. **Query object accuracy**
   - During a query, did the robot pick the correct object?

3. **Target plate accuracy**
   - Did the robot place the object on the requested target plate?

4. **End-to-end query success**
   - Correct object + correct target plate

5. **Full episode success**
   - All query segments completed correctly

This base model should later be compared against:

- no-memory base model
- random-memory model
- oracle-memory model
- prompt-attention-memory model

---

## Important Clarifications

The base model is **not** the final proposed method.

It is a baseline to answer:

> What happens if we train pi0.5 directly on this task without any explicit memory retrieval?

The final proposed method will add visual memory input and attention-based retrieval.

For now, this base model should stay simple:
- no memory image
- no attention retrieval
- no object-name rewriting
- no extra memory module
- original segment instructions only

---

## Final Summary

Train a standard pi0.5 LoRA policy on the plate-memory dataset using segment-level instructions from `events.json`, without any memory input. This gives the no-memory baseline for the task. During inference, the script switches prompts between observe, mix, and query stages, either manually or by fixed timing. This base model is expected to struggle on query segments, which motivates the later memory-conditioned policy.
