"""Transform-faithful token-length audit for the YAM subtask configs.

Iterates every frame of a (non-memory) subtask config's fully transformed dataset with an
oversized token budget, so nothing truncates, and reports the true maxima of

  * the ar=0 context segment  ("Task: {prompt}, State: {state};\\n")      -> max_token_len
  * the ar=1 causal segment   ("{subtask}\\n" + "Action: " + FAST + "|")  -> causal_token_len

Each frame's tokenization is identical wherever it later appears inside a memory sequence
sample (same normalized state, same lookahead subtask, same action chunk), so a per-frame
sweep is exhaustive for the sequence configs too. Norm stats must exist (run
scripts/compute_norm_stats.py first): FAST token counts depend on normalized actions.

Usage:
    uv run scripts/v33_audit_token_lengths.py --config-name pi05_yam_0816
"""

# ruff: noqa: I001 - data_loader (torch) must import before config (tensorflow), or the
# interpreter segfaults; same pinned order as data_loader_test.py.
import dataclasses
import logging

import numpy as np
import torch
import tqdm
import tyro

import openpi.training.data_loader as _data_loader
import openpi.training.config as _config

# Generous ceiling so no frame can truncate; anything close to it would be a data bug.
_AUDIT_TOKEN_LEN = 1024


def main(config_name: str = "pi05_yam_0816", *, num_workers: int = 8, batch_size: int = 32):
    config = _config.get_config(config_name)
    model_config = dataclasses.replace(config.model, max_token_len=_AUDIT_TOKEN_LEN)
    if getattr(model_config, "predict_with_memory", False):
        raise ValueError("audit with the plain (non-memory) companion config; lengths transfer 1:1.")
    data_config = config.data.create(config.assets_dirs, model_config)

    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda items: items,
    )

    max_context = max_causal = 0
    context_argmax = causal_argmax = -1
    index = 0
    for items in tqdm.tqdm(loader, desc="auditing frames"):
        for item in items:
            valid = np.asarray(item["tokenized_prompt_mask"], dtype=bool)
            causal = np.asarray(item["token_ar_mask"], dtype=bool) & valid
            context_len = int((valid & ~causal).sum())
            causal_len = int(causal.sum())
            if context_len > max_context:
                max_context, context_argmax = context_len, index
            if causal_len > max_causal:
                max_causal, causal_argmax = causal_len, index
            index += 1

    if max(max_context, max_causal) >= _AUDIT_TOKEN_LEN - 1:
        raise RuntimeError("audit ceiling reached; a frame tokenizes suspiciously long.")
    print(f"audited {index} frames")
    print(f"max context length: {max_context} (frame {context_argmax}) -> max_token_len must be >= {max_context}")
    print(f"max causal length: {max_causal} (frame {causal_argmax}) -> causal_token_len must be >= {max_causal}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
