"""Public entry point for the Gemini prompt rewriter."""

from __future__ import annotations

import numpy as np

from rewriter.gemini import DEFAULT_MODEL, rewrite_via_gemini


def rewrite(
    image_uint8: np.ndarray,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    """Rewrite a robot instruction using a Gemini VLM call.

    Args:
        image_uint8: (H, W, 3) uint8 keyframe — the retrieved memory image.
        prompt: original instruction string.
        model: Gemini model name (default: gemini-2.5-flash).

    Returns:
        Rewritten prompt, vocabulary-regulated.
    """
    return rewrite_via_gemini(image_uint8, prompt, model=model)
