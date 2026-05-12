"""Gemini client wrapper: structured rewrite call + disk cache + vocab retry.

Reads GEMINI_API_KEY from environment.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pathlib
from typing import Any

import numpy as np
from PIL import Image

from rewriter.vocabulary import (
    OBJECTS_NATURAL,
    PLATES_NATURAL,
    is_in_vocabulary,
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CACHE_DIR = pathlib.Path("rewriter/cache/responses")

_LOG = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You receive an IMAGE (a keyframe) and a PROMPT (a robot instruction).
Rewrite PROMPT into a clearer instruction for the robot, using the
IMAGE as visual context.

Constraint — when you mention an object or a plate, you MUST use the
canonical name from the vocabulary below. Do not invent synonyms,
descriptions, or substitutes.

  Objects: {objects}
  Plates:  {plates}

Output JSON only: {{"rewritten_prompt": "..."}}.""".format(
    objects=", ".join(OBJECTS_NATURAL),
    plates=", ".join(PLATES_NATURAL),
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"rewritten_prompt": {"type": "string"}},
    "required": ["rewritten_prompt"],
}


def _image_bytes(image_uint8: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array to PNG bytes (lossless, stable hash)."""
    if image_uint8.dtype != np.uint8 or image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8, got {image_uint8.shape} {image_uint8.dtype}")
    buf = io.BytesIO()
    Image.fromarray(image_uint8).save(buf, format="PNG")
    return buf.getvalue()


def _cache_key(image_bytes: bytes, user_prompt: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(b"\0")
    h.update(user_prompt.encode("utf-8"))
    h.update(b"\0")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _call_gemini(client, model: str, image_bytes: bytes, user_prompt: str, extra_note: str = "") -> str:
    """One Gemini call. Returns the rewritten_prompt string from the JSON."""
    from google.genai import types

    sys_instruction = SYSTEM_PROMPT + (("\n\n" + extra_note) if extra_note else "")
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            user_prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.0,
        ),
    )
    parsed = json.loads(response.text)
    return str(parsed["rewritten_prompt"])


def rewrite_via_gemini(
    image_uint8: np.ndarray,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    cache_dir: pathlib.Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    max_retries: int = 1,
) -> str:
    """Rewrite a robot prompt using Gemini, with vocabulary enforcement.

    Returns the rewritten string. Caches (request, response) on disk keyed by
    SHA-256(image, prompt, model) so re-running is free.
    """
    cache_dir = pathlib.Path(cache_dir)
    img_bytes = _image_bytes(image_uint8)
    key = _cache_key(img_bytes, user_prompt, model)
    cache_path = cache_dir / f"{key}.json"

    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())["rewritten_prompt"]

    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")
    client = genai.Client(api_key=api_key)

    rewritten = _call_gemini(client, model, img_bytes, user_prompt)
    if not is_in_vocabulary(rewritten) and max_retries > 0:
        reminder = (
            "Your previous output violated the vocabulary constraint. Only use "
            "the canonical object/plate names listed above. Retry."
        )
        rewritten_retry = _call_gemini(client, model, img_bytes, user_prompt, extra_note=reminder)
        if is_in_vocabulary(rewritten_retry):
            rewritten = rewritten_retry
        else:
            _LOG.warning("off-vocabulary after retry: %r", rewritten_retry)
            rewritten = rewritten_retry  # surface model's best attempt

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "model": model,
        "user_prompt": user_prompt,
        "rewritten_prompt": rewritten,
    }))
    return rewritten
