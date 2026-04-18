"""
VLM Feedback Verifier
=====================
Gọi LLM lần 2 để kiểm tra (verify) xem output mà LLM chính đã gen ra có đúng không,
dựa trên: ảnh có labeled ID, input text (task), và output text (action đề xuất).

Returns a dict:
    {
        "is_correct": bool,
        "reason": str,               # giải thích tại sao đúng/sai
        "corrected_id": int | None,  # nếu sai, gợi ý id đúng
        "corrected_class": str | None
    }
"""

import json
import logging
import re

from utils.config import call_llm_with_retry
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")

_VERIFIER_SYSTEM_PROMPT = """\
You are a quality-checker for a robotic bin-picking system with a parallel gripper.

Your job: verify whether the proposed action is reasonable given the labeled image and the user's task.

## What to check
1. **Correct target identification**: Does the chosen object match the user's description (color, type, position)?
2. **Clear physical blocking**: Is another object clearly STACKED ON TOP of the chosen object, physically preventing a gripper from reaching it? Only count objects that are truly resting on or covering the chosen object.
3. **If target is blocked**: Is the proposed obstacle actually one of the objects physically on top of the target?

## IMPORTANT: Avoid false occlusion claims
- Objects placed NEXT TO each other (side by side) are NOT occluding each other.
- Objects that merely TOUCH at the edges are NOT occluding each other.
- Only flag occlusion when one object is clearly ON TOP of another, physically blocking gripper access from above.
- When in doubt, assume the object IS free. Do not over-analyze.

## Response format
If CORRECT:
{
  "is_correct": true,
  "reason": "<brief explanation>",
  "corrected_id": null,
  "corrected_class": null
}

If INCORRECT:
{
  "is_correct": false,
  "reason": "<explain what is wrong>",
  "corrected_id": <correct object_id or null>,
  "corrected_class": "<correct class name or null>"
}

Respond ONLY with valid JSON.
"""


def verify_vlm_output(
    labeled_image_b64: str,
    input_text: str,
    vlm_output_text: str,
) -> dict:
    """
    Verify the VLM's proposed action.

    Args:
        labeled_image_b64: Base64-encoded labeled image (Molmo output with ID badges).
        input_text:        The user's task description (e.g. "take the red bowl on the left").
        vlm_output_text:   Raw text output from the main VLM (e.g. "[1, red bowl]").

    Returns:
        dict with keys: is_correct, reason, corrected_id, corrected_class
    """
    user_content = [
        {
            "type": "text",
            "text": (
                f"User task: {input_text}\n\n"
                f"Proposed action from VLM: {vlm_output_text}\n\n"
                "Look at the labeled image and verify whether this proposed action is correct."
            )
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{labeled_image_b64}"}
        }
    ]

    messages = [
        {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = call_llm_with_retry(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            top_p=1,
        )
        # Extract JSON from response (may be wrapped in markdown code fences)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw)

        # Normalise keys
        return {
            "is_correct": bool(result.get("is_correct", True)),
            "reason": str(result.get("reason", "")),
            "corrected_id": result.get("corrected_id"),
            "corrected_class": result.get("corrected_class"),
        }

    except Exception as e:
        logging.warning(f"[VLMFeedback] Verifier failed: {e}. Assuming correct to avoid blocking.")
        return {
            "is_correct": True,
            "reason": f"Verifier error (skipped): {e}",
            "corrected_id": None,
            "corrected_class": None,
        }
