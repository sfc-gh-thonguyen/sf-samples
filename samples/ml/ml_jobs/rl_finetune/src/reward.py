"""Reward functions for Medical SOAP RL training.

This module implements reward functions for evaluating SOAP note generation,
including JSON parsing validation and optional LLM-as-judge evaluation.
"""

import json
import logging
import re

logger = logging.getLogger("MedicalSOAPReward")

SOAP_KEYS = {"S", "O", "A", "P"}


def extract_json_from_response(response: str) -> dict | None:
    """Extract JSON object from model response.

    Handles common cases like markdown code fences and extra whitespace.
    """
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(json_pattern, response)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(response[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def validate_soap_json(parsed: dict) -> bool:
    """Validate that parsed JSON has exactly the required SOAP keys."""
    if not isinstance(parsed, dict):
        return False

    if set(parsed.keys()) != SOAP_KEYS:
        return False

    for key in SOAP_KEYS:
        if not isinstance(parsed[key], str):
            return False

    return True


def medical_soap_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    pred_S: str,
    pred_O: str,
    pred_A: str,
    pred_P: str,
    **kwargs,
) -> float:
    """Compute reward for SOAP note generation.

    Returns:
        Reward value between 0.0 and 1.0:
        - 0.0: Invalid JSON or missing/wrong keys
        - 0.5: Valid JSON structure with correct keys
        - 0.75: Valid JSON with all non-empty sections
        - 1.0: Valid JSON with substantial content in all sections
    """
    try:
        parsed = extract_json_from_response(completions)

        if parsed is None:
            logger.debug("Failed to parse JSON from response")
            return 0.0

        if not validate_soap_json(parsed):
            logger.debug(f"Invalid SOAP structure. Keys found: {parsed.keys()}")
            return 0.0

        reward = 0.5

        non_empty_count = sum(
            1 for key in SOAP_KEYS if parsed[key] and len(parsed[key].strip()) > 0
        )

        if non_empty_count == 4:
            reward = 0.75

            substantial_count = sum(
                1 for key in SOAP_KEYS if len(parsed[key].strip()) >= 20
            )
            if substantial_count == 4:
                reward = 1.0

        return reward

    except Exception as e:
        logger.warning(f"Exception in medical_soap_reward_fn: {e}", exc_info=True)
        return 0.0


def medical_soap_reward_fn_strict(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    pred_S: str,
    pred_O: str,
    pred_A: str,
    pred_P: str,
    **kwargs,
) -> float:
    """Strict binary reward function for SOAP note generation.

    Returns 1.0 only if the output is valid JSON with all SOAP keys and
    non-empty content in each section. Otherwise returns 0.0.
    """
    try:
        parsed = extract_json_from_response(completions)

        if parsed is None:
            return 0.0

        if not validate_soap_json(parsed):
            return 0.0

        for key in SOAP_KEYS:
            if len(parsed[key].strip()) < 10:
                return 0.0

        return 1.0

    except Exception:
        return 0.0
