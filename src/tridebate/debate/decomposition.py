"""Claim decomposition into atomic, independently-verifiable sub-claims."""

import json
import logging

from tridebate.llm_client import LLMClient
from tridebate.models import Claim

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You break a factual claim into its atomic sub-claims: the \
separate factual components that must each be true for the whole claim to be \
true. Separate distinct events, reasons, causes, dates, numbers, and \
attributions into their own sub-claims. A compound claim is true only if every \
sub-claim is true, so each false or unsupported component must be checkable on \
its own.

For example, "X withdrew from the tournament after testing positive for the \
virus" has two sub-claims: that X withdrew from the tournament, and that the \
reason was a positive test for the virus.

Respond only in JSON with this structure:
{"sub_claims": ["<sub-claim 1>", "<sub-claim 2>", ...]}"""


def decompose_claim(claim: Claim, llm_client: LLMClient) -> list[str]:
    """Return the atomic sub-claims of a claim, or the whole claim if none are found."""
    user_prompt = f"Claim: {claim.text}\n\nBreak this claim into its atomic sub-claims."
    raw_response = llm_client.generate(SYSTEM_PROMPT, user_prompt)

    try:
        parsed = json.loads(raw_response)
        sub_claims = parsed["sub_claims"]
        if isinstance(sub_claims, list) and all(isinstance(s, str) for s in sub_claims):
            cleaned = [s.strip() for s in sub_claims if s.strip()]
            if cleaned:
                return cleaned
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Failed to parse decomposition response: %s", raw_response)

    return [claim.text]