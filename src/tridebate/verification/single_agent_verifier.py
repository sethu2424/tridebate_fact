"""Single-agent claim verification using live search evidence.

This module implements the baseline verification stage: a claim is
checked against live web search results using a single LLM call, with
no adversarial debate and no source credibility filtering applied.
"""

import json
import logging

from tridebate.llm_client import LLMClient
from tridebate.models import Claim, SearchResult, Verdict, VerdictResult
from tridebate.search_client import SearchClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fact-checking assistant. You will be given a \
claim and a set of evidence retrieved from the web. Assess whether the \
evidence supports, refutes, or is insufficient to judge the claim.

Respond only in JSON with the following structure:
{"verdict": "Supported" | "Refuted" | "Not Enough Evidence" | "Conflicting Evidence", \
"reasoning": "<concise justification citing the evidence>"}"""


def _format_evidence(results: list[SearchResult]) -> str:
    """Convert search results into a numbered evidence block for the prompt."""
    if not results:
        return "No evidence was found."

    lines = [
        f"[{i}] {r.title}\n{r.content}\nSource: {r.url}"
        for i, r in enumerate(results, start=1)
    ]
    return "\n\n".join(lines)


def _parse_verdict(raw_response: str) -> tuple[Verdict, str]:
    """Parse the LLM's JSON response into a Verdict and reasoning string.

    Falls back to Not Enough Evidence if parsing fails, since an
    unparseable response provides no reliable signal either way.
    """
    try:
        parsed = json.loads(raw_response)
        verdict = Verdict(parsed["verdict"])
        reasoning = parsed.get("reasoning", "")
        return verdict, reasoning
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Failed to parse LLM response: %s", raw_response)
        return Verdict.NOT_ENOUGH_EVIDENCE, "Failed to parse model response."


def verify_claim(
    claim: Claim, search_client: SearchClient, llm_client: LLMClient
) -> VerdictResult:
    """Verify a claim using live search evidence and a single LLM call."""
    evidence = search_client.search(claim.text)
    evidence_block = _format_evidence(evidence)

    user_prompt = f"Claim: {claim.text}\n\nEvidence:\n{evidence_block}"
    raw_response = llm_client.generate(SYSTEM_PROMPT, user_prompt)
    verdict, reasoning = _parse_verdict(raw_response)

    return VerdictResult(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        verdict=verdict,
        reasoning=reasoning,
        sources_used=evidence,
    )
