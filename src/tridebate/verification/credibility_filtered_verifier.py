"""Single-agent claim verification with source credibility signals."""

import json
import logging

from tridebate.credibility import CredibilityAssessor, CredibilityResult
from tridebate.llm_client import LLMClient
from tridebate.models import Claim, SearchResult, Verdict, VerdictResult
from tridebate.search_client import SearchClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fact-checking assistant. You will be given a \
claim and a set of evidence retrieved from the web. Each piece of evidence is \
prefixed with a credibility tag indicating how trustworthy its source is:

- [TRUSTED SOURCE]: a source with an established reliable reputation.
- [NO KNOWN ISSUES]: a source with no documented credibility problems.
- [UNVERIFIED SOURCE]: a source that is neither established nor documented, and \
should be treated with caution.
- [MIXED RELIABILITY]: a source known to be inconsistent in reliability.
- [KNOWN MISINFORMATION SOURCE]: a source documented as spreading misinformation, \
which should carry little weight.

Weigh the evidence according to these tags when reaching your verdict.

Respond only in JSON with the following structure:
{"verdict": "Supported" | "Refuted" | "Not Enough Evidence" | "Conflicting Evidence", \
"reasoning": "<concise justification citing the evidence>"}"""


def _format_evidence(
    results: list[SearchResult], assessments: list[CredibilityResult]
) -> str:
    if not results:
        return "No evidence was found."

    lines = [
        f"[{i}] {assessment.tag} {r.title}\n{r.content}\nSource: {r.url}"
        for i, (r, assessment) in enumerate(zip(results, assessments), start=1)
    ]
    return "\n\n".join(lines)


def _summarise_credibility(assessments: list[CredibilityResult]) -> dict:
    total = len(assessments)

    tier_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for a in assessments:
        tier_counts[a.tier.name] = tier_counts.get(a.tier.name, 0) + 1
        source_counts[a.source_of_rating] = source_counts.get(a.source_of_rating, 0) + 1

    unverified = sum(1 for a in assessments if a.tag == "[UNVERIFIED SOURCE]")
    conflicts = sum(1 for a in assessments if a.cred1_mbfc_conflict)

    return {
        "total_sources": total,
        "rated_by_cred1": source_counts.get("cred1", 0),
        "rated_by_mbfc": source_counts.get("mbfc", 0),
        "rated_by_tranco": source_counts.get("tranco", 0),
        "unverified": unverified,
        "cred1_mbfc_conflicts": conflicts,
        "tier_distribution": tier_counts,
    }


def _parse_verdict(raw_response: str) -> tuple[Verdict, str]:
    """Parse the JSON response, falling back to Not Enough Evidence on failure."""
    try:
        parsed = json.loads(raw_response)
        verdict = Verdict(parsed["verdict"])
        reasoning = parsed.get("reasoning", "")
        return verdict, reasoning
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Failed to parse LLM response: %s", raw_response)
        return Verdict.NOT_ENOUGH_EVIDENCE, "Failed to parse model response."


def verify_claim(
    claim: Claim,
    search_client: SearchClient,
    llm_client: LLMClient,
    assessor: CredibilityAssessor,
) -> VerdictResult:
    """Verify a claim using credibility-annotated live search evidence."""
    evidence = search_client.search(claim.text)
    assessments = [assessor.assess(r.url) for r in evidence]
    evidence_block = _format_evidence(evidence, assessments)

    user_prompt = f"Claim: {claim.text}\n\nEvidence:\n{evidence_block}"
    raw_response = llm_client.generate(SYSTEM_PROMPT, user_prompt)
    verdict, reasoning = _parse_verdict(raw_response)

    return VerdictResult(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        verdict=verdict,
        reasoning=reasoning,
        sources_used=evidence,
        credibility_stats=_summarise_credibility(assessments),
    )