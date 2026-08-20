"""Condition D verification: claim decomposition, then adversarial debate over
credibility-tagged evidence, judged holistically against the sub-claim checklist."""

from tridebate.credibility import CredibilityAssessor, CredibilityResult
from tridebate.debate.debate_loop import run_debate
from tridebate.debate.decomposition import decompose_claim
from tridebate.debate.judge import judge_debate
from tridebate.llm_client import LLMClient
from tridebate.models import Claim, SearchResult, VerdictResult
from tridebate.search_client import SearchClient

SOURCES_PER_ROUND = 10


def _accumulating_slices(items: list, rounds: int) -> list[list]:
    """Build growing prefixes: round k sees the first k * SOURCES_PER_ROUND items."""
    return [items[: SOURCES_PER_ROUND * (k + 1)] for k in range(rounds)]


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


def verify_claim(
    claim: Claim,
    search_client: SearchClient,
    llm_client: LLMClient,
    assessor: CredibilityAssessor,
    rounds: int = 3,
) -> tuple[VerdictResult, dict]:
    """Verify a claim via decomposition-guided, credibility-aware debate."""
    sub_claims = decompose_claim(claim, llm_client)

    evidence = search_client.search(
        claim.text, max_results=SOURCES_PER_ROUND * rounds
    )
    assessments = [assessor.assess(r.url) for r in evidence]

    evidence_per_round = _accumulating_slices(evidence, rounds)
    assessments_per_round = _accumulating_slices(assessments, rounds)

    turns = run_debate(
        claim, evidence_per_round, llm_client, assessments_per_round, sub_claims
    )
    verdict, reasoning = judge_debate(
        claim, turns, evidence, llm_client, assessments, sub_claims
    )

    result = VerdictResult(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        verdict=verdict,
        reasoning=reasoning,
        sources_used=evidence,
        credibility_stats=_summarise_credibility(assessments),
    )

    transcript = {
        "claim_id": claim.claim_id,
        "claim_text": claim.text,
        "sub_claims": sub_claims,
        "rounds": rounds,
        "sources_retrieved": len(evidence),
        "turns": turns,
        "judge_verdict": verdict.value,
        "judge_reasoning": reasoning,
    }

    return result, transcript