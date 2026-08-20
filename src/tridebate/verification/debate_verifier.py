"""Condition B verification: adversarial debate over live search evidence."""

from tridebate.debate.debate_loop import run_debate
from tridebate.debate.judge import judge_debate
from tridebate.llm_client import LLMClient
from tridebate.models import Claim, SearchResult, VerdictResult
from tridebate.search_client import SearchClient

SOURCES_PER_ROUND = 10


def _accumulating_slices(
    evidence: list[SearchResult], rounds: int
) -> list[list[SearchResult]]:
    """Build growing evidence prefixes: round k sees the first k * SOURCES_PER_ROUND."""
    return [evidence[: SOURCES_PER_ROUND * (k + 1)] for k in range(rounds)]


def verify_claim(
    claim: Claim,
    search_client: SearchClient,
    llm_client: LLMClient,
    rounds: int = 3,
) -> tuple[VerdictResult, dict]:
    """Verify a claim through debate, returning the verdict and a debate transcript."""
    evidence = search_client.search(
        claim.text, max_results=SOURCES_PER_ROUND * rounds
    )
    evidence_per_round = _accumulating_slices(evidence, rounds)

    turns = run_debate(claim, evidence_per_round, llm_client)
    verdict, reasoning = judge_debate(claim, turns, evidence, llm_client)

    result = VerdictResult(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        verdict=verdict,
        reasoning=reasoning,
        sources_used=evidence,
    )

    transcript = {
        "claim_id": claim.claim_id,
        "claim_text": claim.text,
        "rounds": rounds,
        "sources_retrieved": len(evidence),
        "turns": turns,
        "judge_verdict": verdict.value,
        "judge_reasoning": reasoning,
    }

    return result, transcript