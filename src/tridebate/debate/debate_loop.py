"""Orchestration of the multi-round adversarial debate.

Each round consists of a Proposer turn followed by a Critic turn. Both agents
see the transcript accumulated so far, so later turns rebut earlier ones. The
loop is agnostic to how evidence is selected; it receives, for each round, the
evidence slice both agents argue over, and optionally the credibility
assessments for that slice and the claim's sub-claims.
"""

from tridebate.credibility import CredibilityResult
from tridebate.debate.agents import CRITIC, PROPOSER, run_agent
from tridebate.llm_client import LLMClient
from tridebate.models import Claim, SearchResult

DebateTurn = dict[str, object]


def run_debate(
    claim: Claim,
    evidence_per_round: list[list[SearchResult]],
    llm_client: LLMClient,
    assessments_per_round: list[list[CredibilityResult]] | None = None,
    sub_claims: list[str] | None = None,
) -> list[DebateTurn]:
    """Run the debate and return the transcript as ordered, round-tagged turns.

    One round is played per entry in evidence_per_round. Credibility assessments
    and sub-claims are optional; when supplied they are passed to the agents.
    """
    transcript: list[DebateTurn] = []
    history: list[tuple[str, str]] = []

    for round_index, round_evidence in enumerate(evidence_per_round, start=1):
        round_assessments = (
            assessments_per_round[round_index - 1]
            if assessments_per_round is not None
            else None
        )

        proposer_argument = run_agent(
            PROPOSER, claim, round_evidence, history, llm_client,
            round_assessments, sub_claims,
        )
        history.append((PROPOSER.name, proposer_argument))
        transcript.append(
            {"round": round_index, "speaker": PROPOSER.name, "argument": proposer_argument}
        )

        critic_argument = run_agent(
            CRITIC, claim, round_evidence, history, llm_client,
            round_assessments, sub_claims,
        )
        history.append((CRITIC.name, critic_argument))
        transcript.append(
            {"round": round_index, "speaker": CRITIC.name, "argument": critic_argument}
        )

    return transcript