"""Debater agents for adversarial claim verification.

The Proposer defends the claim and the Critic attacks it. Both share one
implementation and differ only in the side they are assigned. Each agent reasons
about what the claim actually asserts and argues whether the evidence, by
meaning, establishes its substance.

Credibility is optional. When source assessments are supplied, each source is
annotated with a credibility tag the agents may use in their reasoning.
"""

from dataclasses import dataclass

from tridebate.credibility import CredibilityResult
from tridebate.llm_client import LLMClient
from tridebate.models import Claim, SearchResult


@dataclass(frozen=True)
class AgentRole:
    name: str
    stance: str


PROPOSER = AgentRole(
    name="Proposer",
    stance="argue that the evidence establishes the claim and that it is true",
)

CRITIC = AgentRole(
    name="Critic",
    stance="argue that the evidence does not establish the claim, or contradicts it, and that it is false",
)

_BASE_STANCE = """You are the {name} in a structured debate about whether a \
factual claim is true. Your assigned role is to {stance}.

Reason about what the claim actually asserts - its central facts and their \
meaning - and argue whether the evidence establishes that meaning. Judge by \
meaning, not by matching words: evidence can support or contradict a claim using \
different wording, and a claim can state something the evidence phrases \
differently. Focus on whether the substance of the claim is borne out by the \
evidence. Do not let an incidental detail that does not change whether the claim \
is essentially true or false decide your argument; concentrate on the central \
assertion.

You are a committed advocate, like a lawyer representing one side in court. \
Argue your assigned side as strongly as the evidence allows. You may acknowledge \
inconvenient evidence, but do not switch sides or argue for your opponent's \
position."""

_SUB_CLAIM_GUIDANCE = """

To help you reason thoroughly, the claim has been broken into its component \
parts, listed below. Use them to make sure you consider the whole claim - \
including its reasons, causes, dates, numbers, and attributions - not only its \
easiest part. Treat them as a thinking aid, not a checklist of words to match."""

_CREDIBILITY_GUIDANCE = """

Each piece of evidence is prefixed with a credibility tag indicating how \
trustworthy its source is, from [TRUSTED SOURCE] down to [KNOWN MISINFORMATION \
SOURCE]. Use these tags in your reasoning: give more weight to trustworthy \
sources, and point out when the opposing side relies on unreliable ones."""

_RULES = """

Rules:
- Ground every point in the numbered evidence; cite sources by their number.
- If there is a previous round, first rebut the opposing side's most recent \
argument, then add new points.
- Be specific and concise.
- Do not output a verdict; only argue your assigned position. The judge decides."""


def _build_system_prompt(
    role: AgentRole, with_credibility: bool, with_sub_claims: bool
) -> str:
    prompt = _BASE_STANCE.format(name=role.name, stance=role.stance)
    if with_sub_claims:
        prompt += _SUB_CLAIM_GUIDANCE
    if with_credibility:
        prompt += _CREDIBILITY_GUIDANCE
    return prompt + _RULES


def _format_sub_claims(sub_claims: list[str]) -> str:
    return "\n".join(f"  ({i}) {s}" for i, s in enumerate(sub_claims, start=1))


def _format_evidence(
    results: list[SearchResult],
    assessments: list[CredibilityResult] | None,
) -> str:
    if not results:
        return "No evidence was found."
    if assessments is None:
        return "\n\n".join(
            f"[{i}] {r.title}\n{r.content}\nSource: {r.url}"
            for i, r in enumerate(results, start=1)
        )
    return "\n\n".join(
        f"[{i}] {assessment.tag} {r.title}\n{r.content}\nSource: {r.url}"
        for i, (r, assessment) in enumerate(zip(results, assessments), start=1)
    )


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "This is the opening round; there is no prior argument yet."
    return "\n\n".join(f"{name}: {argument}" for name, argument in history)


def run_agent(
    role: AgentRole,
    claim: Claim,
    evidence: list[SearchResult],
    history: list[tuple[str, str]],
    llm_client: LLMClient,
    assessments: list[CredibilityResult] | None = None,
    sub_claims: list[str] | None = None,
) -> str:
    """Produce one debate argument for the given role."""
    system_prompt = _build_system_prompt(
        role,
        with_credibility=assessments is not None,
        with_sub_claims=sub_claims is not None,
    )
    sections = [f"Claim: {claim.text}"]
    if sub_claims is not None:
        sections.append(f"Component parts of the claim:\n{_format_sub_claims(sub_claims)}")
    sections.append(f"Evidence:\n{_format_evidence(evidence, assessments)}")
    sections.append(f"Debate so far:\n{_format_history(history)}")
    sections.append(f"Present your argument as the {role.name}.")

    user_prompt = "\n\n".join(sections)
    return llm_client.generate(system_prompt, user_prompt)