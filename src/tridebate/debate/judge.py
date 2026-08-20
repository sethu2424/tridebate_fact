"""Judge for the adversarial debate.

Reads the full debate transcript and the evidence, then returns a single verdict
on the whole claim by weighing which side's reasoning better accounts for
whether the evidence establishes the claim's meaning. Credibility and sub-claims
are optional.
"""

import json
import logging
import re

from tridebate.credibility import CredibilityResult
from tridebate.llm_client import LLMClient
from tridebate.models import Claim, SearchResult, Verdict

logger = logging.getLogger(__name__)

_BASE_INTRO = """You are the judge of a structured debate about whether a \
factual claim is true. A Proposer has argued that the evidence establishes the \
claim, and a Critic has argued that it does not or contradicts it, over several \
rounds.

Your task is to weigh the two sides' reasoning and decide which better accounts \
for whether the evidence establishes what the claim actually asserts. Judge by \
meaning, not by matching words: consider whether the substance of the claim - \
its central facts and their meaning - is borne out by the evidence, regardless \
of exact wording. An incidental detail that does not change whether the claim is \
essentially true or false should not decide the verdict."""

_SUB_CLAIM_GUIDANCE = """

The claim's component parts are listed below to help you consider it in full - \
including reasons, causes, dates, numbers, and attributions - rather than only \
its most obvious part. Treat them as a thinking aid, not a checklist of words to \
match."""

_CREDIBILITY_GUIDANCE = """

Each source carries a credibility tag and a numeric weight from 0.0 to 1.0, \
where higher means more trustworthy. Weigh each side's case by the credibility \
of the sources it relies on: a position resting on trusted, high-weight sources \
should outweigh one resting on low-weight or flagged sources, even if the latter \
is argued more forcefully."""

_INSTRUCTIONS = """

Reach a decision and commit to it. Choose:
- "Supported" if, on balance, the evidence establishes what the claim asserts.
- "Refuted" if the evidence contradicts what the claim asserts, or shows a \
central part of it to be false.
- "Conflicting Evidence" if there is credible evidence on both sides that cannot \
be reconciled.
- "Not Enough Evidence" only when there is genuinely no relevant evidence bearing \
on the central assertion; do not use this label merely because the evidence is \
incomplete or a minor detail is unaddressed.

Respond only in JSON with this structure:
{"verdict": "Supported" | "Refuted" | "Not Enough Evidence" | "Conflicting Evidence", \
"reasoning": "<concise justification weighing both sides against the evidence>"}"""

DebateTurn = dict[str, object]


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


def _format_credibility_reference(assessments: list[CredibilityResult]) -> str:
    return "\n".join(
        f"[{i}] {a.tag} weight={a.weight:.1f} ({a.domain})"
        for i, a in enumerate(assessments, start=1)
    )


def _format_sub_claims(sub_claims: list[str]) -> str:
    return "\n".join(f"  ({i}) {s}" for i, s in enumerate(sub_claims, start=1))


def _format_transcript(transcript: list[DebateTurn]) -> str:
    return "\n\n".join(
        f"Round {turn['round']} {turn['speaker']}: {turn['argument']}"
        for turn in transcript
    )


def _build_system_prompt(with_credibility: bool, with_sub_claims: bool) -> str:
    prompt = _BASE_INTRO
    if with_sub_claims:
        prompt += _SUB_CLAIM_GUIDANCE
    if with_credibility:
        prompt += _CREDIBILITY_GUIDANCE
    return prompt + _INSTRUCTIONS


_VERDICT_PATTERN = re.compile(
    r'"verdict"\s*:\s*"(Supported|Refuted|Not Enough Evidence|Conflicting Evidence)"'
)
_REASONING_PATTERN = re.compile(r'"reasoning"\s*:\s*"(.*)"', re.DOTALL)


def _parse_verdict(raw_response: str) -> tuple[Verdict, str]:
    """Extract the verdict and reasoning, tolerating malformed JSON.

    The judge often produces long reasoning containing unescaped newlines and
    quotes, which breaks strict JSON parsing. A clean parse is tried first; if it
    fails, the verdict is recovered by regex, since it is always one of four fixed
    strings. Only genuinely unrecoverable responses fall back to Not Enough
    Evidence.
    """
    try:
        parsed = json.loads(raw_response)
        return Verdict(parsed["verdict"]), parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    match = _VERDICT_PATTERN.search(raw_response)
    if match:
        verdict = Verdict(match.group(1))
        reasoning_match = _REASONING_PATTERN.search(raw_response)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        return verdict, reasoning

    logger.warning("Failed to recover verdict from judge response: %s", raw_response)
    return Verdict.NOT_ENOUGH_EVIDENCE, "Failed to parse judge response."


def judge_debate(
    claim: Claim,
    transcript: list[DebateTurn],
    evidence: list[SearchResult],
    llm_client: LLMClient,
    assessments: list[CredibilityResult] | None = None,
    sub_claims: list[str] | None = None,
) -> tuple[Verdict, str]:
    """Return the judge's verdict and reasoning for the whole claim."""
    system_prompt = _build_system_prompt(
        with_credibility=assessments is not None,
        with_sub_claims=sub_claims is not None,
    )

    sections = [f"Claim: {claim.text}"]
    if sub_claims is not None:
        sections.append(f"Component parts of the claim:\n{_format_sub_claims(sub_claims)}")
    sections.append(f"Evidence:\n{_format_evidence(evidence, assessments)}")
    if assessments is not None:
        sections.append(
            f"Source credibility:\n{_format_credibility_reference(assessments)}"
        )
    sections.append(f"Debate transcript:\n{_format_transcript(transcript)}")
    sections.append("Weigh both sides and deliver your verdict.")

    user_prompt = "\n\n".join(sections)
    raw_response = llm_client.generate(system_prompt, user_prompt)
    return _parse_verdict(raw_response)