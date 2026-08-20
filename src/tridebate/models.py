"""Core data structures shared across the verification pipeline."""

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    SUPPORTED = "Supported"
    REFUTED = "Refuted"
    NOT_ENOUGH_EVIDENCE = "Not Enough Evidence"
    CONFLICTING_EVIDENCE = "Conflicting Evidence"


@dataclass
class SearchResult:
    title: str
    url: str
    content: str


@dataclass
class Claim:
    claim_id: int
    text: str
    ground_truth_label: str | None = None


@dataclass
class VerdictResult:
    claim_id: int
    claim_text: str
    verdict: Verdict
    reasoning: str
    sources_used: list[SearchResult] = field(default_factory=list)
    credibility_stats: dict | None = None
