"""Source credibility assessment for the TriDebate-Fact verification pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from urllib.parse import urlparse

from tranco import Tranco


class Tier(IntEnum):
    """Trust tier assigned to a source. Lower is more trustworthy."""

    RELIABLE = 1
    NO_KNOWN_ISSUES = 2
    MIXED = 3
    UNTRUSTED = 4


TIER_WEIGHTS: dict[Tier, float] = {
    Tier.RELIABLE: 1.0,
    Tier.NO_KNOWN_ISSUES: 0.7,
    Tier.MIXED: 0.3,
    Tier.UNTRUSTED: 0.1,
}

TIER_TAGS: dict[Tier, str] = {
    Tier.RELIABLE: "[TRUSTED SOURCE]",
    Tier.NO_KNOWN_ISSUES: "[NO KNOWN ISSUES]",
    Tier.MIXED: "[MIXED RELIABILITY]",
    Tier.UNTRUSTED: "[KNOWN MISINFORMATION SOURCE]",
}

CRED1_UNTRUSTED_CATEGORIES = frozenset(
    {"fake", "conspiracy", "unreliable", "satire", "rumor"}
)

MBFC_TIERS: dict[str, Tier] = {
    "high": Tier.RELIABLE,
    "mixed": Tier.MIXED,
    "low": Tier.UNTRUSTED,
}

UNVERIFIED_TIER2_WEIGHT = 0.4
UNVERIFIED_TIER2_TAG = "[UNVERIFIED SOURCE]"


@dataclass(frozen=True)
class CredibilityResult:
    """The credibility profile attached to a single source."""

    domain: str
    tier: Tier
    weight: float
    tag: str
    source_of_rating: str
    cred1_category: str | None
    mbfc_factual: str | None
    tranco_rank: int | None
    cred1_mbfc_conflict: bool


def extract_domain(url: str) -> str:
    """Return the registrable host of a URL, lowercased and without a leading www."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


class CredibilityAssessor:
    """Assesses source credibility using CRED-1, MBFC, and Tranco in priority order.

    CRED-1 (documented misinformation) is authoritative and checked first, then
    MBFC factual-reporting ratings, then Tranco popularity as an establishment
    signal for domains absent from both catalogues. All three resources are
    loaded once at construction and reused across assessments.
    """

    def __init__(
        self, cred1_path: str | Path, mbfc_path: str | Path
    ) -> None:
        self._cred1 = self._load_cred1(Path(cred1_path))
        self._mbfc = self._load_mbfc(Path(mbfc_path))
        self._tranco = Tranco(cache=True, cache_dir=".tranco").list()

    @staticmethod
    def _load_cred1(path: Path) -> dict[str, str]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {
                row["domain"].strip().lower(): row["category"].strip().lower()
                for row in reader
                if row.get("domain")
            }

    @staticmethod
    def _load_mbfc(path: Path) -> dict[str, str]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {
                row["source"].strip().lower(): row["factual_reporting"].strip().lower()
                for row in reader
                if row.get("source") and row.get("factual_reporting")
            }

    def assess(self, url: str) -> CredibilityResult:
        domain = extract_domain(url)
        cred1_category = self._cred1.get(domain)
        mbfc_factual = self._mbfc.get(domain)
        conflict = self._is_conflict(cred1_category, mbfc_factual)

        if cred1_category is not None:
            tier = self._cred1_tier(cred1_category)
        elif mbfc_factual in MBFC_TIERS:
            tier = MBFC_TIERS[mbfc_factual]
        else:
            return self._assess_by_popularity(
                domain, cred1_category, mbfc_factual, conflict
            )

        source = "cred1" if cred1_category is not None else "mbfc"
        return CredibilityResult(
            domain=domain,
            tier=tier,
            weight=TIER_WEIGHTS[tier],
            tag=TIER_TAGS[tier],
            source_of_rating=source,
            cred1_category=cred1_category,
            mbfc_factual=mbfc_factual,
            tranco_rank=self._rank(domain),
            cred1_mbfc_conflict=conflict,
        )

    def _assess_by_popularity(
        self,
        domain: str,
        cred1_category: str | None,
        mbfc_factual: str | None,
        conflict: bool,
    ) -> CredibilityResult:
        rank = self._rank(domain)
        if rank is not None:
            weight, tag = TIER_WEIGHTS[Tier.NO_KNOWN_ISSUES], TIER_TAGS[Tier.NO_KNOWN_ISSUES]
        else:
            weight, tag = UNVERIFIED_TIER2_WEIGHT, UNVERIFIED_TIER2_TAG

        return CredibilityResult(
            domain=domain,
            tier=Tier.NO_KNOWN_ISSUES,
            weight=weight,
            tag=tag,
            source_of_rating="tranco",
            cred1_category=cred1_category,
            mbfc_factual=mbfc_factual,
            tranco_rank=rank,
            cred1_mbfc_conflict=conflict,
        )

    @staticmethod
    def _cred1_tier(category: str) -> Tier:
        if category == "reliable":
            return Tier.RELIABLE
        if category == "mixed":
            return Tier.MIXED
        if category in CRED1_UNTRUSTED_CATEGORIES:
            return Tier.UNTRUSTED
        return Tier.NO_KNOWN_ISSUES

    @staticmethod
    def _is_conflict(cred1_category: str | None, mbfc_factual: str | None) -> bool:
        if cred1_category is None or mbfc_factual is None:
            return False
        cred1_bad = cred1_category in CRED1_UNTRUSTED_CATEGORIES
        return cred1_bad and mbfc_factual == "high"

    def _rank(self, domain: str) -> int | None:
        rank = self._tranco.rank(domain)
        return rank if rank != -1 else None