"""Command-line entry point for running credibility-gated claim verification
over a batch of claims."""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tridebate.config import load_settings
from tridebate.credibility import CredibilityAssessor
from tridebate.llm_client import LLMClient
from tridebate.models import Claim
from tridebate.search_client import SearchClient
from tridebate.verification.credibility_filtered_verifier import verify_claim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CRED1_PATH = (
    Path(__file__).parent.parent / "external" / "cred-1" / "data" / "cred1_current.csv"
)
DEFAULT_MBFC_PATH = (
    Path(__file__).parent.parent / "external" / "mbfc" / "mbfc.csv"
)


def load_claims(claims_path: Path, limit: int | None = None) -> list[Claim]:
    """Load claims from a JSON file, optionally limited to the first N entries."""
    with open(claims_path) as f:
        raw_claims = json.load(f)
    if limit is not None:
        raw_claims = raw_claims[:limit]
    return [
        Claim(
            claim_id=item.get("claim_id", i),
            text=item["claim"],
            ground_truth_label=item.get("label"),
        )
        for i, item in enumerate(raw_claims)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run credibility-gated claim verification.")
    parser.add_argument("--claims", type=Path, required=True, help="Path to claims JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for results.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N claims.")
    parser.add_argument("--cred1-path", type=Path, default=DEFAULT_CRED1_PATH, help="Path to the CRED-1 CSV.")
    parser.add_argument("--mbfc-path", type=Path, default=DEFAULT_MBFC_PATH, help="Path to the MBFC CSV.")
    args = parser.parse_args()

    settings = load_settings()
    search_client = SearchClient(settings)
    llm_client = LLMClient(settings)
    assessor = CredibilityAssessor(args.cred1_path, args.mbfc_path)

    claims = load_claims(args.claims, limit=args.limit)
    args.output.mkdir(parents=True, exist_ok=True)

    for i, claim in enumerate(claims, start=1):
        output_file = args.output / f"{claim.claim_id}.json"
        if output_file.exists():
            logger.info("[%d/%d] Claim %d already processed, skipping", i, len(claims), claim.claim_id)
            continue
        logger.info("[%d/%d] Processing claim %d: %s", i, len(claims), claim.claim_id, claim.text[:60])
        try:
            result = verify_claim(claim, search_client, llm_client, assessor)
            with open(output_file, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)
        except Exception:
            logger.exception("Failed to process claim %d", claim.claim_id)
            continue

    logger.info("Verification run complete.")


if __name__ == "__main__":
    main()