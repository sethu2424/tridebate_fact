"""Command-line entry point for running debate-based claim verification
over a batch of claims."""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tridebate.config import load_settings
from tridebate.llm_client import LLMClient
from tridebate.models import Claim
from tridebate.search_client import SearchClient
from tridebate.verification.debate_verifier import verify_claim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description="Run debate-based claim verification.")
    parser.add_argument("--claims", type=Path, required=True, help="Path to claims JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for results.")
    parser.add_argument("--transcripts", type=Path, default=None, help="Output directory for debate transcripts.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N claims.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of debate rounds.")
    args = parser.parse_args()

    transcripts_dir = args.transcripts or args.output.parent / f"{args.output.name}_transcripts"

    settings = load_settings()
    search_client = SearchClient(settings)
    llm_client = LLMClient(settings)

    claims = load_claims(args.claims, limit=args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    for i, claim in enumerate(claims, start=1):
        output_file = args.output / f"{claim.claim_id}.json"
        if output_file.exists():
            logger.info("[%d/%d] Claim %d already processed, skipping", i, len(claims), claim.claim_id)
            continue
        logger.info("[%d/%d] Processing claim %d: %s", i, len(claims), claim.claim_id, claim.text[:60])
        try:
            result, transcript = verify_claim(claim, search_client, llm_client, rounds=args.rounds)
            with open(output_file, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)
            with open(transcripts_dir / f"{claim.claim_id}.json", "w") as f:
                json.dump(transcript, f, indent=2, default=str)
        except Exception:
            logger.exception("Failed to process claim %d", claim.claim_id)
            continue

    logger.info("Verification run complete.")


if __name__ == "__main__":
    main()