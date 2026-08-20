"""Evaluation script for Condition C results against AVeriTeC ground truth."""
import glob
import json
from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, f1_score

RESULTS_DIR = "results/condition_c"
GROUND_TRUTH_PATH = "/scratch/pk01140/dissertation/DebatingTruth/data/AVeriTeC/dev.json"


def normalize(label: str) -> str:
    """Map any verdict label variant onto AVeriTeC's four canonical categories."""
    if not label:
        return "Not Enough Evidence"
    normalized = label.strip().lower().replace("verdict.", "")
    if "support" in normalized:
        return "Supported"
    if "refut" in normalized:
        return "Refuted"
    if "not_enough" in normalized or "not enough" in normalized or "nei" in normalized:
        return "Not Enough Evidence"
    if "conflict" in normalized:
        return "Conflicting Evidence"
    return label.strip()


def load_ground_truth(path: str) -> dict[int, str]:
    with open(path) as f:
        data = json.load(f)
    return {i: normalize(item["label"]) for i, item in enumerate(data)}


def summarise_credibility(results: list[dict]) -> None:
    total_sources = 0
    rated_by_cred1 = 0
    rated_by_mbfc = 0
    rated_by_tranco = 0
    unverified = 0
    conflicts = 0
    tier_totals: Counter[str] = Counter()
    claims_with_stats = 0

    for result in results:
        stats = result.get("credibility_stats")
        if not stats:
            continue
        claims_with_stats += 1
        total_sources += stats.get("total_sources", 0)
        rated_by_cred1 += stats.get("rated_by_cred1", 0)
        rated_by_mbfc += stats.get("rated_by_mbfc", 0)
        rated_by_tranco += stats.get("rated_by_tranco", 0)
        unverified += stats.get("unverified", 0)
        conflicts += stats.get("cred1_mbfc_conflicts", 0)
        tier_totals.update(stats.get("tier_distribution", {}))

    print("Credibility layer statistics")
    print(f"Claims with credibility stats: {claims_with_stats}")
    print(f"Total sources assessed: {total_sources}")

    if total_sources:
        print(
            f"Rated by CRED-1: {rated_by_cred1} ({rated_by_cred1 / total_sources:.2%})"
        )
        print(
            f"Rated by MBFC: {rated_by_mbfc} ({rated_by_mbfc / total_sources:.2%})"
        )
        print(
            f"Rated by Tranco fallback: {rated_by_tranco} "
            f"({rated_by_tranco / total_sources:.2%})"
        )
        print(
            f"Tagged UNVERIFIED: {unverified} ({unverified / total_sources:.2%})"
        )
        print(f"CRED-1 / MBFC conflicts: {conflicts}")

    print("Tier distribution across all sources:")
    for tier, count in sorted(tier_totals.items()):
        share = count / total_sources if total_sources else 0
        print(f"  {tier}: {count} ({share:.2%})")


def main() -> None:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    predictions, truths, mismatches, results = [], [], [], []

    for filepath in glob.glob(f"{RESULTS_DIR}/*.json"):
        with open(filepath) as f:
            result = json.load(f)
        results.append(result)
        claim_id = result.get("claim_id")
        if claim_id not in ground_truth:
            continue
        predicted = normalize(str(result.get("verdict", "")))
        truth = ground_truth[claim_id]
        predictions.append(predicted)
        truths.append(truth)
        if predicted != truth:
            mismatches.append(
                {
                    "claim_id": claim_id,
                    "claim": result.get("claim_text", ""),
                    "predicted": predicted,
                    "ground_truth": truth,
                }
            )

    accuracy = accuracy_score(truths, predictions)
    macro_f1 = f1_score(truths, predictions, average="macro", zero_division=0)
    print(f"Total claims evaluated: {len(predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()
    print(classification_report(truths, predictions, zero_division=0))

    with open("condition_c_mismatches.json", "w") as f:
        json.dump(mismatches, f, indent=2)
    print(f"{len(mismatches)} mismatches saved to condition_c_mismatches.json")
    print()
    summarise_credibility(results)


if __name__ == "__main__":
    main()