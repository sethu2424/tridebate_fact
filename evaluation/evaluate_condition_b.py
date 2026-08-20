"""Evaluation script for Condition B results against AVeriTeC ground truth."""
import glob
import json

from sklearn.metrics import accuracy_score, classification_report, f1_score

RESULTS_DIR = "results/condition_b"
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


def main() -> None:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    predictions, truths, mismatches = [], [], []

    for filepath in glob.glob(f"{RESULTS_DIR}/*.json"):
        with open(filepath) as f:
            result = json.load(f)
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

    with open("condition_b_mismatches.json", "w") as f:
        json.dump(mismatches, f, indent=2)
    print(f"{len(mismatches)} mismatches saved to condition_b_mismatches.json")


if __name__ == "__main__":
    main()