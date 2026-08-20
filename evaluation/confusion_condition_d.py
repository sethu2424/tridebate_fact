"""Confusion matrix and error-pair breakdown for Condition D results."""
import glob
import json

from sklearn.metrics import confusion_matrix

RESULTS_DIR = "results/condition_d"
GROUND_TRUTH_PATH = "/scratch/pk01140/dissertation/DebatingTruth/data/AVeriTeC/dev.json"

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"]


def normalize(label: str) -> str:
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
    predictions, truths = [], []

    for filepath in glob.glob(f"{RESULTS_DIR}/*.json"):
        with open(filepath) as f:
            result = json.load(f)
        claim_id = result.get("claim_id")
        if claim_id not in ground_truth:
            continue
        predictions.append(normalize(str(result.get("verdict", ""))))
        truths.append(ground_truth[claim_id])

    matrix = confusion_matrix(truths, predictions, labels=LABELS)

    header = "True \\ Pred".ljust(22) + "".join(label[:12].ljust(14) for label in LABELS)
    print(header)
    for i, label in enumerate(LABELS):
        row = label[:20].ljust(22) + "".join(str(matrix[i][j]).ljust(14) for j in range(len(LABELS)))
        print(row)

    print()
    print("Off-diagonal error pairs, most frequent first:")
    errors = []
    for i, true_label in enumerate(LABELS):
        for j, pred_label in enumerate(LABELS):
            if i != j and matrix[i][j] > 0:
                errors.append((matrix[i][j], true_label, pred_label))
    for count, true_label, pred_label in sorted(errors, reverse=True):
        print(f"  {count:4d}  {true_label}  ->  {pred_label}")


if __name__ == "__main__":
    main()