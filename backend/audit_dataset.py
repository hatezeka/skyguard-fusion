import argparse
import json
from pathlib import Path

from app.datasets.yolo_validator import validate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="datasets/drone_bird",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    root = Path(args.dataset).resolve()
    report = validate_dataset(root)

    result = {
        "dataset": str(root),
        "valid": report.valid,
        "images": report.images,
        "labels": report.labels,
        "annotations": report.annotations,
        "duplicate_groups": report.duplicate_groups,
        "errors": list(report.errors),
    }

    print(json.dumps(result, indent=2))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )

    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
