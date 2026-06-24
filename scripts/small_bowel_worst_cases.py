from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-eval", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    data = json.loads(Path(args.official_eval).read_text(encoding="utf-8"))
    rows = [row for row in data["volumes"] if row["class"] == "small_bowel"]
    rows = sorted(rows, key=lambda row: (row["combined_proxy"], row["dice_3d"]))[: args.limit]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fold",
        "case",
        "day",
        "num_slices",
        "target_voxels",
        "pred_voxels",
        "dice_3d",
        "hd95_mm",
        "hausdorff_score_proxy",
        "combined_proxy",
        "config",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    out_md = Path(args.out_md)
    lines = [
        "# Small Bowel Worst Cases",
        "",
        f"Source: `{args.official_eval}`",
        "",
        "| Rank | Fold | Case | Day | Dice 3D | HD95 mm | H-score | Combined | Target voxels | Pred voxels |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        hd = row["hd95_mm"]
        lines.append(
            f"| {rank} | {row['fold']} | {row['case']} | {row['day']} | "
            f"{row['dice_3d']:.5f} | {'' if hd is None else f'{hd:.2f}'} | "
            f"{row['hausdorff_score_proxy']:.5f} | {row['combined_proxy']:.5f} | "
            f"{row['target_voxels']} | {row['pred_voxels']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved report: {out_md}")


if __name__ == "__main__":
    main()
