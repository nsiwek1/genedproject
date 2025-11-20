#!/usr/bin/env python3
"""
Generate instruction-response chat data from segmented Analects and Mencius snippets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS_PATH = ROOT / "data" / "segments" / "all_segments.jsonl"
OUTPUT_DIR = ROOT / "data" / "instruction_pairs"

SYSTEM_PROMPTS: Dict[str, str] = {
    "Confucius": "You are Confucius, the Master who teaches through ritual, study, and the patient refinement of virtue.",
    "Mencius": "You are Mencius, defender of the innate sprouts of virtue and counselor of humane kingship.",
}

USER_TEMPLATES: Dict[str, str] = {
    "Confucius": "Master, what guidance do you offer in {work} {reference}?",
    "Mencius": "Master Meng, what is your teaching in {work} {reference}?",
}


def read_segments(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def build_pair(entry: dict) -> dict:
    philosopher = entry["philosopher"]
    work = entry["work"]
    reference = entry.get("reference")
    text = entry["text"].strip()
    system_prompt = SYSTEM_PROMPTS[philosopher]
    user_question = USER_TEMPLATES[philosopher].format(work=work, reference=reference)

    if philosopher == "Confucius" and not text.lower().startswith("the master"):
        assistant_text = f"The Master replied, {text}"
    elif philosopher == "Mencius" and not text.lower().startswith(("mengzi", "mengzi", "mengzi said")):
        assistant_text = f"Mengzi answered, {text}"
    else:
        assistant_text = text

    return {
        "philosopher": philosopher,
        "source_reference": f"{work} {reference}",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert segmented passages into instruction-response pairs.")
    parser.add_argument(
        "--segments",
        type=Path,
        default=SEGMENTS_PATH,
        help="Path to combined segments JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write auto-generated instruction pairs.",
    )
    args = parser.parse_args()

    entries_by_philosopher: Dict[str, List[dict]] = {"Confucius": [], "Mencius": []}
    for entry in read_segments(args.segments):
        pair = build_pair(entry)
        entries_by_philosopher[entry["philosopher"]].append(pair)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for philosopher, pairs in entries_by_philosopher.items():
        output_path = args.output_dir / f"{philosopher.lower()}_auto.jsonl"
        with output_path.open("w", encoding="utf-8") as fh:
            for record in pairs:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[philosopher] = len(pairs)
        print(f"Wrote {len(pairs)} records to {output_path}")

    total = sum(counts.values())
    print(f"Generated {total} total instruction pairs.")


if __name__ == "__main__":
    main()

