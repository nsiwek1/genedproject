#!/usr/bin/env python3
"""
Utility to segment Analects (Confucius) and Mencius source texts into
JSONL datasets with per-saying metadata suitable for downstream labeling.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]


FOOTNOTE_PATTERN = re.compile(r"^\d+\.(?!\d)")
INLINE_FOOTNOTE_PATTERN = re.compile(r"(?<!\s)\d+")


@dataclass
class SourceConfig:
    philosopher: str
    work: str
    text_path: Path
    segment_pattern: re.Pattern
    book_pattern: re.Pattern = re.compile(r"^Book\s+([A-Za-z]+)")


CONFUCIUS_CONFIG = SourceConfig(
    philosopher="Confucius",
    work="Analects",
    text_path=ROOT / "confuncius.txt",
    segment_pattern=re.compile(r"^(?P<book>\d+)\.(?P<section>\d+)\s+"),
)


MENCIUS_CONFIG = SourceConfig(
    philosopher="Mencius",
    work="Mencius",
    text_path=ROOT / "mencius.txt",
    segment_pattern=re.compile(r"^(?P<book>\d+[AB])(?P<section>\d+)\s+"),
)


def normalize_whitespace(lines: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("\ufeff"):
            line = line.lstrip("\ufeff")
        if not line:
            continue
        cleaned.append(line)
    return cleaned


def append_to_buffer(buffer: List[str], line: str) -> None:
    if not line:
        return
    cleaned = INLINE_FOOTNOTE_PATTERN.sub("", line)
    if buffer and buffer[-1].endswith("-"):
        buffer[-1] = buffer[-1][:-1] + cleaned.lstrip()
    else:
        buffer.append(cleaned)


def segment_text(config: SourceConfig) -> List[dict]:
    lines = normalize_whitespace(config.text_path.read_text(encoding="utf-8").splitlines())
    entries: List[dict] = []
    current_book: Optional[str] = None
    current_ref: Optional[str] = None
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_ref, current_book
        if current_ref and buffer:
            text = " ".join(buffer).strip()
            if text:
                entries.append(
                    {
                        "philosopher": config.philosopher,
                        "work": config.work,
                        "book": current_book,
                        "reference": current_ref,
                        "text": text,
                    }
                )
        buffer = []

    for line in lines:
        book_match = config.book_pattern.match(line)
        if book_match:
            current_book = book_match.group(1)
            continue

        if line.isupper():
            continue

        match = config.segment_pattern.match(line)
        if match:
            flush()
            current_ref = match.group(0).strip()
            remainder = line[match.end() :].strip()
            append_to_buffer(buffer, remainder)
        else:
            if FOOTNOTE_PATTERN.match(line):
                continue
            if buffer:
                append_to_buffer(buffer, line)

    flush()
    return entries


def write_jsonl(entries: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Segment classical Chinese philosophy texts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "segments",
        help="Directory for JSONL outputs.",
    )
    parser.add_argument(
        "--combined",
        type=Path,
        default=ROOT / "data" / "segments" / "all_segments.jsonl",
        help="Combined JSONL output path.",
    )
    args = parser.parse_args()

    outputs = []
    for cfg in (CONFUCIUS_CONFIG, MENCIUS_CONFIG):
        entries = segment_text(cfg)
        output_path = args.output_dir / f"{cfg.philosopher.lower()}_segments.jsonl"
        write_jsonl(entries, output_path)
        outputs.extend(entries)
        print(f"Wrote {len(entries)} entries to {output_path}")

    write_jsonl(outputs, args.combined)
    print(f"Wrote {len(outputs)} combined entries to {args.combined}")


if __name__ == "__main__":
    main()

