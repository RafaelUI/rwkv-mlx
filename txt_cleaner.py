#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

NONRUS_CHARS = re.compile(r"[іїєґўљњћџ]")
URL_PATTERN = re.compile(r"https?://|www\.|\.com\b|\.ru/|\.org\b", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")
REPEATED_PUNCT = re.compile(r"[!?]{3,}|[,]{4,}")
DIGIT_LINE = re.compile(r"^\s*[\d\.\,\;\:\-\(\)\[\]]{3,}\s*$")

BAD_PATTERNS: list[str] = [
    "ингибитор", "ферментов", "дыхательной цепи", "цианид",
    "молекул", "нуклеотид", "аминокислот", "хромосом", "геном",
    "катализатор", "электролит", "полимер", "изотоп", "реагент",
    "нейромедиатор", "синапс", "рецептор", "нейрон", "аксон",
    "антиген", "антитело", "вирион", "патоген",
    "митохондри", "рибосом", "цитоплазм", "эндоплазматическ",
    "квант", "фотон", "электрон вольт", "магнитный поток",
    "интеграл от", "производная функци",
    "тензор", "дивергенц",
    "алгоритм", "процессор", "компилятор", "байт", "гигабайт",
    "протокол tcp", "http://", "html", "css", "javascript",
    "база данных", "sql", "функция main", "return false",
    "import os", "def ", "class ", "print(",
    "цикл for", "цикл while",
    "dolo malo", "отягощающий вину", "согласно статье",
    "настоящим договором", "пункт 1.1", "пункт 2.",
    "приложение №", "протокол №", "исх. №", "вх. №",
    "в соответствии с федеральным", "постановление правительства",
    "арбитражный суд", "истец", "ответчик",
    "pinky world", "скидка %", "бесплатно!", "акция!",
    "заказать сейчас", "перейти на сайт", "подписаться",
    "наш телеграм", "вконтакте", "инстаграм", "тикток",
    "промокод", "кэшбэк", "бонус за регистрацию",
    "окт. вдова", "л, старости", "старости.",
    "train-0000", ".parquet", ".jsonl",
    "```", "====", "----",
    "побочные эффекты", "противопоказани", "дозировка мг",
    "клинические испытани", "плацебо",
    "хирургическое вмешательств",
    "валовый внутренний продукт", "инфляци",
    "процентная ставка", "кредитный рейтинг",
    "ценные бумаги", "фондовый рынок",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a text file from garbage lines using preconfigured Russian-content filters."
    )
    parser.add_argument("input_file", help="Path to the input .txt file.")
    parser.add_argument(
        "--output",
        "-o",
        default="cleaned.txt",
        help="Path to the output cleaned .txt file. Default: cleaned.txt",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file with cleaned content.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep empty lines in the output.",
    )
    return parser.parse_args()


def is_bad_line(line: str) -> tuple[bool, str | None]:
    stripped = line.strip()
    if not stripped:
        return False, None
    if NONRUS_CHARS.search(line):
        return True, "nonrus"
    if URL_PATTERN.search(line):
        return True, "url"
    if EMAIL_PATTERN.search(line):
        return True, "email"
    if REPEATED_PUNCT.search(line):
        return True, "repeated_punct"
    if DIGIT_LINE.match(line):
        return True, "digit_line"

    text = line.lower()
    for pattern in BAD_PATTERNS:
        if pattern in text:
            return True, "bad_pattern"
    return False, None


def clean_file(input_path: Path, output_path: Path, keep_empty: bool) -> tuple[int, int, int, dict[str, int]]:
    kept = 0
    removed = 0
    removed_counts: dict[str, int] = {
        "empty": 0,
        "nonrus": 0,
        "url": 0,
        "email": 0,
        "repeated_punct": 0,
        "digit_line": 0,
        "bad_pattern": 0,
    }

    with input_path.open("r", encoding="utf-8", errors="replace") as source:
        total_lines = sum(1 for _ in source)

    progress_interval = max(1, total_lines // 100)

    with input_path.open("r", encoding="utf-8", errors="replace") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for index, line in enumerate(source, start=1):
            if not line.strip():
                if keep_empty:
                    target.write(line)
                    kept += 1
                else:
                    removed += 1
                    removed_counts["empty"] += 1
            else:
                bad, reason = is_bad_line(line)
                if bad:
                    removed += 1
                    removed_counts[reason or "bad_pattern"] += 1
                else:
                    target.write(line.rstrip("\n") + "\n")
                    kept += 1

            if index % progress_interval == 0 or index == total_lines:
                percent = index * 100 // total_lines if total_lines else 100
                print(
                    f"Progress: {index}/{total_lines} lines ({percent}%)",
                    end="\r",
                    file=sys.stderr,
                    flush=True,
                )

    print(file=sys.stderr)
    return kept, removed, total_lines, removed_counts


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists() or not input_path.is_file():
        raise SystemExit(f"Input file does not exist or is not a file: {input_path}")

    output_path = Path(args.output) if not args.inplace else input_path
    if args.inplace and output_path == input_path:
        temp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        kept, removed, total_lines, removed_counts = clean_file(
            input_path, temp_path, args.keep_empty
        )
        temp_path.replace(input_path)
    else:
        kept, removed, total_lines, removed_counts = clean_file(
            input_path, output_path, args.keep_empty
        )

    print(f"Cleaned '{input_path}' -> '{output_path}'")
    print(f"Lines processed: {total_lines}")
    print(f"Kept: {kept}")
    print(f"Removed: {removed}")
    print("Removal breakdown:")
    for reason, count in removed_counts.items():
        if count:
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
