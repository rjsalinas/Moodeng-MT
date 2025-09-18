import argparse
import csv
import os
import re
from typing import Iterable


NUMERIC_ONLY_RE = re.compile(r"^\s*\d+\.?\s*$")

# Broad emoji and pictograph ranges
# Includes emoticons, dingbats, transport/map symbols, miscellaneous symbols/pictographs, supplemental symbols, flags, etc.
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # Flags
    "\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess, Symbols & Pictographs Extended-A
    "\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
    "\U00002700-\U000027BF"  # Dingbats
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002500-\U000025FF"  # Box Drawing & Geometric
    "]"
)

VARIATION_SELECTORS_RE = re.compile("[\uFE00-\uFE0F]")
ZWJ_RE = re.compile("\u200D")


def remove_emojis(text: str) -> str:
    if text is None:
        return ""
    # Remove variation selectors and zero-width joiners first
    text = VARIATION_SELECTORS_RE.sub("", text)
    text = ZWJ_RE.sub("", text)
    # Remove emojis/pictographs
    text = EMOJI_RE.sub("", text)
    # Normalize whitespace
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    return text.strip()


def is_meaningless(text: str) -> bool:
    if text is None:
        return True
    stripped = text.strip()
    if stripped == "":
        return True
    if NUMERIC_ONLY_RE.match(stripped):
        return True
    return False


def clean_rows(rows: Iterable[dict]) -> Iterable[dict]:
    for row in rows:
        raw_text = row.get("text", "")
        cleaned_text = remove_emojis(raw_text)
        if not is_meaningless(cleaned_text):
            yield {"id": row.get("id", ""), "text": cleaned_text}


def main() -> None:
    default_input = os.path.join("archiveCSVs", "dataset_tweettaglish_part5.csv")
    default_output = os.path.join("archiveCSVs", "dataset_tweettaglish_part5_clean.csv")

    parser = argparse.ArgumentParser(description="Remove meaningless rows from dataset CSV.")
    parser.add_argument("--input", "-i", default=default_input, help="Path to input CSV with columns id,text")
    parser.add_argument("--output", "-o", default=default_output, help="Path to write cleaned CSV")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="If set, drop id column and output only a single 'text' column.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8", newline="") as f_in, open(
        args.output, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["text"] if args.text_only else ["id", "text"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        kept = 0
        for row in clean_rows(reader):
            if args.text_only:
                writer.writerow({"text": row["text"]})
            else:
                writer.writerow(row)
            kept += 1
        print(f"Wrote cleaned CSV: {args.output} (kept={kept}, text_only={args.text_only})")


if __name__ == "__main__":
    main()


