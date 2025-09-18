import argparse
import csv
import json
import os
from typing import List, Dict, Any


def load_json_records(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    # Collapse newlines and excessive whitespace for CSV cleanliness
    text = text.replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def write_csv_id_text(records: List[Dict[str, Any]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"], extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            tweet_id = str(rec.get("id", "")).strip()
            text = sanitize_text(rec.get("text", ""))
            if not tweet_id and not text:
                continue
            writer.writerow({"id": tweet_id, "text": text})


def main() -> None:
    default_input = os.path.join(
        "apify_scraper",
        "dataset_tweettaglish-extraction---remaining-links-part-5_2025-08-14_01-05-28-859.json",
    )
    default_output = os.path.join("archiveCSVs", "tweets_id_text_part5.csv")

    parser = argparse.ArgumentParser(
        description="Extract id and text fields from Apify JSON into a CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=default_input,
        help="Path to Apify JSON file (array of tweet-like objects)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=default_output,
        help="Path to output CSV file (will contain columns: id,text)",
    )
    args = parser.parse_args()

    records = load_json_records(args.input)
    write_csv_id_text(records, args.output)
    print(f"Wrote id,text CSV: {args.output}")


if __name__ == "__main__":
    main()


