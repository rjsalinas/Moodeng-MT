import argparse
import os
import sys
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus-parallel-txt")
WORK_DIR = os.path.join(PROJECT_ROOT, "python-script", "annotated-preprocess")
os.makedirs(WORK_DIR, exist_ok=True)


def _read_parallel_files(
    split: str,
) -> pd.DataFrame:
    """Read parallel cleaned files for a given split into a DataFrame.

    Columns: id, filipino (tl), english_mt (en)
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be one of: train, val")

    tl_path = os.path.join(CORPUS_DIR, f"{split}.tl.cleaned")
    en_path = os.path.join(CORPUS_DIR, f"{split}.en.cleaned")

    if not os.path.exists(tl_path) or not os.path.exists(en_path):
        raise FileNotFoundError(
            f"Missing cleaned files for split '{split}'. Expected: {tl_path} and {en_path}"
        )

    tl = pd.read_csv(
        tl_path,
        sep="\t",
        engine="python",
        header=None,
        names=["filipino"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    en = pd.read_csv(
        en_path,
        sep="\t",
        engine="python",
        header=None,
        names=["english_mt"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    if len(tl) != len(en):
        raise ValueError(
            f"Mismatched line counts for split '{split}': tl={len(tl)} en={len(en)}"
        )

    df = pd.concat([tl, en], axis=1)
    df.insert(0, "id", [f"{split}-{i}" for i in range(len(df))])
    return df


def load_corpora(splits: Optional[List[str]]) -> pd.DataFrame:
    """Load one or multiple splits; default is all available splits."""
    chosen = splits or ["train", "val"]
    frames: List[pd.DataFrame] = []
    for sp in chosen:
        frames.append(_read_parallel_files(sp))
    df = pd.concat(frames, ignore_index=True)
    return df


def flag_suspicious_translations(df: pd.DataFrame) -> pd.DataFrame:
    """Flag translations that might need human review.

    Adds columns: needs_review (bool), flag_reasons (str)
    """
    keyword_checks = {
        "hindi": ["not", "don't", "no"],
        "bat": ["why"],
        "ano": ["what"],
        "wala": ["nothing", "none", "no"],
        "gusto": ["want", "like"],
    }

    flags = []
    for _, row in df.iterrows():
        src = str(row["filipino"]).lower()
        tgt = str(row["english_mt"]).lower()

        flag_reasons: List[str] = []

        # Missing expected keywords
        for tagalog_word, expected_eng in keyword_checks.items():
            if tagalog_word in src:
                if not any(eng_word in tgt for eng_word in expected_eng):
                    flag_reasons.append(
                        f"Missing {tagalog_word} -> {','.join(expected_eng)}"
                    )

        # Too short / too long
        tgt_len = len(tgt.split())
        src_len = len(src.split())
        if tgt_len <= 2:
            flag_reasons.append("Translation too short")
        if src_len > 0 and tgt_len > src_len * 3:
            flag_reasons.append("Translation suspiciously long")

        flags.append(
            {
                "needs_review": len(flag_reasons) > 0,
                "flag_reasons": "; ".join(flag_reasons) if flag_reasons else None,
            }
        )

    flag_df = pd.DataFrame(flags)
    df = pd.concat([df.reset_index(drop=True), flag_df], axis=1)
    return df


def get_temp_annotation_path(suffix: str = "") -> str:
    base = os.path.join(WORK_DIR, "annotations_temp")
    return f"{base}{suffix}.csv" if suffix else f"{base}.csv"


def get_final_annotation_path() -> str:
    return os.path.join(WORK_DIR, "annotations_manual.csv")


def _load_existing_annotations(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=["id", "filipino", "english_mt", "english_final", "was_edited", "flag_reasons"])  # type: ignore


def manual_annotation_interface(df_flagged: pd.DataFrame, temp_path: str) -> pd.DataFrame:
    """Interactive CLI interface with autosave and resume.

    Returns the combined annotations DataFrame (existing + new edits for flagged rows).
    """
    existing = _load_existing_annotations(temp_path)
    annotated_ids = set(existing["id"].tolist()) if not existing.empty else set()

    annotated_rows: List[dict] = []

    total = len(df_flagged)
    for idx, row in df_flagged.iterrows():
        if row["id"] in annotated_ids:
            continue

        print(f"\n--- Sentence {len(annotated_rows) + 1}/{total} ---")
        print(f"ID: {row['id']}")
        print(f"Filipino: {row['filipino']}")
        print(f"MT Output: {row['english_mt']}")
        if "flag_reasons" in row and isinstance(row["flag_reasons"], str) and row["flag_reasons"]:
            print(f"Flag Reason: {row['flag_reasons']}")

        print("\nOptions:")
        print("[Enter] Keep MT output | Type new translation | 'skip' | 'quit'")

        try:
            user_input = input("Your choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = "quit"

        if user_input.lower() in {"q", "quit", ":q"}:
            print("\nExiting and saving progress...")
            break
        if user_input.lower() == "skip":
            continue

        if user_input == "":
            final_translation = row["english_mt"]
            was_edited = False
        else:
            final_translation = user_input
            was_edited = True

        annotated_rows.append(
            {
                "id": row["id"],
                "filipino": row["filipino"],
                "english_mt": row["english_mt"],
                "english_final": final_translation,
                "was_edited": str(was_edited),
                "flag_reasons": row.get("flag_reasons", ""),
            }
        )

        # Autosave incrementally
        combined = pd.concat([existing, pd.DataFrame(annotated_rows)], ignore_index=True)
        combined.drop_duplicates(subset=["id"], keep="last", inplace=True)
        combined.to_csv(temp_path, index=False)
        print(f"[saved] {temp_path}")

    # Final combined annotations
    combined = pd.concat([existing, pd.DataFrame(annotated_rows)], ignore_index=True)
    combined.drop_duplicates(subset=["id"], keep="last", inplace=True)
    combined.to_csv(temp_path, index=False)
    print(f"[final save] {temp_path}")
    return combined


def cmd_flag(args: argparse.Namespace) -> None:
    df = load_corpora(args.splits)
    df = flag_suspicious_translations(df)
    flagged = df[df.get("needs_review", False) == True]  # noqa: E712

    out_path = os.path.join(WORK_DIR, "flagged_for_review.csv")
    flagged.to_csv(out_path, index=False)
    print(f"Flagged {flagged.shape[0]} out of {df.shape[0]} sentences for review")
    print(f"Saved: {out_path}")


def _choose_source_for_annotation(args: argparse.Namespace) -> pd.DataFrame:
    # Prefer explicitly provided CSV of flagged items, else recompute on-the-fly
    if args.flagged_csv and os.path.exists(args.flagged_csv):
        df = pd.read_csv(args.flagged_csv, dtype=str, keep_default_na=False)
    else:
        df_all = load_corpora(args.splits)
        df = flag_suspicious_translations(df_all)
        df = df[df.get("needs_review", False) == True]  # noqa: E712
    return df.reset_index(drop=True)


def cmd_annotate(args: argparse.Namespace) -> None:
    df_flagged = _choose_source_for_annotation(args)
    if df_flagged.empty:
        print("No flagged items to annotate. You're all set!")
        return

    suffix = f"_{args.splits[0]}" if args.splits and len(args.splits) == 1 else ""
    temp_path = args.temp or get_temp_annotation_path(suffix)
    manual_annotation_interface(df_flagged, temp_path)


def cmd_stats(args: argparse.Namespace) -> None:
    temp_path = args.temp or get_temp_annotation_path()
    df = _load_existing_annotations(temp_path)
    if df.empty:
        print("No annotations found.")
        return
    total = df.shape[0]
    edited = (df["was_edited"].astype(str) == "True").sum()
    kept = total - edited
    print(f"Total annotated: {total}")
    print(f"Kept MT output: {kept}")
    print(f"Edited by human: {edited}")


def _merge_with_existing(final_path: str, additions: pd.DataFrame) -> pd.DataFrame:
    existing = _load_existing_annotations(final_path)
    merged = pd.concat([existing, additions], ignore_index=True)
    merged.drop_duplicates(subset=["id"], keep="last", inplace=True)
    return merged


def cmd_export(args: argparse.Namespace) -> None:
    temp_path = args.temp or get_temp_annotation_path()
    final_path = args.out or get_final_annotation_path()

    if not os.path.exists(temp_path):
        print(f"No temp annotations found at {temp_path}")
        return

    additions = pd.read_csv(temp_path, dtype=str, keep_default_na=False)
    merged = _merge_with_existing(final_path, additions)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    merged.to_csv(final_path, index=False)
    print(f"Exported merged annotations to {final_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for flagging and annotating translations without touching final cleaned corpora.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # flag
    p_flag = sub.add_parser("flag", help="Run QC and save flagged_for_review.csv")
    p_flag.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=None,
        help="Which splits to load (default: train val)",
    )
    p_flag.set_defaults(func=cmd_flag)

    # annotate
    p_ann = sub.add_parser("annotate", help="Interactive annotator over flagged items")
    p_ann.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=None,
        help="Which splits to load (default: train val)",
    )
    p_ann.add_argument(
        "--flagged-csv",
        default=os.path.join(WORK_DIR, "flagged_for_review.csv"),
        help="Optional path to a precomputed flagged CSV",
    )
    p_ann.add_argument(
        "--temp",
        default=None,
        help="Path to temp annotations CSV (autosave/resume)",
    )
    p_ann.set_defaults(func=cmd_annotate)

    # stats
    p_stats = sub.add_parser("stats", help="Show progress stats from temp annotations")
    p_stats.add_argument(
        "--temp",
        default=None,
        help="Path to temp annotations CSV",
    )
    p_stats.set_defaults(func=cmd_stats)

    # export
    p_export = sub.add_parser("export", help="Merge temp annotations into a final CSV")
    p_export.add_argument(
        "--temp",
        default=None,
        help="Path to temp annotations CSV",
    )
    p_export.add_argument(
        "--out",
        default=None,
        help="Destination final annotations CSV (default: annotated-preprocess/annotations_manual.csv)",
    )
    p_export.set_defaults(func=cmd_export)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


