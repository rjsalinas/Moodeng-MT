import argparse
import os
import sys
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict

import pandas as pd

HF_REPO_ID = "propanda02/TweetTaglish-SalinTala"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus-parallel-txt")
WORK_DIR = os.path.join(PROJECT_ROOT, "python-script", "annotated-preprocess")
os.makedirs(WORK_DIR, exist_ok=True)


def _read_parallel_files(
    split: str,
) -> pd.DataFrame:
    """Read parallel cleaned files for a given split into a DataFrame.

    Columns: row_id, id, filipino (tl), english_mt (en)
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
    # Add row_id as auto-incremented identifier starting from 1
    df.insert(0, "row_id", range(1, len(df) + 1))
    # Add original id format for compatibility
    df.insert(1, "id", [f"{split}-{i}" for i in range(len(df))])
    return df

def load_from_huggingface(splits: Optional[List[str]] = None) -> pd.DataFrame:
    """Load dataset from Hugging Face Hub with enhanced row identification."""
    print(f"Loading dataset from Hugging Face: {HF_REPO_ID}")
    
    chosen_splits = splits or ["train", "validation"]

    all_data = []
    global_row_id = 1  # Global counter for row_id across splits
    
    for split_name in chosen_splits:
        split_data = load_dataset(HF_REPO_ID, split=split_name)
        split_df = split_data.to_pandas()
        split_df['filipino'] = split_df['translation'].apply(lambda x: x['tl'])
        split_df['english_mt'] = split_df['translation'].apply(lambda x: x['en'])
        
        # Add row_id as auto-incremented identifier
        split_df['row_id'] = range(global_row_id, global_row_id + len(split_df))
        global_row_id += len(split_df)
        
        # Add original id format for compatibility
        split_df['id'] = [f"{split_name}-{i}" for i in range(len(split_df))]
        split_df = split_df[['row_id', 'id', 'filipino', 'english_mt']]
        all_data.append(split_df)

    df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(df)} sentences from Hugging Face (row_id: 1-{len(df)})")
    return df

def load_corpora(source: str = "local", splits: Optional[List[str]] = None) -> pd.DataFrame:
    """Load corpora from either local files or Hugging Face."""
    if source == "hf":
        return load_from_huggingface(splits)
    elif source == "local":
        chosen = splits or ["train", "val"]
        frames: List[pd.DataFrame] = []
        global_row_id = 1  # Global counter for row_id across splits
        
        for sp in chosen:
            split_df = _read_parallel_files(sp)
            # Update row_id to be globally unique
            split_df['row_id'] = range(global_row_id, global_row_id + len(split_df))
            global_row_id += len(split_df)
            frames.append(split_df)
            
        df = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(df)} sentences from local files (row_id: 1-{len(df)})")
        return df
    else:
        raise ValueError("source must be either 'local' or 'hf'")

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
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        # Ensure row_id column exists and is properly typed
        if 'row_id' not in df.columns and 'id' in df.columns:
            # Try to extract row_id from id format (split-number)
            df['row_id'] = df['id'].str.extract(r'-(\d+)$')[0].astype(int) + 1
        return df
    return pd.DataFrame(columns=["row_id", "id", "filipino", "english_mt", "english_final", "was_edited", "flag_reasons"])  # type: ignore


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
        print(f"Row ID: {row['row_id']} | ID: {row['id']}")
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
                "row_id": row["row_id"],
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
    df = load_corpora(source=args.source, splits=args.splits)
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
        # Ensure row_id exists
        if 'row_id' not in df.columns and 'id' in df.columns:
            df['row_id'] = df['id'].str.extract(r'-(\d+)$')[0].astype(int) + 1
    else:
        df_all = load_corpora(source=args.source, splits=args.splits)
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
    if 'row_id' in df.columns:
        print(f"Row ID range: {df['row_id'].min()}-{df['row_id'].max()}")


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

def push_annotations_to_hf(
    annotations_df: pd.DataFrame,
    repo_id: str = HF_REPO_ID,
    splits: Optional[List[str]] = None,
    branch: str = "annotations"
) -> None:
    """Push annotated data back to Hugging Face Hub, keeping splits intact.
    
    SAFETY: Always pushes to a non-main branch to prevent accidental overwrites.
    Uses a consistent 'annotations' branch for all pushes.
    """
    # Safety check: never push to main/master branch
    if branch.lower() in ["main", "master"]:
        branch = "annotations"
        print(f"⚠️  Safety override: Changed branch to 'annotations' to protect main branch")
    
    print(f"🚀 Pushing annotations to {repo_id}@{branch}...")
    print(f"📝 Total annotations to push: {len(annotations_df)}")

    splits = splits or ["train", "validation"]

    # Load the existing dataset from main branch
    try:
        dataset_dict = load_dataset(repo_id)
    except Exception as e:
        print(f"❌ Failed to load dataset {repo_id}: {e}")
        return

    updated_splits = {}

    for split in splits:
        if split not in dataset_dict:
            print(f"⚠️ Split '{split}' not found in repo, skipping.")
            continue

        # Load split as pandas and add synthetic id
        split_ds = dataset_dict[split].to_pandas()
        split_ds["id"] = [f"{split}-{i}" for i in range(len(split_ds))]

        # Annotated rows for this split
        split_annotations = annotations_df[annotations_df["id"].str.startswith(split)]

        if split_annotations.empty:
            print(f"ℹ️  No annotations for split '{split}', keeping original.")
            updated_splits[split] = split_ds
            continue

        print(f"📊 Found {len(split_annotations)} annotations for split '{split}'")

        # Merge annotations
        merged = split_ds.merge(
            split_annotations[["id", "english_final"]],
            on="id",
            how="left",
            suffixes=("", "_annot"),
        )
        # Use human edit if present, else MT
        merged["en"] = merged["english_final"].fillna(merged["translation"].apply(lambda x: x["en"]))
        merged["tl"] = merged["translation"].apply(lambda x: x["tl"])

        # Back to HF format
        updated_splits[split] = pd.DataFrame({
            "translation": merged.apply(lambda r: {"tl": r["tl"], "en": r["en"]}, axis=1)
        })

    # Convert to DatasetDict
    hf_ready = DatasetDict({
        split: Dataset.from_pandas(df.reset_index(drop=True)) for split, df in updated_splits.items()
    })

    # Push to the annotations branch (creates it if it doesn't exist)
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"Update annotations - {len(annotations_df)} total annotations ({timestamp})"
        
        hf_ready.push_to_hub(
            repo_id, 
            private=False, 
            commit_message=commit_msg, 
            revision=branch
        )
        print(f"✅ Successfully pushed annotations to {repo_id}@{branch}")
        print(f"📈 Updated with {len(annotations_df)} annotations")
        print(f"🔗 View your changes at: https://huggingface.co/datasets/{repo_id}/tree/{branch}")
    except Exception as e:
        print(f"❌ Failed to push to Hugging Face: {e}")


def cmd_sync_hf(args: argparse.Namespace) -> None:
    """Sync local annotations with Hugging Face."""
    if args.direction == "pull":
        print("📥 Pulling latest data from Hugging Face...")
        df = load_from_huggingface(splits=args.splits)
        local_path = get_final_annotation_path()
        df.to_csv(local_path, index=False)
        print(f"💾 Saved to: {local_path}")
    elif args.direction == "push":
        final_path = get_final_annotation_path()
        if not os.path.exists(final_path):
            print(f"❌ No annotations found at {final_path}")
            return
        
        df = pd.read_csv(final_path, dtype=str, keep_default_na=False)
        
        print(f"🛡️  Safety mode: Pushing to 'annotations' branch (not main)")
        push_annotations_to_hf(df, branch="annotations", splits=args.splits)
    else:
        print("❌ Invalid direction. Use 'pull' or 'push'")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for flagging and annotating translations with enhanced row identification.",
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

    parser.add_argument(
        "--source",
        choices=["local", "hf"],
        default="local",
        help="Data source: 'local' for local files, 'hf' for Hugging Face",
    )

    p_sync = sub.add_parser("sync", help="Sync with Hugging Face Hub")
    p_sync.add_argument(
        "direction",
        choices=["pull", "push"],
        help="Direction: 'pull' from HF or 'push' to HF",
    )
    p_sync.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation"],
        default=None,
        help="Which splits to load (default: train validation)",
    )
    p_sync.set_defaults(func=cmd_sync_hf)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())