import pandas as pd
import os

def process_corpus(csv_file, src_out, tgt_out):
    # Load CSV
    df = pd.read_csv(csv_file)

    # Sanity check for required columns
    if not {"src", "tgt"}.issubset(df.columns):
        raise ValueError("CSV file must contain 'src' and 'tgt' columns")

    # Remove duplicate pairs (src, tgt)
    before = len(df)
    df = df.drop_duplicates(subset=["src", "tgt"])
    after = len(df)
    print(f"Removed {before - after} duplicate records.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(src_out), exist_ok=True)

    # Write to separate files
    df["src"].to_csv(src_out, index=False, header=False)
    df["tgt"].to_csv(tgt_out, index=False, header=False)

    print(f"Written {after} records to '{src_out}' and '{tgt_out}'.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_csv = os.path.join(script_dir, "annotated-preprocess", "translated_tweets.csv")
    out_dir = os.path.join(project_root, "corpus-parallel-txt")
    src_out = os.path.join(out_dir, "train.en.cleaned")
    tgt_out = os.path.join(out_dir, "train.tl.cleaned")

    process_corpus(input_csv, src_out, tgt_out)
