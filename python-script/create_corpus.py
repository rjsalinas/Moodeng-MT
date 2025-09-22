import pandas as pd

def process_corpus(csv_file, src_out="train.en.cleaned", tgt_out="train.tl.cleaned"):
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

    # Write to separate files
    df["src"].to_csv(src_out, index=False, header=False)
    df["tgt"].to_csv(tgt_out, index=False, header=False)

    print(f"Written {after} records to '{src_out}' and '{tgt_out}'.")

if __name__ == "__main__":
    process_corpus("translated_tweets.csv")
