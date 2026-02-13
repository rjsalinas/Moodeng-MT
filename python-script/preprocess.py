import json
import pandas as pd
import regex as re
import os

def clean_tweet_text(text):
    """
    Cleans the tweet text by removing URLs, mentions, hashtags, emojis,
    and other irrelevant characters.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|x\.com\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF" 
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^\w\s.,?!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    """
    Main function to read, process, and save the data from all JSON files in the dataset directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset')
    output_csv_path = os.path.join(script_dir, 'init-preprocess', 'cleaned_tweets.csv')
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    cleaned_texts = []

    # Process all .json files in the dataset directory
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            json_file_path = os.path.join(dataset_dir, filename)
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Error reading {json_file_path}, skipping.")
                continue

            for record in data:
                if 'text' in record and record['text']:
                    cleaned_text = clean_tweet_text(record['text'])
                    if len(cleaned_text.split()) > 2:
                        cleaned_texts.append(cleaned_text)

    if cleaned_texts:
        df = pd.DataFrame(cleaned_texts, columns=['src'])
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Successfully processed {len(cleaned_texts)} records from all JSON files.")
        print(f"Cleaned data saved to {output_csv_path}")
    else:
        print("No valid text data found to process.")

if __name__ == '__main__':
    main()