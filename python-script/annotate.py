import pandas as pd
import google.generativeai as genai
import os
import json
import time
from datetime import datetime
from tqdm import tqdm

try:
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key="<API_KEY>")
except TypeError:
    print("\nERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set your API key and restart the script.\n")
    exit()

generation_config = {
#   "temperature": 0.4,
  "top_p": 1,
  "top_k": 1,
#   "max_output_tokens": 2048,
}

safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


def create_batch_prompt(tweets_list):
    """Formats a list of tweets into a numbered list for the prompt."""
    formatted_tweets = "\n".join([f"{i+1}. \"{tweet}\"" for i, tweet in enumerate(tweets_list)])
    
    prompt = f"""
    You are an expert linguist and translator, fluent in both English and Filipino, specializing in modern internet slang (Taglish). Your task is to translate a batch of Taglish tweets into natural, grammatically correct, and semantically equivalent English sentences. Preserve the original sentiment and meaning.

    IMPORTANT: Your response MUST be a single, valid JSON object.
    The keys of the JSON object should be the original numbers of the tweets (as strings, e.g., "1", "2").
    The values should be the corresponding English translations.

    **Input Tweets:**
    {formatted_tweets}

    **Output JSON:**
    """
    return prompt

def translate_tweet_batch(tweets_list, max_retries=3, backoff_seconds=2.0):
    """Sends a batch of tweets to the Gemini API and parses the JSON response.

    Returns a tuple: (translations_dict, last_raw_response_text_or_none).
    """
    if not tweets_list:
        return {}, None

    prompt = create_batch_prompt(tweets_list)
    last_raw_response = None

    for attempt_number in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            raw_text = response.text if getattr(response, "text", None) else str(response)
            cleaned_response = raw_text.strip().replace("```json", "").replace("```", "").strip()
            translations = json.loads(cleaned_response)
            return translations, None
        except json.JSONDecodeError:
            last_raw_response = raw_text if 'raw_text' in locals() else last_raw_response
            print(f"  [!] JSONDecodeError on attempt {attempt_number}/{max_retries}.")
        except Exception as e:
            last_raw_response = raw_text if 'raw_text' in locals() else last_raw_response
            print(f"  [!] API error on attempt {attempt_number}/{max_retries}: {e}")

        if attempt_number < max_retries:
            sleep_seconds = backoff_seconds * (2 ** (attempt_number - 1))
            time.sleep(sleep_seconds)

    if last_raw_response is not None:
        print(f"  [!] Failed to decode/parse after {max_retries} attempts. Last response shown below.\n--- RESPONSE START ---\n{last_raw_response}\n--- RESPONSE END ---")
    return {}, last_raw_response

def _append_failed_log(log_csv_path, record_dict):
    """Append a failure record to a CSV log, creating the file with header if needed."""
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    columns = list(record_dict.keys())
    file_exists = os.path.exists(log_csv_path)
    df_row = pd.DataFrame([record_dict], columns=columns)
    if file_exists:
        # append without header
        df_row.to_csv(log_csv_path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(log_csv_path, mode='w', header=True, index=False)


def process_csv_in_batches(input_csv_path, output_csv_path, source_column, target_column, batch_size=50, failed_log_csv_path=None):
    """
    Reads a CSV, translates a specified column in batches, and saves the results.
    """
    print(f"Loading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    if target_column not in df.columns:
        df[target_column] = ""

    print(f"Starting translation process for column '{source_column}'...")

    # Prepare failed log path
    if failed_log_csv_path is None:
        failed_log_csv_path = os.path.join(os.path.dirname(output_csv_path), 'failed_batches.csv')
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
        batch_end = min(i + batch_size, len(df))
        batch_df = df.iloc[i:batch_end]

        tweets_to_translate = batch_df[batch_df[target_column].isnull() | (batch_df[target_column] == "")][source_column].tolist()
        
        if not tweets_to_translate:
            # print(f"Batch {i//batch_size + 1}: Skipping, all tweets already translated.")
            continue
            
        translations_dict, last_response_text = translate_tweet_batch(tweets_to_translate)

        if not translations_dict:
            print(f"  [!] Warning: Received no valid translations for batch starting at index {i}.")
            # Log failure details for targeted reprocessing
            failed_row_indices = [idx for idx, row in batch_df.iterrows() if pd.isnull(row[target_column]) or row[target_column] == ""]
            _append_failed_log(
                failed_log_csv_path,
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "batch_start_index": i,
                    "batch_end_index": batch_end,
                    "num_requested": len(tweets_to_translate),
                    "num_returned": 0,
                    "row_indices": "|".join(map(str, failed_row_indices)),
                    "reason": "empty_translations",
                },
            )
            # Immediate checkpoint save
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            continue

        current_tweet_index = 0
        for idx, row in batch_df.iterrows():
             if pd.isnull(row[target_column]) or row[target_column] == "":
                current_tweet_index += 1
                translation_key = str(current_tweet_index)
                if translation_key in translations_dict:
                    df.loc[idx, target_column] = translations_dict[translation_key]

        if (i // batch_size + 1) % 5 == 0:
            print(f"\nSaving intermediate progress to {output_csv_path}...")
            df.to_csv(output_csv_path, index=False)
        
        time.sleep(1)

    print("\nTranslation complete!")
    print(f"Saving final results to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(script_dir, 'init-preprocess', 'cleaned_tweets.csv') # Change to cleaned_tweets.csv for FULL DATASET
    OUTPUT_FILE = os.path.join(script_dir, 'annotated-preprocess', 'translated_tweets.csv')
    SOURCE_TEXT_COLUMN = 'src'
    TARGET_TEXT_COLUMN = 'tgt'
    BATCH_SIZE = 50

    process_csv_in_batches(
        input_csv_path=INPUT_FILE,
        output_csv_path=OUTPUT_FILE,
        source_column=SOURCE_TEXT_COLUMN,
        target_column=TARGET_TEXT_COLUMN,
        batch_size=BATCH_SIZE
    )