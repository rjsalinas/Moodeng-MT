import json
import csv
import os

def extract_tweet_data(json_file_path, csv_output_path):
    """
    Extract only id and text fields from the JSON file and save to CSV.
    
    Args:
        json_file_path (str): Path to the input JSON file
        csv_output_path (str): Path for the output CSV file
    """
    try:
        # Read the JSON file
        print(f"Reading JSON file: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        print(f"Found {len(data)} tweets in the JSON file")
        
        # Write to CSV with only id and text columns
        print(f"Writing to CSV file: {csv_output_path}")
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header
            writer.writerow(['id', 'text'])
            
            # Write data rows
            for tweet in data:
                tweet_id = tweet.get('id', '')
                tweet_text = tweet.get('text', '')
                
                # Clean the text by removing newlines and extra whitespace
                tweet_text = ' '.join(tweet_text.split())
                
                writer.writerow([tweet_id, tweet_text])
        
        print(f"Successfully extracted {len(data)} tweets to {csv_output_path}")
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    # Input JSON file path
    json_file = "dataset_tweettaglish-extraction---remaining-links_2025-08-13_14-15-28-079.json"
    
    # Output CSV file path
    csv_file = "tweets_id_text_only.csv"
    
    # Check if input file exists
    if not os.path.exists(json_file):
        print(f"Error: Input file '{json_file}' not found in current directory")
        print("Please make sure the JSON file is in the same directory as this script")
        return
    
    # Extract data
    extract_tweet_data(json_file, csv_file)
    
    # Display sample of the output
    if os.path.exists(csv_file):
        print("\nSample of the generated CSV:")
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:6]  # First 6 lines (header + 5 data rows)
            for line in lines:
                print(line.strip())

if __name__ == "__main__":
    main()
