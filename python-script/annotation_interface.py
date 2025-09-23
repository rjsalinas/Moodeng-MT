import os
import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder

def clear_console():
    """Clear the console output"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """
    Main function to run the interactive annotation CLI.
    """
    parser = argparse.ArgumentParser(description="Interactive CLI for annotating translations.")
    parser.add_argument("--repo_id", type=str, default="propanda02/TweetTaglish-SalinTala", help="Hugging Face repository ID.")
    parser.add_argument("--base_branch", type=str, default="annotated-updates", help="The branch to pull the dataset from.")
    parser.add_argument("--new_branch", type=str, default="annotated-updates", help="The branch to push the new dataset to.")
    args = parser.parse_args()

    # --- 1. Load the dataset ---
    print(f"Loading dataset from '{args.repo_id}', branch '{args.base_branch}'...")
    try:
        main_ds = load_dataset(args.repo_id, split="train", revision=args.base_branch)
        print(f"\rDataset loaded successfully. Total items: {len(main_ds)}")
    except Exception as e:
        print(f"\rError loading dataset: {e}")
        return

    # --- 2. Define keyword checks ---
    keyword_checks = {
        "hindi": ["not", "don't", "no"],
        "bat": ["why"],
        "ano": ["what"],
        "wala": ["nothing", "none", "no"],
        "gusto": ["want", "like"],
    }

    # --- 3. Annotation loop ---
    updated_translations = []
    # Keep track of original indices to splice the dataset later
    processed_ids = set()
    total_items = len(main_ds)
    processed_count = 0

    try:
        for item in main_ds:
            processed_count += 1
            print(f"\rProcessing item {processed_count}/{total_items} (ID: {item['id']})", end="", flush=True)
            tl_text = item['translation']['tl']
            en_text = item['translation']['en']
            item_id = item['id']
            processed_ids.add(item_id) # Track that this item has been seen

            flagged = False
            for keyword, checks in keyword_checks.items():
                if keyword in tl_text.lower().split():
                    for check in checks:
                        if check in en_text.lower().split():
                            flagged = True
                            break
                if flagged:
                    break

            if flagged:
                clear_console()
                print(f"Processing: {processed_count}/{total_items} | ID: {item_id}")
                print(f"\n--- Flagged for Review: {item_id} ---")
                print(f"Tagalog: {tl_text}")
                print(f"English: {en_text}")

                action = input("\nAction (a: accept, e: edit, d: delete, s: skip, q: quit and save): ").lower()

                if action == 'q':
                    print("Quitting annotation and saving progress...")
                    updated_translations.append(item)
                    break # Exit the loop
                # if action is a or enter key, accept
                elif action == 'a' or action == '':
                    updated_translations.append(item)
                elif action == 'e':
                    new_en = input("Enter new English translation: ")
                    item['translation']['en'] = new_en
                    updated_translations.append(item)
                elif action == 'd':
                    continue  # Skip adding to the updated list
                else: # 's' or any other key
                    updated_translations.append(item)
            else:
                updated_translations.append(item)

    except KeyboardInterrupt:
        print(f"\r\nInterrupted by user. Saving progress...")

    # --- 4. Combine with the rest of the dataset ---
    print(f"\r\nProcessing complete. {processed_count}/{total_items} items processed.")
    # Get the items that were not processed
    unprocessed_items = [item for item in main_ds if item['id'] not in processed_ids]
    final_data = updated_translations + unprocessed_items

    # --- 5. Save and push the updated dataset ---
    if final_data:
        print(f"\rSaving and pushing changes... ({len(final_data)} items)")
        # Convert to a new Dataset object
        
        # Correctly structure the data for Dataset.from_dict
        data_dict = {
            "id": [item["id"] for item in final_data],
            "translation": [{"tl": item["translation"]["tl"], "en": item["translation"]["en"]} for item in final_data]
        }
        
        updated_ds = Dataset.from_dict(data_dict)

        try:
            api = HfApi()
            print(f"\rCreating and pushing to new branch: '{args.new_branch}'...")
            api.create_branch(
                repo_id=args.repo_id,
                branch=args.new_branch,
                repo_type="dataset",
                token=HfFolder.get_token(),
                exist_ok=True, # Allow the branch to exist already
            )
            updated_ds.push_to_hub(
                repo_id=args.repo_id,
                commit_message="Apply partial annotations",
                revision=args.new_branch
            )
            print(f"\r✅ Changes pushed to branch '{args.new_branch}' successfully.")
        except Exception as e:
            print(f"\r❌ Error pushing to hub: {e}")

if __name__ == "__main__":
    main()