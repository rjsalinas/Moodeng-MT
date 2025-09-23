from datasets import load_dataset, Dataset, DatasetDict
import os

# Define the file paths to load the parallel files
data_files = {
    'train': {
        'tl': 'train.tl',
        'en': 'train.en'
    },
    'validation': {
        'tl': 'val.tl',
        'en': 'val.en'
    }
}

# Load the data as a raw text dataset
raw_datasets = load_dataset('text', data_files=data_files)

# Combine the parallel files into a single dataset with 'translation' feature
def combine_parallel_corpus(batch):
    return {'translation': {'tl': batch['tl'], 'en': batch['en']}}

# Apply the function to create a new, structured dataset
combined_datasets = raw_datasets.map(
    combine_parallel_corpus,
    batched=True,
    remove_columns=['tl', 'en']
)

# Rename the splits to standard names
combined_datasets['train'] = combined_datasets['train'].rename_column('translation', 'translation')
combined_datasets['validation'] = combined_datasets['validation'].rename_column('translation', 'translation')

# Make sure you are logged in to Hugging Face from your terminal:
# huggingface-cli login

# Define the ID of the EXISTING dataset repository on the Hub
# Replace "your_username/your_dataset_name" with your actual repo ID
repo_id = "propanda02/TweetTaglish-SalinTala"

# Push the dataset to the Hub
# The `repo_id` is the only argument needed for an existing repo
combined_datasets.push_to_hub(repo_id)

print(f"Dataset successfully pushed to https://huggingface.co/datasets/{repo_id}")