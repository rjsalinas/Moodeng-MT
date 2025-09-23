from datasets import load_dataset, Dataset, DatasetDict
import os

# Load the parallel text files separately
print("Loading Tagalog text files...")
train_tl = load_dataset('text', data_files='../corpus-parallel-txt/train.tl.cleaned')['train']
val_tl = load_dataset('text', data_files='../corpus-parallel-txt/val.tl.cleaned')['train']

print("Loading English text files...")
train_en = load_dataset('text', data_files='../corpus-parallel-txt/train.en.cleaned')['train']
val_en = load_dataset('text', data_files='../corpus-parallel-txt/val.en.cleaned')['train']

# Create datasets with translation format + explicit ID
print("Creating train dataset...")
train_dataset = Dataset.from_dict({
    'id': list(range(len(train_tl))),
    'translation': [
        {'tl': tl_text, 'en': en_text} 
        for tl_text, en_text in zip(train_tl['text'], train_en['text'])
    ]
})

print("Creating validation dataset...")
val_dataset = Dataset.from_dict({
    'id': list(range(len(val_tl))),
    'translation': [
        {'tl': tl_text, 'en': en_text} 
        for tl_text, en_text in zip(val_tl['text'], val_en['text'])
    ]
})

# Create the combined dataset dictionary
combined_datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# Make sure you are logged in to Hugging Face from your terminal:
# huggingface-cli login

# Define the ID of the EXISTING dataset repository on the Hub
repo_id = "propanda02/TweetTaglish-SalinTala"

# Push the dataset to the Hub on MAIN branch
combined_datasets.push_to_hub(repo_id)

print(f"✅ Dataset successfully pushed to https://huggingface.co/datasets/{repo_id}")
