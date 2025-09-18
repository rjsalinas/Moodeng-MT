```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_prepare_data(csv_path, tokenizer, src_lang="en_XX", tgt_lang="ur_PK"):
    """Loads data from a CSV and prepares it for mBART fine-tuning."""
    
    # 1. Load data from CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)
    # Assuming columns are named 'English' and 'Urdu'
    df.rename(columns={"English": "en", "Urdu": "ur"}, inplace=True)
    
    # 2. Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # 3. Set up the tokenizer for the specific source and target languages
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    def tokenize_function(examples):
        """Tokenize the text."""
        inputs = tokenizer(examples["en"], truncation=True, padding="max_length", max_length=128)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["ur"], truncation=True, padding="max_length", max_length=128)
        
        inputs["labels"] = labels["input_ids"]
        return inputs

    # 4. Apply tokenization to the entire dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["en", "ur"])
    
    # You would typically split this into train/validation sets
    # For example:
    # train_val_split = tokenized_dataset.train_test_split(test_size=0.1)
    # train_dataset = train_val_split['train']
    # eval_dataset = train_val_split['test']
    
    return tokenized_dataset #, train_dataset, eval_dataset
```

### 2. Model Training (`train.sh`)

The `fairseq-train` command and its many arguments can be mapped to `Seq2SeqTrainingArguments` and `Seq2SeqTrainer`.

Here is how the parameters from `train.sh` translate:

| `train.sh` Argument | `transformers` Equivalent (`Seq2SeqTrainingArguments`) |
| :--- | :--- |
| `CHECKPOINTS_DIR` | `output_dir` |
| `EPOCH` | `num_train_epochs` |
| `LEARNING_RATE` | `learning_rate` |
| `OPTIMIZADOR` | `optim` (e.g., 'adamw_torch') |
| `LOSS` & `--label-smoothing` | `label_smoothing_factor` |
| `--max-tokens` | `per_device_train_batch_size` & `gradient_accumulation_steps` |
| `--warmup-updates` | `warmup_steps` |
| `--save-interval-updates` | `save_steps` |
| `--validate-interval-updates` | `eval_steps` |
| `mBART_MODEL` | The model identifier passed to `from_pretrained` |

Here's the corresponding Python code for training:

```python
from transformers import (
    MBartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# --- 1. Load Model and Tokenizer ---
model_name = "facebook/mbart-large-50"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- 2. Load and Prepare Data (using function from step 1) ---
# Assuming you have a 'train.csv' and 'valid.csv'
train_dataset = load_and_prepare_data("path/to/train.csv", tokenizer, src_lang="en_XX", tgt_lang="tl_XX")
eval_dataset = load_and_prepare_data("path/to/valid.csv", tokenizer, src_lang="en_XX", tgt_lang="tl_XX")


# --- 3. Define Training Arguments ---
# This translates the arguments from train.sh
training_args = Seq2SeqTrainingArguments(
    output_dir="../mbart/checkpoints_py", # CHECKPOINTS_DIR
    num_train_epochs=10, # EPOCH
    per_device_train_batch_size=4, # Adjust based on MAX_TOKENS and GPU memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # Simulates a larger batch size (4*4*num_gpus)
    learning_rate=3e-5, # LEARNING_RATE (Note: 3e-4 is high, 3e-5 is more common)
    lr_scheduler_type="polynomial", # SCHEDULER
    warmup_steps=2500, # warmup-updates
    optim="adamw_torch", # OPTIMIZADOR
    weight_decay=0.01,
    label_smoothing_factor=0.2, # --label-smoothing
    
    logging_dir='./logs',
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=5000, # --validate-interval-updates
    save_strategy="steps",
    save_steps=5000, # --save-interval-updates
    save_total_limit=10, # Corresponds to --keep-interval-updates
    
    fp16=True, # Use mixed-precision training
    predict_with_generate=True,
    push_to_hub=False,
)

# --- 4. Initialize Trainer ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 5. Start Training ---
trainer.train()

# --- 6. Save the final model ---
trainer.save_model("../mbart/final_model_py")
```

### 3. Translation/Inference (`translate.sh` and `prep-inference.sh`)

Generating translations is much simpler in `transformers`. You can use the `pipeline` helper, which handles tokenization, model generation, and decoding for you.


This replaces both `prep-inference.sh` (data prep) and `translate.sh` (generation).

```python
from transformers import pipeline

# Path to your fine-tuned model
model_path = "../mbart/final_model_py" 

# Load the translation pipeline
translator = pipeline(
    "translation", 
    model=model_path,
    tokenizer=model_path,
    src_lang="en_XX", 
    tgt_lang="tl_XX",
    device=0 # Use the first GPU
)

# Translate a single sentence
text_to_translate = "This is a test of the translation model."
result = translator(text_to_translate, max_length=50)
print(result)

# Translate a list of sentences
texts = [
    "How are you?",
    "The weather is nice today."
]
results = translator(texts, max_length=50)
for res in results:
    print(res)
```

By following this structure, you can effectively replicate the entire workflow from your bash scripts in a more integrated and often easier-to-debug Python environment.

```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_prepare_data(csv_path, tokenizer, src_lang="en_XX", tgt_lang="ur_PK"):
    """Loads data from a CSV and prepares it for mBART fine-tuning."""
    
    # 1. Load data from CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)
    # Assuming columns are named 'English' and 'Urdu'
    df.rename(columns={"English": "en", "Urdu": "ur"}, inplace=True)
    
    # 2. Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # 3. Set up the tokenizer for the specific source and target languages
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    def tokenize_function(examples):
        """Tokenize the text."""
        inputs = tokenizer(examples["en"], truncation=True, padding="max_length", max_length=128)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["ur"], truncation=True, padding="max_length", max_length=128)
        
        inputs["labels"] = labels["input_ids"]
        return inputs

    # 4. Apply tokenization to the entire dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["en", "ur"])
    
    # You would typically split this into train/validation sets
    # For example:
    # train_val_split = tokenized_dataset.train_test_split(test_size=0.1)
    # train_dataset = train_val_split['train']
    # eval_dataset = train_val_split['test']
    
    return tokenized_dataset #, train_dataset, eval_dataset
```

### 2. Model Training (`train.sh`)

The `fairseq-train` command and its many arguments can be mapped to `Seq2SeqTrainingArguments` and `Seq2SeqTrainer`.

Here is how the parameters from `train.sh` translate:

| `train.sh` Argument | `transformers` Equivalent (`Seq2SeqTrainingArguments`) |
| :--- | :--- |
| `CHECKPOINTS_DIR` | `output_dir` |
| `EPOCH` | `num_train_epochs` |
| `LEARNING_RATE` | `learning_rate` |
| `OPTIMIZADOR` | `optim` (e.g., 'adamw_torch') |
| `LOSS` & `--label-smoothing` | `label_smoothing_factor` |
| `--max-tokens` | `per_device_train_batch_size` & `gradient_accumulation_steps` |
| `--warmup-updates` | `warmup_steps` |
| `--save-interval-updates` | `save_steps` |
| `--validate-interval-updates` | `eval_steps` |
| `mBART_MODEL` | The model identifier passed to `from_pretrained` |

Here's the corresponding Python code for training:

```python
from transformers import (
    MBartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# --- 1. Load Model and Tokenizer ---
model_name = "facebook/mbart-large-50"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- 2. Load and Prepare Data (using function from step 1) ---
# Assuming you have a 'train.csv' and 'valid.csv'
train_dataset = load_and_prepare_data("path/to/train.csv", tokenizer, src_lang="en_XX", tgt_lang="tl_XX")
eval_dataset = load_and_prepare_data("path/to/valid.csv", tokenizer, src_lang="en_XX", tgt_lang="tl_XX")


# --- 3. Define Training Arguments ---
# This translates the arguments from train.sh
training_args = Seq2SeqTrainingArguments(
    output_dir="../mbart/checkpoints_py", # CHECKPOINTS_DIR
    num_train_epochs=10, # EPOCH
    per_device_train_batch_size=4, # Adjust based on MAX_TOKENS and GPU memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # Simulates a larger batch size (4*4*num_gpus)
    learning_rate=3e-5, # LEARNING_RATE (Note: 3e-4 is high, 3e-5 is more common)
    lr_scheduler_type="polynomial", # SCHEDULER
    warmup_steps=2500, # warmup-updates
    optim="adamw_torch", # OPTIMIZADOR
    weight_decay=0.01,
    label_smoothing_factor=0.2, # --label-smoothing
    
    logging_dir='./logs',
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=5000, # --validate-interval-updates
    save_strategy="steps",
    save_steps=5000, # --save-interval-updates
    save_total_limit=10, # Corresponds to --keep-interval-updates
    
    fp16=True, # Use mixed-precision training
    predict_with_generate=True,
    push_to_hub=False,
)

# --- 4. Initialize Trainer ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 5. Start Training ---
trainer.train()

# --- 6. Save the final model ---
trainer.save_model("../mbart/final_model_py")

```

### 3. Translation/Inference (`translate.sh` and `prep-inference.sh`)

Generating translations is much simpler in `transformers`. You can use the `pipeline` helper, which handles tokenization, model generation, and decoding for you.

This replaces both `prep-inference.sh` (data prep) and `translate.sh` (generation).

```python
from transformers import pipeline

# Path to your fine-tuned model
model_path = "../mbart/final_model_py" 

# Load the translation pipeline
translator = pipeline(
    "translation", 
    model=model_path,
    tokenizer=model_path,
    src_lang="en_XX", 
    tgt_lang="tl_XX",
    device=0 # Use the first GPU
)

# Translate a single sentence
text_to_translate = "This is a test of the translation model."
result = translator(text_to_translate, max_length=50)
print(result)

# Translate a list of sentences
texts = [
    "How are you?",
    "The weather is nice today."
]
results = translator(texts, max_length=50)
for res in results:
    print(res)
```

By following this structure, you can effectively replicate the entire workflow from your bash scripts in a more integrated and often easier-to-