from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

model_name = "t5-small"  # You can choose t5-small, t5-large, etc.
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load datasets
spider_dataset = load_dataset('xlangai/spider', split='train')
wikisql_dataset = load_dataset('Salesforce/wikisql', split='train', trust_remote_code=True)
b_mc2 = load_dataset('b-mc2/sql-create-context', split='train')
gretelai = load_dataset('gretelai/synthetic_text_to_sql', split='train')


# Preprocess the datasets
def preprocess_function(examples):
    inputs = (
        examples.get('question') or
        examples.get('sql_prompt') or
        examples.get('context') or
        examples.get('sql_context')
    )
    # Handle WikiSQL's nested 'sql' dict
    if 'sql' in examples and isinstance(examples['sql'], list) and isinstance(examples['sql'][0], dict):
        targets = [sql.get('human_readable', '') if sql else '' for sql in examples['sql']]
    else:
        targets = (
            examples.get('sql') or
            examples.get('query') or
            examples.get('answer') or
            examples.get('context') or
            examples.get('sql_context') or
            examples.get('sql_explanation')
        )

    # Convert to lists if single string
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(targets, str):
        targets = [targets]

    # If still None or not a list, return empty dict
    if not isinstance(inputs, list) or not isinstance(targets, list):
        return {}

    # Filter out pairs with missing values
    filtered = [(q, t) for q, t in zip(inputs, targets) if q and t]
    if not filtered:
        return {}

    inputs, targets = zip(*filtered)
    inputs = list(inputs)
    targets = list(targets)

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

columns_to_keep = ['input_ids', 'attention_mask', 'labels']
# Apply preprocessing
spider_dataset = spider_dataset.map(preprocess_function, batched=True).remove_columns(
    [col for col in spider_dataset.column_names if col not in columns_to_keep]
)
wikisql_dataset = wikisql_dataset.map(preprocess_function, batched=True).remove_columns(
    [col for col in wikisql_dataset.column_names if col not in columns_to_keep]
)
b_mc2 = b_mc2.map(preprocess_function, batched=True).remove_columns(
    [col for col in b_mc2.column_names if col not in columns_to_keep]
)
gretelai = gretelai.map(preprocess_function, batched=True).remove_columns(
    [col for col in gretelai.column_names if col not in columns_to_keep]
)

# Combine datasets
combined_dataset = concatenate_datasets([spider_dataset, wikisql_dataset, b_mc2, gretelai])
combined_dataset.set_format(type="torch")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
