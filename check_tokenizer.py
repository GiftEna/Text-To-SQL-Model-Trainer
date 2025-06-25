import pandas as pd
from transformers import T5Tokenizer

# Load dataset
csv_path = 'sql_dataset.csv'
df = pd.read_csv(csv_path)

# Initial inspections
print("Sample data:\n", df.head())
print("\nColumn types:\n", df.dtypes)
print("\nAny missing values:", df.isnull().sum())

# Load tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Check tokenization
def check_tokenization(df, tokenizer):
    try:
        inputs_sample = tokenizer(df['input'][0], return_tensors='pt')
        outputs_sample = tokenizer(df['output'][0], return_tensors='pt')
        print("Example tokenized input IDs:", inputs_sample['input_ids'])
        print("Example tokenized output IDs:", outputs_sample['input_ids'])
        return True
    except Exception as e:
        print("Error in tokenization:", str(e))
        return False

# Verify tokenization compatibility
tokenization_check = check_tokenization(df, tokenizer)

if tokenization_check:
    print("Dataset is compatible with your model's tokenizer.")
else:
    print("Dataset has tokenization issues. Review encoding process.")

# Calculate statistics
print("\nBasic Statistics:")
print("Average input length:", df['input'].apply(lambda x: len(x.split())).mean())
print("Average output length:", df['output'].apply(lambda x: len(x.split())).mean())

# Example consistency checks
assert 'input' in df.columns, "'input' column missing in dataset."
assert 'output' in df.columns, "'output' column missing in dataset."
assert not df.isnull().values.any(), "Dataset contains missing values."