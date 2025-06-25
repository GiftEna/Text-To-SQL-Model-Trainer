from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, PeftType
import pandas as pd

model_name = "google/flan-t5-xl"  # You can choose t5-small, t5-large, etc.
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load dataset from CSV
df = pd.read_csv('sql_dataset.csv')
# Inspect the model architecture
# for name, module in model.named_modules():
#     print(name)

# Preprocess function
def preprocess_function(examples):
    inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=512)
    inputs['labels'] = labels['input_ids']
    return inputs

# Apply preprocessing
dataset = Dataset.from_pandas(df)
dataset = dataset.map(preprocess_function)

# Define LoRA configuration
lora_config = LoraConfig(
    peft_type=PeftType.LORA,
    r=8,
    lora_alpha=16,  # Tune this based on training observations and needs
    lora_dropout=0.1,
    target_modules=["q", "k", "v"]
)

# Integrate LoRA
lora_model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./result",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    label_names=["labels"],
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save your trained model
lora_model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

