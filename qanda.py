import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

model_dir = "./finetuned_model"  # Path to your saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = T5ForConditionalGeneration.from_pretrained(model_dir)
model = PeftModel.from_pretrained(base_model, model_dir, is_local=True).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

print("Model and tokenizer loaded successfully.")

def ask_question(question):
    prompt = f"""
        {question}
    """
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    sql = tokenizer.decode(outputs[0])
    return sql

print("Ready to answer questions.")

if __name__ == "__main__":
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not question:
            continue
        answer = ask_question(question)
        print("Answer:", answer)
