from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import torch
import math

pretrained_model_name = "distilgpt2"
finetuned_model_path = "./distilgpt2-fakenews"

# Pretrained model (from Hugging Face hub)
pre_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
pre_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

if pre_tokenizer.pad_token is None:
    pre_tokenizer.pad_token = pre_tokenizer.eos_token

# Fine-tuned model (from local path ✅)
fine_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, local_files_only=True)
fine_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, local_files_only=True)

if fine_tokenizer.pad_token is None:
    fine_tokenizer.pad_token = fine_tokenizer.eos_token

# Generators
pre_generator = pipeline("text-generation", model=pre_model, tokenizer=pre_tokenizer)
fine_generator = pipeline("text-generation", model=fine_model, tokenizer=fine_tokenizer)

set_seed(42)

def generate_fake_news(prompt, max_length=150, temperature=1.0, use_finetuned=True):
    generator = fine_generator if use_finetuned else pre_generator
    output = generator(prompt, max_length=max_length, temperature=temperature, num_return_sequences=1)
    return output[0]['generated_text']

def calculate_perplexity(text, use_finetuned=True):
    if not text.strip():
        return float('nan')

    tokenizer = fine_tokenizer if use_finetuned else pre_tokenizer
    model = fine_model if use_finetuned else pre_model

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize and move tensors to the same device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[1] == 0:
        return float('nan')

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

    return round(math.exp(loss.item()), 2)

