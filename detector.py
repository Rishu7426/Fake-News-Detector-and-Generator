from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch

# Model paths
pretrained_model_name = "distilbert-base-uncased"
finetuned_model_path = "./fakenews_detector_model_final"  # <-- your downloaded fine-tuned model

# Load Tokenizers
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

# Load Pretrained Model
pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)

# Load Fine-Tuned Model
finetuned_model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path, num_labels=2)

# Device
device = 0 if torch.cuda.is_available() else -1

# Pipelines
pretrained_pipeline = TextClassificationPipeline(model=pretrained_model, tokenizer=tokenizer, return_all_scores=True, device=device)
finetuned_pipeline = TextClassificationPipeline(model=finetuned_model, tokenizer=tokenizer, return_all_scores=True, device=device)

def predict_fake_news(text, use_finetuned=False):
    """
    Predict if input text is Fake or Real news using pretrained or fine-tuned model.
    Returns: (label, confidence, source)
    """
    pipeline = finetuned_pipeline if use_finetuned else pretrained_pipeline

    outputs = pipeline(text)
    scores = outputs[0]
    fake_score = scores[0]['score']
    real_score = scores[1]['score']

    label = "Fake" if fake_score > real_score else "Real"
    confidence = round(max(fake_score, real_score) * 100, 2)

    source = "Fine-Tuned Model" if use_finetuned else "Pretrained Model"

    return label, confidence, source
