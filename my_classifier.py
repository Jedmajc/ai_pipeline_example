import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextClassifier:
    def __init__(self, model_path="./model_sentymentu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.eval()

    def clean_text(self, text):
            if not isinstance(text, str):
                return ""
            return text.strip()

    def predict(self, text):
        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            raise ValueError("Text is empty")

        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            probabilities = F.softmax(logits, dim=-1)

        confidence, predicted_index = torch.max(probabilities, dim=1)

        labels = ["neg", "pos"]

        return {
            "label": labels[predicted_index.item()],
            "confidence": round(confidence.item(), 4),
            "all_probabilities": probabilities.tolist()[0]
        }

if __name__=="__main__":
    my_classifier = TextClassifier()

    test_sentences = [
        "This movie was absolutely amazing! The acting was brilliant.",
        "What a terrible waste of time. I hated every minute of it.",
    ]

    for sentence in test_sentences:
        result = my_classifier.predict(sentence)
        print(f"Text: {sentence}")
        print(f"Result: {result['label']} (Confidence: {result['confidence']*100:.1f}%)\n")
