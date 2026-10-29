from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#Pobieranie dataset IMDB (recenzje filmow pozytywne/negatywne)
#1000 probek treningowych / 200 walidacyjnych
dataset = load_dataset("stanfordnlp/imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
validation_dataset = dataset["test"].shuffle(seed=42).select(range(200))

#Inicjalizacja tokenizatora i modelu
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Model do kalsyfikacji ma 2 etykiety: pozytywna / negatywna
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)

def tokenize_fun(examples):
    #dane wejsciowe musza miec ten sam rozmiar
    #truncation: ucina zbyt dlugie teksty
    #padding: dopisuje zera do za krotkich tekstow
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train = train_dataset.map(tokenize_fun, batched=True)
tokenized_validation = validation_dataset.map(tokenize_fun, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    #wybiera najwieksze zwrocone prawdopodobienstwo
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./model_sentymentu")
tokenizer.save_pretrained("./model_sentymentu")