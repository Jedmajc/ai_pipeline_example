from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb")

# Sprawdzamy słownik etykiet dla kolumny "label"
print(dataset["train"].features["label"].names)
# Wynik w konsoli: ['neg', 'pos']