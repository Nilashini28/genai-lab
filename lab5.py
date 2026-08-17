from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I really enjoyed this movie!",
    "The service was terrible and disappointing.",
    "The product works as expected."
]

results = classifier(texts)

for text, r in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {r['label']}, Score: {r['score']:.3f}")
    print()