import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch.nn.functional as F

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

LABELS = list(model.config.id2label.values())

class BertSentiment:
    def __init__(self):
        pass

    def import_data_from_cache(self, cache_path: str):
        self.df = pd.read_csv(cache_path)
        self.df["headline"] = self.df["headline"].astype(str)
        self.df = (
            self.df
            .dropna(subset=["price_change_pct"])
            .reset_index(drop=True)
        )
        # print(self.df[45:60])
        # exit(0)

    def predict_with_probs(self, texts):
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        df = pd.DataFrame(
            probs,
            columns=[f"prob_{l}" for l in LABELS]
        )

        df["predicted_label"] = [LABELS[i] for i in probs.argmax(axis=1)]
        # df["confidence"] = probs.max(axis=1)

        return df


def main():
    bert = BertSentiment()
    bert.import_data_from_cache("./financial_news_with_prices.csv")
    df = bert.predict_with_probs(bert.df["headline"].tolist())
    print(df.head())

    df['price_change_pct'] = bert.df['price_change_pct']
    df.to_csv("bert_predictions.csv", index=False)

if __name__ == "__main__":
    main()
