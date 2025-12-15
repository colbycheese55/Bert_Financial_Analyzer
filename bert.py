import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
training_args = {"learning_rate": 2e-5, "per_device_train_batch_size": 32, "num_train_epochs": 4, "weight_decay": 0.01,}


class Bert:
    def __init__(self):
        pass

    def import_data_from_cache(self, cache_path: str) -> None:
        self.df = pd.read_csv(cache_path, sep=',')

        # convert sentiment (positive, neutral, negative) to numerical values
        sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.df['sentiment'] = self.df['sentiment'].map(sentiment_mapping)

    def tokenize_function(self, batch):
        return tokenizer(
            batch["headline"],
            padding="max_length",
            truncation=True,
            max_length=32,
        )
    
    def train(self, train_dataset, eval_dataset):
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="./bert_model", **training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate(eval_dataset)
        print(metrics)

    def split_data(self):
        train_size = int(0.8 * len(self.df))
        shuffled_df = self.df.sample(frac=1, random_state=42)
        train_dataset = shuffled_df.iloc[:train_size]
        eval_dataset = shuffled_df.iloc[train_size:]
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        print(classification_report(labels, preds, target_names=["negative","neutral","positive"]))

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }
    
def train_bert_model():
    bert = Bert()
    bert.import_data_from_cache('./financial_news_with_prices.csv')
    train_dataset, eval_dataset = bert.split_data()

    train_dataset = Dataset.from_pandas(train_dataset)
    eval_dataset  = Dataset.from_pandas(eval_dataset)
    
    train_dataset = train_dataset.map(bert.tokenize_function, batched=True)
    eval_dataset  = eval_dataset.map(bert.tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("sentiment", "labels")
    eval_dataset  = eval_dataset.rename_column("sentiment", "labels")
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    eval_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    bert.train(train_dataset, eval_dataset)



if __name__ == "__main__":
    train_bert_model()