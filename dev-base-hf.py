import os

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from grouphug import AutoMultiTaskModel, ClassificationHeadConfig, DatasetFormatter, LMHeadConfig, MultiTaskTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = ""


tweet_emotion = load_dataset("tweet_eval", "emotion")
tweet_emotion["train"] = tweet_emotion["train"].select(range(1))
tweet_emotion["test"] = tweet_emotion["train"].select(range(1))

base_model = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(base_model)
formatter = DatasetFormatter().tokenize()
data = formatter.apply(tweet_emotion, tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=4)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


training_args = TrainingArguments(
    output_dir="../output",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    logging_steps=100,
    save_strategy="no",
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data["data", "train"],
    eval_dataset=data["data", "test"],
    args=training_args,
    compute_metrics=compute_metrics,
)
trainer.train()


# In[ ]:
