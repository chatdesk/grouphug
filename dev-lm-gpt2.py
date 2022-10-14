import os

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments

from examples.utils import compute_classification_metrics
from grouphug import AutoMultiTaskModel, ClassificationHeadConfig, DatasetFormatter, LMHeadConfig, MultiTaskTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = ""

tweet_emotion = load_dataset("tweet_eval", "emotion").rename_column("label", "emotion")

base_model = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model)

formatter = DatasetFormatter().tokenize()
data = formatter.apply(tweet_emotion, tokenizer=tokenizer)

head_configs = [ClassificationHeadConfig.from_data(data, "emotion", classifier_hidden_size=32)]

training_args = TrainingArguments(
    output_dir="../output",
    evaluation_strategy="epoch",
    num_train_epochs=10,
    logging_steps=100,
    save_strategy="no",
)

lm_head = [LMHeadConfig(weight=0.2, causal_language_modelling=True)]

model = AutoMultiTaskModel.from_pretrained(base_model, head_configs + lm_head, formatter=formatter, tokenizer=tokenizer)
model.config.is_decoder = True
model.head_configs[0].detached = False
model.model_parallel = False
trainer = MultiTaskTrainer(
    model=model,
    tokenizer=tokenizer,
    train_data=data[:, "train"],
    eval_data=data[:, "test"],
    eval_heads=["emotion"],
    compute_metrics=compute_classification_metrics,
    args=training_args,
)
trainer.train()