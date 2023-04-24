#!/usr/bin/env python3

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_metric



################################### BERT



### Load Data
plr_train = pd.read_csv("./data/binary/plr_train_b.csv")
plr_train = plr_train.iloc[: , 1:]
plr_test = pd.read_csv("./data/binary/plr_test_b.csv")
plr_test = plr_test.iloc[: , 1:]
non_plr_test = pd.read_csv("./data/binary/non_plr_test_b.csv")
non_plr_test = non_plr_test.iloc[: , 1:]
uk_test = pd.read_csv("./data/binary/smpc_test_b.csv")
uk_test = uk_test.iloc[: , 1:]



### Tokenize Data
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

train_data = Dataset.from_pandas(plr_train)
train_data = train_data.train_test_split(test_size=0.20)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = train_data.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

print(train_dataset)
print(test_dataset)

plr_test_data = Dataset.from_pandas(plr_test)
tokenized_plr_test_dataset = plr_test_data.map(tokenize_function, batched=True)

non_plr_test_data = Dataset.from_pandas(non_plr_test)
tokenized_non_plr_test_dataset = non_plr_test_data.map(tokenize_function, batched=True)

uk_test_data = Dataset.from_pandas(uk_test)
tokenized_uk_test_dataset = uk_test_data.map(tokenize_function, batched=True)



### Model  (HuggingFace Trainer)
label_num = len(set(plr_train["label"]))

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=label_num)

training_args = TrainingArguments(output_dir="test_trainer", logging_strategy="epoch", save_strategy="epoch", evaluation_strategy="epoch", load_best_model_at_end=True, num_train_epochs=10, per_device_train_batch_size=32, per_device_eval_batch_size=32)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./results/binary/models/bert_b/')



### Graph Loss
history = trainer.state.log_history

file = open("./results/binary/train_logs/tlog_bert_b.txt", "w")
file.write(repr(history))
file.close

history = str(history)
df = pd.DataFrame(eval(history))

loss_df = df[["epoch", "loss"]]
loss_df = loss_df.dropna()
print(loss_df)

eval_loss = df[["epoch", "eval_loss"]]
eval_loss = eval_loss.dropna()
print(eval_loss)

eval_accuracy = df[["epoch", "eval_accuracy"]]
eval_accuracy = eval_accuracy.dropna()
print(eval_accuracy)

x1 = loss_df.iloc[:, 0].tolist()
y1 = loss_df.iloc[:, 1].tolist()

x2 = eval_loss.iloc[:, 0].tolist()
y2 = eval_loss.iloc[:, 1].tolist()

x3 = eval_accuracy.iloc[:, 0].tolist()
y3 = eval_accuracy.iloc[:, 1].tolist()

plt.plot(x1, y1, label = "Loss")
plt.plot(x2, y2, label = "Eval Loss")
plt.plot(x3, y3, label = "Eval Accuracy")
plt.title("Loss & Accuracy over 10 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss & Accuracy")
plt.legend()
plt.show()
plt.savefig('./results/binary/graphs/graph_bert_b.png')
plt.clf()


### Evaluate Model
test_results = {}


### Plr Test Accuracy
pred = trainer.predict(tokenized_plr_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test Accuracy:", output)
test_results["PLR Test Accuracy"] = output


### Non-Plr Test Accuracy
pred = trainer.predict(tokenized_non_plr_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test Accuacy:", output)
test_results["Non-PLR Test Accuracy"] = output


### UK SmPC Test Accuracy
pred = trainer.predict(tokenized_uk_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test Accuracy:", output)
test_results["UK SmPC Test Accuracy"] = output


# PLR TEST - LABEL 0 - Indications
plr_test_0 = plr_test[plr_test["label"] == 0]
plr_test_data_0 = Dataset.from_pandas(plr_test_0)
tokenized_plr_test_dataset_0 = plr_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_plr_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test: Label 0 Accuracy:", output)
test_results["PLR Test: Label 0 Accuracy"] = output


# PLR TEST - LABEL 1 - Warnings
plr_test_1 = plr_test[plr_test["label"] == 1]
plr_test_data_1 = Dataset.from_pandas(plr_test_1)
tokenized_plr_test_dataset_1 = plr_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_plr_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test: Label 1 Accuracy:", output)
test_results["PLR Test: Label 1 Accuracy"] = output


# Non-PLR TEST - LABEL 0 - Indications
non_plr_test_0 = non_plr_test[non_plr_test["label"] == 0]
non_plr_test_data_0 = Dataset.from_pandas(non_plr_test_0)
tokenized_non_plr_test_dataset_0 = non_plr_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_non_plr_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test: Label 0 Accuracy:", output)
test_results["Non-PLR Test: Label 0 Accuracy"] = output


# Non-PLR TEST - LABEL 1 - Warnings
non_plr_test_1 = non_plr_test[non_plr_test["label"] == 1]
non_plr_test_data_1 = Dataset.from_pandas(non_plr_test_1)
tokenized_non_plr_test_dataset_1 = non_plr_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_non_plr_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test: Label 1 Accuracy:", output)
test_results["Non-PLR Test: Label 1 Accuracy"] = output


# UK SmPC TEST - LABEL 0 - Indications
uk_test_0 = uk_test[uk_test["label"] == 0]
uk_test_data_0 = Dataset.from_pandas(uk_test_0)
tokenized_uk_test_dataset_0 = uk_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_uk_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test: Label 0 Accuracy:", output)
test_results["UK SmPC Test: Label 0 Accuracy"] = output


# UK SmPC TEST - LABEL 1 - Warnings
uk_test_1 = uk_test[uk_test["label"] == 1]
uk_test_data_1 = Dataset.from_pandas(uk_test_1)
tokenized_uk_test_dataset_1 = uk_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_uk_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test: Label 1 Accuracy:", output)
test_results["UK SmPC Test: Label 1 Accuracy"] = output


file = open("./results/binary/eval_logs/elog_bert_b.txt", "w")
file.write(repr(test_results))
file.close



################################### ALBERT



### Load Data
plr_train = pd.read_csv("./data/binary/plr_train_b.csv")
plr_train = plr_train.iloc[: , 1:]
plr_test = pd.read_csv("./data/binary/plr_test_b.csv")
plr_test = plr_test.iloc[: , 1:]
non_plr_test = pd.read_csv("./data/binary/non_plr_test_b.csv")
non_plr_test = non_plr_test.iloc[: , 1:]
uk_test = pd.read_csv("./data/binary/smpc_test_b.csv")
uk_test = uk_test.iloc[: , 1:]



### Tokenize Data
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

train_data = Dataset.from_pandas(plr_train)
train_data = train_data.train_test_split(test_size=0.20)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = train_data.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

print(train_dataset)
print(test_dataset)

plr_test_data = Dataset.from_pandas(plr_test)
tokenized_plr_test_dataset = plr_test_data.map(tokenize_function, batched=True)

non_plr_test_data = Dataset.from_pandas(non_plr_test)
tokenized_non_plr_test_dataset = non_plr_test_data.map(tokenize_function, batched=True)

uk_test_data = Dataset.from_pandas(uk_test)
tokenized_uk_test_dataset = uk_test_data.map(tokenize_function, batched=True)



### Model  (HuggingFace Trainer)
label_num = len(set(plr_train["label"]))

model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=label_num)

training_args = TrainingArguments(output_dir="test_trainer", logging_strategy="epoch", save_strategy="epoch", evaluation_strategy="epoch", load_best_model_at_end=True, num_train_epochs=10, per_device_train_batch_size=32, per_device_eval_batch_size=32)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./results/binary/models/albert_b/')



### Graph Loss
history = trainer.state.log_history

file = open("./results/binary/train_logs/tlog_albert_b.txt", "w")
file.write(repr(history))
file.close

history = str(history)
df = pd.DataFrame(eval(history))

loss_df = df[["epoch", "loss"]]
loss_df = loss_df.dropna()
print(loss_df)

eval_loss = df[["epoch", "eval_loss"]]
eval_loss = eval_loss.dropna()
print(eval_loss)

eval_accuracy = df[["epoch", "eval_accuracy"]]
eval_accuracy = eval_accuracy.dropna()
print(eval_accuracy)

x1 = loss_df.iloc[:, 0].tolist()
y1 = loss_df.iloc[:, 1].tolist()

x2 = eval_loss.iloc[:, 0].tolist()
y2 = eval_loss.iloc[:, 1].tolist()

x3 = eval_accuracy.iloc[:, 0].tolist()
y3 = eval_accuracy.iloc[:, 1].tolist()

plt.plot(x1, y1, label = "Loss")
plt.plot(x2, y2, label = "Eval Loss")
plt.plot(x3, y3, label = "Eval Accuracy")
plt.title("Loss & Accuracy over 10 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss & Accuracy")
plt.legend()
plt.show()
plt.savefig('./results/binary/graphs/graph_albert_b.png')
plt.clf()


### Evaluate Model
test_results = {}


### Plr Test Accuracy
pred = trainer.predict(tokenized_plr_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test Accuracy:", output)
test_results["PLR Test Accuracy"] = output


### Non-Plr Test Accuracy
pred = trainer.predict(tokenized_non_plr_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test Accuacy:", output)
test_results["Non-PLR Test Accuracy"] = output


### UK SmPC Test Accuracy
pred = trainer.predict(tokenized_uk_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test Accuracy:", output)
test_results["UK SmPC Test Accuracy"] = output


# PLR TEST - LABEL 0 - Indications
plr_test_0 = plr_test[plr_test["label"] == 0]
plr_test_data_0 = Dataset.from_pandas(plr_test_0)
tokenized_plr_test_dataset_0 = plr_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_plr_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test: Label 0 Accuracy:", output)
test_results["PLR Test: Label 0 Accuracy"] = output


# PLR TEST - LABEL 1 - Warnings
plr_test_1 = plr_test[plr_test["label"] == 1]
plr_test_data_1 = Dataset.from_pandas(plr_test_1)
tokenized_plr_test_dataset_1 = plr_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_plr_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test: Label 1 Accuracy:", output)
test_results["PLR Test: Label 1 Accuracy"] = output


# Non-PLR TEST - LABEL 0 - Indications
non_plr_test_0 = non_plr_test[non_plr_test["label"] == 0]
non_plr_test_data_0 = Dataset.from_pandas(non_plr_test_0)
tokenized_non_plr_test_dataset_0 = non_plr_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_non_plr_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test: Label 0 Accuracy:", output)
test_results["Non-PLR Test: Label 0 Accuracy"] = output


# Non-PLR TEST - LABEL 1 - Warnings
non_plr_test_1 = non_plr_test[non_plr_test["label"] == 1]
non_plr_test_data_1 = Dataset.from_pandas(non_plr_test_1)
tokenized_non_plr_test_dataset_1 = non_plr_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_non_plr_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test: Label 1 Accuracy:", output)
test_results["Non-PLR Test: Label 1 Accuracy"] = output


# UK SmPC TEST - LABEL 0 - Indications
uk_test_0 = uk_test[uk_test["label"] == 0]
uk_test_data_0 = Dataset.from_pandas(uk_test_0)
tokenized_uk_test_dataset_0 = uk_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_uk_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test: Label 0 Accuracy:", output)
test_results["UK SmPC Test: Label 0 Accuracy"] = output


# UK SmPC TEST - LABEL 1 - Warnings
uk_test_1 = uk_test[uk_test["label"] == 1]
uk_test_data_1 = Dataset.from_pandas(uk_test_1)
tokenized_uk_test_dataset_1 = uk_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_uk_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test: Label 1 Accuracy:", output)
test_results["UK SmPC Test: Label 1 Accuracy"] = output


file = open("./results/binary/eval_logs/elog_albert_b.txt", "w")
file.write(repr(test_results))
file.close



################################### DistilBERT



### Load Data
plr_train = pd.read_csv("./data/binary/plr_train_b.csv")
plr_train = plr_train.iloc[: , 1:]
plr_test = pd.read_csv("./data/binary/plr_test_b.csv")
plr_test = plr_test.iloc[: , 1:]
non_plr_test = pd.read_csv("./data/binary/non_plr_test_b.csv")
non_plr_test = non_plr_test.iloc[: , 1:]
uk_test = pd.read_csv("./data/binary/smpc_test_b.csv")
uk_test = uk_test.iloc[: , 1:]



### Tokenize Data
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

train_data = Dataset.from_pandas(plr_train)
train_data = train_data.train_test_split(test_size=0.20)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = train_data.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

print(train_dataset)
print(test_dataset)

plr_test_data = Dataset.from_pandas(plr_test)
tokenized_plr_test_dataset = plr_test_data.map(tokenize_function, batched=True)

non_plr_test_data = Dataset.from_pandas(non_plr_test)
tokenized_non_plr_test_dataset = non_plr_test_data.map(tokenize_function, batched=True)

uk_test_data = Dataset.from_pandas(uk_test)
tokenized_uk_test_dataset = uk_test_data.map(tokenize_function, batched=True)



### Model  (HuggingFace Trainer)
label_num = len(set(plr_train["label"]))

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=label_num)

training_args = TrainingArguments(output_dir="test_trainer", logging_strategy="epoch", save_strategy="epoch", evaluation_strategy="epoch", load_best_model_at_end=True, num_train_epochs=10, per_device_train_batch_size=32, per_device_eval_batch_size=32)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./results/binary/models/distilbert_b/')



### Graph Loss
history = trainer.state.log_history

file = open("./results/binary/train_logs/tlog_distilbert_b.txt", "w")
file.write(repr(history))
file.close

history = str(history)
df = pd.DataFrame(eval(history))

loss_df = df[["epoch", "loss"]]
loss_df = loss_df.dropna()
print(loss_df)

eval_loss = df[["epoch", "eval_loss"]]
eval_loss = eval_loss.dropna()
print(eval_loss)

eval_accuracy = df[["epoch", "eval_accuracy"]]
eval_accuracy = eval_accuracy.dropna()
print(eval_accuracy)

x1 = loss_df.iloc[:, 0].tolist()
y1 = loss_df.iloc[:, 1].tolist()

x2 = eval_loss.iloc[:, 0].tolist()
y2 = eval_loss.iloc[:, 1].tolist()

x3 = eval_accuracy.iloc[:, 0].tolist()
y3 = eval_accuracy.iloc[:, 1].tolist()

plt.plot(x1, y1, label = "Loss")
plt.plot(x2, y2, label = "Eval Loss")
plt.plot(x3, y3, label = "Eval Accuracy")
plt.title("Loss & Accuracy over 10 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss & Accuracy")
plt.legend()
plt.show()
plt.savefig('./results/binary/graphs/graph_distilbert_b.png')
plt.clf()


### Evaluate Model
test_results = {}


### Plr Test Accuracy
pred = trainer.predict(tokenized_plr_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test Accuracy:", output)
test_results["PLR Test Accuracy"] = output


### Non-Plr Test Accuracy
pred = trainer.predict(tokenized_non_plr_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test Accuacy:", output)
test_results["Non-PLR Test Accuracy"] = output


### UK SmPC Test Accuracy
pred = trainer.predict(tokenized_uk_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test Accuracy:", output)
test_results["UK SmPC Test Accuracy"] = output


# PLR TEST - LABEL 0 - Indications
plr_test_0 = plr_test[plr_test["label"] == 0]
plr_test_data_0 = Dataset.from_pandas(plr_test_0)
tokenized_plr_test_dataset_0 = plr_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_plr_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test: Label 0 Accuracy:", output)
test_results["PLR Test: Label 0 Accuracy"] = output


# PLR TEST - LABEL 1 - Warnings
plr_test_1 = plr_test[plr_test["label"] == 1]
plr_test_data_1 = Dataset.from_pandas(plr_test_1)
tokenized_plr_test_dataset_1 = plr_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_plr_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test: Label 1 Accuracy:", output)
test_results["PLR Test: Label 1 Accuracy"] = output


# Non-PLR TEST - LABEL 0 - Indications
non_plr_test_0 = non_plr_test[non_plr_test["label"] == 0]
non_plr_test_data_0 = Dataset.from_pandas(non_plr_test_0)
tokenized_non_plr_test_dataset_0 = non_plr_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_non_plr_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test: Label 0 Accuracy:", output)
test_results["Non-PLR Test: Label 0 Accuracy"] = output


# Non-PLR TEST - LABEL 1 - Warnings
non_plr_test_1 = non_plr_test[non_plr_test["label"] == 1]
non_plr_test_data_1 = Dataset.from_pandas(non_plr_test_1)
tokenized_non_plr_test_dataset_1 = non_plr_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_non_plr_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test: Label 1 Accuracy:", output)
test_results["Non-PLR Test: Label 1 Accuracy"] = output


# UK SmPC TEST - LABEL 0 - Indications
uk_test_0 = uk_test[uk_test["label"] == 0]
uk_test_data_0 = Dataset.from_pandas(uk_test_0)
tokenized_uk_test_dataset_0 = uk_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_uk_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test: Label 0 Accuracy:", output)
test_results["UK SmPC Test: Label 0 Accuracy"] = output


# UK SmPC TEST - LABEL 1 - Warnings
uk_test_1 = uk_test[uk_test["label"] == 1]
uk_test_data_1 = Dataset.from_pandas(uk_test_1)
tokenized_uk_test_dataset_1 = uk_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_uk_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test: Label 1 Accuracy:", output)
test_results["UK SmPC Test: Label 1 Accuracy"] = output


file = open("./results/binary/eval_logs/elog_distilbert_b.txt", "w")
file.write(repr(test_results))
file.close



################################### RoBERTa



### Load Data
plr_train = pd.read_csv("./data/binary/plr_train_b.csv")
plr_train = plr_train.iloc[: , 1:]
plr_test = pd.read_csv("./data/binary/plr_test_b.csv")
plr_test = plr_test.iloc[: , 1:]
non_plr_test = pd.read_csv("./data/binary/non_plr_test_b.csv")
non_plr_test = non_plr_test.iloc[: , 1:]
uk_test = pd.read_csv("./data/binary/smpc_test_b.csv")
uk_test = uk_test.iloc[: , 1:]



### Tokenize Data
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

train_data = Dataset.from_pandas(plr_train)
train_data = train_data.train_test_split(test_size=0.20)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = train_data.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

print(train_dataset)
print(test_dataset)

plr_test_data = Dataset.from_pandas(plr_test)
tokenized_plr_test_dataset = plr_test_data.map(tokenize_function, batched=True)

non_plr_test_data = Dataset.from_pandas(non_plr_test)
tokenized_non_plr_test_dataset = non_plr_test_data.map(tokenize_function, batched=True)

uk_test_data = Dataset.from_pandas(uk_test)
tokenized_uk_test_dataset = uk_test_data.map(tokenize_function, batched=True)



### Model  (HuggingFace Trainer)
label_num = len(set(plr_train["label"]))

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=label_num)

training_args = TrainingArguments(output_dir="test_trainer", logging_strategy="epoch", save_strategy="epoch", evaluation_strategy="epoch", load_best_model_at_end=True, num_train_epochs=10, per_device_train_batch_size=32, per_device_eval_batch_size=32)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./results/binary/models/roberta_b/')



### Graph Loss
history = trainer.state.log_history

file = open("./results/binary/train_logs/tlog_roberta_b.txt", "w")
file.write(repr(history))
file.close

history = str(history)
df = pd.DataFrame(eval(history))

loss_df = df[["epoch", "loss"]]
loss_df = loss_df.dropna()
print(loss_df)

eval_loss = df[["epoch", "eval_loss"]]
eval_loss = eval_loss.dropna()
print(eval_loss)

eval_accuracy = df[["epoch", "eval_accuracy"]]
eval_accuracy = eval_accuracy.dropna()
print(eval_accuracy)

x1 = loss_df.iloc[:, 0].tolist()
y1 = loss_df.iloc[:, 1].tolist()

x2 = eval_loss.iloc[:, 0].tolist()
y2 = eval_loss.iloc[:, 1].tolist()

x3 = eval_accuracy.iloc[:, 0].tolist()
y3 = eval_accuracy.iloc[:, 1].tolist()

plt.plot(x1, y1, label = "Loss")
plt.plot(x2, y2, label = "Eval Loss")
plt.plot(x3, y3, label = "Eval Accuracy")
plt.title("Loss & Accuracy over 10 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss & Accuracy")
plt.legend()
plt.show()
plt.savefig('./results/binary/graphs/graph_roberta_b.png')
plt.clf()


### Evaluate Model
test_results = {}


### Plr Test Accuracy
pred = trainer.predict(tokenized_plr_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test Accuracy:", output)
test_results["PLR Test Accuracy"] = output


### Non-Plr Test Accuracy
pred = trainer.predict(tokenized_non_plr_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test Accuacy:", output)
test_results["Non-PLR Test Accuracy"] = output


### UK SmPC Test Accuracy
pred = trainer.predict(tokenized_uk_test_dataset)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test Accuracy:", output)
test_results["UK SmPC Test Accuracy"] = output


# PLR TEST - LABEL 0 - Indications
plr_test_0 = plr_test[plr_test["label"] == 0]
plr_test_data_0 = Dataset.from_pandas(plr_test_0)
tokenized_plr_test_dataset_0 = plr_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_plr_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test: Label 0 Accuracy:", output)
test_results["PLR Test: Label 0 Accuracy"] = output


# PLR TEST - LABEL 1 - Warnings
plr_test_1 = plr_test[plr_test["label"] == 1]
plr_test_data_1 = Dataset.from_pandas(plr_test_1)
tokenized_plr_test_dataset_1 = plr_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_plr_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("PLR Test: Label 1 Accuracy:", output)
test_results["PLR Test: Label 1 Accuracy"] = output


# Non-PLR TEST - LABEL 0 - Indications
non_plr_test_0 = non_plr_test[non_plr_test["label"] == 0]
non_plr_test_data_0 = Dataset.from_pandas(non_plr_test_0)
tokenized_non_plr_test_dataset_0 = non_plr_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_non_plr_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test: Label 0 Accuracy:", output)
test_results["Non-PLR Test: Label 0 Accuracy"] = output


# Non-PLR TEST - LABEL 1 - Warnings
non_plr_test_1 = non_plr_test[non_plr_test["label"] == 1]
non_plr_test_data_1 = Dataset.from_pandas(non_plr_test_1)
tokenized_non_plr_test_dataset_1 = non_plr_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_non_plr_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("Non-PLR Test: Label 1 Accuracy:", output)
test_results["Non-PLR Test: Label 1 Accuracy"] = output


# UK SmPC TEST - LABEL 0 - Indications
uk_test_0 = uk_test[uk_test["label"] == 0]
uk_test_data_0 = Dataset.from_pandas(uk_test_0)
tokenized_uk_test_dataset_0 = uk_test_data_0.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_uk_test_dataset_0)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test: Label 0 Accuracy:", output)
test_results["UK SmPC Test: Label 0 Accuracy"] = output


# UK SmPC TEST - LABEL 1 - Warnings
uk_test_1 = uk_test[uk_test["label"] == 1]
uk_test_data_1 = Dataset.from_pandas(uk_test_1)
tokenized_uk_test_dataset_1 = uk_test_data_1.map(tokenize_function, batched=True)

pred = trainer.predict(tokenized_uk_test_dataset_1)
preds = np.argmax(pred.predictions, axis=-1)
metric = load_metric("accuracy")
output = metric.compute(predictions=preds, references=pred.label_ids)
print("UK SmPC Test: Label 1 Accuracy:", output)
test_results["UK SmPC Test: Label 1 Accuracy"] = output


file = open("./results/binary/eval_logs/elog_roberta_b.txt", "w")
file.write(repr(test_results))
file.close