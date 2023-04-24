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
import shap
import json
import ast



### Load Data
plr_test = pd.read_csv("./data/multi/plr_test_m.csv")
plr_test = plr_test.iloc[: , 1:]
non_plr_test = pd.read_csv("./data/multi/non_plr_test_m.csv")
non_plr_test = non_plr_test.iloc[: , 1:]
uk_test = pd.read_csv("./data/multi/smpc_test_m.csv")
uk_test = uk_test.iloc[: , 1:]


plr_test0 = plr_test[plr_test["label"] == 0]
plr_test0 = plr_test0.head(1000)
plr_test1 = plr_test[plr_test["label"] == 1]
plr_test1 = plr_test1.head(1000)
plr_test2 = plr_test[plr_test["label"] == 2]
plr_test2 = plr_test2.head(1000)
plr_test3 = plr_test[plr_test["label"] == 3]
plr_test3 = plr_test3.head(1000)

non_plr_test0 = non_plr_test[non_plr_test["label"] == 0]
non_plr_test0 = non_plr_test0.head(1000)
non_plr_test1 = non_plr_test[non_plr_test["label"] == 1]
non_plr_test1 = non_plr_test1.head(1000)
non_plr_test2 = non_plr_test[non_plr_test["label"] == 2]
non_plr_test2 = non_plr_test2.head(1000)
non_plr_test3 = non_plr_test[non_plr_test["label"] == 3]
non_plr_test3 = non_plr_test3.head(1000)

uk_test0 = uk_test[uk_test["label"] == 0]
uk_test0 = uk_test0.head(1000)
uk_test1 = uk_test[uk_test["label"] == 1]
uk_test1 = uk_test1.head(1000)
uk_test2 = uk_test[uk_test["label"] == 2]
uk_test2 = uk_test2.head(1000)
uk_test3 = uk_test[uk_test["label"] == 3]
uk_test3 = uk_test3.head(1000)



### Load Model
model = AutoModelForSequenceClassification.from_pretrained("./results/multi/models/bert_m/", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenizer_kwargs = {'padding':True, 'truncation':True, 'max_length':512}



### Calculate Mean Shap Values
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, **tokenizer_kwargs)

explainer = shap.Explainer(pipe)


shap_values_p0 = explainer(plr_test0["text"])
shap.plots.bar(shap_values_p0[:,:,"LABEL_0"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/PLR_L0_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_p1 = explainer(plr_test1["text"])
shap.plots.bar(shap_values_p1[:,:,"LABEL_1"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/PLR_L1_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_p2 = explainer(plr_test2["text"])
shap.plots.bar(shap_values_p2[:,:,"LABEL_2"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/PLR_L2_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_p3 = explainer(plr_test3["text"])
shap.plots.bar(shap_values_p3[:,:,"LABEL_3"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/PLR_L3_SHAP.png", bbox_inches='tight')
plt.clf()



shap_values_np0 = explainer(non_plr_test0["text"])
shap.plots.bar(shap_values_np0[:,:,"LABEL_0"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/NON_PLR_L0_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_np1 = explainer(non_plr_test1["text"])
shap.plots.bar(shap_values_np1[:,:,"LABEL_1"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/NON_PLR_L1_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_np2 = explainer(non_plr_test2["text"])
shap.plots.bar(shap_values_np2[:,:,"LABEL_2"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/NON_PLR_L2_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_np3 = explainer(non_plr_test3["text"])
shap.plots.bar(shap_values_np3[:,:,"LABEL_3"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/NON_PLR_L3_SHAP.png", bbox_inches='tight')
plt.clf()



shap_values_u0 = explainer(uk_test0["text"])
shap.plots.bar(shap_values_u0[:,:,"LABEL_0"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/UK_L0_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_u1 = explainer(uk_test1["text"])
shap.plots.bar(shap_values_u1[:,:,"LABEL_1"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/UK_L1_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_u2 = explainer(uk_test2["text"])
shap.plots.bar(shap_values_u2[:,:,"LABEL_2"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/UK_L2_SHAP.png", bbox_inches='tight')
plt.clf()


shap_values_u3 = explainer(uk_test3["text"])
shap.plots.bar(shap_values_u3[:,:,"LABEL_3"].mean(0), max_display=250, show=False)
plt.tight_layout()
plt.savefig("./shap/UK_L3_SHAP.png", bbox_inches='tight')
plt.clf()