import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


################################################################################### MULTI

results = {}

############################################## TRAINING

plr_train = pd.read_csv('./data/multi/plr_train_m.csv')

plr_train['text'].dropna(inplace=True)
plr_train['text'] = plr_train['text'].astype(str)
plr_train['text'] = [entry.lower() for entry in plr_train['text']]
plr_train['text'] = [word_tokenize(entry) for entry in plr_train['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(plr_train['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    plr_train.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(plr_train['text_final'],plr_train['label'],test_size=0.2)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=100000)
Tfidf_vect.fit(plr_train['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

clf = svm.SVC()
clf.fit(Train_X_Tfidf, Train_Y)

pred = clf.predict(Test_X_Tfidf)

print("PLR Training Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
results["PLR Training Accuracy Score"] = accuracy_score(pred, Test_Y)*100



############################################### PLR TEST

plr_test = pd.read_csv('./data/multi/plr_test_m.csv')

plr_test['text'].dropna(inplace=True)
plr_test['text'] = plr_test['text'].astype(str)
plr_test['text'] = [entry.lower() for entry in plr_test['text']]
plr_test['text'] = [word_tokenize(entry) for entry in plr_test['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(plr_test['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    plr_test.loc[index,'text_final'] = str(Final_words)

Test_X = plr_test['text_final']
Test_Y = plr_test['label']

Test_X_Tfidf = Tfidf_vect.transform(Test_X)

pred = clf.predict(Test_X_Tfidf)

print("PLR Testing Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
results["PLR Testing Accuracy Score"] = accuracy_score(pred, Test_Y)*100



############################################### Non-PLR TEST

non_plr_test = pd.read_csv('./data/multi/non_plr_test_m.csv')

non_plr_test['text'].dropna(inplace=True)
non_plr_test['text'] = non_plr_test['text'].astype(str)
non_plr_test['text'] = [entry.lower() for entry in non_plr_test['text']]
non_plr_test['text'] = [word_tokenize(entry) for entry in non_plr_test['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(non_plr_test['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    non_plr_test.loc[index,'text_final'] = str(Final_words)

Test_X = non_plr_test['text_final']
Test_Y = non_plr_test['label']

Test_X_Tfidf = Tfidf_vect.transform(Test_X)

pred = clf.predict(Test_X_Tfidf)

print("Non-PLR Testing Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
results["Non-PLR Testing Accuracy Score"] = accuracy_score(pred, Test_Y)*100



############################################### UK TEST

uk_test = pd.read_csv('./data/multi/smpc_test_m.csv')

uk_test['text'].dropna(inplace=True)
uk_test['text'] = uk_test['text'].astype(str)
uk_test['text'] = [entry.lower() for entry in uk_test['text']]
uk_test['text'] = [word_tokenize(entry) for entry in uk_test['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(uk_test['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    uk_test.loc[index,'text_final'] = str(Final_words)

Test_X = uk_test['text_final']
Test_Y = uk_test['label']

Test_X_Tfidf = Tfidf_vect.transform(Test_X)

pred = clf.predict(Test_X_Tfidf)

print("UK SmPC Testing Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
results["UK SmPC Testing Accuracy Score"] = accuracy_score(pred, Test_Y)*100

print(results)
file = open("./svm/results_m.txt", "w")
file.write(repr(results))
file.close




########################################################################################## BINARY

results = {}

############################################## TRAINING

plr_train = pd.read_csv('./data/binary/plr_train_b.csv')

plr_train['text'].dropna(inplace=True)
plr_train['text'] = plr_train['text'].astype(str)
plr_train['text'] = [entry.lower() for entry in plr_train['text']]
plr_train['text'] = [word_tokenize(entry) for entry in plr_train['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(plr_train['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    plr_train.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(plr_train['text_final'],plr_train['label'],test_size=0.2)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=100000)
Tfidf_vect.fit(plr_train['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

clf = svm.SVC()
clf.fit(Train_X_Tfidf, Train_Y)

pred = clf.predict(Test_X_Tfidf)

print("PLR Training Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
results["PLR Training Accuracy Score"] = accuracy_score(pred, Test_Y)*100



############################################### PLR TEST

plr_test = pd.read_csv('./data/binary/plr_test_b.csv')

plr_test['text'].dropna(inplace=True)
plr_test['text'] = plr_test['text'].astype(str)
plr_test['text'] = [entry.lower() for entry in plr_test['text']]
plr_test['text'] = [word_tokenize(entry) for entry in plr_test['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(plr_test['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    plr_test.loc[index,'text_final'] = str(Final_words)

Test_X = plr_test['text_final']
Test_Y = plr_test['label']

Test_X_Tfidf = Tfidf_vect.transform(Test_X)

pred = clf.predict(Test_X_Tfidf)

print("PLR Testing Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
results["PLR Testing Accuracy Score"] = accuracy_score(pred, Test_Y)*100



############################################### Non-PLR TEST

non_plr_test = pd.read_csv('./data/binary/non_plr_test_b.csv')

non_plr_test['text'].dropna(inplace=True)
non_plr_test['text'] = non_plr_test['text'].astype(str)
non_plr_test['text'] = [entry.lower() for entry in non_plr_test['text']]
non_plr_test['text'] = [word_tokenize(entry) for entry in non_plr_test['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(non_plr_test['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    non_plr_test.loc[index,'text_final'] = str(Final_words)

Test_X = non_plr_test['text_final']
Test_Y = non_plr_test['label']

Test_X_Tfidf = Tfidf_vect.transform(Test_X)

pred = clf.predict(Test_X_Tfidf)

print("Non-PLR Testing Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
results["Non-PLR Testing Accuracy Score"] = accuracy_score(pred, Test_Y)*100



############################################### UK TEST

uk_test = pd.read_csv('./data/binary/smpc_test_b.csv')

uk_test['text'].dropna(inplace=True)
uk_test['text'] = uk_test['text'].astype(str)
uk_test['text'] = [entry.lower() for entry in uk_test['text']]
uk_test['text'] = [word_tokenize(entry) for entry in uk_test['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(uk_test['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    uk_test.loc[index,'text_final'] = str(Final_words)

Test_X = uk_test['text_final']
Test_Y = uk_test['label']

Test_X_Tfidf = Tfidf_vect.transform(Test_X)

pred = clf.predict(Test_X_Tfidf)

print("UK SmPC Testing Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
results["UK SmPC Testing Accuracy Score"] = accuracy_score(pred, Test_Y)*100

print(results)

file = open("./svm/results_b.txt", "w")
file.write(repr(results))
file.close