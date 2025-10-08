import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('train.txt',sep=';',header=None, names=['text','emotion'])
df.head()
df.isna().sum()
# assign all emotion in number form
unique_emotions = df['emotion'].unique()
emotion_numbers = {}
i = 0
for emotion in unique_emotions:
    emotion_numbers[emotion] = i
    i += 1
df['emotion'] = df['emotion'].map(emotion_numbers)
# convert text into lower case
df['text'] = df['text'].apply(lambda x : x.lower())
# remove punctuation
def remove_punc(txt):
  return txt.translate(str.maketrans('','',string.punctuation))
df['text'] = df['text'].apply(remove_punc)
# remove number form the text
def remove_number(txt):
  new = ""
  for i in txt:
    if not i.isdigit():
      new = new + i
  return new
df['text'] = df['text'].apply(remove_number)
# remove emojis
def remove_emojis(txt):
  new = ""
  for i in txt:
    if i.isascii():
      new = new + i
  return new
df['text'] = df['text'].apply(remove_emojis)
# remove stop words
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
def remove_stop(txt):
  words = txt.split()
  cleaned = []
  for word in words:
    if word not in stop_words:
      cleaned.append(word)
  return " ".join(cleaned)
df['text'] = df['text'].apply(remove_stop)
# Train model
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

nb_model_bow = MultinomialNB()
nb_model_bow.fit(X_train_bow, y_train)

y_pred_bow = nb_model_bow.predict(X_test_bow)
print("Accuracy with Bag of Words:", accuracy_score(y_test, y_pred_bow))
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

nb_model_tfidf = MultinomialNB()
nb_model_tfidf.fit(X_train_tfidf, y_train)

y_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)
print("Accuracy with TF-IDF:", accuracy_score(y_test, y_pred_tfidf))
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_bow, y_train)

y_pred_logistic_bow = logistic_model.predict(X_test_bow)
print("Accuracy with Logistic Regression (Bag of Words):", accuracy_score(y_test, y_pred_logistic_bow))









