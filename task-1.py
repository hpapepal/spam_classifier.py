# ==========================================
# SPAM EMAIL CLASSIFIER USING MACHINE LEARNING
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ==========================================
# LOAD DATASET
# ==========================================

# Make sure spam.csv is in same folder

df = pd.read_csv("spam.csv", encoding='latin-1')

print("Dataset Loaded Successfully\n")


# ==========================================
# SELECT REQUIRED COLUMNS
# ==========================================

df = df[['v1', 'v2']]

df.columns = ['label', 'message']

print(df.head())


# ==========================================
# CONVERT LABELS
# ham = 0
# spam = 1
# ==========================================

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})


# ==========================================
# SPLIT DATA
# ==========================================

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# TRAIN MODEL
# ==========================================

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("\nModel Trained Successfully")


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# CUSTOM MESSAGE TEST
# ==========================================

while True:

    user_message = input("\nEnter a Message: ")

    message_tfidf = vectorizer.transform([user_message])

    prediction = model.predict(message_tfidf)

    if prediction[0] == 1:
        print("Prediction: SPAM MESSAGE")
    else:
        print("Prediction: NOT SPAM")

    choice = input("\nDo you want to test another message? (yes/no): ")

    if choice.lower() != 'yes':
        break


print("\nProject Finished")