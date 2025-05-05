# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from textblob import Word, TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the IMDB dataset
df = pd.read_csv("IMDB Dataset.csv")  # Ensure this file is in your working directory

# Use only the review column
df = df[['review']]

# Basic text preprocessing
sw = nltk.corpus.stopwords.words('english')
df['review'] = df['review'].apply(lambda x: " ".join(word.lower() for word in x.split()))
df['review'] = df['review'].apply(lambda x: " ".join(word for word in x.split() if word not in sw))
df['review'] = df['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Tokenization (optional, for understanding)
df['tokens'] = df['review'].apply(lambda x: nltk.word_tokenize(x))

# Compute sentiment polarity using TextBlob
df['polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Assign sentiment based on polarity
df['Sentiment'] = np.where(df['polarity'] >= 0, 'Positive', 'Negative')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Evaluate
accuracy = model.score(X_test_vect, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, model.predict(X_test_vect)))

# Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test_vect))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# Test the model on a new review
new_doc = ["The college is very good"]
new_doc_vec = vectorizer.transform(new_doc)
prediction = model.predict(new_doc_vec)
print(f"Predicted Sentiment: {prediction[0]}")
