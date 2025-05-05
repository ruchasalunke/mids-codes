import pandas as pd
import numpy as np
from textblob import Word,TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
df=pd.read_csv('Twitter.csv')
df.head()
df=df[['post_text']]
df.head()
sw=stopwords.words('english')
df['post_text']= df['post_text'].apply(lambda x:" ".join(word.lower() for word in x.split()))
df['post_text']= df['post_text'].apply(lambda x:" ".join(word for word in x.split() if word not in sw)) 
df['post_text']= df['post_text'].apply(lambda x:" ".join([Word(word).lemmatize() for word in x.split()]))
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
df["tokens"] = df["post_text"].apply(lambda x: word_tokenize(x))
df['polarity']=df['post_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['Sentiment']=np.where(df['polarity']>= 0,'Positive','Negative')
X_train,X_test,y_train,y_test = train_test_split(df['post_text'],df['Sentiment'],test_size=0.2,random_state=42)   
vectorizer= CountVectorizer()
X_train_vect= vectorizer.fit_transform(X_train)
X_test_vect= vectorizer.transform(X_test)
knn= KNeighborsClassifier()
knn.fit(X_train_vect,y_train)
accuracy=knn.score(X_test_vect,y_test)
print("Accuracy:",round(accuracy*100,2),'%')
print(classification_report(y_test,knn.predict(X_test_vect)))
# Confusion Matrix
cm = confusion_matrix(y_test, knn.predict(X_test_vect))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
