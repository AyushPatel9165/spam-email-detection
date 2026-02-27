import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

 
data = pd.read_csv("spam.csv")

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

 
data['label'] = data['label'].map({'ham':0, 'spam':1})
 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

y = data['label']

 
model = MultinomialNB()
model.fit(X, y)

 
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Spam model saved")