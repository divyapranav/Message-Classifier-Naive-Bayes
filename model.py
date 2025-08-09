import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('spam.csv', encoding='latin-1')
df.head()

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['message'])
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(x_train, y_train)

print(accuracy_score(y_test, model.predict(x_test)))

print(classification_report(y_test, model.predict(x_test)))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)