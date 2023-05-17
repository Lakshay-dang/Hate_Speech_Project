import pandas as pd
import numpy as np
from skilearn.feature_extraction.text import CounterVectorizer
from skilearn.model_selectionc import train_text_split
from skilearn.tree import DecisionTreeClassifer

import re
import nltk
from nltk.uti import pr
stenner = nltk.SnowballStenner("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))

df = pd.read_csv("twitter_data.csv")

print(df.head())   #tail can be used for last 5 

df['labels'] = df['class'].map({0:"Hate Speech Detected", 1:"Offensive language detected" , 3:"Normal text nothing offensive"})
print(df.head())

df = df[['tweet','labels']]
df.head()

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]','', text)
    text = re.sub('https?://\S+|www\.\S+','', text)
    text = re.sub('<.*?>+','', text)
    text = re.sub("'[%]', % re.escape(string.punctuation),'', text")
    text = re.sub('\n','',text)
    text = re.sub('\w\d\w*','', text)
    text = [word for word in text.split('') if word not in stopword]
    text = " ".join(text)
    return text 


df["tweet"] = df["tweet"].apply(clean)
print(df.head())

x = np.array(df["tweet"])
y = np.array(df["labels"])
cv = CounterVectorizer()
x = cv.fit_transform(x)
X_train, X_test, Y_train, Y_test = train_text_split(x,y, test_size= 0.33, random_state= 42)
clf = DecisionTreeClassifer()
clf.fit(X_train, Y_train)


test_data = " I like you"
df = cv.transform([test_data]).toarray()
print(clf.predict(df))
