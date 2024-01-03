import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import string
import pickle
data_fake=pd.read_csv('Fake.csv')
data_true=pd.read_csv('True.csv')
data_fake.head()
data_true.head()
data_fake["class"]=0
data_true["class"]=1
data_fake.shape,data_true.shape
data_merge=pd.concat([data_fake,data_true],axis=0)
data_merge.head(10)
data_merge.columns
data_merge.info()
data_merge.describe()
sns.countplot(x="class",data=data_merge)
data= data_merge.drop(['title','subject','date'],axis=1)
data.columns
def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\W"," ",text)
    text=re.sub('https://S+|www\.\S','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]' %re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\dw*','',text)
    return text
data['text']=data['text'].apply(wordopt)
x=data['text']
y=data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization=TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
from sklearn.linear_model import LogisticRegression

LR=LogisticRegression()
LR.fit(xv_train ,y_train)
pred_lr=LR.predict(xv_test)
LR.score(xv_test,y_test)
print(classification_report(y_test,pred_lr))
def output_lable(n):
    if n==0:
        return "Fake news"
    elif n==1:
        return "Real news"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    
    return print("\n\nLR prediction: {}".format(output_lable(pred_LR[0])))
news=str(input())
manual_testing(news)
pickle.dump(LR,open('pickle.pkl','wb'))