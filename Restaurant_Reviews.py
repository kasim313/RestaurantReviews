# Author Mr. Muhammad Kasim Ali
# Created Date 28/09/2019
# Source Geeks for geeks
  
# Importing packages 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd 
import sys 
import os 
import re

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t")   #Importing data sets 
def textCleansing(data):
    courpus = []
    for i in range(0,len(data)):
        reviews = data['Review'][i].lower() #Converting it into lower case   
        reviews = re.sub(r"\W+"," ",reviews) #Removeing special charector
        reviews = re.sub(r"\d+"," ",reviews) #Removeing digits
        reviews = word_tokenize(reviews)#Converting into word tokens
        ps = PorterStemmer()#Creating instance of port stemming class
        reviews = [ps.stem(w) for w in reviews if w not in set(stopwords.words('english'))] #Stemming or removeing stor words 
        reviews = " ".join(reviews)# Rejoin words into string  
        courpus.append(reviews)#Appending into list    
    return courpus

cv = CountVectorizer()#Creating instance of countvectorizer
X = cv.fit_transform(textCleansing(dataset)).toarray() #Input varaible
Y = dataset.iloc[:,1].values #Out put varaible
X_train , X_test ,Y_train, Y_test = train_test_split(X,Y, test_size = 0.25) #Spliting data sets into trains
model = RandomForestClassifier(n_estimators = 501 , criterion= 'entropy')
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
print(cm)