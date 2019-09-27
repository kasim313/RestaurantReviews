# Author Mr. Muhammad Kasim Ali
# Created Date 28/09/2019
# Source Geeks for geeks
  
# Importing packages 
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
    for i in range(0,len(data)):
        reviews = data['Review'][i].lower() #Converting it into lower case   
        reviews = re.sub(r"\W+"," ",reviews) #Removeing special charector
        reviews = re.sub(r"\d+"," ",reviews) #Removeing digits
        reviews = word_tokenize(reviews)#Converting into word tokens
        ps = PorterStemmer()#Creating instance of port stemming class
        reviews = [ps.stem(w) for w in reviews if w not in set(stopwords.words('english'))] #Stemming or removeing stor words 
        reviews = " ".join(reviews)# Rejoin words into string  
            
              

textCleansing(dataset)