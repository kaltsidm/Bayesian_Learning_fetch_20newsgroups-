#Don't pay attention to the comments lines in this script. I just keep some of them for future using

from sklearn import datasets,model_selection,metrics
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import scipy.sparse
import seaborn as sns
import matplotlib.pyplot as plt


#laod data
textdata=datasets.fetch_20newsgroups()

#select X and y
X=textdata.data
y=textdata.target


#remove all the symbols
import string
string.punctuation

#defining the function to remove punctuation and numbers
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
def remove_numbers(text):
    numberfree="".join([i for i in text if i not in "0123456789"])
    return numberfree

for i in range(len(X)):
    X[i]=remove_punctuation(X[i])
for i in range(len(X)):
    X[i]=remove_numbers(X[i])

#remove all the \n
for i in range(len(X)):
    X[i]=X[i].replace("\n"," ")

model1=TfidfVectorizer()
XX=model1.fit_transform(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(XX,y,test_size=0.33 ,random_state = 0)

#Choosing a=0.05 for the first training
clf=MultinomialNB(alpha=0.05)
model2=clf.fit(X_train,y_train)
y_predicted=clf.predict(X_test)

acc=metrics.accuracy_score(y_test, y_predicted)
f1=metrics.f1_score(y_test,y_predicted,average="macro")
rec=metrics.recall_score(y_test, y_predicted, average="macro")
pr=metrics.precision_score(y_test, y_predicted, average="macro")

#Heat map
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,y_predicted)
sns.heatmap(mat,xticklabels=textdata.target_names,square=True,annot=True,fmt='d',linecolor='white',cmap="Oranges_r",linewidth=.5,yticklabels=textdata.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title("MultinomialNB-Confusion Matrix(a=0.10) "+"[Accuracy:"+str(round(acc, 2))+", F1:"+str(round(f1, 2))+", Recall:"+str(round(rec, 2))+", Precission:"+str(round(pr, 2))+"]")

##make the all the letters small
#for i in range(len(X)):
#    X[i]=X[i].lower()
    
##stopwords
##importing nlp library
#import nltk
#Stop words present in the library
#stopwords = nltk.corpus.stopwords.words('english')
#def remove_stopwords(text):
#    output= [i for i in text if i not in stopwords]
#    return output
#for i in range(len(X)):
#    X[i]=remove_stopwords(X[i])
    
##stemming-stemming doesn't decreases the number of words in the list
##importing the Stemming function from nltk library
#from nltk.stem.porter import PorterStemmer
##defining the object for stemming
#porter_stemmer = PorterStemmer()
##defining a function for stemming
#def stemming(text):
#    stem_text = [porter_stemmer.stem(word) for word in text]
#    return stem_text
#for i in range(len(X)):
#    X[i]=stemming(X[i])

##our data are ready for transformation to numeric
#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer()
#P = vectorizer.fit_transform(X)
#vectorizer.get_feature_names_out()
#P.shape




