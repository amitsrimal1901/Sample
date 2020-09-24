# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:43:44 2019

@author: A Srimal
"""
 

#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
# reading csv file to df abd then converting to a matrix
train_data= pd.read_csv('C:/Users/Amit Srimal/Desktop/DET/ML/Data/AshuData/train_data.csv')
train_label= pd.read_csv('C:/Users/Amit Srimal/Desktop/DET/ML/Data/AshuData/train_label.csv')
test_data= pd.read_csv('C:/Users/Amit Srimal/Desktop/DET/ML/Data/AshuData/test_data.csv')

#plotting the labels count
sns.countplot(x='label', data=train_label)
# creating dataframe using merge 
df= train_data.merge(train_label,on='id',how='left')
# creating PIVOT table
dfp=pd.pivot_table(df,values='id', index=['text'], columns=['label'],aggfunc=np.sum,fill_value=0)
print dfp
# conveting non zero values as 1 to indicate Boolean YES,NO with 0/1
pre_data=dfp.astype(bool).astype(int)
print pre_data # 41569 rows but 15 columns, but text is hidden as INDEX
#identigying labels of data 
labels = list(pre_data.columns.values)
labels = labels[1:] # removing 'text' column as its not a label tag. 1 to end index is selcted
print(labels)

#resetting of text from Index
pre_data.reset_index(level=0, inplace=True) # now 16 columns. text, and 15 labels
'''
Check for text having  ""Constructor" Prelude door locks feature modern, curvy design combined with excellent craftsmanship and stability of operation. Can be used on right-handed as well as left-handed doors. Each set comes with 3 brass keys."Constructor" brandSide: reversibleFits for door thickness: 1.38 in. - 1.75 in.Backset: 2.38 in. - 2.75 in.Comes with 3 brass keys"
Result id is 161324 =, having labels as "Finish, Hardware included"
When we see in df33 first row, its label is marked in 1/0 pattern.
'''
## now processing part
# lets take only some part of data from 41569 rows. say we take 3000 rows randomly
# Else intensive memory usage will happen
data = pre_data.loc[np.random.choice(pre_data.index, size=3000)]

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Cleaning the data
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

# displaying cleaned data here
data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(cleanHtml)
data['text'] = data['text'].apply(cleanPunc)
data['text'] = data['text'].apply(keepAlpha)
data.head()

## Now removing STOP WORDS
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data['text'] = data['text'].apply(removeStopWords)
data.head()

# Stemming part now
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

data['text'] = data['text'].apply(stemming)
data.head()


##DATA modelling part now
## Training & test data split now
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)
print(train.shape) 
print(test.shape) 

# import and instantiate TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=5000,stop_words='english')


# segregating the test, train comment as
train_text = train['text']
train_text.shape
test_text = test['text'] ##train.iloc[:,[1,4]] to get all rows with 1 & 4th columns
test_text.shape 


# learn the vocabulary in the training data, then use it to create a document-term matrix
train_text_dtm = vect.fit_transform(train_text)
# examine the document-term matrix created from X_train
print(train_text_dtm)

# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_text_dtm = vect.transform(test_text)
# examine the document-term matrix from X_test
print(test_text_dtm)
'''
Solving a multi-label classification problem:
By transforming the problem into separate single-class classifier problems. 
This is known as 'problem transformation
We have three methods in here:
    1.Binary Relevance
    2.Classifiers Chain
    3.Label powerset
'''
## Binary Relevance - build a multi-label classifier using Logistic Regression
# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=12.0)

# create submission dataframe from test
submission_df= test # copy of test data set

for lbl in labels:
    print('... Processing {}'.format(lbl))
    y = train[lbl]
    # train the model using X_dtm & y
    model=logreg.fit(train_text_dtm, train[lbl])
    # compute the training accuracy
    y_pred_X = model.predict(test_text_dtm)
    print(y_pred_X)
    print len(y_pred_X)
    test_y_prob = model.predict_proba(test_text_dtm)[:,1]
    print('Test accuracy is {}'.format(accuracy_score(test[lbl], y_pred_X)))
    #print predicted label in the submission header
    submission_df[lbl] = y_pred_X

#writing csv outut file
submission_df.to_csv("C:/Users/Amit Srimal/Desktop/submission_file.csv", index=False)

### function to check for any random text comment

def predict_labels(s):
    s=pd.Series(s)
    #print(s)
    #vectorizer.fit(s) As we have already used the FIT method ONCE.
    s_labels = vect.transform(s)
    #print(s_labels)
    for lbl in labels:
        # lets resuse the model value again 
        model=logreg.fit(train_text_dtm, train[lbl])
        prediction=model.predict(s_labels)# earlier used predict(s2)
        test_y_prob = model.predict_proba(s_labels)[:,1]
        print lbl, 'label probability is around',test_y_prob[0]*100 
        
               
## checking catgory LABELS for some random texts
predict_labels('"Constructor" Prelude door locks feature modern, curvy design combined with excellent craftsmanship and stability of operation. Can be used on right-handed as well as left-handed doors. Each set comes with 3 brass keys."Constructor" brandSide: reversibleFits for door thickness: 1.38 in. - 1.75 in.Backset: 2.38 in. - 2.75 in.Comes with 3 brass keys') 
predict_labels('"The Hangman Self Leveling Flush mount Hanger is a single piece hanger that attaches to the back of your canvas and hangs on the provided Bear Claw Hanger. The inset created allows the canvas to hang flush against the wall. Ideal for unframed canvas art.Also works on wooden framesQuick and easy, hangs flush on wallsSold as an individual kitNow your canvas art, including gale and gallery wraps will look even more beautiful, enhancing the rest of your home wall decor')
















  
