# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:56:11 2019

@author: Amit Srimal
"""

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
ml_data=pd.read_csv("C:/Users/Amit Srimal/Desktop/DET/ML/Data/MultiLabelData/train.csv")    
ml_data.head()    
ml_data.describe()
print("Number of rows in data =",ml_data.shape[0])
print("Number of columns in data =",ml_data.shape[1])    

###STEP 1: EXPLORARTORY ANALYSIS

# Checking for missing values
missing_values_check = ml_data.isnull().sum()
print(missing_values_check) # all zero & hence no NULL value in dataframe.

# calculating % of unlabelled data
unlabelled_ml_data = ml_data[(ml_data['toxic']!=1) & (ml_data['severe_toxic']!=1) & (ml_data['obscene']!=1) & 
                            (ml_data['threat']!=1) & (ml_data['insult']!=1) & (ml_data['identity_hate']!=1)]
#print('Percentage of unlabelled comments is ', len_unlabel/len_ml_data*100)
# around 89% row of ml_data is UNLABELLED.ie having value as ZERO.

## Calculating number of comments under each label
# Comments with no label are considered to be "CLEAN" comments.
# Creating seperate column in dataframe to identify clean comments.
# We use axis=1 to count row-wise and axis=0 to count column wise

rowSums = ml_data.iloc[:,2:].sum(axis=1)
clean_comments_count = (rowSums==0).sum(axis=0)
print("Total number of comments = ",len(ml_data))
print("Number of clean comments = ",clean_comments_count)
print("Number of comments with labels =",(len(ml_data)-clean_comments_count))

# Getting categories
categories = list(ml_data.columns.values)
categories = categories[2:]
print(categories)

##Calculating number of comments in each category
counts = []
for category in categories:
    counts.append((category, ml_data[category].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number of comments'])
print(df_stats)
# plotting the results
sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax= sns.barplot(categories, ml_data.iloc[:,2:].sum().values)

plt.title("Comments in each category", fontsize=24)
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Comment Type ', fontsize=18)

#adding the text labels
rects = ax.patches
labels = ml_data.iloc[:,2:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)

plt.show()

## Calculating number of comments having multiple labels
rowSums = ml_data.iloc[:,2:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[1:]

sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)

plt.title("Comments having multiple labels ")
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Number of labels', fontsize=18)

#adding the text labels
rects = ax.patches
labels = multiLabel_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


##WORDCLOUD representation of most used words in each category of comment
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize=(40,25))

# toxic
subset = ml_data[ml_data.toxic==1]
text = subset.comment_text.values
cloud_toxic = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

#plt.subplot(2, 3, 1)
plt.axis('off')
plt.title("Toxic",fontsize=40)
plt.imshow(cloud_toxic)
plt.show()

###STEP 2: DATA PROCESSING like stemming, removing stop words etc
data = ml_data
data = ml_data.loc[np.random.choice(ml_data.index, size=2000)]
data.shape # 2000 by 8 datafram created.

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
data['comment_text'] = data['comment_text'].str.lower()
data['comment_text'] = data['comment_text'].apply(cleanHtml)
data['comment_text'] = data['comment_text'].apply(cleanPunc)
data['comment_text'] = data['comment_text'].apply(keepAlpha)
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

data['comment_text'] = data['comment_text'].apply(removeStopWords)
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

data['comment_text'] = data['comment_text'].apply(stemming)
data.head()

## Training & test data split now
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)
print(train.shape) ## 111699 by 8
print(test.shape) ## 47872 by 8

# import and instantiate TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=5000,stop_words='english')
vect

# segregating the test, train comment as
train_text = train['comment_text']
train_text.shape # series type (111699L,)
test_text = test['comment_text'] ##train.iloc[:,[1,4]] to get all rows with 1 & 4th columns
test_text.shape # series type as (47872L,)


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

# create submission file
#submission_binary = pd.read_csv('../input/sample_submission.csv')

for category in categories:
    print('... Processing {}'.format(category))
    y = train[category]
    #print(y) 
    #print len(y)
    # train the model using X_dtm & y
    model=logreg.fit(train_text_dtm, train[category])
    # compute the training accuracy
    y_pred_X = model.predict(test_text_dtm)
    #print 'srimal'
    #print len(y_pred_X)
    print(y_pred_X)
    #print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    # compute the predicted probabilities for X_test_dtm
    test_y_prob = model.predict_proba(test_text_dtm)[:,1]
    print test_y_prob
    #print len(test_y_prob)
    #submission_binary[label] = test_y_prob
    #print(submission_binary[label])

#test.iloc[576,1:8]
#test.iloc[21,1:8] 
#test.iloc[36,1:8] 
#test.iloc[575,1:8] 
#test.iloc[330,1:8] 

def predict_labels(s):
    #s=[s]
    s=pd.Series(s)
    #print(s)
    #vectorizer.fit(s) As we have already used the FIT method ONCE.
    s_labels = vect.transform(s)
    #print(s_labels)
    for category in categories:
        # lets resuse the model value again 
        model=logreg.fit(train_text_dtm, train[category])
        prediction=model.predict(s_labels)# earlier used predict(s2)
        test_y_prob = model.predict_proba(s_labels)[:,1]
        print category, 'label probability is around',test_y_prob[0]*100 
        #print test_y_prob[0]
        #print prediction[0]
        #print len(test_y_prob)
        #print prediction
        #print category
        #print('fetching {} label value as'.format(category),prediction[0])
        #print category,'label is',test_y_prob[0]       
        #return prediction
               
## checking catgory LABELS for some random texts
predict_labels('fuck stay fuck away me') # o/p is 101010
predict_labels('hey freepsban mother know get fuck time now') # o/p is 101010
predict_labels('nigga black fuck dick slut killer murder fuck stay fuck away me')
predict_labels('inform band would allow finish write detail retro gecko line us need break comput perhap too')
predict_labels('COCKSUCKER BEFORE YOU PISS AROUND my dick and kill you pieces shit')
predict_labels('worthless piec shit')
predict_labels('die waht day fuck you')
predict_labels('bullshit nazi you die')
predict_labels('sigh appar im surround idiot like chubbo')
predict_labels('hello thanks for the welcome brother')
predict_labels('impressive design brilliant work great')
predict_labels('ya extra ur fuckin extra tryna bryte urself delet revert edit revert ur face cuz seem like u need revertin ugli shit bitch zoe stay away edit u littl hoe thank consider')
predict_labels('tool go fuck get laid')
predict_labels('retard littl bitch suck ball go hell')
"""
## TF-IDF calcluation now
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
print(vectorizer)
vectorizer.fit(train_text)
#vectorizer.fit(test_text)

train_x = vectorizer.transform(train_text)
print(train_x)
train_x.shape # 111699 rows & 3581291 columns as per tf-idf parameters definition
train_y = train.drop(labels = ['id','comment_text'], axis=1)
train_y.shape # 111699 by 6

test_x = vectorizer.transform(test_text)
print(test_x)
test_x.shape # 47872 rows & 3581291 colums as per tf-idf parameters definition
test_y = test.drop(labels = ['id','comment_text'], axis=1)
test_y.shape # 47872 by 6 

## STEP 3: MULTI LABEL CLASSIFICATION
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),])

for category in categories:
    print('**Processing {} comments...**'.format(category))
    
    # Training logistic regression model on train data
    LogReg_pipeline.fit(train_x, train[category])
    #print(LogReg_pipeline.fit(train_x, train[category]))
    
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(test_x)
    print(prediction) # predicted value for test data
    #print(test[category]) # actual value of test data
    #print(test[category]- prediction)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print("\n") 


## FUNCTION to get indivisual caomment catgeory listed as
'''def predict_labels(s, model=LogReg_pipeline):
    s=[s]
    print(s)
    vectorizer.fit(s)
    s = vectorizer.transform(s)
    print(s)
    for category in categories:
        prediction=model.predict(s) # earlier used predict(s2)
        #print prediction[0]
        #print('fetching {} label value as'.format(category),prediction[0])
        print'Label value for',category,'is',prediction[0]'''
        
        
########   ########
#s= raw_input("Enter comment: ")              
def predict_labels(s):
    s=[s]
    print(s)
    #vectorizer.fit(s)
    s_labels = vectorizer.transform(s)
    print(s_labels)
    for category in categories:
        prediction=LogReg_pipeline.predict(s_labels) # earlier used predict(s2)
        #return prediction
        #print prediction[0]
        #print category
        #print('fetching {} label value as'.format(category),prediction[0])
        print category,'label is',prediction[0]       
        #return prediction
               
## checking catgory LABELS for some random texts
print(predict_labels('fuck faggot'))
print(predict_labels('jew jew dont pollut obama articl snot'))
print(predict_labels('Ask Sityush to clean up his behavior than issue me nonsensical warnings...All of my edits are good.  Cunts like you who revert good edits because youre too stupid to understand how to write well , and then revert other edits just because youve decided to bear a playground grudge, are the problem.  Maybe one day youll realise the damage you did to a noble project.  201.215.187.159'))
print(predict_labels('COCKSUCKER BEFORE YOU PISS around ON MY WORK and here no where idiot shit fucking idiot racist'))
print(predict_labels('insult musikanim your fuck shit reason im insult block fuck wikipedia'))
print(predict_labels('realli rebel guy rebelguy time get oral prostitut'))
print(predict_labels('world give mate sad individu kensington even fuck sadder bet american well owrld favour get lostt'))
print(predict_labels('shut liar arent abid wikipedia polici content must base verifi sourc wikipedia neutral point view wikipedia origin research wikipedia cite sourc wikipedia verifi wikipedia reliabl sourc outlin detail error demand cite verifi reliabl sourc claim csssclll'))
print(predict_labels('excus fuck think delet page troll movi make soon feel compel go around destroy peopl hard work creativ page realli hurt anybodi delet realli necessari would anybodi entir fuck world realli give shit page fuck nazi cunt would fuck somebodi work like fuck sandym'))




for category in categories:
    print category
    print ml_data.iloc[6,category]
    print ml_data.iloc[6,:]
    
    
    '''

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
