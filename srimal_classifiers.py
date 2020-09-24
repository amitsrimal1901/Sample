#CLASSIFIFCATION:
""" 
MULTI-CLASS classification means a classification task with more than two classes; 
each label are MUTUALLY exclusive. 
The classification makes the assumption that each sample is assigned to one and only one label.

On the other hand, MULTI-LABEL classification assigns to each sample a set of target labels. 
This can be thought as predicting properties of a data-point that are NOT MUTUALLY exclusive, 
such as Tim Horton are often categorized as both bakery and coffee shop. 
Multi-label text classification has many real world applications such as categorizing businesses on Yelp 
or classifying movies into one or more genre(s).
"""

####### TYPE A: MULTI-CLASS CLASSIFICATION--------------------------------------------------------------------
##More on NAIVE BAYES CLASSIFIFERS @ https://www.youtube.com/watch?v=l3dZ6ZNFjo0
# works on principal of condiitional probability. 
# defines probability of event A, provided event B has occurred. Denoted by p(a|b)
# Maths of Bayes theorem: p(a|b)= p(a)*p(b|a)/p(b)   
    
## Where NB falls;
# Structure as ML-Supervised-Classification-Naive Bayes   
    
## Used for Face Recognition, Weather Prediction, Medical Diagnosis, News Classification etc.
## ADVANTAGES of Naive Bayes classifier:
    #1: Simple & easy implemenation based on Bayes theorem
    #2: Needs less training data
    #3: Handles continous & discrete data effectively.
    #4: Highly scalable with number of predictors & data points.
    #5: fast & hence can be used in real time prediction.
    #6: not sensitive to irrelevant features

## Understanding NB classifier with Demo:
# Problem: Whether person will buy product based on day, discount & free delivery usng NB classifier    
# create LIKELIHOOD table for each input variable against single output(YES/NO). 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # seaborn sits on TOP of pyplot.
from sklearn.datasets import fetch_20newsgroups # this has (train, test) DATA, LABEL & MODEL in all. 6 files
data=fetch_20newsgroups() # loading data set with size 6
# aliter train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#NOTE: data is BUNCH object type; a simple holder object with fields that can be both accessed as python dict keys or object attributes for convenience.
list(data) #o/p column namesa are as:['data', 'filenames', 'target_names', 'target', 'DESCR'] column names
data.target_names # defines the CATEGORIES of dataset which we imported
data.target # returns aray as array([7, 4, 4, ..., 3, 1, 8])
data.target[:10] # gets first 10 integer target for target names
print(data.target_names[data.target[0]]) # o/p is rec.autos
print(data.target_names[data.target[7]]) # o/p is comp.sys.ibm.pc.hardware
print(data.target_names[data.target[16]]) # o/p is comp.graphics

# It is possible to get back the category names for say FIRST 10 as follows:
for t in data.target[:10]:
    print(train.target_names[t])
    
len(data) # o/p is 6 as fetchnew has 6 different files in source.
len(data.filenames) #o/p is 11314
# defining all categories from above target list
categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

# training data on categroeis
train= fetch_20newsgroups(subset='train', categories= categories) # fetching from "train" label of fetchnewgroup
# Testing data on categroeis
test= fetch_20newsgroups(subset='test', categories= categories) # fetching from "test" label of fetchnewgroup
# printing training data main description
print(train.data[5]) # say 5th data news
print(len(train.data)) #o/p is 11314
print(len(test.data)) # 0/p is 7532
# printing test data
print(test.data[5]) # say 5th 

#Let’s print the first lines of the first loaded file:
print("\n".join(train.data[0].split("\n")[:3]))

## FEATURE EXTRACTION
# In order to perform machine learning on text documents, we first need to turn the text content into numerical feature vectors.
"""
Best Apporach is bags of words representation: 
1. Assign a fixed integer id to each word occurring in any document of the training set.
2. For each document #i, count the number of occurrences of each word w and store it in X[i, j] as 
   the value of feature #j where j is the index of word w in the dictionary.
3. The bags of words representation implies that n_features is the number of distinct words in the corpus.

If n_samples == 10000, storing X as a NumPy array of type float32 would require 10000 x 100000 x 4 bytes = 4GB in RAM which is barely manageable on today’s computers.
Fortunately, most values in X will be zeros since for a given document less than a few thousand distinct words will be used. 
For this reason we say that bags of words are typically high-dimensional sparse datasets. We can save a lot of memory by only storing the non-zero parts of the feature vectors in memory
"""

## Importing necessary packages for Tokenizaationation & grouping
'''
Text preprocessing, tokenizing and filtering of stopwords are all included in "CountVectorizer", 
which builds a dictionary of features and transforms documents to feature vectors.
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
print(count_vect)
X_train_counts = count_vect.fit_transform(train.data)
print(X_train_counts)
X_train_counts.shape
# CountVectorizer supports counts of N-grams of words or consecutive characters. 
#Once fitted, the vectorizer has built a dictionary of feature indices:
count_vect.vocabulary_.get(u'algorithm') # o/p is 27366
# The index value of a word in the vocabulary is linked to its frequency in the whole training corpus.

## From OCURRERENCES TO FREQUENCIES
'''
Occurrence count is a good start but there is an issue: longer documents will have higher average count values 
than shorter documents, even though they might talk about the same topics.
To avoid these potential discrepancies, we have
1.Upscale: divide the number of occurrences of each word in a document by the total number of words in the document caled TF
2. Downscale: downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.
This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”
'''
# Both tf and tf–idf can be computed as follows using TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape # o/p is (11314, 130107)
'''
In the above example-code, we firstly use the fit(..) method to fit our estimator to the data and 
secondly the transform(..) method to transform our count-matrix to a tf-idf representation. 
'''
## Above fit & transofrm canbe acheived with SINGLE line as below:
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape # o/p is (11314, 130107)

### TRAINING CLASSIFIER : Approach 1 using ifidf transformer
'''we can train a classifier to try to predict the category of a post. Let’s start with a naïve Bayes classifier, 
which provides a nice baseline for this task. scikit-learn includes several variants of this classifier; 
the one most suitable for word counts is the multinomial variant'''
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, data.target)
'''
To try to predict the outcome on a new document we need to extract the features using almost the same feature 
extracting chain as before. The difference is that we call transform instead of fit_transform on the transformers, since they have already been fit to the training set:
'''
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# predicting using the classiifer
predicted = clf.predict(X_new_tfidf)
print predicted # array([15,  7])
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, data.target_names[category]))
    
## PIPELINE BUILDING:
# In order to make the vectorizer => transformer => classifier easier to work with, scikit-learn provides a Pipeline class that behaves like a compound classifier:
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
# The names vect, tfidf and clf (classifier) are arbitrary
# Now training the model with single comand lone as 
text_clf.fit(data.data, data.target)

## EVALUATION of the performance
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target) # 0.77389 
'''
Let’s see if we can do better with a linear support vector machine (SVM), which is widely regarded as 
one of the best text classification algorithms (although it’s also a bit slower than naïve Bayes). 
We can change the learner by simply plugging a different classifier object into our pipeline
'''
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', 
                     penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
text_clf.fit(data.data, data.target)
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target) # 0.8238
# further more insights around accuracy
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
metrics.confusion_matrix(twenty_test.target, predicted)


### TRAINING CLASSIFIER : Approach 2 using tfidf VECTORIZER
from sklearn.feature_extraction.text import TfidfVectorizer ## TFIDF is Term Frequency times inverse document frequency.
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

## Creating models based on NB  Multinomial NB
model= make_pipeline(TfidfVectorizer(), MultinomialNB())

# traing model with training data
model.fit(train.data, train.target) # train_data is main description & target is assigned category

# creating lables for test data
labels= model.predict(test.data)
print(labels) #o/p is [ 7 11  0 ...  9  3 15] hwich is EUIVALENT to category_id.

## check performance of model
from sklearn.metrics import confusion_matrix
mat= confusion_matrix(test.target, labels) # actual values vs predicted values as feed of MATRIX needed.
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, 
            yticklabels=train.target_names)

# import accuracy module & check accuracy
from sklearn.metrics import accuracy_score
accuracy_score(test.target, labels) # o.p is 0.7738980350504514 ~77% model is accurate
## Else get entire score report 
from sklearn.metrics import classification_report
print(classification_report(test.target, labels)) # for different taget values

## plotting heatmap of confusion matrix data
plt.xlabel('true label')
plt.ylabel('predicted label')

## predicted category based on trained model
def predict_category(s, train=train, model=model):
	pred=model.predict([s])
        print(pred) # retruns the indent number-position of category
	return train.target_names[pred[0]]
	
## checking catgory o/p for test purpose	
predict_category('BMW is better than Audi') # o/p is 'rec.autos'
predict_category('New generation power plant') # o/p is sci.space
predict_category('England beat AUS in cricket Final') # o/p is rec.sport.hockey. Some ERROR here
predict_category('Corruption & Hunger are social issue') # o/p is 'talk.politics.misc'


##-------------------------------------------------------------------------------------------------------
##Classfier NB of news data NewsClassification.csv
import pandas as pd
import seaborn as sns; sns.set()
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix
news_data= pd.read_csv("C:/Users/Amit Srimal/Desktop/DET/ML/Data/NewsClassification.csv") ## 422k with 8 columns
list(news_data) # gets list of column names
news_data.iloc[:,2].head() # reading TITLE of column data
news_data.iloc[:,5].head() # reading CATEGORY of column data
categories= news_data.CATEGORY.unique() # category as ['b', 't', 'e', 'm']
news_data.CATEGORY # returns ENTIRE list &hence UNIQUE is used.
##NOTE: "b" from Business;"e" from Entertainment,"t" from Science and Technology,"m" from Health

# training data
train, test = train_test_split(news_data, test_size=0.2)
print(len(train)) # 337935
print(train.head())
print(len(test)) # 84484
print(len(news_data)) # 422419 i.e. 80-20 ratio data split

## Getting overview of data
print(train.TITLE.iloc[6]) # retruns 6th row of training dataset from column TITLE

## Importing necessary packages for Tokenizaationation & grouping
from sklearn.feature_extraction.text import TfidfVectorizer ## TFIDF is Term Frequency times inverse document frequency.
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

## Creating models based on NB  Multinomial NB
model= make_pipeline(TfidfVectorizer(), MultinomialNB())

# traing model with training data
model.fit(train.TITLE, train.CATEGORY) # train_data is main description & target is assigned category
test['CATEGORY'].tail()
# creating lables for test data
labels= model.predict(test.TITLE)
print(labels) #o/p is ['b' 'b' 'e' ... 'b' 'e' 'm'] hwich is EUIVALENT to category_id.

## check performance of model
from sklearn.metrics import confusion_matrix
mat= confusion_matrix(test.CATEGORY, labels) # actual values vs predicted values as feed of MATRIX needed.
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels='auto',yticklabels='auto')
#Note:x,y ticklables set as AUTO & hence number will appear. We can set labels as train.target_names

# import accuracy module & check accuracy
from sklearn.metrics import accuracy_score
accuracy_score(test.CATEGORY, labels) # o.p is 0.9253941574736044 ~92% model is accurate
# Alternatively geting all SCORE report as below
from sklearn.metrics import classification_report
print(classification_report(test.CATEGORY, labels)) # evaluating the model's other score for all labels

## predicted category based on trained model
def predict_news_category(s, train=train, model=model):
	pred= model.predict([s])
        #print(pred) 
	return pred[0] #fetching the exact category order here
	
## checking catgory o/p for test purpose	
print(predict_news_category('Chronic Pain Drugs Pushes Through Phase 3')) # o/p needs to be m
print(predict_news_category('New journey to Europe')) # o/p belongs to category b 
print(predict_news_category('Fed official says weak data caused by weather, should not slow taper'))  #o/p is 'b'
# from test data: 
print(predict_news_category('You Can Text 911 In an Emergency Starting Today')) # o/p is t
#_______________________________________________________________________________________________________________



####### TYPE B: MULTI-LABEL CLASSIFICATION--------------------------------------------------------------------
#Refhttps://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
# Objective:
    #1. build a multi-label model that’s capable of detecting different types of toxicity like severe toxic, threats, obscenity, insults, and so on.
    #2. A toxic comment might be about any of toxic, severe toxic, obscene, threat, insult or identity hate at the same time or none of the above.
    # eg a movie may have label adventure 60%, sci-fi-20%, horror-10%, drama-10% LABELS
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
print(missing_values_check)

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
data = ml_data.loc[np.random.choice(ml_data.index, size=159571)]
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
# segregating the test, train comment as
train_text = train['comment_text']
train_text.shape # series type (111699L,)
test_text = test['comment_text'] ##train.iloc[:,[1,4]] to get all rows with 1 & 4th columns
test_text.shape # series type as (47872L,)

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
    model=LogReg_pipeline.fit(train_x, train[category])
    print(model)
    
    # calculating test accuracy
    prediction = model.predict(test_x)
    print(prediction) # predicted value for test data
    print(test[category]) # actual value of test data
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print("\n") 


## FUNCTION to get indivisual caomment catgeory listed as
def predict_labels(s):
    #s=[s]
    s=pd.Series(s)
    #print(s)
    #vectorizer.fit(s)
    s_labels = vect.transform(s)
    #print(s_labels)
    for category in categories:
        # lets resuse the model value again 
        model=logreg.fit(train_text_dtm, train[category])
        prediction=model.predict(s_labels)# earlier used predict(s2)
        test_y_prob = model.predict_proba(s_labels)[:,1]
        print category, 'label probability is around',test_y_prob[0] 
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
predict_labels('worthless piec shit')
predict_labels('die waht day fuck you')
predict_labels('bullshit nazi you die')
predict_labels('sigh appar im surround idiot like chubbo')
predict_labels('hello thanks for the welcome brother')
predict_labels('impressive design brilliant work great') 
 
    
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
        print'Label value for',category,'is',prediction[0]
def predict_labels(s,model=model,train=train_text):
    s=[s]
    print(s)
    #vectorizer.fit(s)
    s_labels = vectorizer.transform(s)
    print(s_labels)
    for category in categories:
        prediction=model.predict(s_labels) # earlier used predict(s2)
        #print prediction
        #print category
        #print('fetching {} label value as'.format(category),prediction[0])
        print category,'label is',prediction[0]       
        
               
## checking catgory LABELS for some random texts
print(predict_labels('power hungri administr like suck cock'))
print(predict_labels('Ask Sityush to clean up his behavior than issue me nonsensical warnings...All of my edits are good.  Cunts like you who revert good edits because youre too stupid to understand how to write well , and then revert other edits just because youve decided to bear a playground grudge, are the problem.  Maybe one day youll realise the damage you did to a noble project.  201.215.187.159'))
print(predict_labels('COCKSUCKER BEFORE YOU PISS around ON MY WORK and here no where idiot shit fucking idiot racist'))
print(predict_labels('insult musikanim your fuck shit reason im insult block fuck wikipedia'))
print(predict_labels('realli rebel guy rebelguy time get oral prostitut'))
print(predict_labels('let screwattack fuck pedant wikipedia asshol let nerd page fuck sake'))
print(predict_labels('world give mate sad individu kensington even fuck sadder bet american well owrld favour get lostt'))
'''
























































































































































