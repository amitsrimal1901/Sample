# Text mining / text analytics is process of deriving meaning ful information from natural language text.
# Around 25% is data is strcutured & rest is unstructured in current world scenario.
# NLP is common (venn diagram zone) area of Computer Science, Artificial Intelligence & Human language.
# NLP is usage of CS & AI to understand Human language.

#Applications of NLP:
    #1. Sentiment analysis
    #2. Chatbot
    #3. Speech Recognition
    #4. Machine transalation
    #5. Spell Check
    #6. Keyword search
    #7. Infirmation extraction
    #8. Advertsiement matching: recommending ads based on history.

##Components of NLP:    
    #1. Natural Language undersatdning
    #2. Natural Language Generation
    
## Steps of NLP:
    #Step1: Tokenization: say breaking complete sentence in words 
    #Step2: Stemming: normalize word into its root/ base form.
    #Step3: Lemmatization: advanced than stemming & uses some DICTIONARY say gone, go, going , went mapped to GO.
    #Stpe4: POS tags: identifies verb, noun, adj  etc of words
    #Step5: NameEntityRelationship(NER): say indetfy google as organization, verb, noun etc based on usage case.
            #Three steps as Noun-phrase identifcation, Phrase classification & Entity disambiguition
    #Step6:Chunking: grouping of individual pieces for bigger information 
    
# Package to use here is Python NLTK Natural Language Tool Kit
# Others are Textblob, CoreNLP, Gensim, Spacy etc.
# Should start with TEXTBLOB. Its build on top of NLTK & has great features.
 
##More about textblob:
# TextBlob is that they are just like python strings.
#SExecution Speed : Spacy > TextBlob > NLTK
name= TextBlob('Amit Srimal')
name[2:7] ##o/p is TextBlob("it Sr")
name.upper() ##o/p is TextBlob("AMIT SRIMAL")
name +"" +' Krishna' ## o/p is  TextBlob("Amit Srimal Krishna")

## For the uninitiated – practical work in Natural Language Processing typically uses large bodies of linguistic data, or corpora. To download the necessary corpora, you can run the following command
## Conmmand to download corpora: python -m textblob.download_corpora
# step by step exuction as listed in 6 steps above

#Step1: Tokenization
from textblob import TextBlob
statement = TextBlob("Analytics Vidhya is a great platform to learn data science. It helps community through blogs, hackathons, discussions,etc.")    
print(statement)
# textblob can be tokenized into a sentence and further into words
statement.sentences
# extracting first sentense from statement
statement.sentences[0]
statement.sentences[1] # retruns secind sentence. Needs proper SYNTAX of dot(.)
# Get words from first sentence
words_list= statement.sentences[0].words
print(words_list)
## to print all wordsof para
print (statement.words) # returns o/p words for all sentences in the paragraph.

#Step2: Noun phrase extraction: helps to identify "WHO" part of the sentence.
for np in statement.sentences[0].noun_phrases:
 print (np)
 #Alternatively use below command
statement.sentences[0].noun_phrases # gives a word list f noun

#Step3: Part-of-speech tagging or grammatical tagging is a method to mark words present in a text on the basis of its definition and context.
#  it tells whether a word is a noun, or an adjective, or a verb, etc
for words, tag in statement.sentences[0].tags:
 print (words, tag)
## NOTE: Here, NN represents a noun, DT represents as a determiner,JJ adjective etc. More tags present.
 # Link for tags: https://www.clips.uantwerpen.be/pages/mbsp-tags
 #Alternatively use below command
statement.sentences[0].tags # gives a word list f noun, adj,verbs etc.

#Step4: Words Inflection and Lemmatization
# Inflection is a process of word formation in which characters are added to the base form of a word to express grammatical meanings.
# the words we tokenized from a textblob can be easily changed into singular or plural.
print (statement.sentences[1].words[1]) # o/p is HELPS
print (statement.sentences[1].words[1].singularize()) #o/p is HELP
print (statement.sentences[1].words[2].pluralize()) # o/p is communities
# Lemmeaization
# TextBlob library also offers an in-build object known as Word
from textblob import Word
w= Word("running")
x= w= Word("went")
w.pluralize() # o/p is runningS
w.lemmatize("v") #o/p is run. Mapping to base word
x.lemmatize("v") # o/p is GO. which is base word for go, went, gone

#Step5: N-grams
# A combination of multiple words together are called N-Grams.
# N grams (N > 1) are generally more informative as compared to words, and can be used as features for language modelling
# Accessible using the ngrams function, returns a tuple of n successive words
for ngram in statement.sentences[0].ngrams(2): # if we need to have 2 words in each element.
    print(ngram)

#Step6: Sentiment Analysis
# retruns polarity[-1 to +1 scale], and subjectivity[0 to 1 scale].Both float datatypes.  

##oTher Thinsg
# Getting defintions of words
Word("octopus").definitions    # needs to import WORD pakcage from textblob
   
## Spell check & correction: using correct() fucntion
statement_1= TextBlob('Indian teeam is goood at pleyinf foootball and criket.')
statement_1.correct() # o/p is  TextBlob("Indian team is good at playing football and cricket.")
# Check suggested word fom list & their accuracy using SPELLCHECK function
# Word.spellcheck() method that returns a list of (word, confidence) tuples with spelling suggestions
statement_1.words[5].spellcheck() # o/p says playing, preying, plying with accuracy.

## Word and Noun Phrase Frequencies
statement_1.word_counts['at']  # o/p is 1
# Alternatively using the .(dot) method with WORDS, not word. Attention to KEY WORD, S & BRACKETS
statement_1.words.count('at') # o/p is 1
## Word and Noun Phrase Frequencies
statement_1.words.count('At')  # o/p is 0
## Word and Noun Phrase Frequencies
statement_1.words.count('At',case_sensitive=True )  # Default is False

## Text parsing
statement_1.parse()
#IMP: By default, TextBlob uses pattern’s parser 

##TextBlobs Are Like Python Strings
# Python’s substring syntax lets say for statement_1
statement_1[1:10] # o/p is TextBlob("ndian tee")
statement_1[1:10].upper() #o/p is TextBlob("NDIAN TEE")
statement_1[1:20].find('dian') #o/p is 1, basically returns indexed position.
# comparisons between TextBlobs and strings
apple_blob = TextBlob('apples')
banana_blob = TextBlob('bananas')
apple_blob < banana_blob # o/p is True
apple_blob == 'apples' # o/p is True
apple_blob == 'bananas' #o/p is False
apple_blob != 'grapes' # o/p is True
# concatenate and interpolate TextBlobs and strings
apple_blob + ' and ' + banana_blob #o/p is TextBlob("apples and bananas")
"{0} and {1}".format(apple_blob, banana_blob) # o/p is 'apples and bananas'

## Creating short summary of text.
import random
statement_2 = TextBlob('Industry Industry Society is a thriving community for data driven industry. This platform allows \
people to know more about analytics from its articles, Q&A forum, and learning paths. Also, we help \
professionals & amateurs to sharpen their skillsets by providing a platform to participate in Hackathons.')
# create container for list of nouns (identifier of objects on grammar)
nouns = list()
for word, tag in statement_2.tags:
    if tag == 'NN':
        nouns.append(word.lemmatize()) #lemmatized output of tagged words

##Presenting information about the sentence 
print ("This text is about...")
for item in random.sample(nouns, 5):
    word = Word(item)
    print (word.pluralize()) # retruns list of distinct key values. here 5 distinct key words. Above 5, its gives "Sample larger than population or is negative"

## Transalation & language detetection: Uses Google Translate in background
unknown_language= TextBlob('수송 시스템')  # korean for transport system
unknown_language.detect_language() #o/p is ko
unknown_language.translate(from_lang='ko', to ='en') #o/p is TextBlob("Transportation system")
#Alternatively, we can tell target language. 
unknown_language.translate(to= 'en') #o/p is TextBlob("Transportation system")

aa=TextBlob('Vennligst sjekk internettforbindelse')
aa.translate(to= 'fr') #o/p is TextBlob("S'il vous plaît vérifier la connexion Internet")
aa.translate(to= 'en') #o/p is  TextBlob("Please check internet connection")


### Text classification-------------------------------------------------------------------------------------
## Lets prepare training data first with positive & negative commnets around
training = [
('Tom Holland is a terrible spiderman.','pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
]
testing = [
('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')
]
# Textblob provides in-build classifiers module to create a custom classifier
from textblob import classifiers
classifier = classifiers.NaiveBayesClassifier(training)
# we have used Naive Bayes classifier, but TextBlob also offers Decision tree classifier.
## decision tree classifier
dt_classifier = classifiers.DecisionTreeClassifier(training)
## Now Test & check accuracy of prediction
print (classifier.accuracy(testing))
classifier.show_informative_features(3) # can set other ineteger to check values & impact.
##IMP: if the text contains “is”, then there is a high probability that the statement will be negative.
# let’s check our classifier on a random text
statement_3 = TextBlob('the weather is terrible!', classifier=classifier)
print (statement_3.classify()) #o/p is given as NEGATIVE.
# get the label probability distribution
prob_dist = classifier.prob_classify("the weather is terrible!.")
prob_dist.max() # o/p is neg
round(prob_dist.prob("pos"), 2) # o/p is 0.39
round(prob_dist.prob("neg"), 2) # o/p is 0.61

## ALTERNATIVE way Another way to classify text is to pass a classifier into the constructor of TextBlob and call its classify() method
statement_4 = TextBlob("The beer is good. But the hangover is horrible.", classifier=classifier)
statement_4.classify() # o/p is NEG
## With this approach you can classify sentences within a TextBlob
for s in statement_4.sentences:
     print(s,' is ',s.classify(),' statement') # two o/p for two sentences of the BLOB

## Updating Classifiers with New Data. Constantly feed new data for trainung the model.
new_data = [('She is my best friend.', 'pos'),
         ("I'm happy to have a new friend.", 'pos'),
         ("Stay thirsty, my friend.", 'pos'),
         ("He ain't from around here.", 'neg')]
#updating classifier with new data
classifier.update(new_data)
classifier.accuracy(testing) # o/p is 1.0

## Feature Extractors:
# By default, the NaiveBayesClassifier uses a simple feature extractor that indicates which words in the training set are contained in a document.
#Ref: https://textblob.readthedocs.io/en/dev/classifiers.html#evaluating-classifiers

#SENTIMENT ANALYSIS MODELS-----------------------------------------------------------------------------------   
from textblob import TextBlob
#The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity).
#polarity: how positive or negative your statement is
#Subjectivity: expresses personal feeling, views, beliefs etc. They dont express any sentiments.
# Collecting feedbacks from end users
review1='The movie was okay'
review2='The movie was good'
review3='The movie was very okay'
review4='The movie was awesome'
review5='The movie was pathetic'
# Converting texts to blob
blob1= TextBlob(review1)
blob2= TextBlob(review2)
blob3= TextBlob(review3)
blob4= TextBlob(review4)
blob5= TextBlob(review5)
# Gettings sentiments from blobs
print('Review 1s is having',blob1.sentiment)
print('Review 2s is having',blob2.sentiment)
print('Review 3s is having',blob3.sentiment)
print('Review 4s is having',blob4.sentiment)
print('Review 5s is having',blob5.sentiment)   
## Lets plot the sentiment graph for 5 reviews
r1_xvalue= blob1.sentiment.polarity
r2_xvalue= blob2.sentiment.polarity
r3_xvalue= blob3.sentiment.polarity
r4_xvalue= blob4.sentiment.polarity
r5_xvalue= blob5.sentiment.polarity

import matplotlib.pyplot as plt
import numpy as np
x = np.array([0,1,2,3,5])
y= [r1_xvalue,r2_xvalue,r3_xvalue,r4_xvalue,r5_xvalue]
x_label = ['review 1','review 2','review 3','review 4', 'review 5']
plt.xticks(x, x_label) # naming x axes plot point as per label defintion.
plt.ylabel('Polarity scale')
plt.axis([-3, 10, -2, 2]) #defining scale of graph for visualization
plt.plot(y, linewidth=7.0, color='r') #plotting with custom width & color
plt.show()

############# Enter statement to check the Sentiment polarity
statement= str(input("Enter review for movie Robot 2.0:"))
blob_statement= TextBlob(statement)
polarity= blob_statement.sentiment.polarity
if polarity >= 0.6:
    print("Highly recommending movie with likeniness of ",polarity)
elif polarity >0.4 and polarity <0.6:
    print("Okay sort of movie with likeniness of ",polarity)
else:
    print("Not at all recommending the movie with likeniness of ,",polarity)    

#--------------------------------------------------------------------------------------------------------





