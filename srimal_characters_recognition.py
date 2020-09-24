'''
Entire process consist of 3 steps as:
1:Create a database of handwritten digits.
2:For each handwritten digit in the database, extract HOG features and train a Linear SVM.
3:Use the classifier trained in step 2 to predict digits.
'''
# import the necessary packages
# dataset from kaggle @ https://www.kaggle.com/c/digit-recognizer/data
# each digit represented by 28*28 pixels
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# reading csv file to df abd then converting to a matrix
data= pd.read_csv('C:/Users/Amit Srimal/Desktop/DET/ML/ImageDetection/train.csv')
data_mat=data.as_matrix()
print(data)

# Here x will be intensity label & y will the digits that we need to predict
# lets define a classifier frst
clf=DecisionTreeClassifier()

## Lets make two sets train & test
#First lets have training with correspong x & y labels
xtrain= data_mat[0:21000,1:] # taking first 21000 rows & all columns except first as its label.
train_label=data_mat[0:21000,0] # getting the label, ie digit of pixels

# for testing 
xtest= data_mat[21000:,1:] # represents the x's of test data set
actual_label= data_mat[21000:,0] # actaul resut, which we will match agaunst predicted answers.

# Now lets train model 
clf.fit(xtrain, train_label)

## Now lets analyse any pixel digit say from test
d=xtest[8] # picking 8th index from xtest 
d.shape=(28,28) # converting the element as 28by28 pixels iamge
pt.imshow(d,cmap='gray')
#pt.imshow(255-d,cmap='gray') # 255-d for showing black with white background.
pt.show()

# Now printing & predicting 
select_label_row= int(input("Enter row index from test_set:"))
d=xtest[select_label_row] # picking nth index from xtest 
d.shape=(28,28) # converting the element as 28by28 pixels iamge
pt.imshow(255-d,cmap='gray') # 255-d for showing black with white background.
print('predict digit is',clf.predict([xtest[select_label_row]])) # predict the same index which we picked for test prediction
pt.show()

# Just to confirm,if we take first row of test and check with actaul data
data.iloc[21000,0] # o/p is 6


###----------------------------------------------------------------------------------
# dataset from skelaern load_digits: https://www.youtube.com/watch?v=PO4hePKWIGQ
import numpy as np
import matplotlib.pyplot as pt
from sklearn.datasets import load_digits
digits= load_digits() # images[image for digit] & target[actaul digit]

import pylab as pl
pl.gray()
pl.matshow(digits.images[0]) # displaying the random digits here
pl.show() # retruns image having digit as 1
# ALTERNATIVELY we can have black/gray comibnation
d=digits.images[1]
pl.imshow(255-d,cmap='gray')
pl.show() # retruns image having digit as 1

##Ltes visualize the pixel density here
digits.images[1] # retruns a matrix intensity for digit[1]

# starting modle prediction
import random
from sklearn import ensemble

#Define variables
n_samples = len(digits.images) # 1797 is row count of dataset
x = digits.images.reshape((-1,n_samples))
y = digits.target

#Create random indices 
sample_index=random.sample(range(len(x)),len(x)/5) #20-80
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample and validation images
sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

#Sample and validation targets
sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

#Using the Random Forest Classifier
classifier = ensemble.RandomForestClassifier()

#Fit model with sample data
classifier.fit(sample_images, sample_target)

# printing & preciting digits
Enter_row= int(input("Enter row index from digit_set:"))
d=digits.images[i]  # index of data row here
pl.imshow(255-d,cmap='gray')
pl.show()
#classifier.predict(x[i])


####-------------------------------------------------------------------------------------
## Predicting Numbers from Image data set --------------------------------------
# predict number from give set of image

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd

digits=load_digits() # ( is VERY imp else error is thrown)

# determining the total number of images & labels
print('Image data label:',digits.data.shape) # o/p is (1797,64) & denoting 8by8 pxel images
print('Target data label:',digits.target.shape)  # o/p is (1797,)
#plotting the image files to see the number pattern
plt.figure(figsize=(20,4))
for index,(image,label) in enumerate (zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)), cmap=plt.cm.gray)
    plt.title('training: %i\n' % label, fontzise=20)
# dividing & training the model
x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.23,random_state=2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
# creating logistic model 
logmodel=linear_model.LogisticRegression()
# passing & fitting training data to model
logmodel.fit(x_train,y_train)

## testing of model for one IMAGE now
print(logmodel.predict(x_test[0].reshape(1,-1))) # o/p prediction is number 4 here.
# try predicting few more in series
logmodel.predict(x_test[0:10]) # o/p is array([4, 0, 9, 1, 8, 7, 1, 5, 1, 6])
# test with entire data set
predictions=logmodel.predict(x_test)
# now checking accuracy score
from sklearn.metrics import accuracy_score
score= accuracy_score(y_test,predictions) # ~ 94.2 %
# Checking accuracy with confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,predictions)
print(cm) # more the number in diagonal, more is Accuracy.
# IMP: sum of matrix is equal to no of observation in test df.

## Visualizing confusion matrix in HeatMap graph
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True,fmt='.3f',linewidths=0.5,square=True,cmap='Blues_r')
plt.xlabel('Predicted_label')
plt.ylabel('Actual_label')
all_sample_title='Accuracy_score:{0}'.format(score)
plt.title(all_sample_title,size=15)

## testing o/p now along with comparison b/w actaul & predicted
index=0
classifiedIndex=[]
for predict, actual in zip(predictions,y_test):
    if predict==actual:
        classifiedIndex.append(index)
        index+=1
plt.figure(figsize=(20,9))
for plotIndex, wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1,4,plotIndex+1)
    plt.imshow(np.reshape(x_test[wrong],(8,8)),cmap=plt.cm.gray)
    plt.title('predicted {}, actual {}'.format(predictions[wrong],y_test[wrong]), fontsize=20)


### Predicting numbers from handwritten Image-------------------------------------------------------------
#http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html
# Import the modules
from sklearn.externals import joblib
from sklearn.datasets import fetch_mldata
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np    
    
dataset = fetch_mldata('MNIST original')   
    
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')    
    
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')    
    
    
names(data())   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



