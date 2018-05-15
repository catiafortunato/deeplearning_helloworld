import deepneuralnet as net
import random 
import tflearn.datasets.mnist as mnist
from skimage import io
from fractions import Fraction
import tensorflow as tf
from tflearn.datasets import cifar10
model = net.model
path_to_model = 'final-model.tflearn'
(X, Y), (testX, testY) = cifar10.load_data()
model.load(path_to_model)
# Randomly take an image from the test set
correct=0
test_result=[]
test_label=[]

doFlip = False #Defines if we want to analyse the images normaly or switched
#doFlip = True

print(Y)

def flipY(x):
    a=range(0, 14)
    b=range(14,28)
    pm=x[a]
    sm=x[b]
    x[a]=sm
    x[b]=pm
    return x

def flipX(x):
    a=range(0, 14)
    b=range(14,28)
    pm=x[:,a]
    sm=x[:,b]
    x[:,a]=sm
    x[:,b]=pm
    return x


def count_all(vec):
    buckets = [0] * 10
    for x in range(0,len(vec)):
        if (vec[x]==0):
            buckets[0]=buckets[0]+1
        elif (vec[x]==1):
            buckets[1]=buckets[1]+1
        elif (vec[x]==2):
            buckets[2]=buckets[2]+1
        elif (vec[x]==3):
            buckets[3]=buckets[3]+1
        elif (vec[x]==4):
            buckets[4]=buckets[4]+1
        elif (vec[x]==5):
            buckets[5]=buckets[5]+1
        elif (vec[x]==6):
            buckets[6]=buckets[6]+1
        elif (vec[x]==7):
            buckets[7]=buckets[7]+1
        elif (vec[x]==8):
            buckets[8]=buckets[8]+1
        elif (vec[x]==9):
            buckets[9]=buckets[9]+1

    
    return buckets

for j in range (0,10000):

    x = testX[j].reshape((32, 32, 3))
    result = model.predict([x])[0] # Predict
    prediction = result.tolist().index(max(result)) # The index represents the number predicted in this case
    if(testY[j]==prediction):
    	correct=correct+1
        test_result.append(prediction)
        test_label.append(testY[j])
    else:
        test_result.append(testY[j])
        test_label.append(prediction)
 

accuracytest=float(correct/10000.0)
#wrong_alg=count_all(wrong)

correct_train=0
train_result=[]
train_label=[]

for j in range (0,10000):

    x = X[j].reshape((32, 32, 3))    
    if(doFlip):
        x=flipX(x)
    
    result = model.predict([x])[0] # Predict
    prediction = result.tolist().index(max(result)) # The index represents the number predicted in this case
    if(Y[j]==prediction):
        correct_train=correct_train+1
        train_result.append(prediction)
        train_label.append(Y[j])
    else:
        train_result.append(prediction)
        train_label.append(Y[j])
 
# print(X[1].reshape((28, 28, 1)))

accuracytrain=float(correct_train/10000.0)
#wrong_alg_train=count_all(wrong_train)

confusion_test= tf.confusion_matrix(test_label,test_result)
confusion_train= tf.confusion_matrix(train_label,train_result)

print("Accuracy for test set=", accuracytest)
print("Confusion matrix for test set result")
with tf.Session():
   print('Confusion Matrix: \n\n', tf.Tensor.eval(confusion_test,feed_dict=None, session=None))
print("Accuracy for train set=", accuracytrain)
with tf.Session():
   print('Confusion Matrix: \n\n', tf.Tensor.eval(confusion_train,feed_dict=None, session=None))
#print(wrong_alg_train)

