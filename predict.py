import deepneuralnet as net
import random 
import tflearn.datasets.mnist as mnist
from skimage import io
from fractions import Fraction
import tensorflow as tf
model = net.model
path_to_model = 'final-model.tflearn'
X, Y, testX, testY= mnist.load_data(one_hot=True)
model.load(path_to_model)
# Randomly take an image from the test set
correct=0
test_result=[]
test_label=[]

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

    x = testX[j].reshape((28, 28, 1))
    result = model.predict([x])[0] # Predict
    prediction = result.tolist().index(max(result)) # The index represents the number predicted in this case
    #print("Prediction", prediction)
    count=0
    i=0
    while (testY[j][i]!=1):
    	count=count+1
    	i=i+1
    #print(count)
    if(count==prediction):
    	correct=correct+1
        test_result.append(prediction)
        test_label.append(count)
        
        #print(correct)
    else:
        #wrong.append(count)
        #correct=correct
        test_result.append(count)
        test_label.append(prediction)
 

accuracytest=float(correct/10000.0)
#wrong_alg=count_all(wrong)

correct_train=0
train_result=[]
train_label=[]

for j in range (0,10000):

    x = X[j].reshape((28, 28, 1))
    result = model.predict([x])[0] # Predict
    prediction = result.tolist().index(max(result)) # The index represents the number predicted in this case
    #print("Prediction", prediction)
    count=0
    i=0
    while (Y[j][i]!=1):
        count=count+1
        i=i+1
    #print(count)
    if(count==prediction):
        correct_train=correct_train+1
        train_result.append(prediction)
        train_label.append(count)
    else:
        train_result.append(prediction)
        train_label.append(count)
 

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

