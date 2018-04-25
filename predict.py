import deepneuralnet as net
import random 
import tflearn.datasets.mnist as mnist
from skimage import io
from fractions import Fraction
model = net.model
path_to_model = 'final-model.tflearn'
X, Y, testX, testY= mnist.load_data(one_hot=True)
model.load(path_to_model)
# Randomly take an image from the test set
correct=0
wrong=[]

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
        #print(correct)
    else:
        wrong.append(count)
        correct=correct
 

accuracytest=float(correct/10000.0)
wrong_alg=count_all(wrong)

correct_train=0
wrong_train=[]

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
        #print(correct)
    else:
        wrong_train.append(count)
        correct_train=correct_train
 

accuracytrain=float(correct_train/10000.0)
wrong_alg_train=count_all(wrong_train)

print("Accuracy for test set=", accuracytest)
print(wrong_alg)
print("Accuracy for train set=", accuracytrain)
print(wrong_alg_train)

