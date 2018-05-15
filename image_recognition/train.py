import sys
import tflearn
import deepneuralnet as net
import tflearn.datasets.mnist as mnist
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y,10)
Y_test = to_categorical(Y_test,10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Get the model
model = net.model
# Load data
#X, Y, _, _ = mnist.load_data(one_hot=True)
X = X.reshape([-1, 32, 32, 3])
#testX = testX.reshape([-1, 28, 28, 1])
num_it= int(sys.argv[1])
#model.fit(X, Y, n_epoch=num_it, validation_set=(X, Y), show_metric=True, run_id="deep_nn")
#model.save('final-model.tflearn')

# Train using classifier
#model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=num_it, validation_set=(X, Y), show_metric=True, run_id="deep_nn")
model.save('final-model.tflearn')