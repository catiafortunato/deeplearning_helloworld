import deepneuralnet as net
import tflearn.datasets.mnist as mnist
# Get the model
model = net.model
# Load data
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
model.fit(X, Y, n_epoch=15, validation_set=(testX, testY), show_metric=True, run_id="deep_nn")
model.save('final-model.tflearn')