from neuralnetwork import NeuralNetworkBatch
from neuralnetwork import NeuralNetworkStochastic
from neuralnetwork import NeuralNetworkBiasValue
from neuralnetwork import NeuralNetworkStocasticBiasValue
from neuralnetwork import *

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

hidden_layer_param = (3, 5, 8)
iterations = 200
batch_size = 100
lambdaVal = 0

inputs = np.random.rand(20000, 4)
outputs = np.random.randint(0, 2, [20000, 1])

#inputs = np.array([[0., 0.], [1., 1.]])
#outputs = np.array([[0.], [1.]])


# self made MLP network *BATCH*
print("BATCH")
testBatch = NeuralNetworkBatch.NeuralNetworkBatch(hidden_layer_param, lambdaVal, iterations)
testBatch.train(inputs, outputs, .01, False)

# self made MLP network *STOCHASTIC*
print("STOCHASTIC")
testStochastic = NeuralNetworkStochastic.NeuralNetworkStochastic(hidden_layer_param, lambdaVal, iterations, batch_size)
testStochastic.train(inputs, outputs, .01, False)

# self made MLP network *USING BIAS VALUES RATHER THAN BIAS NODES*
print("USING BIAS VALUES RATHER THAN BIAS NODES")
testBiasVal = NeuralNetworkBiasValue.NeuralNetworkBiasValue(hidden_layer_param, lambdaVal, iterations)
testBiasVal.train(inputs, outputs, .01, False)

# self made MLP network *STOCHASTIC USING BIAS VALUES RATHER THAN BIAS NODES*
print("STOCHASTIC USING BIAS VALUES RATHER THAN BIAS NODES")
testBiasValStochastic = NeuralNetworkStocasticBiasValue.NeuralNetworkStochasticBiasValue(hidden_layer_param, lambdaVal, iterations, batch_size)
testBiasValStochastic.train(inputs, outputs, .01, False)

# scikit-learn MLP network
print("SK-Learn")
x = preprocessing.scale(inputs)
clf = MLPClassifier(hidden_layer_sizes=hidden_layer_param, solver='sgd', alpha=0, activation='logistic', max_iter=iterations)
clf.fit(inputs, outputs.reshape(outputs.shape[0],))
print(clf.loss_)

