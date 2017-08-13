# Basic Neural Network API

Includes 4 different types of building a neural network implementation using gradient descent:
* Batch
* Stochastic
* Separate Bias Values
* Stochastic w/ Separate Bias Values

The method structure as far as train() is inspired by the scikit-learn implementation.

## Usage

Import modules from neuralnetwork package

```
from neuralnetwork import NeuralNetworkBatch
from neuralnetwork import NeuralNetworkStochastic
from neuralnetwork import NeuralNetworkBiasValue
from neuralnetwork import NeuralNetworkStocasticBiasValue
```

Using a batch implantation:
* hidden_layer_param is a tuple with length of the number of hidden layers consisting of the hidden layer sizes
* lambdaVal is the regularization value

```
hidden_layer_param = (3, 5, 8)
lambdaVal = 0
iterations = 200

testBatch = NeuralNetworkBatch.NeuralNetworkBatch(hidden_layer_param, lambdaVal, iterations)
testBatch.train(inputs, outputs, .01, False)
```

Using a stochastic implementation
* hidden_layer_param is a tuple with length of the number of hidden layers consisting of the hidden layer sizes
* lambdaVal is the regularization value
* batch_size is how large each batch should be for each iteration

```
hidden_layer_param = (3, 5, 8)
lambaVal = 0
batch_size = 100
iterations = 200


testStochastic = NeuralNetworkStochastic.NeuralNetworkStochastic(hidden_layer_param, lambdaVal, iterations, batch_size)
testStochastic.train(inputs, outputs, .01, False)
```

##### Example Output from Testing.py

```
Batch
0.709006986103
STOCHASTIC
0.696859893265
USING BIAS VALUES RATHER THAN BIAS NODES
0.696894700716
STOCHASTIC USING BIAS VALUES RATHER THAN BIAS NODES
0.694983312341
SK-Learn
0.693160790281
```

## Built With

* [NumPy](http://www.numpy.org/) - The matrix framework used
* [scikit-learn](http://scikit-learn.org/stable/) - framework testing against

## Authors

* **Michael Sloma**
