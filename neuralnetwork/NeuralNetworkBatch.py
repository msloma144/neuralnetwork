import numpy as np


class NeuralNetworkBatch(object):
    def __init__(self, hidden_layer_sizes, lambda_val, iterations):
        # Storage Arrays
        self.activity = []
        self.activation = []
        self.weights = []
        self.cost_per_layer = []
        self.deltas = []
        self.partial_derivatives = []
        self.partial_derivatives_reg = []

        # More constant variables
        self.num_examples = 0
        self.hidden_layer_sizes = hidden_layer_sizes
        self.iterations = iterations
        self.lambda_val = lambda_val
        self.layers = len(hidden_layer_sizes) + 2

    def sigmoid(self, activity):
        # sigmoid/logistic activation function
        return 1 / (1 + np.exp(-1 * activity))

    def sigmoid_prime(self, activation):
        return activation * (1 - activation)

    def initialize_weights(self, inputs, outputs):
        # sets up the weights for the network and puts them in the weights array
        self.weights.append(np.random.rand(inputs.shape[1] + 1, self.hidden_layer_sizes[0]))
        # make sure that the hidden layer tuple is larger than 1
        if len(self.hidden_layer_sizes) != 1:
            for i in range(len(self.hidden_layer_sizes) - 1):
                self.weights.append(np.random.rand(self.hidden_layer_sizes[i] + 1, self.hidden_layer_sizes[i + 1]))
        self.weights.append(np.random.rand(self.hidden_layer_sizes[len(self.hidden_layer_sizes) - 1] + 1, outputs.shape[1]))

    def add_bias_inputs(self, inputs):
        # adds the bias nodes to the input matrix
        return np.hstack((np.ones([self.num_examples, 1]), inputs))

    def add_bias_activation(self, activation_layer):
        # adds the bias nodes to the activation matrix
        return np.hstack((np.ones([activation_layer.shape[0], 1]), activation_layer))

    def forward_prop(self, inputs):
        # forward propagation
        self.activity = []
        self.activation = []

        # compute activation for the first layer
        self.activity.append(np.dot(inputs, self.weights[0]))
        # compute activation
        self.activation.append(self.sigmoid(self.activity[0]))
        # add the bias nodes for the activation
        self.activation[0] = self.add_bias_activation(self.activation[0])

        # run through and compute the activity and activation for each layer
        for i in range(1, self.layers - 1):
            self.activity.append(np.dot(self.activation[i - 1], self.weights[i]))
            # compute activation
            self.activation.append(self.sigmoid(self.activity[i]))
            # add the bias nodes for the activation
            if i != self.layers - 2: # if the layer is not the output layer...
                self.activation[i] = self.add_bias_activation(self.activation[i])

    def calculate_partials(self, outputs):
        self.deltas = [None] * self.layers
        # calc final layer's delta
        self.deltas[self.layers - 1] = self.activation[len(self.activation) - 1] - outputs
        # calculate the rest of the deltas
        for i in reversed(range(1, self.layers - 1)):
            self.deltas[i] = np.dot(self.deltas[i + 1], np.transpose(self.weights[i])) * self.sigmoid_prime(self.activation[i - 1])
            # remove the bias nodes from the delta
            self.deltas[i] = self.deltas[i][:, 1:]

    def calculate_partial_derivatives(self, inputs):
        self.partial_derivatives = []
        # need to insert the inputs for "activation" on layer 1
        self.activation.insert(0, inputs)
        # compute all the partials
        for i in range(1, len(self.deltas)):
            self.partial_derivatives.append(np.dot(np.transpose(self.activation[i - 1]), self.deltas[i]))

    def calculate_partial_derivatives_regularized(self, inputs):
        self.partial_derivatives_reg = []
        for i in range(0, len(self.partial_derivatives)):
            # compute derivative for bias node
            self.partial_derivatives_reg.append((1 / self.num_examples) * self.partial_derivatives[i][1, :])
            # compute derivative for the rest of the nodes
            reg_temp = (1 / self.num_examples) * (self.partial_derivatives[i][1:, :] + (self.lambda_val * self.weights[i][1:, :]))
            # combine the bias node and the rest of the nodes back together
            self.partial_derivatives_reg[i] = np.vstack((self.partial_derivatives_reg[i], reg_temp))

    def error_function(self, outputs):
        example_cost = np.empty([1, outputs.shape[0]])
        for i in range(0, outputs.shape[0]):
            example_cost[0, i] = np.sum((outputs[i, :] * np.log(self.activation[self.layers - 1][i, :])) + ((1 - outputs[i, :]) * np.log(1 - self.activation[self.layers - 1][i, :])))

        # calculate th sum of the squared weights
        squared_weights_sum = 0
        for i in range(len(self.weights) - 1):
            squared_weights_sum += np.sum(np.power(self.weights[i][1:, :], 2))

        # regularization term
        reg_term = (self.lambda_val/(2 * outputs.shape[0])) * squared_weights_sum

        cost = (-1 * (1 / outputs.shape[0] * np.sum(example_cost))) + reg_term
        return cost

    def adjust_weights(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.partial_derivatives_reg[i]

    def gradient_checker(self, outputs, inputs):
        partial_derv_weight_guess = []
        partial_derv_diff = []
        # the average difference between the back prop and the checker across all weights
        net_avg_diff = 0

        initial_weights = self
        epsilon = .0001
        for i in range(len(self.weights)):
            sub_avg = 0
            partial_derv_weight_guess.append(np.empty(self.weights[i].shape))

            for (x, y), value in np.ndenumerate(self.weights[i]):
                # calculate the upper weight
                self.weights[i][x, y] += epsilon
                self.forward_prop(inputs)
                self.activation.insert(0, inputs)
                upper_error = self.error_function(outputs)

                # calculate the lower weight
                self.weights[i][x, y] -= 2 * epsilon
                self.forward_prop(inputs)
                self.activation.insert(0, inputs)
                low_error = self.error_function(outputs)

                # store the approximated gradient
                partial_derv_weight_guess[i][x, y] = (upper_error - low_error) / (2 * epsilon)
                # set back to original epsilon
                self.weights[i][x, y] += epsilon

            # store all of the differences between the approximation and the back prop gradient
            partial_derv_diff.append(partial_derv_weight_guess[i] - self.partial_derivatives_reg[i])
            # calculate the average difference for the round
            sub_avg = np.sum(np.abs(partial_derv_weight_guess[i] - self.partial_derivatives_reg[i])) / self.partial_derivatives_reg[i].size
            net_avg_diff += sub_avg
            self = initial_weights
        net_avg_diff = net_avg_diff / len(self.weights)
        print("Average Difference: " + str(net_avg_diff))

    def back_propagation(self, inputs, outputs, learning_rate):
        # calculate each partial
        self.calculate_partials(outputs)
        self.calculate_partial_derivatives(inputs)
        self.calculate_partial_derivatives_regularized(inputs)
        #print(self.error_function(outputs))
        # adjust the weights using the gradients and the learning rate
        self.adjust_weights(learning_rate)

    def train(self, inputs, outputs, learning_rate, gradient_check_on):
        self.num_examples = inputs.shape[0]
        # initialize the weight matrices
        self.initialize_weights(inputs, outputs)

        # add in the bias units for the input layer
        inputs = self.add_bias_inputs(inputs)

        # compute forward prop and back prop
        for i in range(0, self.iterations):
            self.forward_prop(inputs)
            self.back_propagation(inputs, outputs, learning_rate)
            # if gradient_check_on = True then check the accuracy of the gradient values
            if gradient_check_on:
                self.gradient_checker(outputs, inputs)
            #print(i)
        # output end error/loss
        print(self.error_function(outputs))
