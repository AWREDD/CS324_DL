import numpy as np
import torch
import matplotlib.pyplot as plt


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.weight = np.zeros([1, n_inputs])
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

    def forward(self, input):
        """
        Predict label fropm input
        Args:
            input: array of dimension equal to n_inputs.
        """
        label = np.matmul(self.weight, input)
        label = np.where(label > 0, 1, -1)
        return label

    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        for i in range(int(self.max_epochs)):
            error = 0
            for idx in range(training_inputs.shape[0]):
                if labels[idx] * self.forward(training_inputs[idx]) < 0:
                    error = error + 1
                    self.weight = self.weight + self.learning_rate * labels[idx] * training_inputs[idx]
            print("accuracy rate of ", i, " training turns is ", 1.0 - error / training_inputs.shape[0])