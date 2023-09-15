from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropy
from sklearn import datasets
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, targets):
    total = predictions.shape[0]
    predictions = np.where(predictions > 0.5, 1, 0)
    acc_cnt = np.sum(predictions * targets)
    return acc_cnt / total


def train(para):
    x, y = datasets.make_moons(n_samples=1000, noise=0.08, shuffle=True, random_state=None)
    x_input = np.expand_dims(x, axis=1)
    y_onehot = np.eye(2)[y]
    y_onehot = np.expand_dims(y_onehot, axis=1)

    x_input_train = x_input[:800, :, :]
    y_onehot_train = y_onehot[:800, :]
    x_input_test = x_input[800:, :, :]
    y_onehot_test = y_onehot[800:, :]
    hidden_array = DNN_HIDDEN_UNITS_DEFAULT.split(",")
    model = MLP(2, hidden_array, 2)
    epoch_set = []
    loss_set = []
    acc_set = []
    acc_train = 0.
    acc_test = 0.
    acc_test_set = []
    idxs = np.random.randint(0, x_input_train.shape[0], size=10)
    tx = x_input_train.take(idxs, axis=0)
    print(tx.shape)
    if para == 'BGD':
        for turn in range(MAX_EPOCHS_DEFAULT):
            pre = model.forward(x_input_train)
            _, loss = CrossEntropy().forward(pre, y_onehot_train)
            loss_avg = np.mean(loss, axis=0)[0]
            acc_train = accuracy(pre, y_onehot_train)
            dout = CrossEntropy().backward(pre, y_onehot_train)
            model.backward(dout)
            if turn > 0 and turn % EVAL_FREQ_DEFAULT == 0:
                pre_test = model.forward(x_input_test)
                acc_test = accuracy(pre_test, y_onehot_test)

                epoch_set.append(turn)
                loss_set.append(loss_avg)
                acc_set.append(acc_train)
                acc_test_set.append(acc_test)
                print("The ", turn, " BGD training turn, loss is: ", loss_avg, "accuracy is: ", acc_train)
                print("train accuracy: ", acc_set[-1], "test accuracy: ", acc_test)
    if para == 'SGD':
        for turn in range(MAX_EPOCHS_DEFAULT):
            idxs = np.random.randint(0, x_input_train.shape[0], size=10)
            tmp_X = x_input_train.take(idxs, axis=0)
            tmp_Y = y_onehot_train.take(idxs, axis=0)

            pre = model.forward(tmp_X)
            _, loss = CrossEntropy().forward(pre, tmp_Y)
            loss_avg = np.mean(loss, axis=0)[0]
            acc_train = accuracy(pre, tmp_Y)
            dout = CrossEntropy().backward(pre, tmp_Y)
            model.backward(dout)
            if turn > 0 and turn % EVAL_FREQ_DEFAULT == 0:
                pre_test = model.forward(x_input_test)
                acc_test = accuracy(pre_test, y_onehot_test)

                epoch_set.append(turn)
                loss_set.append(loss_avg)
                acc_set.append(acc_train)
                acc_test_set.append(acc_test)
                print("The ", turn, " SGD training turn, loss is: ", loss_avg, "accuracy is: ", acc_train)
                print("train accuracy: ", acc_set[-1], "test accuracy: ", acc_test)


def main():
    """
    Main function
    """
    train("BGD")


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
