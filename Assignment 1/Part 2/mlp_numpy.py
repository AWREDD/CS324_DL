from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        self.layers = []
        for index in range(len(n_hidden)):
            if index == 0:
                self.layers.append(Linear(n_inputs, int(n_hidden[index])))
                self.layers.append(ReLU())
            if index == len(n_hidden)-1:
                self.layers.append(Linear(int(n_hidden[index]), n_classes))
            else:
                self.layers.append(Linear(int(n_hidden[index]), int(n_hidden[index+1])))
                self.layers.append(ReLU())
        self.layers.append(SoftMax())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        out = x
        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            layer.update()
        return

