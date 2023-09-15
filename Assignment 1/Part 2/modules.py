import numpy as np
import torch
import math
from sklearn import datasets
import matplotlib.pyplot as plt

LEARNING_RATE_DEFAULT = 0.01


def kaiming(m, h):
    return np.random.randn(m, h) * math.sqrt(2. / m)


def normal(loc, scale, in_features, out_features):
    return np.random.normal(loc=loc, scale=scale, size=(in_features, out_features))


class Linear(object):
    def __init__(self, in_features, out_features):
        self.batch_size = None
        self.input = None
        self.output = None
        self.params = dict()
        self.in_features = in_features
        self.out_features = out_features
        self.params['weight'] = normal(0.0, 0.001, in_features, out_features)
        # self.params['weight'] = kaiming(in_features, out_features)
        self.params['bias'] = 0
        self.dw = None
        self.db = None

    def forward(self, x):
        self.input = x
        # print("linear forward", x.shape, self.params['weight'].shape)
        self.output = np.matmul(x, self.params['weight']) + self.params['bias']
        # print("linear forward output size: ", self.output.shape)
        return self.output

    def backward(self, dout):
        # print(dout)
        self.batch_size = dout.shape[0]
        self.dw = np.mean(np.matmul(self.input.transpose(0, 2, 1), dout), axis=0)  # δw = δg * x
        self.db = np.mean(dout, axis=0)
        grad = np.matmul(dout, self.params['weight'].T)
        return grad

    def update(self):
        # print("update dw", self.dw)
        # print("update db", self.db)
        self.params['weight'] = self.params['weight'] - self.dw * LEARNING_RATE_DEFAULT
        self.params['bias'] = self.params['bias'] - self.db * LEARNING_RATE_DEFAULT
        return


class ReLU(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.where(x > 0, x, 0)
        # print("relu forward output size: ", out.shape)
        return self.out

    def backward(self, dout):
        dout_t = torch.tensor(dout)
        out_t = torch.tensor(self.out)

        idx = torch.where(out_t > 0)
        jd = torch.zeros_like(out_t)
        jd[idx] = 1
        j = torch.diag_embed(jd).squeeze()

        assert dout_t.size(-1) == j.size(-2)
        dx_t = torch.matmul(dout_t, j)
        dx = dx_t.numpy()
        return dx

    def update(self):
        return


class SoftMax(object):
    def __init__(self):
        self.output = None

    def forward(self, x):
        x_max = np.max(x, -1)[:, np.newaxis]
        x = x - x_max
        exp = np.exp(x)
        exp_sum = np.sum(np.exp(x), -1)[:, np.newaxis]
        return exp / exp_sum

    def backward(self, dout):
        return dout

    def update(self):
        return


class CrossEntropy(object):
    def __init__(self):
        self.output = None

    def forward(self, x, y):
        out = -np.sum(y * np.log(x), -1)
        self.output = out
        # print("CE", out.shape)
        return x, out

    def backward(self, x, y):
        batch_size = x.shape[0]
        dx = (x - y)
        return dx

    def update(self):
        return
