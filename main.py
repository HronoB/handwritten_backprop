import sys
import time
from random import random

import pandas as pd
import numpy as np

from show import show_image_from_array


def load_image_from_csv(csv_path='test.csv', index=0):
    df = pd.read_csv(csv_path)

    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of range. File has {len(df)} rows.")

    row = df.iloc[index].values.astype(np.uint8)

    return row[0], list(row[1:])




def printWH(arr):
    print(len(arr[0]), len(arr))




class Network:


    def __init__(self, dimensions = (784, 784, 100, 10)):
        self. dims = dimensions

        self.weights = []
        self.biases = []
        for i in range(len(dimensions)-1):
            self.weights.append(
                np.random.normal(
                size=(dimensions[i+1], dimensions[i]),
                scale=((2 / dimensions[i]) ** 0.5) )
            )
            self.biases.append(np.zeros(dimensions[i+1]))



    def forward(self, inp):
        mx = max(1, *inp)
        inp = list(i/mx for i in inp)

        results = [inp]
        for it in range(len(self.weights)):
            inp = np.dot(self.weights[it], inp)
            inp += self.biases[it]
            inp = list(max(0, i) for i in inp)
            results.append(inp)
        return results


    def backward(self, samples, expected):
        expected = list(np.array(i) for i in expected)
        d_weights = [i.copy() for i in self.weights]
        d_biases = [i.copy() for i in self.biases]

        sum_error = 0

        vecLen = len(expected[0])
        learningRate = 0.05 / len(samples)

        for samp in range(len(samples)):
            r1 = self.forward(samples[samp])
            result = r1[-1]
            delta = 0.0
            for i in range(vecLen):
                delta += (result[i]-expected[samp][i])**2
            delta /= vecLen

            sum_error += delta

            nextDiff = []
            diff = result-expected[samp]
            for it in range(len(self.biases)-1, -1, -1):
                nextDiff = diff
                diff = np.transpose(self.weights[it]) @ nextDiff
                diff = np.array(list((diff[i] if r1[it][i]>0.0001 else 0) for i in range(len(diff))))

                d_weights[it] -= np.outer(nextDiff, r1[it]) * learningRate
                d_biases[it] -= nextDiff * learningRate

        self.weights = d_weights
        self.biases = d_biases

        return sum_error


    def train(self, samples, expected, mxIter = 500, mnErr = 0.2):
        while True:
            if mxIter==0:
                break
            mxIter -= 1

            res = self.backward(samples, expected)

            print(f"iteration {mxIter}, error = {res}", flush=True)
            sys.stdout.flush()

            if res <= mnErr:
                break













net = Network()

if False:
    for i in range(len(net.weights)):
        print(len(net.weights[i][0]), len(net.weights[i]))

if True:
    tm0 = time.time()
    raw = [load_image_from_csv(index = i) for i in range(100)]
    tm1 = time.time()
    print(f"loading took {tm1-tm0} seconds")
    samples = [i[1] for i in raw]
    expected = [[(1.0 if i[0]==j else 0.0) for j in range(10)] for i in raw]

    net.train(samples, expected)

img = load_image_from_csv(index=99)
print(net.forward([i for i in img[1]])[-1])
print("before")
#show_image_from_array(img[1])
print("after")

# ---- Launch drawing window ----
from drawing import launch
launch(net)













































