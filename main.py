import math
import sys
import time
from random import random, randint

import pandas as pd
import numpy as np

from show import *


def printWH(arr):
    print(len(arr[0]), len(arr))




class Network:


    def __init__(self, dimensions = (784, 128, 64, 10)):
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
        inp = list(i/256 for i in inp)

        results = [inp]
        for it in range(len(self.weights)):
            inp = np.dot(self.weights[it], inp)
            inp += self.biases[it]
            if(it+1!=len(self.weights)):
                inp = list(max(0, i) for i in inp)
            results.append(inp)
        return results


    def backward(self, samples, expected, learningRate = 0.2, weightDecay = 0.0005):
        weightDecay = 1-weightDecay
        expected = list(np.array(i) for i in expected)
        d_weights = [i.copy()*weightDecay for i in self.weights]
        d_biases = [i.copy()*weightDecay for i in self.biases]

        sum_error = 0

        vecLen = len(expected[0])
        learningRate /= len(samples)

        for samp in range(len(samples)):
            r1 = self.forward(samples[samp])
            result = r1[-1]
            softmax = [0]*vecLen

            sm = sum(math.exp(i) for i in result)
            for i in range(vecLen):
                softmax[i] = math.exp(result[i])/sm

            delta = 0.0
            for i in range(vecLen):
                delta += -math.log(softmax[i])*expected[samp][i]
            #print(delta, softmax)

            sum_error += delta

            nextDiff = []
            # dL/dy = dL/dSoftMax * dSoftMax/dy
            # dL/dSoftmax = -expected/softmax
            # dSoftmax/dy = e^y*(sm-1)/sm^2
            diff = np.array([ (softmax[i]-expected[samp][i])
                    for i in range(vecLen)])
            for it in range(len(self.biases)-1, -1, -1):
                nextDiff = diff
                diff = np.transpose(self.weights[it]) @ nextDiff
                diff = np.array(list((diff[i] if r1[it][i]>0.0001 else 0) for i in range(len(diff))))

                d_weights[it] -= np.outer(nextDiff, r1[it]) * learningRate
                d_biases[it] -= nextDiff * learningRate

        self.weights = d_weights
        self.biases = d_biases

        return sum_error


    def train(self, samples, expected, mxIter = 500, avErr = 0.005):

        samplesPerIteration = len(samples)

        completionCoeff = 1
        samplesCoeff = 1
        learningCoeff = 0.2

        while True:
            if mxIter==0:
                break
            mxIter -= 1
            currentSamples = []
            currentExpected = []

            pool = list(range(len(samples)))
            for i in range(samplesPerIteration):
                exx = randint(0,len(pool)-1)
                currentSamples.append(samples[pool[exx]])
                currentExpected.append(expected[pool[exx]])
                pool[exx] = pool[-1]
                pool.pop()

            res = self.backward(currentSamples, currentExpected, learningRate=0.2*learningCoeff)
            res /= len(currentSamples)


            if True:
                completionCoeff = min(1.0, max(0.2, 0.1/res))

                samplesCoeff = completionCoeff
                learningCoeff = 1

                samplesPerIteration = int(len(samples)*samplesCoeff)

            print(f"===========================================================\n"
                  f"iteration {mxIter}, average error = {res} (target: {avErr})\n"
                  f"using {samplesPerIteration} samples, step: {learningCoeff}\n"
                  f"{net.getInfo()}",
                  flush=True)
            sys.stdout.flush()

            if res <= avErr:
                break

    def getInfo(self):
        maxWeights = []
        for i in self.weights:
            maxWeights.append(max(max([abs(k) for k in j]) for j in i))
        maxBiases = []
        for i in self.biases:
            maxBiases.append(max(abs(j) for j in i))

        return (f"maxWeights = {', '.join(map(str, maxWeights))}\n"
                f"maxBiases  = {', '.join(map(str, maxBiases))}")











net = Network()

if False:
    for i in range(len(net.weights)):
        print(len(net.weights[i][0]), len(net.weights[i]))

if True:
    tm0 = time.time()
    raw = load_images_from_csv()


    toUse = 2000
    samples = raw[1][0:toUse]
    labels = list(raw[0][0:toUse])
    if True:
        samples = samples+getRandomOffset(getRandomRots(getRandomScale(samples)))
        labels = labels+labels
    print(len(samples), len(labels))
    expected = [[(1.0 if i==j else 0.0) for j in range(10)] for i in labels]


    tm1 = time.time()
    print(f"loading and processing took {tm1-tm0} seconds")

    print(net.getInfo())
    net.train(samples, expected)
    print(net.getInfo())

    # ---- Launch drawing window ----
    from drawing import launch
    launch(net)

imgs = load_images_from_csv(mx = 100)
imgs = imgs[0], getRandomOffset(getRandomRots(getRandomScale(imgs[1])))
print(net.forward(imgs[1][0])[-1])
print("before")
#show_image_from_array([imgs[1][i] for i in range(100) if imgs[0][i]==9])
print("after")















































