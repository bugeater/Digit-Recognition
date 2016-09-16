__author__ = 'anmsingh'

import numpy as np
import random
import os

class Network:
    """
    A Neural Network
    """

    def __init__(self,LayerSize):
        """
        Initialize the Network
        """

        # Layer Info
        self.layerCount=len(LayerSize) - 1
        self.shape=LayerSize

        # I/O Layers
        self.InputLayer=[]
        self.OuputLayer=[]

        # Initialising the Weight Array
        self.weights=[]
        # If Computing for the First Time
        if(os.stat("ComputedWeights_0.txt").st_size == 0):

            for (l1,l2) in zip(LayerSize[:-1],LayerSize[1:]):
                self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))
        else:
            # Load it from the Files
            for index in range(self.layerCount):
                filename = "ComputedWeights_" + str(index) + ".txt"
                self.weights.append(np.loadtxt(filename))


    # Run Method
    def Run(self,input):
        """ Run the Netwotk bsaed on Input Data """

        self.InputLayer=[]
        self.OutputLayer=[]

        for index in range(self.layerCount):
            if index == 0:
                layerinput = self.weights[0].dot(np.vstack([np.ones([1,1]),input]))
            else:
                layerinput = self.weights[index].dot(np.vstack([np.ones([1,1]),self.OutputLayer[-1]]))
            self.InputLayer.append(layerinput)
            self.OutputLayer.append(self.sigmoid(layerinput))

        return self.OutputLayer[-1]

    # Stochastic Gradient Method
    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
            # Writing the computed weights to a file
            self.Write_Computed_Weights_to_file()


    def update_mini_batch(self, mini_batch, eta):

        theta = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_theta= self.TrainEpoch(x, y)
            theta = [nw+dnw for nw, dnw in zip(theta, delta_theta)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights,theta)]

    # TrainEpoch Method
    def TrainEpoch(self,input,target):
        """ Trains The Network For One Epoch """

        delta = []

        # Forward Propagation
        self.Run(input)

        # Compute Deltas
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                ouput_delta = self.OutputLayer[index] - target
                delta.append(ouput_delta * self.sigmoid(self.InputLayer[index],True))
            else:
                delta_pullback = self.weights[index+1].T.dot(delta[-1])
                delta.append(delta_pullback[1:] * self.sigmoid(self.InputLayer[index],True))


        # Compute weight Deltas

        delta_weight = [np.zeros(weight.shape) for weight in self.weights]

        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index
            if index == 0:
                layerOutput = np.vstack([np.ones([1,1]),input])
            else:
                layerOutput = np.vstack([np.ones([1,1]),self.OutputLayer[index]])
            delta_weight[index] = delta[delta_index]*layerOutput.T

            return delta_weight

    # Evaluate Answers for Test data
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.Run(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    # Transfer Function
    def sigmoid(self,x,Derivative=False):
        if not Derivative:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            temp=self.sigmoid(x)
            return temp*(1-temp)


    # Writing Computed Weights To a File
    def Write_Computed_Weights_to_file(self):

        for x in range(self.layerCount):
            filename = "ComputedWeights_" + str(x) + ".txt"
            f=open(filename,"w")
            np.savetxt(filename,self.weights[x],delimiter=' ')
            f.close()



