__author__ = 'anmsingh'

import mnist_loader
import numpy as np
from Network1 import Network as Net

net=Net([784,210,10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net.SGD(training_data, 30, 20, 3.0, test_data=test_data)



