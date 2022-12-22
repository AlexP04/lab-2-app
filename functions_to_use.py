import numpy as np

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

tanh_list = ['\tanh', '\arth', tanh]