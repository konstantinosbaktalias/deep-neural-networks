import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(z, alpha=1):
    return (1 / (1+np.exp(-z))) * alpha

def relu(z, alpha=1):
    z[z < 0] = 0. 
    return z * alpha

def leaky_relu(z, alpha=1):
    z = np.where(z > 0, z, z * 0.01) 
    return z * alpha

def tanh(z, alpha=1):
    return (np.tanh(z)) * alpha

def softmax(z, alpha=1):
    return (np.exp(z) / np.sum(np.exp(z))) * alpha

# Derivatives of activations functions
def sigmoid_prime(a, alpha=1):
    return (a * (1. - a)) *  alpha

def relu_prime(a, alpha=1):
    a[a > 0] = 1. 
    return a * alpha

def leaky_relu_prime(a, alpha=1):
    a[a > 0] = 1.
    a = np.where(a > 0, a, 0.01)
    return a * alpha

def tahn_prime(a, alpha=1):
    return (1 - a**2) * alpha

def softmax_prime(a, alpha=1):
    s = a.reshape(-1, 1)
    j = (np.diagflat(s) - np.dot(s, s.T)).sum(axis=1)
    return j.reshape(j.shape[0], 1) * alpha

# Loss functions
def cross_entropy_loss(pred):
    return -np.log(pred)

def mse_loss(y_true, pred):
    return np.square(y_true - pred).mean()

class NeuralNetwork:
    def __init__(self, params: list):
        self.params = params

        self.weights: list = []
        self.biases: list = []

        self.initialize_params()

    def initialize_params(self):
        for layer_param in self.params:
            # Initialize layer's weights
            self.weights.append(
                np.random.randn(layer_param['neurons_size'], layer_param['input_size']) 
            )
            # Initialize layer's biases
            self.biases.append(np.zeros((layer_param['neurons_size'], 1)))

    def forward_propagation(self, input, back_prop=False):
        outputs: list = []
        
        for i in range(len(self.params)):
            # Getting the layer's input 
            input = outputs[i-1] if len(outputs) > 0 else input
           
            if self.params[i]['activation'] == 'sigmoid':
                activation_func = sigmoid
            elif self.params[i]['activation'] == 'relu':
                activation_func = relu
            elif self.params[i]['activation'] == 'leaky_relu':
                activation_func = leaky_relu
            elif self.params[i]['activation'] == 'tanh':
                activation_func = tanh
            elif self.params[i]['activation'] == 'softmax':
                activation_func = softmax
            else:
                raise Exception(f'Invalid activation function on layer {i+1}!')

            z = self.weights[i] @ input + self.biases[i]
            outputs.append(activation_func(z, 1 if 'activation_alpha' not in self.params[i] else self.params[i]['activation_alpha']))

        if back_prop:
            return outputs
        else:
            return outputs[-1]

    def back_prop(self, X, Y, epochs, learn_rate, loss_func, plot=True):
        total_losses: list = []
        for epoch in range(epochs):
            epoch_losses: float = 0
            for input, y_true in zip(X, Y):
                outputs = self.forward_propagation(input, back_prop=True)

                target = np.argmax(y_true)
                pred = outputs[-1].copy()

                if loss_func == 'MeanSqueredError':
                    epoch_losses += mse_loss(y_true, outputs[-1])
                elif loss_func == 'CrossEntropyLoss':
                    epoch_losses += cross_entropy_loss(pred[target])
                else:
                    raise Exception('Invalid loss funtion!')

                error = pred
                error[target] -= 1.

                for i in reversed(range(len(self.params))):
                    if self.params[i]['activation'] == 'sigmoid':
                        activation_der = sigmoid_prime
                    elif self.params[i]['activation'] == 'relu':
                        activation_der = relu_prime
                    elif self.params[i]['activation'] == 'leaky_relu':
                        activation_der = leaky_relu_prime
                    elif self.params[i]['activation'] == 'tahn':
                        activation_der = tahn_prime
                    elif self.params[i]['activation'] == 'softmax':
                        activation_der = softmax_prime

                    grad = error * activation_der(outputs[i], 1 if 'activation_alpha' not in self.params[i] else self.params[i]['activation_alpha'])

                    if epoch == 0:
                        self.last_grad = grad

                    layer_input = outputs[i-1] if i > 0 else input

                    self.weights[i] -= grad @ layer_input.T * learn_rate
                    self.biases[i] -= grad * learn_rate

                    error = self.weights[i].T @ error

            epoch_avg_loss = epoch_losses / len(X) 
            total_losses.append(epoch_avg_loss)

            print(f'Epoch: #{epoch+1}/{epochs} \n Loss: {epoch_avg_loss}')    

        if plot:
            plt.title('Loss Rate')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid()
            plt.plot(total_losses)
            plt.show()

    def save(self, filename):
        np.save(filename, [self.weights, self.biases])
        print('Model params were saved!')

    def load(self, filename):
        data = np.load(f'{filename}.npy', allow_pickle=True)
        self.weights = data[0]
        self.biases = data[1]
        print('Paramas were loaded!')