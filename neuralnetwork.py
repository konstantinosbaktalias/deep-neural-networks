import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def mse_loss(y_true, pred):
    return np.square(y_true - pred).mean()

class NeuralNetwork:
    def __init__(self, params):
        self.W = self.init_w(params)
        self.b = self.init_b(params)

    def init_w(self, params):
        weights = []
        for i in range(1, len(params)):
            weights.append(np.random.randn(params[i], params[i-1]))
        return np.array(weights)

    def init_b(self, params):
        biases = []
        for i in range(1, len(params)):
            biases.append(np.random.randn(params[i]))
        return np.array(biases)

    # Feed forward
    def forward_prop(self, input):
        outputs = []
        z = np.dot(self.W[0], input) + self.b[0]
        a = sigmoid(z)
        outputs.append(a)
        for i in range(1, len(self.W)):
            z = np.dot(self.W[i], outputs[i-1]) + self.b[i]
            a = sigmoid(z)
            outputs.append(a)

        outputs = np.array(outputs)
        return outputs[-1]

    def back_prop(self, X, Y, epochs, rate, plot=False):
        costs_arr = []
        for epoch in range(epochs):
            costs = []
            # Looping throw the dataset
            for x_input, y_true in zip(X, Y):
                    # Feed forward
                    outputs = []
                    z = np.dot(self.W[0], x_input) + self.b[0]
                    a = sigmoid(z)
                    outputs.append(a)
                    for i in range(1, len(self.W)):
                        z = np.dot(self.W[i], outputs[i-1]) + self.b[i]
                        a = sigmoid(z)
                        outputs.append(a)

                    outputs = np.array(outputs)

                    # Get the cost of the prediction
                    cost =  y_true - outputs[-1]
                    costs.append(cost)

                    # Adjasting the weights and biases for higher accuracy
                    for i in reversed(range(1, len(self.W))):
                        d = (cost * outputs[i] * (1.0 - outputs[i]))
                        self.W[i] += np.dot(d.reshape(d.shape[0], 1), np.transpose(outputs[i-1].reshape(outputs[i-1].shape[0], 1))) * rate
                        self.b[i] += cost * outputs[i] * (1.0 - outputs[i]) * rate
                        cost = np.dot(self.W[i].T, cost)
                    
                    d = (cost * outputs[0] * (1.0 - outputs[0]))
                    self.W[0] += np.dot(d.reshape(d.shape[0], 1), np.transpose(x_input.reshape(x_input.shape[0], 1))) * rate
                    self.b[0] += cost * outputs[0] * (1.0 - outputs[0]) * rate

            costs_arr.append(mse_loss(y_true, outputs[-1]))    
            print(f'Epoch: {epoch+1} | Loss: {mse_loss(y_true, outputs[-1])}')    

        if plot:
            plt.title('Loss Rate')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid()
            plt.plot(costs_arr)
            plt.show()

    def save(self, filename):
        np.save(filename, [self.W, self.b])
        print('Model Saved')

    def load(self, filename):
        data = np.load(f'{filename}.npy', allow_pickle=True)
        self.W = data[0]
        self.b = data[1]
        print('Paramas were loaded')