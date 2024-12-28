import numpy as np


def sig(x: np.array):
    return 1/(1 + np.exp(-x))


def sigPrime(x: np.array):
    return sig(x) * (1-sig(x))


def softmax(x: np.array):
    # Clipping for numerical stability
    x = np.clip(x, -500, 500)
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


class Layer:
    def __init__(self, outputSize: int, inputSize: int, activation="sigmoid"):
        self.size = outputSize
        self.weights = np.random.randn(outputSize, inputSize)
        self.biases = np.random.randn(outputSize, 1)
        self.z = np.zeros(shape=(outputSize, 1))
        self.activation = activation

    def forward(self, inputVector: np.array):
        self.z = (self.weights @ inputVector) + self.biases
        return self.act(self.z)

    def act(self, x: np.array):
        if self.activation == "sigmoid":
            return sig(x)
        elif self.activation == "relu":
            return np.where(x < 0, 0.0, x)
        elif self.activation == "softmax":
            return softmax(x)

    def actPrime(self, x: np.array):
        if self.activation == "sigmoid":
            return sigPrime(x)
        elif self.activation == "relu":
            return np.where(x > 0, 1.0, 0.0)


class MLP:
    def __init__(self, hiddenLayersInput: list[Layer], learnRate: float, lossFunction="mse"):
        self.hiddenLayers: list[Layer] = []
        for i in range(len(hiddenLayersInput) - 1):
            self.hiddenLayers.append(hiddenLayersInput[i])
        self.outputLayer = hiddenLayersInput[-1]
        self.learnRate = learnRate
        self.isTrained = False
        self.lossFunction = lossFunction

    def forwardPass(self, inputVector: np.array):
        outputVector = inputVector
        for layer in self.hiddenLayers:
            outputVector = layer.forward(outputVector)

        return self.outputLayer.forward(outputVector)

    def backProp(self, x, yActual):
        yPred = self.forwardPass(x)
        gradient = None

        if self.lossFunction != "ce":
            gradient = yPred - yActual
            # δ^L = ∇_(a^L)C ⊙ a'^(L)
            delta = gradient * self.outputLayer.actPrime(self.outputLayer.z)
        else:
            delta = yPred - yActual

        # w^L ← w^L - β(δ^L)(a^(L-1))^T
        self.outputLayer.weights -= self.learnRate * (delta @ self.hiddenLayers[-1].act(self.hiddenLayers[-1].z).T)

        # b^{L}_j ← b^{L}_j - β(δ^{L}_j)
        self.outputLayer.biases -= self.learnRate * delta

        for layer in range(len(self.hiddenLayers) - 1, -1, -1):
            # δ^l = a'^(l) ⊙ ((w^(l+1))^T δ^(l+1))
            if layer == len(self.hiddenLayers) - 1:
                delta = self.hiddenLayers[layer].actPrime(self.hiddenLayers[layer].z) * (self.outputLayer.weights.T @ delta)
            else:
                delta = self.hiddenLayers[layer].actPrime(self.hiddenLayers[layer].z) * (self.hiddenLayers[layer+1].weights.T @ delta)

            # w^l ← w^l - β(δ^l)[a^(l-1)]^T
            if layer != 0:
                self.hiddenLayers[layer].weights -= self.learnRate * (delta @ self.hiddenLayers[layer - 1].act(self.hiddenLayers[layer-1].z).T)
            else:
                self.hiddenLayers[layer].weights -= self.learnRate * (delta @ x.T)

            # b^{l}_j ← b^{l}_j - β(δ^{l}_j)
            self.hiddenLayers[layer].biases -= self.learnRate * delta

    def train(self, xTrain, yTrain, epochs: int):
        for _ in range(epochs):
            for i in range(len(xTrain)):
                x = xTrain[i].reshape(-1, 1)
                y = yTrain[i].reshape(-1, 1)
                self.backProp(x, y)

        self.isTrained = True

    def predict(self, x: np.array):
        if not self.isTrained:
            return -1
        else:
            return self.forwardPass(x.reshape(-1,1))

    def test(self, xTest: np.array, yTest: np.array):
        n = len(xTest)
        correct = 0
        for i in range(n):
            prediction = self.predict(xTest[i])
            if yTest[i][np.argmax(prediction)] == 1:
                correct += 1
        return correct/n

    def pruneNode(self, layerIdx, nodeIdx):
        self.hiddenLayers[layerIdx].weights = np.delete(self.hiddenLayers[layerIdx].weights, nodeIdx, axis=0)
        self.hiddenLayers[layerIdx].biases = np.delete(self.hiddenLayers[layerIdx].biases, nodeIdx, axis=0)
        if layerIdx == len(self.hiddenLayers)-1:
            self.outputLayer.weights = np.delete(self.outputLayer.weights, nodeIdx, axis=1)
        else:
            self.hiddenLayers[layerIdx + 1].weights = np.delete(self.hiddenLayers[layerIdx + 1].weights, nodeIdx, axis=1)
