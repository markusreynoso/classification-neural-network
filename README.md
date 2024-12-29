# classification-neural-network

## The Dataset

## Layer object

### Constructor

Initializes a `Layer` object with the following parameters:
| Parameter    | Type   | Default  | Description                                         |
|--------------|--------|----------|-----------------------------------------------------|
| `outputSize`  | `int`  | Required | Number of output features to the layer.             |
| `inputSize` | `int`  | Required | Number of input features from the layer.          |
| `activation` | `str`  | `'sigmoid'` | Activation function to use (`'relu'`, `'sigmoid'`).|

## The Model

Initializes an `MLP` object with the following parameters:
| Parameter    | Type   | Default  | Description                                         |
|--------------|--------|----------|-----------------------------------------------------|
| `hiddenLayersInput`  | `list[Layer]`  | Required | The layers of the model |
| `learnRate` | `float`  | Required | The learning rate of the model |
| `lossFunction` | `str`  | `'mse'` | Loss function to use (`'mse'`, `'ce'`).|

Note: For binary classification tasks, `'mse'` may be used as the resulting gradient ends up being equivalent.
