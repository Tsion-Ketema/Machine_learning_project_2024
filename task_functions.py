from neuralNet import NeuralNetwork


def train_model(train_data, val_data, config, task_type):
    """
    Train a neural network for either CUP (regression) or MONK (classification).

    :param train_data: (X_train, Y_train)
    :param val_data: (X_val, Y_val)
    :param config: Hyperparameter combination
    :param task_type: 'regression' or 'classification'
    :return: Trained model
    """
    X_tr, Y_tr = train_data
    X_vl, Y_vl = val_data

    activations = config['activations']
    if isinstance(activations, str):  # Single activation for all layers
        activations = [activations] * (len(config['hidden_layer_sizes']) - 1)

    model = NeuralNetwork(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        learning_rate=config['learning_rates'],
        epochs=config['epochs'],
        momentum=config['momentum'],
        weight_initialization=config['weight_initialization'],
        regularization=config.get('regularization', ('none', 0)),
        activations=activations,
        task_type=task_type
    )
    model.train(X_tr, Y_tr, X_val=X_vl, y_val=Y_vl)
    # print("Final weights:", model.weights[0][:5])
    return model


def taskCup(train_data, val_data, config):
    return train_model(train_data, val_data, config, task_type='regression')


def taskMonk(train_data, val_data, config):
    return train_model(train_data, val_data, config, task_type='classification')
