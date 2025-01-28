from neuralNet import NeuralNetwork


# def train_model(train_data, val_data, config, task_type):
#     """
#     Train a neural network for either CUP (regression) or MONK (classification).

#     :param train_data: (X_train, Y_train)
#     :param val_data: (X_val, Y_val)
#     :param config: Hyperparameter combination
#     :param task_type: 'regression' or 'classification'
#     :return: Trained model
#     """
#     X_tr, Y_tr = train_data
#     X_vl, Y_vl = val_data

#     activations = config['activations']
#     if isinstance(activations, str):  # Single activation for all layers
#         activations = [activations] * (len(config['hidden_layer_sizes']) - 1)

#     model = NeuralNetwork(
#         hidden_layer_sizes=config['hidden_layer_sizes'],
#         learning_rate=config['learning_rates'],
#         epochs=config['epochs'],
#         momentum=config['momentum'],
#         weight_initialization=config['weight_initialization'],
#         regularization=config.get('regularization', ('none', 0)),
#         activations=activations,
#         task_type=task_type
#     )
#     model.train(X_tr, Y_tr, X_val=X_vl, y_val=Y_vl)
#     return model


# def taskCup(train_data, val_data, config):
#     return train_model(train_data, val_data, config, task_type='regression')


# def taskMonk(train_data, val_data, config):
#     return train_model(train_data, val_data, config, task_type='classification')


def train_model(train_data, val_data, config, task_type, dataset_name=None):
    """
    Train a neural network for either CUP (regression) or MONK (classification).

    :param train_data: (X_train, Y_train)
    :param val_data: (X_val, Y_val)
    :param config: Hyperparameter combination
    :param task_type: 'regression' or 'classification'
    :param dataset_name: Name of the dataset (used for detecting regularization type dynamically)
    :return: Trained model
    """
    X_tr, Y_tr = train_data
    X_vl, Y_vl = val_data

    # Detect regularization dynamically based on dataset_name
    if dataset_name:
        if 'l1' in dataset_name.lower():
            regularization = ('L1', config.get('regularization', (0,))[1])
        elif 'l2' in dataset_name.lower():
            regularization = ('L2', config.get('regularization', (0,))[1])
        else:  # Default to no regularization for 'no-reg' or other cases
            regularization = ('none', 0)
    else:
        regularization = config.get('regularization', ('none', 0))  # Fallback

    # Ensure activations are correctly provided
    activations = config.get('activations', None)
    if activations is None:  # Default activations based on task type
        activations = ['relu'] * (len(config['hidden_layer_sizes']) - 2) + \
            (['sigmoid'] if task_type == 'classification' else ['identity'])

    # Instantiate the model with dynamic regularization
    model = NeuralNetwork(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        learning_rate=config['learning_rates'],
        epochs=config['epochs'],
        momentum=config['momentum'],
        weight_initialization=config['weight_initialization'],
        regularization=regularization,  # Pass dynamically determined regularization
        activations=activations,
        task_type=task_type
    )

    # Train the model
    model.train(X_tr, Y_tr, X_val=X_vl, y_val=Y_vl)
    return model


def taskMonk(train_data, val_data, config, dataset_name=None):
    # Pass dataset_name to train_model for dynamic handling of regularization
    return train_model(train_data, val_data, config, task_type='classification', dataset_name=dataset_name)


def taskCup(train_data, val_data, config, dataset_name=None):
    return train_model(train_data, val_data, config, task_type='regression', dataset_name=dataset_name)
