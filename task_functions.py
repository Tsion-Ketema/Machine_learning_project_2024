
from neuralNet import NeuralNetwork


def train_model(train_data, val_data, config, task_type, dataset_name=None, **kwargs):
    """
    Train a neural network for either CUP (regression) or MONK (classification).

    :param train_data: (X_train, Y_train)
    :param val_data: (X_val, Y_val)
    :param config: Hyperparameter combination
    :param task_type: 'regression' or 'classification'
    :param dataset_name: Name of the dataset (used for detecting regularization type dynamically)
    :param kwargs: Additional dynamic parameters (e.g., dropout_rate, batch_size)
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

    # Instantiate the model with dynamic regularization and additional parameters
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

    # Apply additional parameters (e.g., dropout_rate, batch_size) if provided
    model.dropout_rate = kwargs.get('dropout_rate', 0.0)  # Default: no dropout
    # Default: full batch training
    model.batch_size = kwargs.get('batch_size', None)

    # Train the model
    model.train(X_tr, Y_tr, X_val=X_vl, y_val=Y_vl)
    return model


def taskMonk(train_data, val_data, config, dataset_name, **kwargs):
    """
    Task-specific function for Monk datasets with dynamic updates for Monk-3.
    :param train_data: (X_train, Y_train)
    :param val_data: (X_val, Y_val)
    :param config: Hyperparameter combination
    :param dataset_name: Dataset name for identifying Monk-3
    :param kwargs: Additional parameters for Monk-3
    """
    # Apply dynamic updates for Monk-3 with L2 regularization
    if dataset_name == 'monks-3-l2':
        kwargs['dropout_rate'] = 0.2  # Apply dropout
        kwargs['batch_size'] = 32    # Use batch size

    # Pass dataset_name and additional parameters to train_model
    return train_model(train_data, val_data, config, task_type='classification', dataset_name=dataset_name, **kwargs)


def taskCup(train_data, val_data, config, dataset_name=None):
    # Do not dynamically modify config for CUP
    return train_model(train_data, val_data, config, task_type='regression', dataset_name=dataset_name)
