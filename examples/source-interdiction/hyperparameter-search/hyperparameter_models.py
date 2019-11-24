from random import choice
import numpy as np
from annsa.model_classes import (dnn_model_features,
                                 cnn1d_model_features,
                                 cae_model_features,
                                 DNN,
                                 CNN1D,
                                 CAE,
                                 )
import tensorflow as tf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer

scaler_choices = [make_pipeline(FunctionTransformer(np.log1p, validate=True)),
                  make_pipeline(FunctionTransformer(np.log1p, validate=True),
                                Normalizer(norm='l1')),
                  make_pipeline(FunctionTransformer(np.log1p, validate=True),
                                Normalizer(norm='max')),
                  make_pipeline(FunctionTransformer(np.sqrt, validate=True)),
                  make_pipeline(FunctionTransformer(np.sqrt, validate=True),
                                Normalizer(norm='l1')),
                  make_pipeline(FunctionTransformer(np.sqrt, validate=True),
                                Normalizer(norm='max')), ]


def make_dense_model(all_keys_binarized):
    """
    Makes a random dense model given some parameters.

    Parameters:
    -----------
        all_keys_binarized : list, bool
            List of binarized keys
    Returns:
    --------
        model : object
            A keras model of a dense network.
        model_features : class
            Class that describes the structure of a the dense network
    """
    number_layers = choice([1, 2, 3])
    dense_nodes = 2**np.random.randint(5, 10, number_layers)
    dense_nodes = np.sort(dense_nodes)
    dense_nodes = np.flipud(dense_nodes)
    model_features = dnn_model_features(
        learining_rate=10**np.random.uniform(-4, -1),
        l2_regularization_scale=10**np.random.uniform(-2, 0),
        dropout_probability=np.random.uniform(0, 1),
        batch_size=2**np.random.randint(4, 10),
        output_size=all_keys_binarized.shape[1],
        dense_nodes=dense_nodes,
        activation_function=choice([tf.nn.tanh, tf.nn.relu]),
        output_function=None,
        scaler=choice(scaler_choices))

    model = DNN(model_features)

    return model, model_features


def generate_random_cnn1d_architecture(cnn_filters_choices,
                                       cnn_kernel_choices,
                                       pool_size_choices):
    """
    Makes a random convolutional model features given some parameters.

    Parameters:
    -----------
        cnn_filters_choices : list, int
            List of number of filter to use for each convolutional layer.
        cnn_kernel_choices : list, int
            List of filter lengths to use for each convolutional layer.
        pool_size_choices : list, int
            List of pooling lengths to use for each convolutional layer.

    Returns:
    --------
        model_features : class
            Class that describes the structure of a the 1D convolutional
            network
    """

    cnn_filters = choice(cnn_filters_choices)
    cnn_kernel_choice = choice(cnn_kernel_choices)
    pool_size_choice = choice(pool_size_choices)

    cnn_kernel = cnn_kernel_choice * (len(cnn_filters))
    cnn_strides = (1,) * (len(cnn_filters))
    pool_size = pool_size_choice * (len(cnn_filters))
    pool_strides = (2,) * (len(cnn_filters))

    number_layers = np.random.randint(1, 4)
    dense_nodes = (10 ** np.random.uniform(
        1,
        np.log10(1024 / (2 ** len(cnn_filters))),
        number_layers)).astype('int')
    dense_nodes = np.sort(dense_nodes)
    dense_nodes = np.flipud(dense_nodes)

    model_features = cnn1d_model_features(
        trainable=None,
        learining_rate=None,
        batch_size=None,
        output_size=None,
        scaler=None,
        activation_function=None,
        output_function=None,
        Pooling=None,
        l2_regularization_scale=None,
        dropout_probability=None,
        cnn_filters=cnn_filters,
        cnn_kernel=cnn_kernel,
        cnn_strides=cnn_strides,
        pool_size=pool_size,
        pool_strides=pool_strides,
        dense_nodes=dense_nodes
    )

    return model_features


def make_conv1d_model(all_keys_binarized):
    """
    Makes a random convolutional model and model features using
    predefined parameters.

    Parameters:
    -----------
        all_keys_binarized : list, bool
            List of binarized keys

    Returns:
    --------
    model : object
        A keras model of a convolutional network.
    model_features : class
        Class that describes the structure of a the convolutional network
    """

    cnn_filters_choices = (
        (4, 8),
        (8, 16),
        (16, 32),
        (4,),
        (8,),
        (16,),
        (32,),
        (4, 8, 16),
        (8, 16, 32),
    )

    cnn_kernel_choices = ((2,), (4,), (8,), (16,))
    pool_size_choices = ((2,), (4,), (8,), (16,))

    model_features = generate_random_cnn1d_architecture(
        cnn_filters_choices=cnn_filters_choices,
        cnn_kernel_choices=cnn_kernel_choices,
        pool_size_choices=pool_size_choices,
    )
    model_features.trainable = True
    model_features.learining_rate = 10 ** np.random.uniform(-4, -1)
    model_features.batch_size = 2 ** np.random.randint(4, 6)
    model_features.output_size = all_keys_binarized.shape[1]
    model_features.scaler = choice(scaler_choices)

    model_features.activation_function = tf.nn.relu
    model_features.output_function = None
    model_features.Pooling = tf.layers.MaxPooling1D
    model_features.l2_regularization_scale = 10 ** np.random.uniform(-3, 0)
    model_features.dropout_probability = np.random.uniform(0, 1)
    model_features.pool_strides = ((2, 2, 2))
    number_layers = choice([1, 2, 3])
    dense_nodes = 2 ** np.random.randint(4, 8, number_layers)
    dense_nodes = np.sort(dense_nodes)
    dense_nodes = np.flipud(dense_nodes)
    model_features.dense_nodes = dense_nodes

    model = CNN1D(model_features)

    return model, model_features


def make_cae1d_model():
    """
    Makes a random convolutional model and model features using
    predefined parameters.

    Parameters:
    -----------
        None

    Returns:
    --------
    model : object
        A keras model of a convolutional autoencoder network.
    model_features : class
        Class that describes the structure of a the convolutional autoencoder
        network.
    """

    cnn_filters_encoder_choice = choice([(4, 1),
                                         (64, 1),
                                         (4, 8, 1),
                                         (32, 64, 1),
                                         (4, 8, 16, 1),
                                         (16, 32, 64, 1),
                                         ]
                                        )

    cnn_kernel_encoder_choice = choice([(2, ), (4, ), (8, ), (16, )])
    pool_size_choice = choice([(2, ), (4, ), (8, ), (16, )])

    scaler_choice = choice(scaler_choices)

    num_cnn_filters = len(cnn_filters_encoder_choice)

    model_features = cae_model_features(
        encoder_trainable=True,
        learning_rate=10 ** np.random.uniform(-4, -1),
        batch_size=2 ** np.random.randint(4, 6),
        scaler=scaler_choice,
        activation_function=tf.nn.tanh,
        output_function=None,
        Pooling=tf.layers.MaxPooling1D,
        cnn_filters_encoder=cnn_filters_encoder_choice,
        cnn_kernel_encoder=(cnn_kernel_encoder_choice,) * num_cnn_filters,
        cnn_strides_encoder=(1, ) * num_cnn_filters,
        pool_size_encoder=pool_size_choice * num_cnn_filters,
        pool_strides_encoder=(2, ) * num_cnn_filters,
        cnn_filters_decoder=cnn_filters_encoder_choice,
        cnn_kernel_decoder=(cnn_kernel_encoder_choice,) * num_cnn_filters,
        cnn_strides_decoder=(1, ) * num_cnn_filters)

    model = CAE(model_features)

    return model, model_features
