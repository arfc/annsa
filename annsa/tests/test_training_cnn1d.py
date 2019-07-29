from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes import (generate_random_cnn1d_architecture,
                                 CNN1D)
from annsa.load_dataset import load_dataset

tf.enable_eager_execution()


def construct_cnn1d():
    """
    Constructs a convolutional neural network and tests construction
    functions.

    Returns:
    --------
    model_features : class cnn1d_model_features
        Contains all features of the CNN1D model

    optimizer :
    An Operation that updates the variables in var_list.
    If global_step was not None, that operation also increments
    global_step. See documentation for tf.train.Optimizer

    model : Class CNN1D
        A convolution neural network for finding one dimensional
        features.
    """
    scaler = make_pipeline(FunctionTransformer(np.log1p, validate=False))

    cnn_filters_choices = ((4, 1), (8, 1))  # choose either 4x1 or 8x1 filter
    cnn_kernel_choices = ((8, ), (4, ))  # choose either 8xn or 4xn kernel size
    pool_size_choices = ((8, ), (4, ))
    model_features = generate_random_cnn1d_architecture(cnn_filters_choices,
                                                        cnn_kernel_choices,
                                                        pool_size_choices,)
    model_features.learning_rate = 1e-1
    model_features.trainable = True
    model_features.batch_size = 2**5
    model_features.output_size = 2
    model_features.output_function = None
    model_features.l2_regularization_scale = 1e-1
    model_features.dropout_probability = 0.5
    model_features.scaler = scaler
    model_features.Pooling = tf.layers.MaxPooling1D
    model_features.activation_function = tf.nn.relu

    optimizer = tf.train.AdamOptimizer(model_features.learning_rate)
    model = CNN1D(model_features)
    return model_features, optimizer, model


def test_cnn1d_construction():
    """
    Tests the construction of a convolution neural network.

    Returns: Nothing
    """
    _, _, _ = construct_cnn1d()
    pass


def test_cnn1d_training():
    """
    Testing the convolutional neural network class and training function.
    Returns : Nothing
    """

    tf.reset_default_graph()
    model_features, optimizer, model = construct_cnn1d()
    train_dataset, test_dataset = load_dataset()
    model_features.scaler.fit(train_dataset[0])

    all_loss_train, all_loss_test = model.fit_batch(
        train_dataset,
        test_dataset,
        optimizer,
        num_epochs=1,
        earlystop_patience=0,
        verbose=1,
        print_errors=0,
        obj_cost=model.cross_entropy,
        earlystop_cost_fn=model.f1_error,
        data_augmentation=model.default_data_augmentation,)
    pass

def test_forward_pass():
    """
    Tests that the network is convolving and/or learning.
    """
    pass


def test_fit_batch():

    tf.reset_default_graph()
    model_features, optimizer, model = construct_cnn1d()
    pass


def test_train_earlystop():
    tf.reset_default_graph()
    pass
