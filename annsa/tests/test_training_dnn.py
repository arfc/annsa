from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes import (dnn_model_features,
                                 DNN)
from annsa.load_dataset import load_dataset

tf.enable_eager_execution()


def construct_dnn():
    """
    Constructs a dense neural network and tests construction
    functions.

    Returns:
    --------
    model_features : class dnn_model_features
        Contains all features of the DNN model

    optimizer :
    An Operation that updates the variables in var_list.
    If global_step was not None, that operation also increments
    global_step. See documentation for tf.train.Optimizer

    model : Class DNN
        A dense neural network
    """
    scaler = make_pipeline(FunctionTransformer(np.log1p, validate=False))

    model_features = dnn_model_features(
        learning_rate=1e-1,
        l2_regularization_scale=1e-1,
        dropout_probability=0.5,
        batch_size=2**5,
        output_size=2,
        dense_nodes=[100],
        output_function=None,
        activation_function=tf.nn.relu,
        scaler=scaler)

    optimizer = tf.train.AdamOptimizer(model_features.learning_rate)
    model = DNN(model_features)
    return model_features, optimizer, model


def test_dnn_construction():
    """
    Tests the construction of the dense neural network.
    """
    _, _, _ = construct_dnn()
    pass



#unit tests

#add a test for the following:

#forward_pass

#


#integration tests
def test_dnn_training():
    """
    Testing the dense neural network class and training function.
    """

    tf.reset_default_graph()
    model_features, optimizer, model = construct_dnn()
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



