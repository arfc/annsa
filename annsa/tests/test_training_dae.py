from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import tensorflow as tf
import annsa as an

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes import (dae_model_features,
                                 DAE)

tf.enable_eager_execution()


def load_dataset():

    """
    Generates dummy data using 'sklearn.datasets.make_classification()'. 
    See 'make_classification' documentation for more details.

    Returns:
    -------
    train_dataset : tuple of [train_data, training_keys_binarized]
        Contains the training data and the labels in a binarized
        format.
    test_dataset : tuple of [test_data, testing_keys_binarized]
        Contains the testing data and the labels in a binarized
        format.
    """
    training_dataset = make_classification(n_samples=100,
                                           n_features=1024,
                                           n_informative=200,
                                           n_classes=2)

    testing_dataset = make_classification(n_samples=100,
                                          n_features=1024,
                                          n_informative=200,
                                          n_classes=2)

    mlb = LabelBinarizer()

    training_data = np.abs(training_dataset[0])
    training_keys = training_dataset[1]
    training_keys_binarized = mlb.fit_transform(
        training_keys.reshape([training_data.shape[0], 1]))
    train_dataset = [training_data, training_data]

    testing_data = np.abs(testing_dataset[0])
    testing_keys = testing_dataset[1]
    testing_keys_binarized = mlb.transform(
        testing_keys.reshape([testing_data.shape[0], 1]))
    test_dataset = [testing_data, testing_data]

    return train_dataset, test_dataset


def construct_dae():

    """
    Constructs a dense autoencoder.

    Returns:
    --------
    model_features : class dae_model_features
        Contains all features of the DAE model

    optimizer : 
    An Operation that updates the variables in var_list. 
    If global_step was not None, that operation also increments
    global_step. See documentation for tf.train.Optimizer

    model : Class DAE
    """
    scaler = make_pipeline(FunctionTransformer(np.log1p, validate=False))
    model_features = dae_model_features(
        learning_rate=1e-1,
        l1_regularization_scale=1e-1,
        dropout_probability=0.5,
        batch_size=2**5,
        dense_nodes_encoder=[100],
        dense_nodes_decoder=[],
        scaler=scaler,
        activation_function=tf.nn.relu,
        output_size=1024,
        output_function=None)

    optimizer = tf.train.AdamOptimizer(model_features.learning_rate)
    model = DAE(model_features)
    return model_features, optimizer, model


def test_dae_construction():

    """
    Description
    """
    _, _, _ = construct_dae()
    pass


def test_dae_training():
    """
    Testing the dense neural network class and training function.
    """

    tf.reset_default_graph()
    model_features, optimizer, model = construct_dae()
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
        max_time=3600,
        obj_cost=model.mse,
        earlystop_cost_fn=model.mse,
        data_augmentation=model.default_data_augmentation,)
    pass
