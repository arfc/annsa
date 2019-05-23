from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import tensorflow as tf
import annsa as an

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes import (dnn_model_features,
                                 DNN)

tf.enable_eager_execution()


def load_dataset():
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
    train_dataset = [training_data, training_keys_binarized]

    testing_data = np.abs(testing_dataset[0])
    testing_keys = testing_dataset[1]
    testing_keys_binarized = mlb.transform(
        testing_keys.reshape([testing_data.shape[0], 1]))
    test_dataset = [testing_data, testing_keys_binarized]

    return train_dataset, test_dataset


def construct_dnn():
    scaler = make_pipeline(FunctionTransformer(np.log1p, validate=False))

    model_features = dnn_model_features(
        learning_rate=1e-1,
        l2_regularization_scale=1e-1,
        dropout_probability=0.5,
        batch_size=2**5,
        output_size=2,
        dense_nodes=[100],
        activation_function=tf.nn.relu,
        scaler=scaler)

    optimizer = tf.train.AdamOptimizer(model_features.learning_rate)
    model = DNN(model_features)
    return model_features, optimizer, model


def test_dnn_construction():
    _, _, _ = construct_dnn()
    pass


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
        max_time=3600,
        obj_cost=model.cross_entropy,
        earlystop_cost_fn=model.f1_error,
        data_augmentation=model.default_data_augmentation,)
    pass
