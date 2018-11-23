from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes import dnn_model_features, dnn

tf.enable_eager_execution()


def load_dataset():
    training_dataset = make_classification(n_samples=100,
                                           n_features=1024,
                                           n_informative=200,
                                           n_classes=29)

    testing_dataset = make_classification(n_samples=100,
                                          n_features=1024,
                                          n_informative=200,
                                          n_classes=29)

    return training_dataset, testing_dataset


def test_dnn_training():
    """
    Testing the dense neural network class and training function.
    """
    scaler = make_pipeline(FunctionTransformer(np.log1p, validate=False))
    mlb = LabelBinarizer()

    training_dataset, testing_dataset = load_dataset()
    training_data = np.abs(training_dataset[0])
    training_keys = training_dataset[1]
    training_keys_binarized = mlb.fit_transform(
        training_keys.reshape([training_data.shape[0], 1]))

    testing_data = np.abs(testing_dataset[0])
    testing_keys = testing_dataset[1]
    testing_keys_binarized = mlb.transform(
        testing_keys.reshape([testing_data.shape[0], 1]))

    model_features = dnn_model_features(
        learining_rate=1e-1,
        l2_regularization_scale=1e-1,
        dropout_probability=0.5,
        batch_size=2**5,
        output_size=29,
        dense_nodes=[200, 100, 50],
        scaler=scaler
        )

    model_features.scaler.fit(training_data)
    X_tensor = tf.constant(training_data)
    y_tensor = tf.constant(training_keys_binarized)
    train_dataset_tensor = tf.data.Dataset.from_tensor_slices((X_tensor,
                                                               y_tensor))
    test_dataset = (testing_data, testing_keys_binarized)

    tf.reset_default_graph()
    optimizer = tf.train.AdamOptimizer(model_features.learining_rate)
    model = dnn(model_features)
    all_loss_train, all_loss_test = model.fit_batch(
        train_dataset_tensor,
        test_dataset,
        optimizer,
        num_epochs=1,
        early_stopping_patience=0,
        verbose=1,
        print_errors=False,
        max_time=3600)
    pass


def test_cnn_training():
    """
    Testing the convolution neural network class and training function.
    """
    pass


def test_ae_training():
    """
    Testing the autoencoder training function.
    """
    pass
