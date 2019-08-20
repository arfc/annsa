from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes import (generate_random_cae_architecture,
                                 CAE)
from annsa.load_dataset import load_dataset

tf.enable_eager_execution()


def construct_cae():
    """
    Builds a convolutional autoencoder with a random architecture.

    Returns:
    --------

    model_features : dictionary
        Contains all the features of a particular model.
    optimizer : tensorflow class, optimizer
        A tensorflow function, see tf.train.AdamOptimizer for more
        details.
    model : object class CAE
        A convolutional autoencoder that can be trained, saved, and
        reused.
    """
    scaler = make_pipeline(FunctionTransformer(np.log1p, validate=False))
    model_features = generate_random_cae_architecture(((4, 1), (8, 1)),
                                                      ((8,), (4,)),
                                                      ((8,), (4,)),)
    model_features.encoder_trainable = True
    model_features.learning_rate = 1e-1
    model_features.batch_size = 2**5
    model_features.scaler = scaler
    model_features.activation_function = tf.nn.relu
    model_features.output_function = None
    model_features.Pooling = tf.layers.MaxPooling1D

    optimizer = tf.train.AdamOptimizer(model_features.learning_rate)
    model = CAE(model_features)
    return model_features, optimizer, model


def test_cae_construction():
    """
    Tests the convolutional autoencoder construction.
    """
    _, _, _ = construct_cae()
    pass


def test_cae_training():
    """
    Tests the convolutional neural network class and training function.
    """

    tf.reset_default_graph()
    model_features, optimizer, model = construct_cae()
    train_dataset, test_dataset = load_dataset(
        kind='ae')  # this should fail a test
    model_features.scaler.fit(train_dataset[0])

    all_loss_train, all_loss_test = model.fit_batch(
        train_dataset,
        test_dataset,
        optimizer,
        num_epochs=1,
        earlystop_patience=0,
        verbose=1,
        print_errors=0,
        obj_cost=model.mse,
        earlystop_cost_fn=model.mse,
        data_augmentation=model.default_data_augmentation,)
    pass
