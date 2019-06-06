from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import tensorflow as tf
import annsa as an

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes import (cnn1d_model_features,
                                 generate_random_cnn1d_architecture,
                                 CNN1D)

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

    #make_classification returns [X, y] and because two variables
    #are not being specified, training_dataset and testing_dataset
    #will be tuples of [X, y]. 
    #X is an array of shape [n_samples, n_features] and stores
    #the actual data.
    #y is an array of shape [n_samples] and stores the integer
    #labels for class membership of each sample. 
    #see documentation for make_classification for further details.

    mlb = LabelBinarizer()

    training_data = np.abs(training_dataset[0]) #takes the absolute value of the training data.
    training_keys = training_dataset[1] #takes the labels from training_dataset, which is a tuple of [X, y]
    training_keys_binarized = mlb.fit_transform(
        training_keys.reshape([training_data.shape[0], 1]))
    #fit_transform gives a binary representation of all of the integer labels. 
    train_dataset = [training_data, training_keys_binarized]

    testing_data = np.abs(testing_dataset[0])
    testing_keys = testing_dataset[1]
    testing_keys_binarized = mlb.transform(
        testing_keys.reshape([testing_data.shape[0], 1]))
    test_dataset = [testing_data, testing_keys_binarized]

    return train_dataset, test_dataset


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
    model_features = generate_random_cnn1d_architecture(((4, 1), (8, 1)),
                                                        ((8,), (4,)),
                                                        ((8,), (4,)),)
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

    Note: _ = something --> Initializes 'something', but does not 
                            store anything in memory.
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
        max_time=3600,
        obj_cost=model.cross_entropy,
        earlystop_cost_fn=model.f1_error,
        data_augmentation=model.default_data_augmentation,)
    pass
