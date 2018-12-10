from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import tensorflow as tf
import annsa as an

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes_inheritance import dnn_model_features, DNN
from annsa.results_plotting_functions import hyperparameter_efficiency_plot

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
    train_dataset = [training_data, training_keys_binarized]
    
    testing_data = np.abs(testing_dataset[0])
    testing_keys = testing_dataset[1]
    testing_keys_binarized = mlb.transform(
        testing_keys.reshape([testing_data.shape[0], 1]))
    test_dataset = [testing_data, testing_keys_binarized]

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
    test_dataset = (testing_data, testing_keys_binarized)
    def data_augmentation(input_data):
        return np.random.poisson(input_data).astype(float)
    
    
    tf.reset_default_graph()
    optimizer = tf.train.AdamOptimizer(model_features.learining_rate)
    model = DNN(model_features)
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
        earlystop_cost_fn=model.cross_entropy,
        data_augmentation=data_augmentation,)
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

def test_hyperparameter_efficiency_plot():
    """
    Testing the hyperparameter_efficiency_plot function with a test
    data vector.
    """
    accuracy = np.random.normal(0.8,0.2,64)
    _, _ = hyperparameter_efficiency_plot(accuracy)
    pass

