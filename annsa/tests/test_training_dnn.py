from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import pytest
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from annsa.model_classes import DNN, dnn_model_features

tf.enable_eager_execution()


@pytest.fixture()
def dnn(request):
    '''
    Constructs a dense neural network with dense connections
    initialized to ones.
    '''
    scaler = make_pipeline(FunctionTransformer(np.abs, validate=False))
    model_features = dnn_model_features(
        learning_rate=1e-1,
        l2_regularization_scale=1e-1,
        dropout_probability=0.999,
        batch_size=2**5,
        output_size=2,
        dense_nodes=[10],
        output_function=None,
        activation_function=None,
        scaler=scaler)
    model = DNN(model_features)
    # forward pass to initialize dnn weights
    model.forward_pass(np.ones([1, 1024]), training=False)
    # set weights to ones
    weight_ones = []
    for index, weight in enumerate(model.get_weights()):
        if index % 2 == 0:
            weight_ones.append(np.ones(weight.shape))
        else:
            weight_ones.append(weight)
    model.set_weights(weight_ones)
    return model


# forward pass tests
def test_forward_pass_0(dnn):
    '''case 0: test if output size is correct'''
    output = dnn.forward_pass(np.ones([1, 1024]), training=False)
    assert(output.shape[1] == 2)


@pytest.mark.parametrize('dnn', [[]], indirect=True,)
def test_forward_pass_1(dnn):
    '''case 1: Tests response to a spectrum of all ones
    when weight filters are all one. Note, layer before output has
    10 nodes, each with an activation of 1024. Each of these 10 nodes
    is added into one output, so each output's value is 10240.'''
    output = dnn.forward_pass(np.ones([1, 1024]), training=False)
    output_value = output.numpy()[0][0]
    assert(output_value == 10240)


# loss function tests
def test_loss_fn_0(dnn):
    '''case 0: tests if l2 regularization adds to loss_fn.'''
    loss = dnn.loss_fn(
        input_data=np.ones([1, 1024]),
        targets=np.array([[16384, 16384]]),
        cost=dnn.mse,
        training=False)
    loss = loss
    assert(loss > 0.)


# dropout test
def test_dropout_0(dnn):
    '''case 0: tests that dropout is applied when training.'''
    o_training_false = dnn.forward_pass(np.ones([1, 1024]),
                                        training=False).numpy()
    o_training_true = dnn.forward_pass(np.ones([1, 1024]),
                                       training=True).numpy()
    assert(np.array_equal(o_training_false, o_training_true) is False)


def test_dropout_1(dnn):
    '''case 1: tests that dropout is not applied in inference, when training
    is False.'''
    o_training_false_1 = dnn.forward_pass(np.ones([1, 1024]),
                                          training=False).numpy()
    o_training_false_2 = dnn.forward_pass(np.ones([1, 1024]),
                                          training=False).numpy()
    assert(np.array_equal(o_training_false_1, o_training_false_2))
