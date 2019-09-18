from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import pytest
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

from annsa.model_classes import (BaseClass,
                                 generate_random_cnn1d_architecture,
                                 CNN1D)

tf.enable_eager_execution()


@pytest.fixture(params=[[], [10]])
def cnn1d(request):
    '''
    Constructs a convolutional neural network with filters
    initialized to ones. Fixture params are either zero or one hidden
    dense layer.
    '''
    dense_nodes = request.param
    
    scaler = make_pipeline(FunctionTransformer(np.abs, validate=False))
    model_features = generate_random_cnn1d_architecture(
        cnn_filters_choices=((4, 1),),
        cnn_kernel_choices=((4, ), ),
        pool_size_choices=((4, ), ))
    model_features.learning_rate = 1e-1
    model_features.trainable = True
    model_features.batch_size = 2**5
    model_features.output_size = 2
    model_features.output_function = None
    model_features.l2_regularization_scale = 1e1
    model_features.dropout_probability = 0.5
    model_features.scaler = scaler
    model_features.Pooling = tf.layers.MaxPooling1D
    model_features.activation_function = None
    model_features.dense_nodes = dense_nodes
    model = CNN1D(model_features)
    # forward pass to initialize cnn1d weights
    model.forward_pass(1*np.ones([1, 1024]), training=False)
    # set weights to ones
    weight_ones = []
    for index, weight in enumerate(model.get_weights()):
        if index % 2 == 0:
            weight_ones.append(np.ones(weight.shape))
        else:
            weight_ones.append(weight)
    model.set_weights(weight_ones)
    return model


def test_forward_pass_0(cnn1d):
    '''case 0: test if output size is correct'''
    output = cnn1d.forward_pass(np.ones([1, 1024]), training=False)
    assert(output.shape[1] == 2)


@pytest.mark.parametrize('cnn1d', [[]], indirect=True,)
def test_forward_pass_1(cnn1d):
    '''case 1: Tests response to a spectrum of all ones
    when weight filters are all one. Note, layer before output has an
    activation of 64 in each node and a length of 256. Densely connected
    output connection yields 64*256=16384 for each output node.'''
    output = cnn1d.forward_pass(np.ones([1, 1024]), training=False)
    output_value = output.numpy()[0][0]
    assert(output_value == 16384)

# loss function tests
@pytest.mark.parametrize('cnn1d', [[]], indirect=True,)
def test_loss_fn_0(cnn1d):
    '''case 0: tests if l2 regularization does not add to the loss_fn
    with hidden dense layers.'''
    loss = cnn1d.loss_fn(
        input_data=np.ones([1,1024]),
        targets=np.array([[16384, 16384]]),
        cost=cnn1d.mse,
        training=False)
    loss = loss.numpy()
    assert(loss == 0.)

@pytest.mark.parametrize('cnn1d', [[10]], indirect=True,)
def test_loss_fn_1(cnn1d):
    '''case 1: tests if l2 regularization adds to loss_fn when there are
    dense nodes'''
    loss = cnn1d.loss_fn(
        input_data=np.ones([1,1024]),
        targets=np.array([[16384, 16384]]),
        cost=cnn1d.mse,
        training=False)
    loss = loss
    assert(loss > 0.)

# dropout test
@pytest.mark.parametrize('cnn1d', [[]], indirect=True,)
def test_dropout_1(cnn1d):
    '''case 0: tests that dropout is not applied when there are no dense 
    hidden layers.'''
    o_training_false = cnn1d.forward_pass(np.ones([1,1024]),
                                               training=False).numpy()
    o_training_true = cnn1d.forward_pass(np.ones([1,1024]),
                                             training=True).numpy()
    assert(np.array_equal(o_training_false, o_training_true))

@pytest.mark.parametrize('cnn1d', [[10]], indirect=True,)
def test_dropout_2(cnn1d):
    '''case 0: tests that dropout is applied when there are dense hidden layers'''
    o_training_false = cnn1d.forward_pass(np.ones([1,1024]),
                                               training=False).numpy()
    o_training_true = cnn1d.forward_pass(np.ones([1,1024]),
                                             training=True).numpy()
    assert(np.array_equal(o_training_false, o_training_true) is False)
    
