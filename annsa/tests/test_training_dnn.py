from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import pytest
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits
from annsa.model_classes import DNN, dnn_model_features


tf.enable_eager_execution()


@pytest.fixture(params=[(0.5, 1024),
                        (0.5, 64),
                        (0.999, 1024), ])
def dnn(request):
    '''
    Constructs a dense neural network with dense connections
    initialized to ones.
    '''
    (dropout_probability, input_size) = request.param
    scaler = make_pipeline(FunctionTransformer(np.abs, validate=False))
    model_features = dnn_model_features(
        learning_rate=1e-2,
        l2_regularization_scale=1e-1,
        dropout_probability=dropout_probability,
        batch_size=5,
        output_size=3,
        dense_nodes=[10],
        output_function=None,
        activation_function=None,
        scaler=scaler)
    model = DNN(model_features)
    # forward pass to initialize dnn weights
    model.forward_pass(np.ones([1, input_size]), training=False)
    # set weights to ones
    weight_ones = [np.ones(weight.shape) if (index % 2 == 0) else weight for index,
                   weight in enumerate(model.get_weights())]
    model.set_weights(weight_ones)
    return model


@pytest.fixture()
def toy_dataset():
    '''
    Constructs toy dataset of digits.
    '''
    data, target = load_digits(n_class=3, return_X_y=True)
    mlb = LabelBinarizer()
    targets_binarized = mlb.fit_transform(target)
    return (data, targets_binarized)


# forward pass tests
@pytest.mark.parametrize('dnn',
                         ((0.5, 1024), (0.999, 1024)),
                         indirect=True,)
def test_forward_pass_0(dnn):
    '''case 0: Tests if output size is correct'''
    output = dnn.forward_pass(np.ones([1, 1024]), training=False)
    assert(output.shape[1] == 3)


@pytest.mark.parametrize('dnn',
                         ((0.5, 1024), (0.999, 1024)),
                         indirect=True,)
def test_forward_pass_1(dnn):
    '''case 1: Tests response to a spectrum of all ones
    when weight filters are all one. Note, layer before output has
    10 nodes, each with an activation of 1024. Each of these 10 nodes
    is added into one output, so each output's value is 10240.'''
    output = dnn.forward_pass(np.ones([1, 1024]), training=False)
    output_value = output.numpy()[0][0]
    assert(output_value == 10240)


# loss function tests
@pytest.mark.parametrize('dnn',
                         ((0.5, 1024), (0.999, 1024)),
                         indirect=True,)
def test_loss_fn_0(dnn):
    '''case 0: tests if l2 regularization adds to loss_fn.'''
    loss = dnn.loss_fn(
        input_data=np.ones([1, 1024]),
        targets=np.array([[16384, 16384, 16384]]),
        cost=dnn.mse,
        training=False)
    loss = loss
    assert(loss > 0.)


# dropout tests
@pytest.mark.parametrize('dnn',
                         ((0.999, 1024),),
                         indirect=True,)
def test_dropout_0(dnn):
    '''case 0: tests that dropout is applied when training.'''
    o_training_false = dnn.forward_pass(np.ones([1, 1024]),
                                        training=False).numpy()
    o_training_true = dnn.forward_pass(np.ones([1, 1024]),
                                       training=True).numpy()
    assert(np.array_equal(o_training_false, o_training_true) is False)


@pytest.mark.parametrize('dnn',
                         ((0.999, 1024),),
                         indirect=True,)
def test_dropout_1(dnn):
    '''case 1: tests that dropout is not applied in inference, when training
    is False.'''
    o_training_false_1 = dnn.forward_pass(np.ones([1, 1024]),
                                          training=False).numpy()
    o_training_false_2 = dnn.forward_pass(np.ones([1, 1024]),
                                          training=False).numpy()
    assert(np.array_equal(o_training_false_1, o_training_false_2))


# training tests
@pytest.mark.parametrize('cost', ['mse', 'cross_entropy'])
@pytest.mark.parametrize('dnn',
                         ((0.5, 64),),
                         indirect=True,)
def test_training_0(dnn, toy_dataset, cost):
    '''case 0: test if training on toy dataset reduces errors using
    both error functions'''
    (data, targets_binarized) = toy_dataset
    cost_function = getattr(dnn, cost)
    objective_cost, earlystop_cost = dnn.fit_batch(
        (data, targets_binarized),
        (data, targets_binarized),
        optimizer=tf.train.AdamOptimizer(1e-2),
        num_epochs=2,
        obj_cost=cost_function,
        data_augmentation=dnn.default_data_augmentation,)
    epoch0_error = objective_cost['test'][0].numpy()
    epoch1_error = objective_cost['test'][1].numpy()
    assert(epoch1_error < epoch0_error)
