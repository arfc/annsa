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

@pytest.fixture(scope)
def cnn1d():
    '''
    Constructs a convolutional neural network with random filter initialization.
    '''
    scaler = make_pipeline(FunctionTransformer(np.abs, validate=False))
    model_features = generate_random_cnn1d_architecture(cnn_filters_choices=((4, 1),),
                                                        cnn_kernel_choices=((4,),),
                                                        pool_size_choices=((4,),))
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
    model_features.dense_nodes = []
    model = CNN1D(model_features)
    # forwward pass to initialize cnn1d weights
    model.forward_pass(1*np.ones([1,1024]), training=False)
    return model

@pytest.fixture()
def cnn1d_ones():
    '''
    Constructs a convolutional neural network and replaces filters with ones.
    '''
    weight_ones = []
    for index, weight in enumerate(cnn1d.get_weights()):
        if index %2 == 0:
            weight_ones.append(np.ones(weight.shape))
        else:
            weight_ones.append(weight)
    cnn1d.set_weights(weight_ones)

    return cnn1d

# forward pass tests
def test_forward_pass_0(cnn1d):
    '''case 0: test if output size is correct'''
    output = cnn1d.forward_pass(np.ones([1,1024]), training=False)
    assert(output.shape[1] == 2)

def test_forward_pass_1(cnn1d_ones):
    '''case 1: Tests response to a spectrum of all ones
    when weight filters are all one. Note, layer before output has an 
    activation of 64 in each node and a length of 256. Densely connected
    output connection yields 64*256=16384 for each output node.'''
    output = cnn1d_ones.forward_pass(np.ones([1,1024]), training=False)
    output_node = output.numpy()[0][0]
    assert(output_node == 16384)

# loss function tests

@pytest.mark.parametrize('name', ['Claire', 'Gloria', 'Haley'])
def test_loss_fn_0(cnn1d_ones(dense_nodes)):
    '''case 0: tests if mse cost function works with cnn1d'''
    loss = cnn1d_ones.loss_fn(
        input_data=np.ones([1,1024]),
        targets=np.array([[16384, 16384]]),
        cost=cnn1d_ones.mse,
        training=False)
    loss = loss.numpy()
    if True
        assert(loss == 0.)
    if False
        assert(loss != 0.)

# def test_loss_fn_1(cnn1d):
#     '''case 1: tests if l2 regularization .'''
#     loss = cnn1d_ones.loss_fn(
#         input_data=np.ones([1,1024]),
#         targets=np.array([[16384, 16384]]),
#         cost=cnn1d_ones.mse,
#         training=True)
#     loss = loss.numpy()
#     assert(loss == 0.)
    
def test_loss_fn_1(cnn1d(dense_nodes = [10])):
    '''case 0: tests if l2 regularization adds to loss_fn when there are dense nodes'''
    loss = cnn1d_ones.loss_fn(
        input_data=np.ones([1,1024]),
        targets=np.array([[16384, 16384]]),
        cost=cnn1d_ones.mse,
        training=True)
    loss = loss.numpy()
    assert(loss == 0.)

    
# # dropout test
# @pytest.mark.parametrize('dense_nodes, output', [([], [10]),
#                                                  (True, False)])
# def test_dropout_1(cnn1d):
#     '''case 0: tests that dropout is not applied when there are no dense layers'''
#     o_training_false = cnn1d_ones.forward_pass(np.ones([1,1024]),
#                                                training=False).numpy()
#     o_training_true = cnn1d_ones.forward_pass(np.ones([1,1024]),
#                                              training=True).numpy()
#     assert(o_training_false == o_training_true)

# def test_dropout_2(cnn1d(dense_nodes = [10]):
#     '''case 0: tests that dropout is not applied when there are no dense layers'''
#     o_training_false = cnn1d_ones.forward_pass(np.ones([1,1024]),
#                                                training=False).numpy()
#     o_training_true = cnn1d_ones.forward_pass(np.ones([1,1024]),
#                                              training=True).numpy()
#     assert(o_training_false != o_training_true)
    
# def test_cnn1d_training():
#     """
#     Testing the convolutional neural network class and training function.

#     Returns : Nothing
#     """

#     tf.reset_default_graph()
#     model_features, optimizer, model = construct_cnn1d()
#     train_dataset, test_dataset = load_dataset()
#     model_features.scaler.fit(train_dataset[0])

#     all_loss_train, all_loss_test = model.fit_batch(
#         train_dataset,
#         test_dataset,
#         optimizer,
#         num_epochs=1,
#         earlystop_patience=0,
#         verbose=1,
#         print_errors=0,
#         obj_cost=model.cross_entropy,
#         earlystop_cost_fn=model.f1_error,
#         data_augmentation=model.default_data_augmentation,)
#     pass
