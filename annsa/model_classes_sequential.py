from __future__ import print_function
import tensorflow as tf
import numpy as np
# import tensorflow.contrib.eager as tfe
from sklearn.metrics import f1_score
from tensorflow.image import resize_images
from keras import backend as K
from tensorflow.keras.initializers import he_normal, glorot_normal
from tensorflow.keras.layers import (Dense, Conv1D, Dropout, MaxPool1D, Flatten, Reshape, MaxPool1D, UpSampling1D)
from tensorflow.keras import Sequential
from keras.regularizers import l2
from keras import backend as K
from random import choice

def mean_normalized_kl_divergence(y_true, y_pred):
    '''
    Normalizes vector to a probability distribution before 
    calculating the mean KL divergence.
    '''

    y_true /= K.sum(y_true)
    y_pred /= K.sum(y_pred)
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(y_true * K.log(y_true / y_pred), axis=-1)


def f_precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f_recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    precision = f_precision(y_true, y_pred)
    recall = f_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def compile_model(build_model, model_features):
    '''
    Builds and compiles a machine learning model using model_features. This function
    creates a model ready to train with model.fit(x,y).

    Parameters
    ----------
    model_builder : function
        Function that builds a machine learning model based on model_features.

    model_features : model feature class
        A dictionary of features to be used by the model_builder.
        The dictionary must contain the features expectd by the model_builder.

    Returns
    -------
    model : TensorFlow Model class
        A comiled machine learning model that contains training and inference features.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model

    '''
    model = build_model(model_features)
    model.compile(loss=model_features.loss,
                  optimizer=model_features.optimizer(model_features.learning_rate),
                  metrics=model_features.metrics,)

    return model


def build_dnn_model(model_features):
    """
    Builds a DNN archetecture basd on model_features
    without compilation. Uses the Keras Sequential class and 
    model.add method to add layers.

    Parameters
    ----------
    model_features : model feature class
        A dictionary of features to be used by the model_builder.
        The dictionary must contain the features expectd by the model_builder.

    Returns
    -------
    model : TensorFlow Model class
        A machine learning model that contains training and inference features.
        Note, this model is uncompiled. model.compile needs to be run for this model
        to be trained.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model
    """

    if model_features.activation_function == tf.nn.relu:
        model_features.kernel_initializer = he_normal()
    else:
        model_features.kernel_initializer = glorot_normal()

    model = Sequential()
    model.add(Reshape((model_features.input_dim,),
                      input_shape=(model_features.input_dim,)))
    for nodes in model_features.dense_nodes:
        model.add(
            Dense(
                units=nodes,
                activation=model_features.activation_function,
                kernel_initializer=model_features.kernel_initializer,
                kernel_regularizer=l2(model_features.l2_regularization_scale),))
        model.add(Dropout(model_features.dropout_rate))
    model.add(Dense(model_features.output_size,
                    activation=model_features.output_function,))

    return model


class dnn_model_features(object):
    """
    Defines the features for the dense neural network.

    '__init__' : Constructor
    """

    def __init__(self,
                 learning_rate,
                 l2_regularization_scale,
                 dropout_rate,
                 batch_size,
                 output_size,
                 output_function,
                 dense_nodes,
                 activation_function,
                 scaler,
                 loss=tf.keras.losses.categorical_crossentropy,
                 optimizer=tf.keras.optimizers.Adam,
                 metrics=[f1],
                 input_dim=1024,
                 ):
        """
        Parameters
        ----------
        learning_rate : float
            How much the weights update due to back propagation of the
            error/loss function.
        l2_regularization_scale : float
            The loss penalty for regularization type l2. If the model
            attempts to increase the weights, it will only be accepted
            if there is an equal or greater decrease in the error
            function.
        dropout_rate : float
            The probability that any neuron will be temporarily turned
            off during training. Example: dropout_rate = 0.4
            means there is a 40% probability of the neuron turning off.
        batch_size : int
            'batch_size' is the number of spectra/images being passed
            through the network at once. For reference, one epoch is
            the size of all training data.
        output_size : Array/Tuple
            The desired dimensions of your output, typically [nx1]
        dense_nodes : int
            The desired number of nodes in a dense layer.
        activation_function : Tensorflow activation function.
            Activation function used after each dense layer. Example: tf.nn.relu
        scaler : Sklearn pipeline class
            Sklearn pipeline for preprocessing. 
        input_size : int
            Length of the input to the network.
        loss : tf.keras.losses class
            The primary loss used by the model. Examples are CategoricalCrossentropy,
            MeanSquaredError, and KLDivergence.
            https://www.tensorflow.org/api_docs/python/tf/keras/losses
        optimizer : tf.keras.optimizers class
            The optimizer used by the model. Examples are Adam, SGD, and RMSprop.
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        metrics : tf.keras.metrics class, list
            A list of metrics to monitor and save during training. Examples include 
            Accuracy, Precision, and Recall.
            https://www.tensorflow.org/api_docs/python/tf/keras/metrics
 
        """
        self.learning_rate = learning_rate
        self.l2_regularization_scale = l2_regularization_scale
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.output_size = output_size
        self.output_function = output_function
        self.dense_nodes = dense_nodes
        self.activation_function = activation_function
        self.scaler = scaler
        self.input_dim = input_dim
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        
    def to_dae_model_features(self):
        '''
        Creates a DAE feature class based on the DNN model features.

        Returns:
        -------
        model_features : DAE feature class
            Class capable of constructing a DAE with the corresponding
            DNN archetecture.

        '''
        
        dense_nodes_decoder = self.dense_nodes[1:]
        dense_nodes_decoder = dense_nodes_decoder[::-1]
        
        model_features = dae_model_features(
            learning_rate=self.learning_rate,
            dropout_rate=0.,
            batch_size=self.batch_size,
            dense_nodes_encoder=self.dense_nodes[:],
            dense_nodes_decoder=dense_nodes_decoder,
            scaler=self.scaler,
            activation_function=self.activation_function,
            output_function=None,
            input_dim=self.input_dim,
            output_size=self.input_dim,
            loss=tf.keras.losses.MSE,
            optimizer=self.optimizer,
            metrics=[tf.keras.losses.MSE],)
        
        return model_features

    
def add_conv_pool_layer(model,
                        cnn_filter,
                        cnn_kernel,
                        pool_size,
                        pool_stride,
                        trainable,
                        activation_function,
                        kernel_initializer):
    model.add(
        Conv1D(
            filters=cnn_filter,
            kernel_size=cnn_kernel,
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer,
            activation=activation_function,
            trainable=trainable,))
    model.add(
        MaxPool1D(
            pool_size=pool_size,
            strides=pool_stride,
            padding='same',))
    return model

def add_conv_upsample_layer(
    model,
    cnn_filter,
    cnn_kernel,
    trainable,
    activation_function,
    kernel_initializer):
    model.add(
        UpSampling1D(
            size=2))
    model.add(
        Conv1D(
            filters=cnn_filter,
            kernel_size=cnn_kernel,
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer,
            activation=activation_function,
            trainable=trainable,))
    return model


def build_cnn_model(model_features):
    """
    Builds a 1D-CNN archetecture basd on model_features
    without compilation. Uses the Keras Sequential class and 
    model.add method to add layers.

    Parameters
    ----------
    model_features : model feature class
        A dictionary of features to be used by the model_builder.
        The dictionary must contain the features expectd by the model_builder.

    Returns
    -------
    model : TensorFlow Model class
        A machine learning model that contains training and inference features.
        Note, this model is uncompiled. model.compile needs to be run for this model
        to be trained.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model
    """

    if model_features.activation_function == tf.nn.relu:
        kernel_initializer = he_normal()
    else:
        kernel_initializer = glorot_normal()

    model = Sequential()
    model.add(Reshape((model_features.input_dim, 1),
                      input_shape=(model_features.input_dim,)))
    
    for cnn_filter, cnn_kernel, pool_size, pool_stride in zip(
        model_features.cnn_filters,
        model_features.cnn_kernels,
        model_features.pool_sizes,
        model_features.pool_strides,):
        model = add_conv_pool_layer(
            model,
            cnn_filter,
            cnn_kernel,
            pool_size,
            pool_stride,
            trainable=model_features.trainable,
            activation_function=model_features.activation_function,
            kernel_initializer=kernel_initializer)
    model.add(Flatten())

    for nodes in model_features.dense_nodes:            
        model.add(
            Dense(
                units=nodes, 
                activation=model_features.activation_function,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=l2(model_features.l2_regularization_scale),))
        model.add(Dropout(model_features.dropout_rate))
    model.add(Dense(model_features.output_size,
                    activation=model_features.output_function,))

    return model


class cnn1d_model_features(object):

    """
    Defines the features of a CNN model.
    """

    def __init__(self,
                 learning_rate,
                 trainable,
                 batch_size,
                 output_size,
                 output_function,
                 l2_regularization_scale,
                 dropout_rate,
                 scaler,
                 Pooling,
                 cnn_filters,
                 cnn_kernels,
                 cnn_strides,
                 pool_sizes,
                 pool_strides,
                 dense_nodes,
                 activation_function,
                 loss,
                 optimizer,
                 metrics,
                 ):
        """
        Parameters
        ----------
        learning_rate : float
            How much the weights update due to back propagation of the
            error/loss function.
        trainable : boolean
            If true, optimization will be applied and weights will be
            updated.
            False is used for prediction.
        output_function :

        l2_regularization_scale : float
            The loss penalty for regularization type l2. If the model
            attempts to increase the weights, it will only be accepted
            if there is an equal or greater decrease in the error
            function.
        dropout_rate : float
            The probability that any neuron will be temporarily turned
            off during training. Example: dropout_rate = 0.4
            means there is a 40% probability of the neuron turning off.
        scaler : Tensorflow scaling function
        batch_size : int
            'batch_size' is the number of spectra/images being passed
            through the network at once. For reference, one epoch is
            the size of all training data.
        Pooling : Tensorflow pooling function
        cnn_filters : int
            The number of filters in a convolutional layer.
        cnn_kernels : int or 1D array of type int
            Passing int will assume a square filter of size int x int.
            The values of an array will be taken as the desired dimens-
            ion size of the filter.
        cnn_strides: int
            The stride size of each filter. How far it shifts per
            iteration. Typically stride size is one.
        pool_sizes : int or array/tuple
            'int':
                Creates a square pool.
            'array' or 'tuple':
                Creates a pool the size of elements in your tuple.
                pool_strides : int
            How much the pool shifts per iteration. Typically stride
            is 2.
        output_size : Array/Tuple
            The desired dimensions of your output, typically [nx1]
        dense_nodes : int
            The desired number of nodes in a dense layer.
        activation_function : Tensorflow activation function
            Example: tf.nn.relu
        loss : tf.keras.losses class
            The primary loss used by the model. Examples are CategoricalCrossentropy,
            MeanSquaredError, and KLDivergence.
            https://www.tensorflow.org/api_docs/python/tf/keras/losses
        optimizer : tf.keras.optimizers class
            The optimizer used by the model. Examples are Adam, SGD, and RMSprop.
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        metrics : tf.keras.metrics class, list
            A list of metrics to monitor and save during training. Examples include
            Accuracy, Precision, and Recall.
            https://www.tensorflow.org/api_docs/python/tf/keras/metrics

        """
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.batch_size = batch_size
        self.output_size = output_size
        self.output_function = output_function
        self.l2_regularization_scale = l2_regularization_scale
        self.dropout_rate = dropout_rate
        self.scaler = scaler
        self.Pooling = Pooling
        self.cnn_filters = cnn_filters
        self.cnn_kernels = cnn_kernels
        self.cnn_strides = cnn_strides
        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.dense_nodes = dense_nodes
        self.activation_function = activation_function
        self.loss
        self.optimizer
        self.metrics
        
    def to_cae_model_features(self):
        '''
        Creates a CAE feature class based on the CNN model features.

        Returns:
        -------
        model_features : CAE feature class
            Class capable of constructing a CAE with the corresponding
            CNN archetecture.

        '''
        
        model_features = cae_model_features(
            learning_rate=self.learning_rate,
            encoder_trainable=True,
            batch_size=self.batch_size,
            Pooling = MaxPool1D,
            scaler=self.scaler,
            activation_function=self.activation_function,
            output_function=None,
            input_dim=self.input_dim,
            output_size=self.input_dim,
            cnn_filters_encoder = list(self.cnn_filters) + [1],
            cnn_kernels_encoder = list(self.cnn_kernel) + list(self.cnn_kernel)[-2:-1],
            cnn_strides_encoder =  list(self.cnn_strides) + list(self.cnn_strides)[-2:-1],
            pool_sizes_encoder = list(self.pool_sizes) + list(self.pool_sizes)[-2:-1],
            pool_strides_encoder = list(self.pool_strides) + list(self.pool_strides)[-2:-1],
            cnn_filters_decoder = list(self.cnn_filters) + [1],
            cnn_kernels_decoder = list(self.cnn_kernel) + list(self.cnn_kernel)[-2:-1],
            cnn_strides_decoder = list(self.cnn_strides) + list(self.cnn_strides)[-2:-1],
            loss=tf.keras.losses.MSE,
            optimizer=self.optimizer,
            metrics=[tf.keras.losses.MSE],)
        
        return model_features


def generate_random_cnn1d_architecture(cnn_filters_choices,
                                       cnn_kernels_choices,
                                       pool_sizes_choices):
    """
    Generates a random 1d convolutional neural network based on a
    set of predefined architectures.

    Parameters:
    -----------
    cnn_filters_choices : 1-D array-like or int
        Input a choice of ..............

    cnn_kernel_choices : 1-D array-like or int
        Input a choice of kernel (filter) sizes.

    pool_sizes_choices : 1-D array-like or int
        Input a choice of pooling sizes.

    Returns:
    --------
    model_features : class
        Class that describes the structure of a 1D convolution
        neural network.
    """

    cnn_filters = choice(cnn_filters_choices)
    cnn_kernel_choice = choice(cnn_kernel_choices)
    pool_sizes_choice = choice(pool_sizes_choices)

    cnn_kernel = cnn_kernel_choice * (len(cnn_filters))
    cnn_strides = (1,) * (len(cnn_filters))
    pool_sizes = pool_sizes_choice * (len(cnn_filters))
    pool_strides = (2,) * (len(cnn_filters))

    number_layers = np.random.randint(1, 4)
    dense_nodes = (10**np.random.uniform(1,
                                         np.log10(1024 / (2**len(
                                             cnn_filters))),
                                         number_layers)).astype('int')
    dense_nodes = np.sort(dense_nodes)
    dense_nodes = np.flipud(dense_nodes)

    model_features = cnn1d_model_features(
        trainable=None,
        learning_rate=None,
        batch_size=None,
        output_size=None,
        scaler=None,
        activation_function=None,
        output_function=None,
        Pooling=None,
        l2_regularization_scale=None,
        dropout_rate=None,
        cnn_filters=cnn_filters,
        cnn_kernel=cnn_kernel,
        cnn_strides=cnn_strides,
        pool_sizes=pool_sizes,
        pool_strides=pool_strides,
        dense_nodes=dense_nodes
    )

    return model_features


def build_dae_model(model_features):
    """
    Builds a DAE archetecture basd on model_features
    without compilation. Uses the Keras Sequential class and 
    model.add method to add layers.

    Parameters
    ----------
    model_features : model feature class
        A dictionary of features to be used by the model_builder.
        The dictionary must contain the features expectd by the model_builder.

    Returns
    -------
    model : TensorFlow Model class
        A machine learning model that contains training and inference features.
        Note, this model is uncompiled. model.compile needs to be run for this model
        to be trained.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model
    """

    if model_features.activation_function == tf.nn.relu:
        model_features.kernel_initializer = he_normal()
    else:
        model_features.kernel_initializer = glorot_normal()

    model = Sequential()
    model.add(Reshape((model_features.input_dim,),
                      input_shape=(model_features.input_dim,)))
    for nodes in model_features.dense_nodes_encoder:
        model.add(
            Dense(
                units=nodes,
                activation=model_features.activation_function,
                kernel_initializer=model_features.kernel_initializer,))
        model.add(Dropout(model_features.dropout_rate))
    for nodes in model_features.dense_nodes_decoder:
        model.add(
            Dense(
                units=nodes,
                activation=model_features.activation_function,
                kernel_initializer=model_features.kernel_initializer,))
        model.add(Dropout(model_features.dropout_rate))
    model.add(Dense(model_features.output_size,
                    activation=model_features.output_function,))

    return model


class dae_model_features(object):

    def __init__(self,
                 learning_rate,
                 dropout_rate,
                 batch_size,
                 dense_nodes_encoder,
                 dense_nodes_decoder,
                 scaler,
                 activation_function,
                 output_function,
                 output_size,
                 input_dim,
                 loss,
                 optimizer,
                 metrics,
                 ):
        """
        Parameters
        ----------
        learning_rate : float
            How much the weights update due to back propagation of the
            error/loss function.
        output_function :

        l2_regularization_scale : float
            The loss penalty for regularization type l2. If the model
            attempts to increase the weights, it will only be accepted
            if there is an equal or greater decrease in the error
            function.
        dropout_rate : float
            The probability that any neuron will be temporarily turned
            off during training. Example: dropout_rate = 0.4
            means there is a 40% probability of the neuron turning off.
        scaler : Tensorflow scaling function
        batch_size : int
            'batch_size' is the number of spectra/images being passed
            through the network at once. For reference, one epoch is
        output_size : Array/Tuple
            The desired dimensions of your output, typically [nx1]
        dense_nodes_encoder : int
            The desired number of nodes in a dense layer of the encoder.
        dense_nodes_decoder : int
            The desired number of nodes in a dense layer of the decoder.
        activation_function : Tensorflow activation function
            Example: tf.nn.relu
        input_dim : int
            Size of the input
        loss : tf.keras.losses class
            The primary loss used by the model. Examples are CategoricalCrossentropy,
            MeanSquaredError, and KLDivergence.
            https://www.tensorflow.org/api_docs/python/tf/keras/losses
        optimizer : tf.keras.optimizers class
            The optimizer used by the model. Examples are Adam, SGD, and RMSprop.
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        metrics : tf.keras.metrics class, list
            A list of metrics to monitor and save during training. Examples include
            Accuracy, Precision, and Recall.
            https://www.tensorflow.org/api_docs/python/tf/keras/metrics
        """
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_nodes_encoder = dense_nodes_encoder
        self.dense_nodes_decoder = dense_nodes_decoder
        self.scaler = scaler
        self.activation_function = activation_function
        self.output_function = output_function
        self.output_size = output_size
        self.input_dim = input_dim
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics


def build_cae_model(model_features):
    """
    Builds a DAE archetecture basd on model_features
    without compilation. Uses the Keras Sequential class and 
    model.add method to add layers.

    Parameters
    ----------
    model_features : model feature class
        A dictionary of features to be used by the model_builder.
        The dictionary must contain the features expectd by the model_builder.

    Returns
    -------
    model : TensorFlow Model class
        A machine learning model that contains training and inference features.
        Note, this model is uncompiled. model.compile needs to be run for this model
        to be trained.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model
    """

    if model_features.activation_function == tf.nn.relu:
        model_features.kernel_initializer = he_normal()
    else:
        model_features.kernel_initializer = glorot_normal()

    model = Sequential()
    model.add(Reshape((model_features.input_dim, 1),
                      input_shape=(model_features.input_dim,)))
    # encoder
    for cnn_filter, cnn_kernel, pool_size, pool_stride in zip(
        model_features.cnn_filters_encoder,
        model_features.cnn_kernels_encoder,
        model_features.pool_sizes_encoder,
        model_features.pool_strides_encoder,):
        model = add_conv_pool_layer(
            model,
            cnn_filter,
            cnn_kernel,
            pool_size,
            pool_stride,
            trainable=model_features.encoder_trainable,
            activation_function=model_features.activation_function,
            kernel_initializer=model_features.kernel_initializer)
    # decoder
    for cnn_filter, cnn_kernel, pool_size, pool_stride in zip(
        model_features.cnn_filters_decoder,
        model_features.cnn_kernels_decoder,
        model_features.pool_sizes_decoder,
        model_features.pool_strides_decoder,):
        model = add_conv_upsample_layer(
            model,
            cnn_filter,
            cnn_kernel,
            trainable=True,
            activation_function=model_features.activation_function,
            kernel_initializer=model_features.kernel_initializer)
#     model.add(Conv1D(filters=model_features.cnn_filters_decoder[-1],
#                      kernel_size=model_features.cnn_kernels_decoder[-1],
#                      strides=1,
#                      padding='same',
#                      kernel_initializer=model_features.kernel_initializer,
#                      activation=model_features.activation_function,
#                      trainable=True,))
#     # final layer
    model.add(Reshape((model_features.input_dim,)))
        
    return model


class cae_model_features(object):

    def __init__(self,
                 learning_rate,
                 encoder_trainable,
                 batch_size,
                 scaler,
                 activation_function,
                 output_function,
                 Pooling,
                 cnn_filters_encoder,
                 cnn_kernels_encoder,
                 cnn_strides_encoder,
                 pool_sizes_encoder,
                 pool_strides_encoder,
                 cnn_filters_decoder,
                 cnn_kernels_decoder,
                 cnn_strides_decoder,
                 output_size,
                 input_dim,
                 loss,
                 optimizer,
                 metrics,
                 ):
        """
        @Author: sam dotson
        Parameters
        ----------
        learning_rate : float
            How much the weights update due to back propagation of the
            error/loss function.
        encoder_trainable : boolean
            If true, optimization will be applied and weights will be
            updated.
            False is used for prediction.
        output_function :

        scaler : Tensorflow scaling function
        batch_size : int
            'batch_size' is the number of spectra/images being passed
            through the network at once. For reference, one epoch is
            the size of all training data.
        Pooling : Tensorflow pooling function
        cnn_filters_encoder : int
            The number of filters in a convolutional layer.
        cnn_kernels_encoder : int or 1D array of type int
            Passing int will assume a square filter of size int x int.
            The values of an array will be taken as the desired dimens-
            ion size of the filter.
        cnn_strides_encoder: int
            The stride size of each filter. How far it shifts per
            iteration. Typically stride size is one.
        pool_sizes_encoder : int or array/tuple
            'int':
                Creates a square pool.
            'array' or 'tuple':
                Creates a pool the size of elements in your tuple.
        pool_strides_encoder : int
            How much the pool shifts per iteration. Typically stride
            is 2.
        cnn_filters_decoder : int
            The number of filters in a convolutional layer.
        cnn_kernels_decoder : int or 1D array of type int
            Passing int will assume a square filter of size int x int.
            The values of an array will be taken as the desired dimens-
            ion size of the filter.
        cnn_strides_decoder : int
            The stride size of each filter. How far it shifts per
            iteration. Typically stride size is one.
        output_size : Array/Tuple
            The desired dimensions of your output, typically [nx1]
        dense_nodes : int
            The desired number of nodes in a dense layer.
        activation_function : Tensorflow activation function
            Example: tf.nn.relu
        output_size : Array/Tuple
            The desired dimensions of your output, typically [nx1]
        input_dim : int
            Size of the input
        loss : tf.keras.losses class
            The primary loss used by the model. Examples are CategoricalCrossentropy,
            MeanSquaredError, and KLDivergence.
            https://www.tensorflow.org/api_docs/python/tf/keras/losses
        optimizer : tf.keras.optimizers class
            The optimizer used by the model. Examples are Adam, SGD, and RMSprop.
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        metrics : tf.keras.metrics class, list
            A list of metrics to monitor and save during training. Examples include
            Accuracy, Precision, and Recall.
            https://www.tensorflow.org/api_docs/python/tf/keras/metrics
        """
        self.learning_rate = learning_rate
        self.encoder_trainable = encoder_trainable
        self.batch_size = batch_size
        self.scaler = scaler
        self.activation_function = activation_function
        self.output_function = output_function
        self.Pooling = Pooling
        self.cnn_filters_encoder = cnn_filters_encoder
        self.cnn_kernels_encoder = cnn_kernels_encoder
        self.cnn_strides_encoder = cnn_strides_encoder
        self.pool_sizes_encoder = pool_sizes_encoder
        self.pool_strides_encoder = pool_strides_encoder
        self.cnn_filters_decoder = cnn_filters_decoder
        self.cnn_kernels_decoder = cnn_kernels_decoder
        self.cnn_strides_decoder = cnn_strides_decoder
        self.pool_sizes_decoder = pool_sizes_encoder
        self.pool_strides_decoder = pool_strides_encoder
        self.output_size = output_size
        self.input_dim = input_dim
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics


def generate_random_cae_architecture(cnn_filters_encoder_choices,
                                     cnn_kernels_encoder_choices,
                                     pool_sizes_encoder_choices):
    """
    Generates a random convolutional autoencoder based on given architecture
    choices.

    Parameters
    ----------
    cnn_filters_encoder_choices : list, int
        List of lists containing number of filters in each layer.
    cnn_kernels_encoder_choices: list, int
        Kernel sizes
    pool_sizes_encoder_choices: list, int
        Pooling sizes
    Returns
    -------
    cae_model_features : class
        Class that describes the structure of a CAE.

    """

    cnn_filters_encoder_choice = np.random.randint(
        len(cnn_filters_encoder_choices))
    cnn_kernels_encoder_choice = np.random.randint(
        len(cnn_kernels_encoder_choices))
    pool_sizes_encoder_choice = np.random.randint(
        len(pool_sizes_encoder_choices))

    # #############
    # ## Encoder ##
    # #############
    cnn_filters_encoder = cnn_filters_encoder_choices[
        cnn_filters_encoder_choice]
    cnn_kernels_encoder = cnn_kernels_encoder_choices[
        cnn_kernels_encoder_choice] * (len(cnn_filters_encoder_choices))
    cnn_strides_encoder = (1,) * (len(cnn_filters_encoder_choices))
    pool_sizes_encoder = pool_sizes_encoder_choices[pool_sizes_encoder_choice] * (
        len(cnn_filters_encoder_choices))
    pool_strides_encoder = (2,) * (len(cnn_filters_encoder_choices))

    # #############
    # ## Decoder ##
    # #############
    cnn_filters_decoder = cnn_filters_encoder
    cnn_kernels_decoder = cnn_kernels_encoder
    cnn_strides_decoder = cnn_strides_encoder

    model_features = cae_model_features(
        encoder_trainable=None,
        learning_rate=None,
        batch_size=None,
        scaler=None,
        activation_function=None,
        output_function=None,
        Pooling=None,
        cnn_filters_encoder=cnn_filters_encoder,
        cnn_kernels_encoder=cnn_kernels_encoder,
        cnn_strides_encoder=cnn_strides_encoder,
        pool_sizes_encoder=pool_sizes_encoder,
        pool_strides_encoder=pool_strides_encoder,
        cnn_filters_decoder=cnn_filters_decoder,
        cnn_kernels_decoder=cnn_kernels_decoder,
        cnn_strides_decoder=cnn_strides_decoder)

    return model_features


def train_earlystop(training_data,
                    training_keys,
                    testing_data,
                    testing_keys,
                    model,
                    optimizer,
                    num_epochs,
                    obj_cost,
                    earlystop_cost_fn,
                    earlystop_patience,
                    data_augmentation,
                    not_learning_patience=0,
                    not_learning_threshold=0,
                    verbose=True,
                    augment_testing_data=False,
                    fit_batch_verbose=5,
                    record_train_errors=False,):
    """
    @Author: Sam Dotson
    Trains the model to stop early to avoid overfitting.

    Parameters:
    -----------
    training_data : numpy matrix, float
        Data is a [nxm] numpy matrix of unprocessed gamma-ray spectra
    training_keys : list
        [nx1] list of keys that correspond to your training data.
    testing_data : numpy matrix
        Data is a [nxm] numpy matrix of unprocessed gamma-ray spectra
    testing_keys : list
        [nx1] list of keys that correspond to your testing data.
    model : object
        The model you are currently training.
    optimizer : tensorflow optimizer in tf.train.*
    num_epochs : int
        Total number of epochs training is allowed to run.
    obj_cost : string
        Main cost function the algorithm minimizes. examples are
        'self.mse' or 'self.cross_entropy'.
    earlystop_cost_fn : Tensorflow earlystop function
        Cost function used for early stopping. Examples are
        ``self.f1_error``, ``self.mse``, and ``self.cross_entropy``.
    earlystop_patience : int, optional
        Number of epochs training is allowed to run without improvment.
    data_augmentation : function
            Function used to apply data augmentation each training iteration.
    not_learning_patience : int, optional
        Max number of epochs to wait before checking if model is not
        learning.
    not_learning_threshold : float, optional
        If error at epoch ``not_learning_patience`` is above this, training
    verbose : int, optional
        Frequency that the errors are printed per iteration.
    augment_testing_data : boolean, optional
        Decides whether to augment testing data. Default is False.
    fit_batch_verbose : int, optional
        The frequency that the output of fit_batch is printed.
    record_train_errors : boolean, optional
        Decides whether to record training error. Default is False.
        If True, will print model errors after each epoch.

    Returns:
    --------
    costfunctionerr_test : array-like

    earlystoperr_test : array-like
    """

    costfunctionerr_test, earlystoperr_test = [], []

    objective_cost, earlystop_cost = model.fit_batch(
        (training_data, training_keys),
        (testing_data, testing_keys),
        optimizer,
        num_epochs=num_epochs,
        verbose=fit_batch_verbose,
        obj_cost=obj_cost,
        earlystop_cost_fn=earlystop_cost_fn,
        earlystop_patience=earlystop_patience,
        not_learning_patience=not_learning_patience,
        not_learning_threshold=not_learning_threshold,
        data_augmentation=data_augmentation,
        augment_testing_data=augment_testing_data,
        print_errors=True,
        record_train_errors=False,)
    
    
    # if length less than earlystop_patience, not_learning_patience was caught
    if len(objective_cost['test']) < earlystop_patience:
        costfunctionerr_test.append(objective_cost['test'][-1])
        earlystoperr_test.append(earlystop_cost['test'][-1])
    else:
        costfunctionerr_test.append(
            objective_cost['test'][-earlystop_patience])
        earlystoperr_test.append(earlystop_cost['test'][-earlystop_patience])

    if verbose is True:
        print("Test error at early stop: Objectives fctn: {0:.2f} Early stop"
              "fctn: {0:.2f}".format(float(costfunctionerr_test[-1]),
                                     float(earlystoperr_test[-1])))

    return costfunctionerr_test, earlystoperr_test
