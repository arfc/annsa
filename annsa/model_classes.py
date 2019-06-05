from __future__ import print_function
import pickle
import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tensorflow.image import resize_images
from tensorflow.keras.initializers import he_normal, glorot_normal
import time
from random import choice

# ##############################################################
# ##############################################################
# ##############################################################
# ######################### Base Class #########################
# ##############################################################
# ##############################################################
# ##############################################################


class BaseClass(object):
    def __init__(self):
        pass

    def predict_class(self, input_data, training=False):
        """
        Predicts the class (one-hot format) of some data.

        Uses the model to predict the class of some data. When
        predicting class, training needs to be false to avoid
        applying dropout.

        Parameters:
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        -------
        class_predictions : int
            [nxl] matrix of class predictions. n is number of samples, l is
            number of classes
        """
        model_predictions = self.forward_pass(input_data, training=training)
        class_predictions = tf.argmax(model_predictions, axis=-1)
        return class_predictions

    def cross_entropy(self, input_data, targets, training):
        """
        Computes the cross entropy error on some data and target.

        Parameters:
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
            [nxl] matrix of target outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.
        training : bool
            Turns training on or off for optional features.

        Returns:
        -------
        cross_entropy_loss : float
            The cross entropy between the
            model's prediction given the inputs and the ground-truth
            target.
        """

        logits = self.forward_pass(input_data, training=training)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=targets,
                logits=logits))
        return cross_entropy_loss

    def mse(self, input_data, targets, training):
        """
        Computes the mean squared error on some data and target.

        Parameters:
        -----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        target : narray, int
            [nxl] matrix of target outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        --------
        mean_squared_error : float
            The mean squared error between the model's prediction given
            the inputs and the ground-truth target.
        """
        # check if targets are a spectrum
        if targets.shape[1] > 1:
            targets_scaled = self.scaler.transform(targets)
        model_predictions = self.forward_pass(input_data, training=training)
        return tf.losses.mean_squared_error(targets_scaled, model_predictions)

    def f1_error(self, input_data, targets, training=False, average='micro'):
        """
        Computes the f1 score on some data and target.

        From the sklearn documentation:
            'micro' calculates metrics globally by counting the total true
            positives, false negatives and false positives.

            'macro' calculates metrics for each label, and find their
            unweighted mean. This does not take label imbalance into account.

            'weighted' calculates metrics for each label, and find their
            average weighted by support (the number of true instances for
            each label). This alters macro to account for label imbalance; it
            can result in an F-score that is not between precision and recall.

            'samples' calculates metrics for each instance, and find their
            average (only meaningful for multilabel classification where this
            differs from accuracy_score).

        Parameters:
        -----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        target : narray, int
            [nxl] matrix of target outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.
        training : bool, optional
            Turns training on or off for optional features.
        average : string, optional
            Type of averaging used by sklearns f1 score function.

        Returns:
        --------
        f1_error: float
            The f1_score of the model  on some data implemented using sklearn.

        """
        class_predictions = self.predict_class(input_data, training)
        class_truth = tf.argmax(targets, axis=1)
        f1_error = 1.0 - f1_score(class_truth,
                                  class_predictions,
                                  average='micro')
        return f1_error

    def default_data_augmentation(self, x):
        """
        Default data augmentation is an identity function.

        Parameters:
        -----------
        x : list, float
            Input data

        Returns:
        --------
        x : list, float
            Input data

        """
        return x

    def poisson_data_augmentation(self, x):
        """
        Returns input data with poisson noise for data augmentation.

        Parameters:
        -----------
        x : list, float
            Input data

        Returns:
        -------
        x : list, float
            Poisson sampled input data

        """
        return np.random.poisson(x)

    def grads_fn(self, input_data, target, cost):
        """
        Dynamically computes the gradients of the loss value
        with respect to the parameters of the model, in each
        forward pass.

        Parameters:
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        target : narray, int
            [nxl] matrix of target outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.
        cost : function
            Main cost function the algorithm minimizes. examples are
            'self.mse' or 'self.cross_entropy'.

        Returns:
        -------
        gradient : float
            The gradient of the loss function with respect to the
            weights.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target, cost)
        gradient = tape.gradient(loss, self.variables)
        return gradient

    def train_epoch(self,
                    train_dataset_tensor,
                    obj_cost,
                    optimizer,
                    data_augmentation):

        """
        Trains model on a single epoch using mini-batch training.

        Parameters:
        ----------
        train_dataset_tensor : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        obj_cost : function
            Main cost function the algorithm minimizes. examples are
            'self.mse' or 'self.cross_entropy'.
        optimizer : TensorFlow optimizer class
            The optimizer to use to optimize the algorithm. Choices include
            ``tf.train.AdamOptimizer``, ``tf.train.GradientDescentOptimizer``.
        data_augmentation : function
            Function used to apply data augmentation each training iteration.

        Returns:
        -------
        None
        """
        for (input_data, target) in tfe.Iterator(
                train_dataset_tensor.shuffle(int(1e8)).batch(self.batch_size)):
                input_data = data_augmentation(input_data)
                # check if data_augmentation returns separate source and
                # background
                if input_data.shape[1] == 2:
                    target = input_data[:, 1]
                    input_data = input_data[:, 0]
                grads = self.grads_fn(input_data,
                                      target,
                                      obj_cost)
                optimizer.apply_gradients(zip(grads, self.variables))
        return None

    def check_earlystop(self, earlystop_cost, earlystop_patience):
        """
        Checks if early stop condition is met and either continues or
            stops training.

        Parameters:
        ----------
        earlystop_cost : narray, float
            Array of cost values for each iteration used for early stopping.
        earlystop_patience : int
            Number of epochs training is allowed to run without improvment.
            If 0, training will run until max_time or num_epochs is passed.

        Returns:
        -------
        earlystop_flag: bool
            If true will end training. If false training continues.
        """
        earlystop_flag = 0
        argmin_error_in_patience_range = np.argmin(
            earlystop_cost[-earlystop_patience:])
        if (argmin_error_in_patience_range == 0):
            earlystop_flag = 1
        return earlystop_flag

    def fit_batch(self,
                  train_dataset,
                  test_dataset,
                  optimizer,
                  num_epochs=50,
                  verbose=50,
                  print_errors=True,
                  earlystop_patience=0,
                  max_time=300,
                  not_learning_patience=0,
                  not_learning_threshold=0,
                  obj_cost=None,
                  earlystop_cost_fn=None,
                  data_augmentation=None,
                  augment_testing_data=False,
                  record_train_errors=False,):
        """
        Function used to train the model.

        Parameters:
        ----------
        train_dataset : list, float, int
            Two element list of [data, keys] where data
            is a [nxm] numpy matrix of unprocessed gamma-ray spectra
            and keys are a [nxl] matrix of  target outputs.
        test_dataset : list, float, int
            Two element list of [data, keys] where data
            is a [nxm] numpy matrix of unprocessed gamma-ray spectra
            and keys are a [nxl] matrix of  target outputs.
        optimizer : TensorFlow optimizer class
            The optimizer to use to optimize the algorithm. Choices include
            ``tf.train.AdamOptimizer``, ``tf.train.GradientDescentOptimizer``.
        num_epochs : int, optional
            Total number of epochs training is allowed to run.
        verbose : int, optional
            Frequency that the errors are printed per iteration.
        print_errors : bool, optional
            If true, will print model errors after each epoch.
        earlystop_patience : int, optional
            Number of epochs training is allowed to run without improvment.
            If 0, training will run until max_time or num_epochs is passed.
        max_time : int, optional
            Max time in seconds training is allowed to run.
        not_learning_patience : int, optional
            Max number of epochs to wait before checking if model is not
            learning.
        not_learning_threshold : float, optional
            If error at epoch ``not_learning_patience`` is above this, training
            stops.
        earlystop_cost_fctn : function
            Cost function used for early stopping. Examples are
            ``self.f1_error``, ``self.mse``, and ``self.cross_entropy``.
        data_augmentation : function
            Function used to apply data augmentation each training iteration.

        Returns
        -------
        [objective_cost, earlystop_cost]: list of dictionaries
            If true will end training. If false training continues.
        """
        if not_learning_patience > earlystop_patience:
            print('Early stop patience must be greater than not learning'
                  'patience. Setting not learning patience to early stop')
            not_learning_patience = earlystop_patience

        earlystop_cost = {'train': [], 'test': []}
        objective_cost = {'train': [], 'test': []}

        train_dataset_tensor = tf.data.Dataset.from_tensor_slices(
            (tf.constant(train_dataset[0]), tf.constant(train_dataset[1])))

        time_start = time.time()
        for epoch in range(num_epochs):
            # Train through one epoch
            self.train_epoch(train_dataset_tensor,
                             obj_cost,
                             optimizer,
                             data_augmentation)
            if record_train_errors:
                training_data_aug = data_augmentation(train_dataset[0])
            if augment_testing_data:
                testing_data_aug = data_augmentation(test_dataset[0])
            else:
                testing_data_aug = test_dataset[0]

            training_key = train_dataset[1]
            testing_key = test_dataset[1]

            # check if data_augmentation returns separate source and background
            if record_train_errors:
                if training_data_aug.shape[1] == 2:
                    training_key = training_data_aug[:, 1]
                    training_data_aug = training_data_aug[:, 0]
            if testing_data_aug.shape[1] == 2:
                testing_key = testing_data_aug[:, 1]
                testing_data_aug = testing_data_aug[:, 0]

            # Record errors at each epoch
            if earlystop_patience:
                if record_train_errors:
                    earlystop_cost['train'].append(
                        earlystop_cost_fn(training_data_aug,
                                          training_key,
                                          training=False))
                else:
                    earlystop_cost['train'].append(0)
                earlystop_cost['test'].append(
                    earlystop_cost_fn(testing_data_aug,
                                      testing_key,
                                      training=False))
            else:
                earlystop_cost['train'].append(0)
                earlystop_cost['test'].append(0)
            if record_train_errors:
                objective_cost['train'].append(
                    self.loss_fn(training_data_aug,
                                 training_key,
                                 obj_cost,
                                 training=False))
            else:
                objective_cost['train'].append(0)
            objective_cost['test'].append(
                self.loss_fn(testing_data_aug,
                             testing_key,
                             obj_cost,
                             training=False))
            # Print erros at end of epoch
            if (print_errors and ((epoch+1) % verbose == 0)) is True:
                print('Epoch %d: CostFunc loss: %3.2f %3.2f, '
                      'EarlyStop loss: %3.2f %3.2f' % (
                          epoch+1,
                          objective_cost['train'][-1],
                          objective_cost['test'][-1],
                          earlystop_cost['train'][-1],
                          earlystop_cost['test'][-1]))
            # Apply early stopping with patience
            if (earlystop_patience and
                (epoch > earlystop_patience) and
                self.check_earlystop(earlystop_cost['test'],
                                     earlystop_patience)):
                break
            # Apply early stopping if not learning
            if (not_learning_patience and
               (epoch > not_learning_patience) and
               (earlystop_cost['test'][-1] > not_learning_threshold)):
                break

        return [objective_cost, earlystop_cost]

# ##############################################################
# ##############################################################
# ##############################################################
# ##################### Dense Architecture #####################
# ##############################################################
# ##############################################################
# ##############################################################


class DNN(tf.keras.Model, BaseClass):
    """
    Defines dense neural network structure, loss functions, training functions.

    DOCSTRING

    Under the class -- list the member functions and a short
    summary of what they do!

    Functions:
    ----------

    '__init__' : constructor
    'forward_pass' : Runs a forward pass throught the network
    'loss_fn' : 

    

    """
    def __init__(self, model_features):
        """
        Initializes dnn structure with model features.

        Parameters:
        -----------
        model_features : Class dnn_model_features

        """
        super(DNN, self).__init__()
        """
        Define here the layers used during the forward-pass of the neural
        network.

        """
        self.l2_regularization_scale = model_features.l2_regularization_scale
        dropout_probability = model_features.dropout_probability
        self.dense_nodes = model_features.dense_nodes
        self.batch_size = model_features.batch_size
        self.scaler = model_features.scaler
        output_size = model_features.output_size
        activation_function = model_features.activation_function
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=self.l2_regularization_scale)

        if activation_function == tf.nn.relu:
            kernel_initializer = he_normal()
        else:
            kernel_initializer = glorot_normal()

        # Define hidden layers.
        self.dense_layers = {}
        self.drop_layers = {}
        for layer, nodes in enumerate(self.dense_nodes):

            self.dense_layers[str(layer)] = tf.layers.Dense(
                nodes,
                activation=activation_function,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=regularizer)
            self.drop_layers[str(layer)] = tf.layers.Dropout(
                dropout_probability)
        self.output_layer = tf.layers.Dense(output_size, activation=None)

    def forward_pass(self, input_data, training):
        """
        Runs a forward-pass through the network.

        Parameters:
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        -------
        logits : tensor
            Output layer of the network.

        """
        x = self.scaler.transform(input_data)
        x = tf.reshape(x, [-1, 1, x.shape[1]])
        for layer, nodes in enumerate(self.dense_nodes):
            x = self.dense_layers[str(layer)](x)
            x = self.drop_layers[str(layer)](x, training=training)
        logits = self.output_layer(x)
        return logits

    def loss_fn(self, input_data, targets, cost, training=True):
        """
        Defines the loss function, including regularization, used during
        training.

        Parameters:
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        target : narray, int
            [nxl] matrix of target outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.
        cost : string
            Main cost function the algorithm minimizes. examples are
            'self.mse' or 'self.cross_entropy'.
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        -------
        loss : TensorFlow float
            The value of all network losses computed during training

        """
        loss = cost(input_data, targets, training)
        if self.l2_regularization_scale > 0:
            for layer, nodes in enumerate(self.dense_layers):
                loss += self.dense_layers[str(layer)].losses

        return loss


class dnn_model_features(object): #DOCSTRING
	"""
	Defines the features for the dense neural network.
	"""


    def __init__(self, learning_rate,
                 l2_regularization_scale,
                 dropout_probability,
                 batch_size,
                 output_size,
                 dense_nodes,
                 activation_function,
                 scaler
                 ):

        """
        @author: Sam Dotson

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
        dropout_probability : float
            The probability that any neuron will be temporarily turned 
            off during training. Example: dropout_probability = 0.4 
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
            Example: tf.nn.relu
        scaler : tensorflow scaling function
            See documentation for tensorflow scaling

        """
        self.learning_rate = learning_rate
        self.l2_regularization_scale = l2_regularization_scale
        self.dropout_probability = dropout_probability
        self.batch_size = batch_size
        self.output_size = output_size
        self.dense_nodes = dense_nodes
        self.activation_function = activation_function
        self.scaler = scaler

# ##############################################################
# ##############################################################
# ##############################################################
# ################# Convolutional Architecture #################
# ##############################################################
# ##############################################################
# ##############################################################


class CNN1D(tf.keras.Model, BaseClass):
    """
    DOCSTRING

    Under the class -- list the member functions and a short
    summary of what they do!
    """

    def __init__(self, model_features):
        super(CNN1D, self).__init__()
        """
        Define here the layers used during the forward-pass of the neural
        network.

        """

        #=========================Notes======================#
        #
        #
        #
        #
        #
        #
        #
        #
        #================Delete this section later===========#
        
        self.batch_size = model_features.batch_size
        output_size = model_features.output_size
        self.scaler = model_features.scaler
        trainable = model_features.trainable
        activation_function = model_features.activation_function
        output_function = model_features.output_function
        cnn_filters = model_features.cnn_filters
        cnn_kernel = model_features.cnn_kernel
        cnn_strides = model_features.cnn_strides
        pool_size = model_features.pool_size
        pool_strides = model_features.pool_strides
        Pooling = model_features.Pooling
        output_size = model_features.output_size
        dense_nodes = model_features.dense_nodes
        self.l2_regularization_scale = model_features.l2_regularization_scale
        dropout_probability = model_features.dropout_probability

        regularizer = tf.contrib.layers.l2_regularizer(
            scale=self.l2_regularization_scale)

        if activation_function == tf.nn.relu:
            kernel_initializer = he_normal()
        else:
            kernel_initializer = glorot_normal()

        # Define hidden layers for encoder
        self.conv_layers = {}
        self.pool_layers = {}
        for layer in range(len(cnn_filters)):
            self.conv_layers[str(layer)] = tf.layers.Conv1D(
                filters=cnn_filters[layer],
                kernel_size=cnn_kernel[layer],
                strides=1,
                padding='same',
                kernel_initializer=kernel_initializer,
                activation=activation_function,
                trainable=trainable)
            self.pool_layers[str(layer)] = Pooling(
                pool_size=pool_size[layer],
                strides=pool_strides[layer],
                padding='same')

        self.dense_layers = {}
        self.drop_layers = {}
        for layer in range(len(dense_nodes)):
            self.dense_layers[str(layer)] = tf.layers.Dense(
                dense_nodes[layer],
                activation=activation_function,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=regularizer)
            self.drop_layers[str(layer)] = tf.layers.Dropout(
                dropout_probability)
        self.output_layer = tf.layers.Dense(output_size,
                                            activation=output_function)

    def loss_fn(self, input_data, targets, cost, training=True):
        """
        Defines the loss function, including regularization, used during
        training.

        Parameters:
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        target : narray, int
            [nxl] matrix of target outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.
        cost : string
            Main cost function the algorithm minimizes. examples are
            'self.mse' or 'self.cross_entropy'.
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        -------
        loss : TensorFlow float
            The value of all network losses computed during training

        """
        loss = cost(input_data, targets, training)
        if self.l2_regularization_scale > 0:
            for layer in self.dense_layers.keys():
                loss += self.dense_layers[layer].losses
        return loss

    def forward_pass(self, input_data, training):
        """ 
        Runs a forward-pass through the network. Outputs are defined by
        'output_layer' in the model's structure. The scaler is applied
        here.

        Parameters:
        -----------
        input_data : [nxm] matrix of unprocessed gamma-ray spectra. n is
            number of samples, m is length of a spectrum
        training : Binary (True or False). If true, dropout is applied.
            When training weights this needs to be true for dropout to
            work.

        Returns:
        --------
        logits : [nxl] matrix of model outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.

        """
        x = self.scaler.transform(input_data)
        x = tf.reshape(x, [-1, x.shape[1], 1])
        for layer in self.conv_layers.keys():
            x = self.conv_layers[str(layer)](x)
            x = self.pool_layers[str(layer)](x)
        x = tf.layers.flatten(x)
        for layer in self.dense_layers.keys():
            x = self.dense_layers[str(layer)](x)
            x = self.drop_layers[str(layer)](x, training)
        logits = self.output_layer(x)
        return logits


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
                 dropout_probability,
                 scaler,
                 Pooling,
                 cnn_filters,
                 cnn_kernel,
                 cnn_strides,
                 pool_size,
                 pool_strides,
                 dense_nodes,
                 activation_function,
                 ):

        """
        @Author: Sam Dotson

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
        dropout_probability : float
            The probability that any neuron will be temporarily turned 
            off during training. Example: dropout_probability = 0.4 
            means there is a 40% probability of the neuron turning off. 
        scaler : Tensorflow scaling function
        batch_size : int 
            'batch_size' is the number of spectra/images being passed 
            through the network at once. For reference, one epoch is 
            the size of all training data. 
        Pooling : Tensorflow pooling function
        cnn_filters : int
            The number of filters in a convolutional layer.
        cnn_strides: int
            The stride size of each filter. How far it shifts per 
            iteration. Typically stride size is one. 
        pool_size : int or array/tuple
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

        """
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.batch_size = batch_size
        self.output_size = output_size
        self.output_function = output_function
        self.l2_regularization_scale = l2_regularization_scale
        self.dropout_probability = dropout_probability
        self.scaler = scaler
        self.Pooling = Pooling
        self.cnn_filters = cnn_filters
        self.cnn_kernel = cnn_kernel
        self.cnn_strides = cnn_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.dense_nodes = dense_nodes
        self.activation_function = activation_function


def generate_random_cnn1d_architecture(cnn_filters_choices,
                                       cnn_kernel_choices,
                                       pool_size_choices):
    """
    Generates a random 1d convolutional neural network based on a 
    set of predefined architectures.

    @author: Sam Dotson

    Parameters:
    ----------- 
    cnn_filters_choices : 1-D array-like or int
        Input a choice of ..............

    cnn_kernel_choices : 1-D array-like or int
        Input a choice of kernel (filter) sizes.

    pool_size_choices : 1-D array-like or int
        Input a choice of pooling sizes.

    Returns:
    --------
    model_features : class
        Class that describes the structure of a 1D convolution 
        neural network.
    """

    cnn_filters = choice(cnn_filters_choices)
    cnn_kernel_choice = choice(cnn_kernel_choices)
    pool_size_choice = choice(pool_size_choices)

    cnn_kernel = cnn_kernel_choice*(len(cnn_filters))
    cnn_strides = (1,)*(len(cnn_filters))
    pool_size = pool_size_choice*(len(cnn_filters))
    pool_strides = (2,)*(len(cnn_filters))

    number_layers = np.random.randint(1, 4)
    dense_nodes = (10**np.random.uniform(1,
                                         np.log10(1024/(2**len(
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
            dropout_probability=None,
            cnn_filters=cnn_filters,
            cnn_kernel=cnn_kernel,
            cnn_strides=cnn_strides,
            pool_size=pool_size,
            pool_strides=pool_strides,
            dense_nodes=dense_nodes
            )

    return model_features


# ##############################################################
# ##############################################################
# ##############################################################
# ##################### Dense Autoencoder ######################
# ##############################################################
# ##############################################################
# ##############################################################


class DAE(tf.keras.Model, BaseClass):
    """
    DOCSTRING

    Under the class -- list the member functions and a short
    summary of what they do!

    """


    def __init__(self, model_features):
        super(DAE, self).__init__()
        """
        Define here the layers used during the forward-pass of the neural
        network.

        """
        self.batch_size = model_features.batch_size
        self.scaler = model_features.scaler
        dropout_probability = model_features.dropout_probability
        activation_function = model_features.activation_function
        output_function = model_features.output_function
        output_size = model_features.output_size

        self.l1_regularization_scale = model_features.l1_regularization_scale
        self.regularizer = tf.contrib.layers.l1_regularizer(
            scale=self.l1_regularization_scale)
        self.dense_nodes_encoder = model_features.dense_nodes_encoder
        self.dense_nodes_decoder = model_features.dense_nodes_decoder

        if activation_function == tf.nn.relu:
            kernel_initializer = he_normal()
        else:
            kernel_initializer = glorot_normal()

        # Define Hidden layers for encoder
        self.dense_layers_encoder = {}
        self.dropout_layers_encoder = {}
        for layer, nodes in enumerate(self.dense_nodes_encoder):
            self.dense_layers_encoder[str(layer)] = tf.layers.Dense(
                nodes,
                activation=activation_function,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=self.regularizer)
            self.dropout_layers_encoder[str(layer)] = tf.layers.Dropout(
                dropout_probability)

        # Define Hidden layers for decoder
        self.dense_layers_decoder = {}
        self.dropout_layers_decoder = {}
        for layer, nodes in enumerate(self.dense_nodes_decoder):
            self.dense_layers_decoder[str(layer)] = tf.layers.Dense(
                nodes,
                activation=activation_function,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=self.regularizer)
            self.dropout_layers_decoder[str(layer)] = tf.layers.Dropout(
                dropout_probability)

        # Output layer. No activation.
        self.output_layer = tf.layers.Dense(output_size,
                                            activation=output_function)

    def encoder(self, input_data, training=True):
        """
        Runs a forward-pass through only the encoder.
        Note, training is currently not used here.

        Parameters:
        -----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        --------
        encoding : tensor
            The DAE's encoding of the input.

        """
        x = self.scaler.transform(input_data)
        x = tf.reshape(x, [-1, x.shape[1]])
        for layer, nodes in enumerate(self.dense_nodes_encoder):
            x = self.dense_layers_encoder[str(layer)](x)
            x = self.dropout_layers_encoder[str(layer)](x, training)
        encoding = x
        return encoding

    def decoder(self, encoding, training=True):
        """
        Runs a forward-pass through only the decoder.
        Note, training is currently not used here.

        Parameters:
        -----------
        encoding : 2D tensor, float
            Input tensor of shape (n_samples, size_encoding).
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        --------
        decoding : tensor
            The DAE's decoding of the encoding.

        """
        x = encoding
        for layer, nodes in enumerate(self.dense_nodes_decoder):
            x = self.dense_layers_decoder[str(layer)](x)
            x = self.dropout_layers_decoder[str(layer)](x, training)
        decoding = self.output_layer(x)
        return decoding

    def total_activity(self, input_data, training=False):
        """ 
        Calculates the total network activity (l1 activation) on
        some input data.
            
        Parameters:
        -----------
            input_data : 2D tensor of shape (n_samples, n_features).
        
        Returns:
        --------
        average_activity : float
            Average total l1 activation.

        """
        activity = 0
        x = self.scaler.transform(input_data)
        x = tf.reshape(x, [-1, x.shape[1]])
        for layer, nodes in enumerate(self.dense_nodes_encoder):
            x = self.dense_layers_encoder[str(layer)](x)
            activity += np.sum(np.abs(x))
            x = self.dropout_layers_encoder[str(layer)](x, training)
        for layer, nodes in enumerate(self.dense_nodes_decoder):
            x = self.dense_layers_decoder[str(layer)](x)
            activity += np.sum(np.abs(x))
            x = self.dropout_layers_encoder[str(layer)](x, training)
        average_activity = activity/int(input_data.shape[0])
        return average_activity

    def forward_pass(self, input_data, training):
        """
        Runs a forward-pass through the encoder and decoder.

        Parameters:
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        -------
        decoding : tensor
            The DAE's decoding of the encoding.

        """
        encoding = self.encoder(input_data, training)
        decoding = self.decoder(encoding, training)
        return decoding

    def loss_fn(self, input_data, targets, cost, training=True):
        """
        Defines the loss function, including regularization, used during
        training.

        Parameters:
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        target : narray, int
            [nxl] matrix of target outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.
        cost : string
            Main cost function the algorithm minimizes. examples are
            'self.mse' or 'self.cross_entropy'.
        training : bool, optional
            Turns training on or off for optional features.

        Returns:
        -------
        loss : TensorFlow float
            The value of all network losses computed during training

        """
        loss = cost(input_data, targets, training)
        loss += self.l1_regularization_scale * self.total_activity(input_data)
        return loss


class dae_model_features(object):

    def __init__(self,
                 learning_rate,
                 l1_regularization_scale,
                 dropout_probability,
                 batch_size,
                 dense_nodes_encoder,
                 dense_nodes_decoder,
                 scaler,
                 activation_function,
                 output_function,
                 output_size,
                 ):
        #DOCSTRING
        self.learning_rate = learning_rate
        self.l1_regularization_scale = l1_regularization_scale
        self.dropout_probability = dropout_probability
        self.batch_size = batch_size
        self.dense_nodes_encoder = dense_nodes_encoder
        self.dense_nodes_decoder = dense_nodes_decoder
        self.scaler = scaler
        self.activation_function = activation_function
        self.output_function = output_function
        self.output_size = output_size

# ##############################################################
# ##############################################################
# ##############################################################
# ################## Convolution Autoencoder ###################
# ##############################################################
# ##############################################################
# ##############################################################


class CAE(tf.keras.Model, BaseClass): #DOCSTRING
    """
    DOCSTRING

    Under the class -- list the member functions and a short
    summary of what they do!
    
    """
    def __init__(self, model_features):
        super(CAE, self).__init__()
        """
        Define here the layers used during the forward-pass of the neural
        network.

        """
        self.batch_size = model_features.batch_size
        self.scaler = model_features.scaler
        encoder_trainable = model_features.encoder_trainable
        activation_function = model_features.activation_function
        output_function = model_features.output_function
        cnn_filters_encoder = model_features.cnn_filters_encoder
        cnn_kernel_encoder = model_features.cnn_kernel_encoder
        cnn_strides_encoder = model_features.cnn_strides_encoder
        pool_size_encoder = model_features.pool_size_encoder
        pool_strides_encoder = model_features.pool_strides_encoder
        cnn_filters_decoder = model_features.cnn_filters_decoder
        cnn_kernel_decoder = model_features.cnn_kernel_decoder
        cnn_strides_decoder = model_features.cnn_strides_decoder
        Pooling = model_features.Pooling

        if activation_function == tf.nn.relu:
            kernel_initializer = he_normal()
        else:
            kernel_initializer = glorot_normal()

        # Define hidden layers for encoder
        self.conv_layers_encoder = {}
        self.pool_layers_encoder = {}
        for layer in range(len(cnn_filters_encoder)):
            self.conv_layers_encoder[str(layer)] = tf.layers.Conv1D(
                filters=cnn_filters_encoder[layer],
                kernel_size=cnn_kernel_encoder[layer],
                strides=1,
                padding='same',
                kernel_initializer=kernel_initializer,
                activation=activation_function,
                trainable=encoder_trainable)
            self.pool_layers_encoder[str(layer)] = Pooling(
                pool_size=pool_size_encoder[layer],
                strides=pool_strides_encoder[layer],
                padding='same')

        # self.conv_layers_encoder[str(layer+1)] = tf.layers.Conv1D(
        #     filters=cnn_filters_encoder[-1],
        #     kernel_size=cnn_kernel_encoder[-1],
        #     strides=cnn_strides_encoder[-1],
        #     padding='same',
        #     kernel_initializer=he_normal(),
        #     activation=activation_function,
        #     trainable=encoder_trainable)

        # Define hidden layers for encoder
        self.conv_layers_decoder = {}
        for layer in range(len(cnn_filters_decoder)-1):
            self.conv_layers_decoder[str(layer)] = tf.layers.Conv1D(
                filters=cnn_filters_decoder[layer],
                kernel_size=cnn_kernel_decoder[layer],
                strides=cnn_strides_decoder[layer],
                padding='same',
                kernel_initializer=kernel_initializer,
                activation=activation_function)
        self.conv_layers_decoder[str(layer+1)] = tf.layers.Conv1D(
            filters=cnn_filters_decoder[-1],
            kernel_size=cnn_kernel_decoder[-1],
            strides=cnn_strides_decoder[-1],
            padding='same',
            kernel_initializer=he_normal(),
            activation=output_function)

    def encoder(self, input_data, training=True):
        """
        Runs a forward-pass through only the encoder.
        Note, training is currently not used here.

        Parameters
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        training : bool, optional
            Turns training on or off for optional features.

        Returns
        -------
        encoding : tensor
            The CAE's encoding of the input.

        """
        x = self.scaler.transform(input_data)
        x = tf.reshape(x, [-1, x.shape[1], 1])
        layer_list = list(self.conv_layers_encoder.keys())
        layer_list.sort()
        for layer in layer_list:
            x = self.conv_layers_encoder[str(layer)](x)
            x = self.pool_layers_encoder[str(layer)](x)
        encoding = x

        return encoding

    def decoder(self, encoding, training=True):
        """
        Runs a forward-pass through only the decoder.
        Note, training is currently not used here.

        Parameters
        ----------
        encoding : 2D tensor, float
            Input tensor of shape (n_samples, size_encoding).
        training : bool, optional
            Turns training on or off for optional features.

        Returns
        -------
        decoding : tensor
            The CAE's decoding of the encoding.

        """
        x = encoding
        layer_list = list(self.conv_layers_decoder.keys())
        layer_list.sort()
        for layer in layer_list:
            x = tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
            # upscale image by 2x
            x = resize_images(x, [x.shape[1]*2, 1])
            x = tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))
            x = self.conv_layers_decoder[str(layer)](x)
            'decoder conv ' + str(x.shape)
        decoding = tf.reshape(x, (x.shape[0], x.shape[1]))
        'decoder FINAL ' + str(x.shape)
        return decoding

    def forward_pass(self, input_data, training):
        """
        Runs a forward-pass through the encoder and decoder.

        Parameters
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        training : bool, optional
            Turns training on or off for optional features.

        Returns
        -------
        decoding : tensor
            The CAE's decoding of the encoding.

        """
        encoding = self.encoder(input_data, training)
        decoding = self.decoder(encoding, training)
        return decoding

    def loss_fn(self, input_data, targets, cost, training=True):
        """
        Defines the loss function, including regularization, used during
        training.

        Parameters
        ----------
        input_data : 2D tensor, float
            Input tensor of shape (n_samples, n_features). Tensor is
            unprocessed gamma-ray spectra (counts per channel).
        target: narray, int
            [nxl] matrix of target outputs. n is number of samples,
            same as n in input. l is the number of elements in each
            output . If using one-hot encoding l is equal to number
            of classes. If used as autoencoder l is equal to m.
        cost: string
            Main cost function the algorithm minimizes. examples are
            'self.mse' or 'self.cross_entropy'.
        training : bool, optional
            Turns training on or off for optional features.

        Returns
        -------
        loss : TensorFlow float
            The value of all network losses computed during training

        """
        loss = cost(input_data, targets, training)
        return loss


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
                 cnn_kernel_encoder,
                 cnn_strides_encoder,
                 pool_size_encoder,
                 pool_strides_encoder,
                 cnn_filters_decoder,
                 cnn_kernel_decoder,
                 cnn_strides_decoder
                 ):
        #DOCSTRING
        self.learning_rate = learning_rate
        self.encoder_trainable = encoder_trainable
        self.batch_size = batch_size
        self.scaler = scaler
        self.activation_function = activation_function
        self.output_function = output_function
        self.Pooling = Pooling
        self.cnn_filters_encoder = cnn_filters_encoder
        self.cnn_kernel_encoder = cnn_kernel_encoder
        self.cnn_strides_encoder = cnn_strides_encoder
        self.pool_size_encoder = pool_size_encoder
        self.pool_strides_encoder = pool_strides_encoder
        self.cnn_filters_decoder = cnn_filters_decoder
        self.cnn_kernel_decoder = cnn_kernel_decoder
        self.cnn_strides_decoder = cnn_strides_decoder


def generate_random_cae_architecture(cnn_filters_encoder_choices,
                                     cnn_kernel_encoder_choices,
                                     pool_size_encoder_choices):
    """
    Generates a random convolutional autoencoder based on given architecture
    choices.

    Parameters
    ----------
    cnn_filters_encoder_choices : list, int
        List of lists containing number of filters in each layer.
    cnn_kernel_encoder_choices: list, int
        Kernel sizes
    pool_size_encoder_choices: list, int
        Pooling sizes
    Returns
    -------
    cae_model_features : class
        Class that describes the structure of a CAE.

    """

    cnn_filters_encoder_choice = np.random.randint(
        len(cnn_filters_encoder_choices))
    cnn_kernel_encoder_choice = np.random.randint(
        len(cnn_kernel_encoder_choices))
    pool_size_encoder_choice = np.random.randint(
        len(pool_size_encoder_choices))

    # #############
    # ## Encoder ##
    # #############
    cnn_filters_encoder = cnn_filters_encoder_choices[
        cnn_filters_encoder_choice]
    cnn_kernel_encoder = cnn_kernel_encoder_choices[
        cnn_kernel_encoder_choice]*(len(cnn_filters_encoder_choices))
    cnn_strides_encoder = (1,)*(len(cnn_filters_encoder_choices))
    pool_size_encoder = pool_size_encoder_choices[pool_size_encoder_choice]*(
        len(cnn_filters_encoder_choices))
    pool_strides_encoder = (2,)*(len(cnn_filters_encoder_choices))

    # #############
    # ## Decoder ##
    # #############
    cnn_filters_decoder = cnn_filters_encoder
    cnn_kernel_decoder = cnn_kernel_encoder
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
            cnn_kernel_encoder=cnn_kernel_encoder,
            cnn_strides_encoder=cnn_strides_encoder,
            pool_size_encoder=pool_size_encoder,
            pool_strides_encoder=pool_strides_encoder,
            cnn_filters_decoder=cnn_filters_decoder,
            cnn_kernel_decoder=cnn_kernel_decoder,
            cnn_strides_decoder=cnn_strides_decoder)

    return model_features

# ##############################################################
# ##############################################################
# ##############################################################
# #################### Training Functions ######################
# ##############################################################
# ##############################################################
# ##############################################################


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
    #DOCSTRING

    costfunctionerr_test, earlystoperr_test = [], []

    tf.reset_default_graph()
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


def save_model(folder_name, model_id, model, model_features): #DOCSTRING
    saver = tfe.Saver(model.variables)
    saver.save(folder_name+'/'+model_id)
    with open(folder_name+'/'+model_id+'_features', 'w') as f:
        pickle.dump(model_features, f)


def load_model(model_folder,
               model_id,
               model_class,
               training_data_length=1024,
               training_key_length=57):
    #DOCSTRING
    # load model features (number of layers, nodes)
    with open('./'+model_folder+'/'+model_id+'_features') as f:
        new_model_features = pickle.load(f)

    # Initialize variables by running a single training iteration
    tf.reset_default_graph()
    optimizer = tf.train.AdamOptimizer(new_model_features.learning_rate)
    model = model_class(new_model_features)

    dummy_data = np.ones([10, training_data_length])

    X_tensor = tf.constant(dummy_data)
    y_tensor = tf.constant(np.ones([10, training_key_length]))
    dummy_train_dataset = tf.data.Dataset.from_tensor_slices((X_tensor,
                                                              y_tensor))
    dummy_test_dataset = (dummy_data, np.ones([10, training_key_length]))

    _, _ = model.fit_batch(dummy_train_dataset,
                           dummy_test_dataset,
                           optimizer,
                           num_epochs=1,
                           verbose=1,
                           print_errors=False)

    # Restore saved variables
    saver = tfe.Saver(model.variables)
    saver.restore('./'+model_folder+'/'+model_id)

    return model, new_model_features.scaler


class_isotopes = ['Am241',
                  'Ba133',
                  'Co57',
                  'Co60',
                  'Cs137',
                  'Cr51',
                  'Eu152',
                  'Ga67',
                  'I123',
                  'I125',
                  'I131',
                  'In111',
                  'Ir192',
                  'U238',
                  'Lu177m',
                  'Mo99',
                  'Np237',
                  'Pd103',
                  'Pu239',
                  'Pu240',
                  'Ra226',
                  'Se75',
                  'Sm153',
                  'Tc99m',
                  'Xe133',
                  'Tl201',
                  'Tl204',
                  'U233',
                  'U235',
                  'shielded_Am241',
                  'shielded_Ba133',
                  'shielded_Co57',
                  'shielded_Co60',
                  'shielded_Cs137',
                  'shielded_Cr51',
                  'shielded_Eu152',
                  'shielded_Ga67',
                  'shielded_I123',
                  # 'shielded_I125',
                  # Removed due to max gamma energy being too weak.
                  # Any shielding fully attenuates.
                  'shielded_I131',
                  'shielded_In111',
                  'shielded_Ir192',
                  'shielded_U238',
                  'shielded_Lu177m',
                  'shielded_Mo99',
                  'shielded_Np237',
                  'shielded_Pd103',
                  'shielded_Pu239',
                  'shielded_Pu240',
                  'shielded_Ra226',
                  'shielded_Se75',
                  'shielded_Sm153',
                  'shielded_Tc99m',
                  'shielded_Xe133',
                  'shielded_Tl201',
                  'shielded_Tl204',
                  'shielded_U233',
                  'shielded_U235']
