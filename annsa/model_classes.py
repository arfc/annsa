import pickle
import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
from sklearn.model_selection import StratifiedKFold
from tensorflow.image import resize_images
from tensorflow.keras.initializers import he_normal
import time

# ##############################################################
# ##############################################################
# ##############################################################
# ##################### Dense Archetecture #####################
# ##############################################################
# ##############################################################
# ##############################################################


class dnn(tf.keras.Model):
    """Defines dense NN structure and training functions.

    """
    def __init__(self, model_features):
        """Initializes dnn structure with model features.

        Args:
            model_features: Class that contains variables
            to construct the dense neural network.

        """
        super(dnn, self).__init__()
        """ Define here the layers used during the forward-pass
            of the neural network.
        """

        self.l2_regularization_scale = model_features.l2_regularization_scale
        dropout_probability = model_features.dropout_probability
        self.dense_nodes = model_features.dense_nodes
        self.batch_size = model_features.batch_size
        self.scaler = model_features.scaler
        output_size = model_features.output_size
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=self.l2_regularization_scale)

        # Define hidden layers.
        self.dense_layers = {}
        self.drop_layers = {}
        for layer, nodes in enumerate(self.dense_nodes):

            self.dense_layers[layer] = tf.layers.Dense(
                nodes,
                activation=tf.nn.relu,
                kernel_initializer=he_normal(),
                kernel_regularizer=regularizer)
            self.drop_layers[layer] = tf.layers.Dropout(dropout_probability)
        self.output_layer = tf.layers.Dense(output_size, activation=None)

    def predict_logits(self, input_data, training=True):
        """ Runs a forward-pass through the network. Only outputs logits for
            loss function. This is because
            tf.nn.softmax_cross_entropy_with_logits_v2 calculates softmax
            internally. Note, training is true here to turn dropout on.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
        """
        x = self.scaler.transform(input_data)
        x = tf.reshape(x, [-1, 1, x.shape[1]])
        for layer, nodes in enumerate(self.dense_nodes):
            x = self.dense_layers[layer](x)
            x = self.drop_layers[layer](x, training)
        logits = self.output_layer(x)

        return logits

    def predict(self, input_data, training=False):
        """ Runs a forward-pass through the network and uses softmax output.
            Dropout training is off, this is not used for gradient calculations
            in loss function.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
        """
        return tf.nn.softmax(self.predict_logits(self, input_data, training))

    def loss_fn(self, input_data, target, training=True):
        """ Defines the loss function used during
            training.
        """
        logits = self.predict_logits(input_data, training)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=target,
                logits=logits))
        loss = cross_entropy_loss
        if self.l2_regularization_scale > 0:
            for layer, nodes in enumerate(self.dense_layers):
                loss += self.dense_layers[layer].losses

        return loss

    def grads_fn(self, input_data, target):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)

        return tape.gradient(loss, self.variables)

    def fit_batch(self,
                  train_dataset,
                  test_dataset,
                  optimizer,
                  num_epochs=50,
                  verbose=50,
                  print_errors=True,
                  early_stopping_patience=0,
                  max_time=300):
        """ Function to train the model, using the selected optimizer and
            for the desired number of epochs. Uses early stopping with
            patience.
        """
        all_loss_train = []
        all_loss_test = []
        time_start = time.time()
        early_stopping_flag = False
        for epoch in range(num_epochs):
            for (input_data, target) in tfe.Iterator(
                            train_dataset.shuffle(1e8).batch(self.batch_size)):
                input_data = np.random.poisson(input_data).astype(float)
                grads = self.grads_fn(input_data, target)
                optimizer.apply_gradients(zip(grads, self.variables))
                all_loss_train.append(
                    self.loss_fn(input_data, target, training=False).numpy())
                if early_stopping_patience == 0:
                    all_loss_test.append(self.loss_fn(test_dataset[0],
                                                      test_dataset[1],
                                                      training=False).numpy())

            # Save error for early stopping
            if early_stopping_patience != 0:
                all_loss_test.append(self.loss_fn(test_dataset[0],
                                                  test_dataset[1],
                                                  training=False).numpy())
                tmp_min_test_error = all_loss_test[-1]
                if epoch == 0:
                    patience_counter = 0
                    min_test_error = tmp_min_test_error
                elif (epoch > 0) and (tmp_min_test_error < min_test_error):
                    min_test_error = tmp_min_test_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                time_taken = time.time()-time_start
                if ((patience_counter >= early_stopping_patience) or
                        (time_taken > max_time)):
                    early_stopping_flag = True

            if (print_errors and
                    ((epoch == 0) | ((epoch+1) % verbose == 0))) is True:
                print('Loss at epoch %d: %3.2f %3.2f' % (epoch+1,
                                                         all_loss_train[-1],
                                                         all_loss_test[-1]))
            if early_stopping_flag is True:
                break
        return all_loss_train, all_loss_test


class dnn_model_features(object):
    def __init__(self, learining_rate,
                 l2_regularization_scale,
                 dropout_probability,
                 batch_size,
                 output_size,
                 dense_nodes,
                 scaler
                 ):
        self.learining_rate = learining_rate
        self.l2_regularization_scale = l2_regularization_scale
        self.dropout_probability = dropout_probability
        self.batch_size = batch_size
        self.output_size = output_size
        self.dense_nodes = dense_nodes
        self.scaler = scaler

# ##############################################################
# ##############################################################
# ##############################################################
# ################# Convolutional Archetecture #################
# ##############################################################
# ##############################################################
# ##############################################################


class cnn1d(tf.keras.Model):
    def __init__(self, model_features):
        super(cnn1d, self).__init__()
        """ Define here the layers used during the forward-pass
            of the neural network.
        """
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

        # Define hidden layers for encoder
        self.conv_layers = {}
        self.pool_layers = {}
        for layer in range(len(cnn_filters)):
            self.conv_layers[str(layer)] = tf.layers.Conv1D(
                filters=cnn_filters[layer],
                kernel_size=cnn_kernel[layer],
                strides=1,
                padding='same',
                kernel_initializer=he_normal(),
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
                activation=tf.nn.relu,
                kernel_initializer=he_normal(),
                kernel_regularizer=regularizer)
            self.drop_layers[str(layer)] = tf.layers.Dropout(
                dropout_probability)
        self.output_layer = tf.layers.Dense(output_size,
                                            activation=output_function)

    def predict_logits(self, input_data, training=True):
        """ Runs a forward-pass through the network. Only outputs logits for
            loss function. This is because
            tf.nn.softmax_cross_entropy_with_logits_v2 calculates softmax
            internally. Note, training is true here to turn dropout on.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
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

    def predict(self, input_data, training=False):
        """ Runs a forward-pass through the network and uses softmax output.
            Dropout training is off, this is not used for gradient calculations
            in loss function.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
        """
        return tf.nn.softmax(self.predict_logits(self, input_data, training))

    def loss_fn(self, input_data, target, training=True):
        """ Defines the loss function used during
            training.
        """
        logits = self.predict_logits(input_data, training)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=target,
                logits=logits))
        loss = cross_entropy_loss

        if self.l2_regularization_scale > 0:
            for layer in self.dense_layers.keys():
                loss += self.dense_layers[layer].losses
        return loss

    def grads_fn(self, input_data, target):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)

        return tape.gradient(loss, self.variables)

    def fit_batch(self,
                  train_dataset,
                  test_dataset,
                  optimizer,
                  num_epochs=50,
                  verbose=50,
                  print_errors=True,
                  early_stopping_patience=0,
                  max_time=300):
        """ Function to train the model, using the selected optimizer and
            for the desired number of epochs. Uses early stopping with
            patience.
        """
        all_loss_train = []
        all_loss_test = []
        time_start = time.time()
        early_stopping_flag = False
        for epoch in range(num_epochs):
            for (input_data, target) in tfe.Iterator(
                            train_dataset.shuffle(1e8).batch(self.batch_size)):
                input_data = np.random.poisson(input_data).astype(float)
                grads = self.grads_fn(input_data, target)
                optimizer.apply_gradients(zip(grads, self.variables))
                all_loss_train.append(
                    self.loss_fn(input_data, target, training=False).numpy())
                if early_stopping_patience == 0:
                    all_loss_test.append(self.loss_fn(test_dataset[0],
                                                      test_dataset[1],
                                                      training=False).numpy())

            # Save error for early stopping
            if early_stopping_patience != 0:
                all_loss_test.append(self.loss_fn(test_dataset[0],
                                                  test_dataset[1],
                                                  training=False).numpy())
                tmp_min_test_error = all_loss_test[-1]
                if epoch == 0:
                    patience_counter = 0
                    min_test_error = tmp_min_test_error
                elif (epoch > 0) and (tmp_min_test_error < min_test_error):
                    min_test_error = tmp_min_test_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                time_taken = time.time()-time_start
                if ((patience_counter >= early_stopping_patience) or
                        (time_taken > max_time)):
                    early_stopping_flag = True

            if (print_errors and
                    ((epoch == 0) | ((epoch+1) % verbose == 0))) is True:
                print('Loss at epoch %d: %3.2f %3.2f' % (epoch+1,
                                                         all_loss_train[-1],
                                                         all_loss_test[-1]))
            if early_stopping_flag is True:
                break
        return all_loss_train, all_loss_test


class cnn1d_model_features(object):

    def __init__(self,
                 learining_rate,
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
        self.learining_rate = learining_rate
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


def generate_random_cnn1d_architecture():
    """ Generates a random 1d convolutional neural network based on a set of
        predefined architectures.

        inputs: None

        outputs: cae_model_features class
    """
    cnn_filters_choices = ((4, 8, 1),
                           (4, 8, 16, 1),
                           (4, 8, 16, 32, 1),
                           (8, 16, 1),
                           (8, 16, 32, 1),
                           (8, 16, 32, 64, 1),
                           (16, 32, 1),
                           (16, 32, 64, 1),
                           (16, 32, 64, 128, 1))
    # cnn_filters_choices = ((4, 1),
    #                        (8, 1),
    #                        (16, 1),
    #                        (32, 1))
    cnn_kernel_choices = ((2,), (4,), (8,), (16,))
    pool_size_choices = ((2,), (4,), (8,), (16,))
    cnn_filters_choice = np.random.randint(
        len(cnn_filters_choices))
    cnn_kernel_choice = np.random.randint(
        len(cnn_kernel_choices))
    pool_size_choice = np.random.randint(
        len(pool_size_choices))

    cnn_filters = cnn_filters_choices[
        cnn_filters_choice]
    cnn_kernel = cnn_kernel_choices[
        cnn_kernel_choice]*(len(cnn_filters_choices))
    cnn_strides = (1,)*(len(cnn_filters_choices))
    pool_size = pool_size_choices[pool_size_choice]*(
        len(cnn_filters_choices))
    pool_strides = (2,)*(len(cnn_filters_choices))

    number_layers = np.random.randint(1, 4)
    dense_nodes = (10**np.random.uniform(1,
                                         np.log10(1024/(2**len(
                                             cnn_filters_choices))),
                                         number_layers)).astype('int')
    dense_nodes = np.sort(dense_nodes)
    dense_nodes = np.flipud(dense_nodes)

    model_features = cnn1d_model_features(
            trainable=None,
            learining_rate=None,
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


class dae(tf.keras.Model):
    def __init__(self, model_features):
        super(dae, self).__init__()
        """ Define here the layers used during the forward-pass
            of the neural network.
        """
        self.batch_size = model_features.batch_size
        self.scaler = model_features.scaler
        dropout_probability = model_features.dropout_probability
        activation_function = model_features.activation_function
        output_function = model_features.output_function

        self.l1_regularization_scale = model_features.l1_regularization_scale
        self.regularizer = tf.contrib.layers.l1_regularizer(
            scale=self.l1_regularization_scale)
        self.dense_nodes_encoder = model_features.dense_nodes_encoder
        self.dense_nodes_decoder = model_features.dense_nodes_decoder

        # Define Hidden layers for encoder
        self.dense_layers_encoder = {}
        self.dropout_layers_encoder = {}
        for layer, nodes in enumerate(self.dense_nodes_encoder):
            self.dense_layers_encoder[str(layer)] = tf.layers.Dense(
                nodes,
                activation=activation_function,
                kernel_initializer=he_normal(),
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
                kernel_initializer=he_normal(),
                kernel_regularizer=self.regularizer)
            self.dropout_layers_decoder[str(layer)] = tf.layers.Dropout(
                dropout_probability)

        # Output layer. No activation.
        self.output_layer = tf.layers.Dense(1024, activation=output_function)

    def encoder(self, input_data, training=True):
        """ Runs a forward-pass through the network. Only outputs logits for
            loss function. This is because
            tf.nn.softmax_cross_entropy_with_logits_v2 calculates softmax
            internally.
            Note, training is true here to turn dropout on.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
        """
        if training:
            x = np.random.poisson(input_data).astype(float)
        x = self.scaler.transform(input_data)
        x = tf.reshape(x, [-1, x.shape[1]])
        for layer, nodes in enumerate(self.dense_nodes_encoder):
            x = self.dense_layers_encoder[str(layer)](x)
            x = self.dropout_layers_encoder[str(layer)](x, training)
        encoding = x
        return encoding

    def decoder(self, encoding, training=True):
        """ Runs a forward-pass through the network. Only outputs logits for
            loss function. This is because
            tf.nn.softmax_cross_entropy_with_logits_v2 calculates softmax
            internally.
            Note, training is true here to turn dropout on.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
        """
        x = encoding
        for layer, nodes in enumerate(self.dense_nodes_decoder):
            x = self.dense_layers_decoder[str(layer)](x)
            x = self.dropout_layers_decoder[str(layer)](x, training)
        decoding = self.output_layer(x)
        return decoding

    def loss_fn(self, input_data, target, training=True):
        """ Defines the loss function used during
            training.
        """
        encoding = self.encoder(input_data, training)
        decoding = self.decoder(encoding, training)
        loss = tf.losses.mean_squared_error(
            labels=self.scaler.transform(target),
            predictions=decoding)
        return loss

    def grads_fn(self, input_data, target):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)

    def fit_batch(self,
                  train_dataset,
                  test_dataset,
                  optimizer,
                  num_epochs=50,
                  verbose=50,
                  print_errors=True,
                  early_stopping_patience=0,
                  max_time=300):
        """ Function to train the model, using the selected optimizer and
            for the desired number of epochs. Uses early stopping with
            patience.
        """
        all_loss_train = []
        all_loss_test = []
        time_start = time.time()
        early_stopping_flag = False
        for epoch in range(num_epochs):
            for (input_data, target) in tfe.Iterator(
                            train_dataset.shuffle(1e8).batch(self.batch_size)):
                input_data = np.random.poisson(input_data).astype(float)
                grads = self.grads_fn(input_data, target)
                optimizer.apply_gradients(zip(grads, self.variables))
                all_loss_train.append(
                    self.loss_fn(input_data, target, training=False).numpy())
                if early_stopping_patience == 0:
                    all_loss_test.append(self.loss_fn(test_dataset[0],
                                                      test_dataset[1],
                                                      training=False).numpy())

            # Save error for early stopping
            if early_stopping_patience != 0:
                all_loss_test.append(self.loss_fn(test_dataset[0],
                                                  test_dataset[1],
                                                  training=False).numpy())
                tmp_min_test_error = all_loss_test[-1]
                if epoch == 0:
                    patience_counter = 0
                    min_test_error = tmp_min_test_error
                elif (epoch > 0) and (tmp_min_test_error < min_test_error):
                    min_test_error = tmp_min_test_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                time_taken = time.time()-time_start
                if ((patience_counter >= early_stopping_patience) or
                        (time_taken > max_time)):
                    early_stopping_flag = True

            if (print_errors and
                    ((epoch == 0) | ((epoch+1) % verbose == 0))) is True:
                print('Loss at epoch %d: %3.2f %3.2f' % (epoch+1,
                                                         all_loss_train[-1],
                                                         all_loss_test[-1]))
            if early_stopping_flag is True:
                break
        return all_loss_train, all_loss_test


class dae_model_features(object):

    def __init__(self,
                 learining_rate,
                 l1_regularization_scale,
                 dropout_probability,
                 batch_size,
                 dense_nodes_encoder,
                 dense_nodes_decoder,
                 scaler,
                 activation_function,
                 output_function,
                 ):
        self.learining_rate = learining_rate
        self.l1_regularization_scale = l1_regularization_scale
        self.dropout_probability = dropout_probability
        self.batch_size = batch_size
        self.dense_nodes_encoder = dense_nodes_encoder
        self.dense_nodes_decoder = dense_nodes_decoder
        self.scaler = scaler
        self.activation_function = activation_function
        self.output_function = output_function


# ##############################################################
# ##############################################################
# ##############################################################
# ################## Convolution Autoencoder ###################
# ##############################################################
# ##############################################################
# ##############################################################

class cae(tf.keras.Model):
    def __init__(self, model_features):
        super(cae, self).__init__()
        """ Define here the layers used during the forward-pass
            of the neural network.
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

        # Define hidden layers for encoder
        self.conv_layers_encoder = {}
        self.pool_layers_encoder = {}
        for layer in range(len(cnn_filters_encoder)):
            self.conv_layers_encoder[str(layer)] = tf.layers.Conv1D(
                filters=cnn_filters_encoder[layer],
                kernel_size=cnn_kernel_encoder[layer],
                strides=1,
                padding='same',
                kernel_initializer=he_normal(),
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
                kernel_initializer=he_normal(),
                activation=activation_function)
        self.conv_layers_decoder[str(layer+1)] = tf.layers.Conv1D(
            filters=cnn_filters_decoder[-1],
            kernel_size=cnn_kernel_decoder[-1],
            strides=cnn_strides_decoder[-1],
            padding='same',
            kernel_initializer=he_normal(),
            activation=output_function)

    def encoder(self, input_data, training=True):
        """ Runs a forward-pass through the network. Only outputs logits for
            loss function. This is because
            tf.nn.softmax_cross_entropy_with_logits_v2 calculates softmax
            internally.
            Note, training is true here to turn dropout on.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
        """
        x = self.scaler.transform(input_data)
        x = tf.reshape(x, [-1, x.shape[1], 1])
        layer_list = self.conv_layers_encoder.keys()
        layer_list.sort()
        for layer in layer_list:
            x = self.conv_layers_encoder[str(layer)](x)
            x = self.pool_layers_encoder[str(layer)](x)
        encoding = x

        return encoding

    def decoder(self, encoding, training=True):
        """ Runs a forward-pass through the network. Only outputs logits for
            loss function. This is because
            tf.nn.softmax_cross_entropy_with_logits_v2 calculates softmax
            internally.
            Note, training is true here to turn dropout on.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
        """
        x = encoding
        layer_list = self.conv_layers_decoder.keys()
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

    def loss_fn(self, input_data, target, training=True):
        """ Defines the loss function used during
            training.
        """
        encoding = self.encoder(input_data, training)
        decoding = self.decoder(encoding, training)
        loss = tf.losses.mean_squared_error(
            labels=self.scaler.transform(target),
            predictions=decoding)
        return loss

    def grads_fn(self, input_data, target):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)

    def fit_batch(self,
                  train_dataset,
                  test_dataset,
                  optimizer,
                  num_epochs=50,
                  verbose=50,
                  print_errors=True,
                  early_stopping_patience=0,
                  max_time=300):
        """ Function to train the model, using the selected optimizer and
            for the desired number of epochs. Uses early stopping with
            patience.
        """
        all_loss_train = []
        all_loss_test = []
        time_start = time.time()
        early_stopping_flag = False
        for epoch in range(num_epochs):
            for (input_data, target) in tfe.Iterator(
                            train_dataset.shuffle(1e8).batch(self.batch_size)):
                input_data = np.random.poisson(input_data).astype(float)
                grads = self.grads_fn(input_data, target)
                optimizer.apply_gradients(zip(grads, self.variables))
                all_loss_train.append(
                    self.loss_fn(input_data, target, training=False).numpy())
                if early_stopping_patience == 0:
                    all_loss_test.append(self.loss_fn(test_dataset[0],
                                                      test_dataset[1],
                                                      training=False).numpy())

            # Save error for early stopping
            if early_stopping_patience != 0:
                all_loss_test.append(self.loss_fn(test_dataset[0],
                                                  test_dataset[1],
                                                  training=False).numpy())
                tmp_min_test_error = all_loss_test[-1]
                if epoch == 0:
                    patience_counter = 0
                    min_test_error = tmp_min_test_error
                elif (epoch > 0) and (tmp_min_test_error < min_test_error):
                    min_test_error = tmp_min_test_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                time_taken = time.time()-time_start
                if ((patience_counter >= early_stopping_patience) or
                        (time_taken > max_time)):
                    early_stopping_flag = True

            if (print_errors and
                    ((epoch == 0) | ((epoch+1) % verbose == 0))) is True:
                print('Loss at epoch %d: %3.2f %3.2f' % (epoch+1,
                                                         all_loss_train[-1],
                                                         all_loss_test[-1]))
            if early_stopping_flag is True:
                break
        return all_loss_train, all_loss_test


class cae_model_features(object):

    def __init__(self,
                 learining_rate,
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
        self.learining_rate = learining_rate
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


def generate_random_cae_architecture():
    """ Generates a random convolutional autoencoder based on a set of
        predefined architectures.

        inputs: None

        outputs: cae_model_features class
    """
    cnn_filters_encoder_choices = ((4, 8, 1),
                                   (4, 8, 16, 1),
                                   (4, 8, 16, 32, 1),
                                   (8, 16, 1),
                                   (8, 16, 32, 1),
                                   (8, 16, 32, 64, 1),
                                   (16, 32, 1),
                                   (16, 32, 64, 1),
                                   (16, 32, 64, 128, 1))
    # cnn_filters_encoder_choices = ((4, 1),
    #                                (8, 1),
    #                                (16, 1),
    #                                (32, 1))
    cnn_kernel_encoder_choices = ((2,), (4,), (8,), (16,))
    pool_size_encoder_choices = ((2,), (4,), (8,), (16,))
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
            learining_rate=None,
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


def train_kfolds(training_data,
                 training_keys,
                 number_folds,
                 num_epochs,
                 model_class,
                 model_features,
                 verbose=True):

    skf = StratifiedKFold(n_splits=number_folds, shuffle=True, random_state=1)

    errors_train = []
    errors_test = []

    for train_index, test_index in skf.split(training_data, training_keys):

        # only fit scaler to training data
        model_features.scaler.fit(training_data[train_index])
        X_tensor = tf.constant(training_data[train_index])
        y_tensor = tf.constant(training_keys[train_index])
        train_dataset_tensor = tf.data.Dataset.from_tensor_slices((X_tensor,
                                                                   y_tensor))
        # not iterating through test dataset, don't need to put in TF DataSet
        test_dataset = (training_data[test_index], training_keys[test_index])

        tf.reset_default_graph()
        optimizer = tf.train.AdamOptimizer(model_features.learining_rate)
        model = model_class(model_features)
        all_loss_train, all_loss_test = model.fit_batch(train_dataset_tensor,
                                                        test_dataset,
                                                        optimizer,
                                                        num_epochs=1,
                                                        verbose=1,
                                                        print_errors=False)
        if verbose is True:
            print("training loss: {0:.2f}  testing loss: {0:.2f}".format(
                all_loss_train[-1], all_loss_test[-1]))
        errors_train.append(all_loss_train)
        errors_test.append(all_loss_test)
    if verbose is True:
        print(("final average training loss: {0:.2f} "
              "final average testing loss: {0:.2f}").format(
                    np.average(errors_train, axis=0)[-1],
                    np.average(errors_test, axis=0)[-1]))

    return np.average(errors_train, axis=0)[-1],
    np.average(errors_test, axis=0)[-1]


def train_earlystopping(training_data,
                        training_keys,
                        testing_data,
                        testing_keys,
                        model_class,
                        model_features,
                        num_epochs,
                        early_stopping_patience,
                        verbose=True,
                        fit_batch_verbose=5):

    # only fit scaler to training data
    X_tensor = tf.constant(training_data)
    y_tensor = tf.constant(training_keys)
    train_dataset_tensor = tf.data.Dataset.from_tensor_slices((X_tensor,
                                                               y_tensor))
    # not iterating through test dataset, don't need to put in TF DataSet
    test_dataset = (testing_data, testing_keys)

    tf.reset_default_graph()
    optimizer = tf.train.AdamOptimizer(model_features.learining_rate)
    model = model_class(model_features)
    all_loss_train, all_loss_test = model.fit_batch(
        train_dataset_tensor,
        test_dataset,
        optimizer,
        num_epochs=num_epochs,
        verbose=fit_batch_verbose,
        early_stopping_patience=early_stopping_patience,
        print_errors=True)
    if verbose is True:
        print("training loss: {0:.2f}  testing loss: {0:.2f}".format(
            float(all_loss_train[-1]), float(all_loss_test[-1])))

    return all_loss_test


def save_model(folder_name, model_id, model, model_features):
    saver = tfe.Saver(model.variables)
    saver.save(folder_name+'/'+model_id)
    with open(folder_name+'/'+model_id+'_features', 'w') as f:
        pickle.dump(model_features, f)


def load_model(model_folder,
               model_id,
               model_class,
               training_data_length=1024,
               training_key_length=57):

    # load model features (number of layers, nodes)
    with open('./'+model_folder+'/'+model_id+'_features') as f:
        new_model_features = pickle.load(f)

    # Initialize variables by running a single training iteration
    tf.reset_default_graph()
    optimizer = tf.train.AdamOptimizer(new_model_features.learining_rate)
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
