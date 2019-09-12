import matplotlib.pyplot as plt
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import numpy as np
import pandas as pd
from random import choice

from annsa.model_classes import train_earlystop, CNN1D, DNN, DAE, CAE
from annsa.template_sampling import *
from annsa.load_dataset import load_easy, load_full, dataset_to_spectrakeys
from annsa.load_pretrained_network import (load_features,
                                           save_features,
                                           load_pretrained_dae_into_dnn,
                                           load_pretrained_cae_into_cnn,) 


def training_wrapper(GPU_device_id = str(0),
                     model_id_save_as = 'learningcurve-cnn-easy',
                     architecture_id = '../hyperparameter_search/hyperparameter-search-results/CNN-kfoldseasy_1',
                     model_class_id = 'CNN1D',
                     testing_dataset_id = '../../source-interdiction/dataset_generation/validation_dataset_200keV_log10time_1000.npy',
                     data_augmentation = False,
                     train_sizes = None,
                     dense_nodes = None,
                     weight_id = None,
                     fit_batch_verbose = 10,
                     total_networks = 1,
                     training_dataset_id = '../../source-interdiction/dataset_generation/testing_dataset_200keV_log10time_10000.npy',
                     difficulty_setting = 'easy',
                     num_epochs=1000,
                     earlystop_patience=50,
                     dnn_features_filename=None,
                     cnn_features_filename=None,
                     replacement_index=None,
                    ):

    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device_id

    import tensorflow as tf
    import tensorflow.contrib.eager as tfe
    tf.enable_eager_execution()

    # ## Training Data Construction

    if data_augmentation:
        background_dataset = pd.read_csv('../../source-interdiction/training_testing_data/background_template_dataset.csv')
        source_dataset = pd.read_csv('../../source-interdiction/training_testing_data/shielded_templates_200kev_dataset.csv')
        if difficulty_setting == 'easy':
            source_dataset, training_spectra, training_keys = load_easy(source_dataset, background_dataset)
        if difficulty_setting == 'full':
            source_dataset, training_spectra, training_keys = load_full(source_dataset, background_dataset)

    elif training_dataset_id:
        training_dataset = np.load(training_dataset_id)
        training_spectra, training_keys = dataset_to_spectrakeys(training_dataset)

    else:
        print('No training set specified')


    # ## Load testing dataset

    testing_dataset = np.load(testing_dataset_id)
    testing_spectra, testing_keys = dataset_to_spectrakeys(testing_dataset)


    # ## Load Model

    model_class = eval(model_class_id)
    
    if 'dae' in architecture_id.lower():

        if dnn_features_filename:
            model_features = load_features(dnn_features_filename)
            activation_function = model_features.activation_function 
            l2_regularization_scale = model_features.l2_regularization_scale
            dropout_probability = model_features.dropout_probability
        else:
            dropout_probability = 0.0 
            l2_regularization_scale = 0.0
            dropout_probability = 0.0
        
        model, model_features = load_pretrained_dae_into_dnn(dae_features_filename=architecture_id,
                                                             dae_weights_filename=weight_id,
                                                             dnn_dense_nodes=dense_nodes,
                                                             output_function=None,
                                                             activation_function=activation_function,
                                                             l2_regularization_scale=l2_regularization_scale,
                                                             dropout_probability=dropout_probability,)
    elif 'cae' in architecture_id.lower():

        if cnn_features_filename:
            model_features = load_features(cnn_features_filename)
            activation_function = model_features.activation_function 
            l2_regularization_scale = model_features.l2_regularization_scale
            dropout_probability = model_features.dropout_probability
        else:
            dropout_probability = 0.0 
            l2_regularization_scale = 0.0
            dropout_probability = 0.0
    
        model, model_features = load_pretrained_cae_into_cnn(cae_features_filename=architecture_id,
                                                             cae_weights_filename=weight_id,
                                                             cnn_dense_nodes=dense_nodes,
                                                             activation_function=activation_function,
                                                             output_function=None,
                                                             l2_regularization_scale=l2_regularization_scale,
                                                             dropout_probability=dropout_probability,)

    else: 
        model_features = load_features(architecture_id)
        if model_class_id == 'DNN':
            model_features.output_function = None
        model = model_class(model_features)
    
    

    # ## Define Online Data Augmentation Training Parameters


    if difficulty_setting == 'full':
            integration_time, background_cps, signal_to_background, calibration = online_data_augmentation_full()
    elif difficulty_setting == 'easy':
        integration_time, background_cps, signal_to_background, calibration = online_data_augmentation_easy()
    else:
        print('augmentation setting not in template_sampling.py')



    if data_augmentation:
        online_data_augmentation = online_data_augmentation_vanilla(background_dataset,
                                         background_cps,
                                         integration_time,
                                         signal_to_background,
                                         calibration)
    else:
        online_data_augmentation = model.default_data_augmentation


    # ## Train network


    mlb=LabelBinarizer()

    all_errors_test = []
    all_errors_train = []

    if train_sizes:
        for train_size in train_sizes:
            print('\n\nRunning through training size '+str(train_size))
            k_folds_errors_test = []
            k_folds_errors_train = []
            sss = StratifiedShuffleSplit(n_splits=5, train_size=train_size)
            k = 0
            if replacement_index:
                k = replacement_index

            for train_index, _ in sss.split(training_spectra, training_keys):
                print('Running through fold '+str(k))
                training_keys_binarized = mlb.fit_transform(training_keys.reshape([training_keys.shape[0],1]))
                testing_keys_binarized = mlb.transform(testing_keys)
                model = model_class(model_features)
                optimizer = tf.train.AdamOptimizer(model_features.learining_rate)

                _, f1_error_test = train_earlystop(
                    training_spectra[train_index],
                    training_keys_binarized[train_index],
                    testing_spectra,
                    testing_keys_binarized,
                    model,
                    optimizer,
                    num_epochs=num_epochs,
                    verbose=True,
                    fit_batch_verbose=fit_batch_verbose,
                    obj_cost=model.cross_entropy,
                    earlystop_cost_fn=model.f1_error,
                    earlystop_patience=earlystop_patience,
                    not_learning_patience=0,
                    not_learning_threshold=0,
                    data_augmentation=online_data_augmentation,
                    augment_testing_data=False,
                    record_train_errors=False,)

                f1_error_train = model.f1_error(training_spectra[train_index],
                                                training_keys_binarized[train_index])
                k_folds_errors_test.append(f1_error_test)
                k_folds_errors_train.append(f1_error_train)
                if model_id_save_as:
                    model.save_weights('./final-models/'+model_id_save_as+'_trainsize_'+str(train_size)+'_checkpoint_'+str(k))
                k += 1               
                replacement_index_str = ''
                if replacement_index:
                    replacement_index_str = str(replacement_index)
                    continue
               
            all_errors_test.append(k_folds_errors_test)
            all_errors_train.append(k_folds_errors_train)
            if model_id_save_as:
                np.save('./final-models/final_test_errors_'+model_id_save_as+replacement_index_str+'_trainsize_'+str(train_size),
                        all_errors_test)
                np.save('./final-models/final_train_errors_'+model_id_save_as+replacement_index_str+'_trainsize_'+str(train_size),
                        all_errors_train)
                save_features(model_features,
                              './final-models/'+model_id_save_as+'-features')

    else:
        training_keys_binarized = mlb.fit_transform(training_keys.reshape([training_keys.shape[0],1]))
        testing_keys_binarized = mlb.transform(testing_keys)
        for network_id in range(total_networks):
            model = model_class(model_features)
            optimizer = tf.train.AdamOptimizer(model_features.learining_rate)

            _, f1_error_test = train_earlystop(
                training_spectra,
                training_keys_binarized,
                testing_spectra,
                testing_keys_binarized,
                model,
                optimizer,
                num_epochs=num_epochs,
                verbose=True,
                fit_batch_verbose=fit_batch_verbose,
                obj_cost=model.cross_entropy,
                earlystop_cost_fn=model.f1_error,
                earlystop_patience=earlystop_patience,
                not_learning_patience=0,
                not_learning_threshold=0,
                data_augmentation=online_data_augmentation,
                augment_testing_data=False,
                record_train_errors=False,)
 
            if not data_augmentation and model_id_save_as:
                f1_error_train = model.f1_error(training_spectra,
                                                training_keys_binarized)
                all_errors_train.append(f1_error_train)

            all_errors_test.append(f1_error_test)            
            if model_id_save_as:
                model.save_weights('./final-models/'+model_id_save_as+'_checkpoint-'+str(network_id))
                np.save('./final-models/final_test_errors_'+model_id_save_as+'-'+str(network_id), all_errors_test)
                if not data_augmentation:
                    np.save('./final-models/final_train_errors_'+model_id_save_as+'-'+str(network_id), all_errors_train)
                save_features(model_features,
                              './final-models/'+model_id_save_as+'-features')
