import matplotlib.pyplot as plt
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import numpy as np
import pandas as pd
from random import choice

from annsa.model_classes import train_earlystop, CNN1D, DNN
from annsa.template_sampling import *
from annsa.load_dataset import load_easy, load_full, dataset_to_spectrakeys
from annsa.load_pretrained_network import (load_features,
                                           load_pretrained_dae_into_dnn,
                                           load_pretrained_cae_into_cnn,) 


def training_wrapper(GPU_device_id = str(0),
                     model_id_save_as = 'learningcurve-cnn-easy',
                     architecture_id = '../hyperparameter_search/hyperparameter-search-results/CNN-kfoldseasy_1',
                     model_class_id = 'CNN1D',
                     testing_dataset_id = '../../source-interdiction/dataset_generation/validation_dataset_200keV_log10time_1000.npy',
                     data_augmentation = False,

                        # default 
                     train_sizes = None,
                     dense_nodes = None,
                     weight_id = None,
                     training_dataset_id = '../../source-interdiction/dataset_generation/testing_dataset_200keV_log10time_10000.npy',
                     difficulty_setting = 'easy',):

    
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
    
    if 'DAE' in architecture_id:
        model, model_features = load_pretrained_dae_into_dnn(dae_features_filename = architecture_id,
                                                             dae_weights_filename = weight_id,
                                                             dnn_dense_nodes = dense_nodes,)
    elif 'CAE' in architecture_id:
        model, model_features = load_pretrained_cae_into_cnn(cae_features_filename = architecture_id,
                                                             cae_weights_filename = weight_id,
                                                             cnn_dense_nodes = dense_nodes,)

    else: 
        model_features = load_features(architecture_id)
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

    all_errors = []

    if train_sizes:
        for train_size in train_sizes:
            print('\n\nRunning through training size '+str(train_size))
            k_folds_errors = []

            sss = StratifiedShuffleSplit(n_splits=5, train_size=train_size)
            k = 0
            for train_index, _ in sss.split(training_spectra, training_keys):
                print('Running through fold '+str(k))
                training_keys_binarized = mlb.fit_transform(training_keys.reshape([training_keys.shape[0],1]))
                testing_keys_binarized = mlb.transform(testing_keys)
                model = model_class(model_features)
                optimizer = tf.train.AdamOptimizer(model_features.learining_rate)

                _, f1_error = train_earlystop(
                    training_spectra[train_index],
                    training_keys_binarized[train_index],
                    testing_spectra,
                    testing_keys_binarized,
                    model,
                    optimizer,
                    num_epochs=1000,
                    verbose=True,
                    fit_batch_verbose=10,
                    obj_cost=model.cross_entropy,
                    earlystop_cost_fn=model.f1_error,
                    earlystop_patience=50,
                    not_learning_patience=0,
                    not_learning_threshold=0,
                    data_augmentation=online_data_augmentation,
                    augment_testing_data=False,
                    record_train_errors=False,)

                k_folds_errors.append(f1_error)
                if model_id_save_as:
                    model.save_weights('./final-models/'+model_id_save_as+'_trainsize_'+str(train_size)+'_checkpoint_'+str(k))
                k += 1

            all_errors.append(k_folds_errors)
            np.save('./final-models/final_test_errors_'+model_id_save_as, all_errors)

    else:
        training_keys_binarized = mlb.fit_transform(training_keys.reshape([training_keys.shape[0],1]))
        testing_keys_binarized = mlb.transform(testing_keys)
        model = model_class(model_features)
        optimizer = tf.train.AdamOptimizer(model_features.learining_rate)

        _, f1_error = train_earlystop(
            training_spectra[train_index],
            training_keys_binarized[train_index],
            testing_spectra,
            testing_keys_binarized,
            model,
            optimizer,
            num_epochs=1000,
            verbose=True,
            fit_batch_verbose=10,
            obj_cost=model.cross_entropy,
            earlystop_cost_fn=model.f1_error,
            earlystop_patience=50,
            not_learning_patience=0,
            not_learning_threshold=0,
            data_augmentation=online_data_augmentation,
            augment_testing_data=False,
            record_train_errors=False,)

        k_folds_errors.append(f1_error)
        if model_id_save_as:
            model.save_weights('./final-models/'+model_id_save_as+'_trainsize_'+str(train_size)+'_checkpoint_'+str(k))
        k += 1

        all_errors.append(k_folds_errors)
        np.save('./final-models/final_test_errors_'+model_id_save_as, all_errors)

