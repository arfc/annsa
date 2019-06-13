import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.pipeline import make_pipeline


def load_easy(source_dataset, background_dataset):
    source_dataset = source_dataset[source_dataset['fwhm']==7.5]
    source_dataset = source_dataset[source_dataset['sourcedist']==175.0]
    source_dataset = source_dataset[source_dataset['sourceheight']==100.0]

    # remove 80% shielding
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=13.16]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=11.02]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=1.61]

    # remove 60% shielding
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=7.49]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=6.28]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=0.92]

    # remove 40% shielding
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=4.18]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=3.5]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=0.51]

    # remove 20% shielding
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=1.82]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=1.53]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=0.22]

    # remove empty spectra
    zero_count_indicies = np.argwhere(np.sum(source_dataset.values[:,6:],axis=1) == 0).flatten()

    print('indicies dropped: ' +str(zero_count_indicies))

    source_dataset.drop(source_dataset.index[zero_count_indicies], inplace=True)

    # Add empty spectra for background
    blank_spectra = []
    for fwhm in set(source_dataset['fwhm']):
        num_examples = source_dataset[(source_dataset['fwhm']==fwhm) &
                                      (source_dataset['isotope']==source_dataset['isotope'].iloc()[0])].shape[0]
        for k in range(num_examples):
            blank_spectra_tmp = [0]*1200
            blank_spectra_tmp[5] = fwhm
            blank_spectra_tmp[0] = 'background'
            blank_spectra_tmp[3] = 'background'
            blank_spectra.append(blank_spectra_tmp)

    source_dataset = source_dataset.append(pd.DataFrame(blank_spectra,
                                                        columns=source_dataset.columns))

    spectra_dataset = source_dataset.values[:,5:].astype('float64')
    all_keys = source_dataset['isotope'].values
    
    return source_dataset, spectra_dataset, all_keys


def load_full(source_dataset, background_dataset):
    source_dataset = source_dataset[(source_dataset['fwhm']==7.0) | 
                                    (source_dataset['fwhm']==7.5) |
                                    (source_dataset['fwhm']==8.0)]

    source_dataset = source_dataset[(source_dataset['sourcedist']==50.5) | 
                                    (source_dataset['sourcedist']==175.0) | 
                                    (source_dataset['sourcedist']==300.0)]

    source_dataset = source_dataset[(source_dataset['sourceheight']==50.0) |
                                    (source_dataset['sourceheight']==100.0) |
                                    (source_dataset['sourceheight']==150.0)]

    # remove 80% shielding
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=13.16]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=11.02]
    source_dataset = source_dataset[source_dataset['shieldingdensity']!=1.61]

    # remove empty spectra
    zero_count_indicies = np.argwhere(np.sum(source_dataset.values[:,6:],axis=1) == 0).flatten()

    print('indicies dropped: ' +str(zero_count_indicies))

    source_dataset.drop(source_dataset.index[zero_count_indicies], inplace=True)

    # Add empty spectra for background
    blank_spectra = []
    for fwhm in set(source_dataset['fwhm']):
        num_examples = source_dataset[(source_dataset['fwhm']==fwhm) &
                                      (source_dataset['isotope']==source_dataset['isotope'].iloc()[0])].shape[0]
        for k in range(num_examples):
            blank_spectra_tmp = [0]*1200
            blank_spectra_tmp[5] = fwhm
            blank_spectra_tmp[0] = 'background'
            blank_spectra_tmp[3] = 'background'
            blank_spectra.append(blank_spectra_tmp)

    source_dataset = source_dataset.append(pd.DataFrame(blank_spectra,
                                                        columns=source_dataset.columns))

    spectra_dataset = source_dataset.values[:,5:].astype('float64')
    all_keys = source_dataset['isotope'].values
    
    return source_dataset, spectra_dataset, all_keys


def load_dataset(kind='nn'):
    """
    @Author: Sam Dotson
    Generates dummy data using 'sklearn.datasets.make_classification()'. 
    See 'make_classification' documentation for more details.

    Parameters:
    kind : string, optional
        A string describing what kind of neural network this dataset
        will be used for. Default is 'nn.'
        Accepts: 
        'nn' (standard convolution or dense neural networks)
        'ae' (autoencoder)


    Returns:
    -------
    train_dataset : tuple of [train_data, training_keys_binarized]
        Contains the training data and the labels in a binarized
        format.
    test_dataset : tuple of [test_data, testing_keys_binarized]
        Contains the testing data and the labels in a binarized
        format. 
    """

    training_dataset = make_classification(n_samples=100,
                                           n_features=1024,
                                           n_informative=200,
                                           n_classes=2)

    testing_dataset = make_classification(n_samples=100,
                                          n_features=1024,
                                          n_informative=200,
                                          n_classes=2)

    mlb = LabelBinarizer()

    #transform the training data
    training_data = np.abs(training_dataset[0])
    training_keys = training_dataset[1]
    training_keys_binarized = mlb.fit_transform(
        training_keys.reshape([training_data.shape[0], 1]))
    
    #transform the testing data
    testing_data = np.abs(testing_dataset[0])
    testing_keys = testing_dataset[1]
    testing_keys_binarized = mlb.transform(
        testing_keys.reshape([testing_data.shape[0], 1]))

    if kind == 'nn':
        test_dataset = [testing_data, testing_keys_binarized]
        train_dataset = [training_data, training_keys_binarized]

    elif kind == 'ae':
        train_dataset = [training_data, training_data]
        test_dataset = [testing_data, testing_data]

    return train_dataset, test_dataset
