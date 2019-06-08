import pickle
import numpy as np
import pandas as pd


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


def dataset_to_spectrakeys(dataset):
    '''
    Loads a dataset into spectra and corresponding isotope keys 
    '''
    
    source_spectra = np.array(dataset.item()['sources'],dtype='float64')
    background_spectra = np.array(dataset.item()['backgrounds'],dtype='float64')
    
    spectra = np.random.poisson(np.add(source_spectra,background_spectra))
    keys = np.array(dataset.item()['keys'])

    return spectra, keys
