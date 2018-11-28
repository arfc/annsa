from __future__ import print_function
import numpy as np
from numpy import genfromtxt
import random
from scipy import signal
import datetime
import os
from scipy.ndimage.interpolation import zoom
import tensorflow as tf
import tensorflow.contrib.eager as tfe


def write_time_and_date():
    os.environ['TZ'] = 'CST6CDT'
    return "Time" + datetime.datetime.now().strftime("_%H_%M_%S_") + \
           "Date" + datetime.datetime.now().strftime("_%Y_%m_%d")


def results2(res, number_isotopes_displayed):

    index = [i[0] for i in sorted(enumerate(res), key=lambda x:x[1])]
    index = list(reversed(index))
    for i in range(number_isotopes_displayed):
        print((isotopes[index[i]], round(res[index[i]], 3)))


def load_template_spectra_from_folder(parent_folder,
                                      spectrum_identifier,
                                      LLD=10):
    '''
    inputs: partent_folder, spectrum_identifier
    output: dictionary containing all template spectra from a folder.

    Load template spectrum data into a dictionary. This allows templates from
    different folders to be loaded into different dictionaries.

    '''

    temp_dict = {}

    def normalize_spectrum(ID):
        temp_spectrum = read_spectrum(parent_folder + ID + spectrum_identifier)
        temp_spectrum[0:LLD] = 0
        return temp_spectrum/np.max(temp_spectrum)

    for i in range(len(isotopes)):
        # Fixes background spectra name issue
        if i >= len(isotopes)-3:
            spectrum_identifier = ''

        temp_dict[isotopes[i]] = normalize_spectrum(isotopes_GADRAS_ID[i])

    return temp_dict


def RepresentsInt(s):
    '''
    Helper funtion to see if a string represents an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False


def zoom_spectrum(spectrum, zoom_strength):
    spectrum = np.abs(zoom(spectrum, zoom_strength))
    if zoom_strength < 1.0:
        spectrum = np.lib.pad(spectrum,
                              (0, 1024-spectrum.shape[0]),
                              'constant',
                              constant_values=0)
    if zoom_strength > 1.0:
        spectrum = spectrum[0:1024]

    return spectrum


def read_spectrum(filename):
    '''
    Reads spectrum from .spe files.
    Works with silver detector and GADRAS formatted spectra.
    '''
    spectrum = np.empty(1024)

    with open(filename) as f:

        content = f.readlines()

        if RepresentsInt(content[8]):
            for i in range(1024):
                # spectra begins at index 8
                spectrum[i] = int(content[8+i])
        else:
            for i in range(1024):
                # spectra begins at index 12
                spectrum[i] = int(content[12+i])

    return spectrum


def create_simplex(number_samples, number_categories):
    # make an empty array
    k = np.zeros([number_samples, number_categories+1])
    # Make a sorted array of random variables
    a = np.sort(np.random.uniform(0,
                                  1,
                                  [number_samples, number_categories-1]),
                axis=1)
    # Zero pad left side
    k[:, 0] = 0
    # Put sorted array in new array
    k[:, 1:number_categories] = a
    # One pad right side
    k[:, number_categories] = 1
    # Take the difference of adjacent elements
    temp_simplex = np.diff(k)
    return temp_simplex


def shuffle_simplex(simplex):
    # last term from simpex is always background, by convention
    temp = random.sample(simplex[:-1], len(simplex)-1)

    # 29 here because 29 isotopes plus one background super-isotope
    shuffled_array = np.pad(temp, [0, 29-len(temp)], 'constant')

    np.random.shuffle(shuffled_array)

    # Add background
    shuffled_array = np.append(shuffled_array, simplex[-1])

    return shuffled_array


def visualize_simplex(key, index1, index2):
    new_list = []

    for i in range(len(key)):
        if key[i][index1] > 0 and key[i][index2] > 0:
            new_list.append([key[i][index1], key[i][index2]])

    plt.scatter(np.array(new_list)[:, 0], np.array(new_list)[:, 1])
    return new_list


def RepresentsInt(s):
    '''
    Helper funtion to see if a string represents an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False


isotopes = [
    'Am241',
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
    'Back_Th',
    'Back_U',
    'Back_K',
            ]


isotopes_GADRAS_ID = [
    '241AM',
    '133BA',
    '57CO',
    '60CO',
    '137CS',
    '51CR',
    '152EU',
    '67GA',
    '123I',
    '125I',
    '131I',
    '111IN',
    '192IR',
    '238U',
    '177MLU',
    '99MO',
    '237NP',
    '103PD',
    '239PU',
    '240PU',
    '226RA',
    '75SE',
    '153SM',
    '99TCM',
    '133XE',
    '201TL',
    '204TL',
    '233U',
    '235U',
    'ThoriumInSoil.spe',
    'UraniumInSoil.spe',
    'PotassiumInSoil.spe',
                 ]


isotopes_sources_GADRAS_ID = [
    '241AM',
    '133BA',
    '57CO',
    '60CO',
    '137CS',
    '51CR',
    '152EU',
    '67GA',
    '123I',
    '125I',
    '131I',
    '111IN',
    '192IR',
    '238U',
    '177MLU',
    '99MO',
    '237NP',
    '103PD',
    '239PU',
    '240PU',
    '226RA',
    '75SE',
    '153SM',
    '99TCM',
    '133XE',
    '201TL',
    '204TL',
    '233U',
    '235U'
                 ]


def sample_spectrum(iso_DRF, ncounts):
    '''
    Input:
    isoDRF: the 1024x1 vector containing the spectrum to be sampled.
            Does not need to be normalized.
    Output:
    ncounts: the 1024x1 vector containing the sampled spectrum.

    Method:
    Normalize isoDRF, and it is effectively a probability density function
    Calculate the cumulative distribution function
    Generate uniform random numbers to sample the cdf
    '''

    pdf = iso_DRF/sum(iso_DRF)
    cdf = np.cumsum(pdf)

    # take random samples and generate spectrum
    t_all = np.random.rand(np.int(ncounts))
    spec = pdf*0
    for t in t_all:
        pos = np.argmax(cdf > t)
        spec[pos] = spec[pos]+1
    return spec
