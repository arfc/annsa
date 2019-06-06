import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import tensorflow as tf


def random_background_template_with_FWHM(background_dataset, FWHM, cosmic=0):
    """
    Parameters:
    -----------
        background_dataset : pandas dataframe
            The spectrums template
        FWHM : float
            Desired FWHM parameter
        cosmic : bool (optional)
            Choice to include cosmic radiation
    Returns:
    --------
        random_background_spectrum : vector
            The full background spectrum
    """

    background_choices = background_dataset[
        (background_dataset['fwhm'] == FWHM) &
        (background_dataset['cosmic'] == cosmic)]
    random_background = background_choices.sample()
    random_background_spectrum = random_background.values[0][3:]

    return random_background_spectrum


def rebin_spectrum(spectrum_template, a=0, b=1, c=0):
    """
    Rebins spectrum based on quadratic rebinning. Returns a 1024 channel
    spectrum.

    Parameters:
    -----------
        spectrum_template : vector (1x1194)
            The spectrums template
        a : float
            Constant rebinning term
        b : float
            Linear rebinning term
        c : float
            Quadratic rebinning term
    Returns:
    --------
        rebinned_spectrum_template : vector (1x1194)
            The rebinned spectrum template
    """

    new_bin_positions = a + b*np.arange(1194) + c*np.arange(1194)**2

    spectrum_template = griddata(np.arange(1194),
                                 spectrum_template,
                                 new_bin_positions,
                                 method='cubic',
                                 fill_value=0.0)
    spectrum_template[spectrum_template < 0] = 0
    return spectrum_template[:1024]


def poisson_sample_template(template, total_counts):
    """
    Parameters:
    -----------
        template : vector
            The spectrums template
        total_counts : int
            The total expected counts in a spectrum
    Returns:
    --------
        template_poiss_sampled : vector
            The poisson sampled spectrum
    """

    template_probability = template/np.sum(template)
    template_poiss_sampled = np.random.poisson(total_counts *
                                               template_probability)

    return template_poiss_sampled


def apply_LLD(spectrum, LLD=10):
    """
    Parameters:
    -----------
        spectrum : vector
            The spectrum
        LLD : int
            The channel where the low level discriminator is applied
    Returns:
    --------
        spectrum : vector
            The spectrum with LLD channelsset to 0
    """
    spectrum[0:LLD] = 0
    return spectrum


def make_random_spectrum(source_data,
                         background_dataset,
                         background_cps=120.0,
                         integration_time=600.0,
                         signal_to_background=1.0,
                         calibration=[0, 1.0, 0],
                         LLD=10,
                         **kwargs,):
    """
    Parameters:
    -----------
        source_spectrum : vector
            Vector containing the FWHM and spectrum for
        background_dataset :
            bla
    Returns:
    --------
        source_spectrum : vector
            The 1024 length source spectrum
        background_spectrum : vector
            The 1024 length background spectrum
    """
    a = calibration[0]
    b = calibration[1]
    c = calibration[2]

    # Make source spectrum
    source_counts = background_cps*integration_time*signal_to_background
    if type(source_data) == np.ndarray:
        source_data = tf.convert_to_tensor(source_data)
    if tf.contrib.framework.is_tensor(source_data):
        # if single spectrum
        source_spectrum = source_data
        fwhm = source_spectrum.numpy()[0]
        source_spectrum = source_spectrum.numpy()[1:]
        if np.count_nonzero(source_spectrum) > 0:
            source_spectrum = rebin_spectrum(source_spectrum, a, b, c)
            source_spectrum = apply_LLD(source_spectrum, LLD)
            source_spectrum /= np.sum(source_spectrum)
            source_spectrum *= source_counts
        else:
            source_spectrum = source_spectrum[:1024]
    else:
        # if dataset of spectra
        for key, value in kwargs.items():
            source_data = source_data[source_data[key] == value]

        # if (source_data['isotope'] != 'background' and
        #    np.count_nonzero(source_spectrum) == 1024):
        #    # resample if template is non-background and empty
        source_spectrum = source_data.sample().values[0][6:]
        source_spectrum = rebin_spectrum(source_spectrum, a, b, c)
        source_spectrum = apply_LLD(source_spectrum, LLD)
        source_spectrum /= np.sum(source_spectrum)
        source_spectrum *= source_counts
        if 'fwhm' in kwargs:
            fwhm = kwargs['fwhm']

    # Make background spectrum
    background_spectrum = random_background_template_with_FWHM(
        background_dataset,
        fwhm,
        cosmic=0)
    background_counts = background_cps*integration_time
    background_spectrum = rebin_spectrum(background_spectrum, a, b, c)
    background_spectrum = apply_LLD(background_spectrum, LLD)
    background_spectrum /= np.sum(background_spectrum)
    background_spectrum *= background_counts

    return source_spectrum, background_spectrum


'''
def make_random_spectrum(source_spectrum,
                         background_dataset,
                         background_cps=120.0,
                         integration_time=600.0,
                         signal_to_background=1.0,
                         calibration=[0, 1.0, 0],
                         LLD=10):
    ''
    inputs:
        source_spectrum : vector
            Vector containing the FWHM and spectrum for
        background_dataset :
            bla
    returns:
        source_spectrum : vector
            The 1024 length source spectrum
        background_spectrum : vector
            The 1024 length background spectrum
    ''
    a = calibration[0]
    b = calibration[1]
    c = calibration[2]

    if type(source_spectrum) == np.ndarray:
        source_spectrum = tf.convert_to_tensor(source_spectrum)

    fwhm = source_spectrum.numpy()[0]

    # Make source spectrum
    source_spectrum = source_spectrum.numpy()[1:]

    if np.count_nonzero(source_spectrum) > 0:
        source_counts = background_cps*integration_time*signal_to_background
        source_spectrum = rebin_spectrum(source_spectrum, a, b, c)
        source_spectrum = apply_LLD(source_spectrum, LLD)
        source_spectrum /= np.sum(source_spectrum)
        source_spectrum *= source_counts
    else:
        source_spectrum = source_spectrum[:1024]

    # Make background spectrum
    background_spectrum = random_background_template_with_FWHM(
        background_dataset,
        fwhm,
        cosmic=0)
    background_counts = background_cps*integration_time
    background_spectrum = rebin_spectrum(background_spectrum, a, b, c)
    background_spectrum = apply_LLD(background_spectrum, LLD)
    background_spectrum /= np.sum(background_spectrum)
    background_spectrum *= background_counts

    return source_spectrum, background_spectrum
'''


def online_data_augmentation_vanilla(background_dataset,
                                     background_cps,
                                     integration_time,
                                     signal_to_background,
                                     calibration,):
    """
    Uses premade datasets to generate new ones for data augmentation.

    PARAM

    RET
    """
    def online_data_augmentation(input_data):
        """
        Augments data using a template dataset.

        PARAM

        RET
        """
        output_data = []
        for source_data in input_data:
            if type(source_data) == 'numpy.ndarray':
                source_data = tf.convert_to_tensor(source_data)
            source_spectrum, background_spectrum = make_random_spectrum(
                source_data,
                background_dataset,
                background_cps=background_cps(),
                integration_time=integration_time(),
                signal_to_background=signal_to_background(),
                calibration=calibration())
            source_spectrum_poiss = np.random.poisson(source_spectrum)
            background_spectrum_poiss = np.random.poisson(background_spectrum)
            output_data.append(
                tf.cast(source_spectrum_poiss+background_spectrum_poiss,
                        tf.double))
        return tf.convert_to_tensor(output_data)
    return online_data_augmentation


def online_data_augmentation_ae(background_dataset,
                                background_cps,
                                integration_time,
                                signal_to_background,
                                calibration,
                                background_subtracting=True):
    """
    Augments datasets for autoencoders.

    PARAM

    RET
    """
    def online_data_augmentation(input_data):
        """
        Augments data using a template dataset.

        PARAM

        RET
        """
        output_data = []
        for source_data in input_data:
            if type(source_data) == 'numpy.ndarray':
                source_data = tf.convert_to_tensor(source_data)
            source_spectrum, background_spectrum = make_random_spectrum(
                source_data,
                background_dataset,
                background_cps=background_cps(),
                integration_time=integration_time(),
                signal_to_background=signal_to_background(),
                calibration=calibration())
            source_spectrum_poiss = np.random.poisson(source_spectrum)
            background_spectrum_poiss = np.random.poisson(background_spectrum)
            if background_subtracting:
                output_data.append(
                    [tf.cast(source_spectrum_poiss+background_spectrum_poiss,
                             tf.double),
                     tf.cast(source_spectrum,
                             tf.double)])
            else:
                output_data.append(
                    [tf.cast(source_spectrum_poiss+background_spectrum_poiss,
                             tf.double),
                     tf.cast(source_spectrum+background_spectrum,
                             tf.double)])
        return tf.convert_to_tensor(output_data)
    return online_data_augmentation
