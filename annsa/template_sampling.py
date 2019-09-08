import numpy as np
from scipy.interpolate import griddata
import tensorflow as tf


def random_background_template_with_FWHM(background_dataset, FWHM, cosmic=0):
    """
    Parameters:
    -----------
        background_dataset : dataframe
            contains the background template data
        FWHM : float
            Desired FWHM parameter
        cosmic : bool (optional)
            Choice to include cosmic radiation
    Returns:
    --------
        random_background_spectrum : vector
            The full background spectrum template
    """

    background_choices = background_dataset[
        (background_dataset['fwhm'] == FWHM) &
        (background_dataset['cosmic'] == cosmic)]
    random_background = background_choices.sample()
    random_background_spectrum = random_background.values[0][3:]

    return random_background_spectrum


def rebin_spectrum(spectrum_template, a=0, b=1, c=0, output_len=1024):
    """
    Rebins spectrum based on second order polynomial rebinning.

    Parameters:
    -----------
        spectrum_template : vector (1x1194)
            The spectral template
        a : float, optional
            Constant rebinning term. Also known as offset.
        b : float, optional
            Linear rebinning term. Also known as gain.
        c : float, optional
            Quadratic rebinning term. Also known as the non-linear term.
        output_len : int, optional
            Length of output spectrum
    Returns:
    --------
        rebinned_spectrum_template : vector (1x1024)
            The rebinned spectrum template
    """
    spectrum_template = spectrum_template.flatten()
    spec_len = len(spectrum_template)
    new_bin_positions = a
    new_bin_positions += b * np.arange(spec_len)
    new_bin_positions += c * np.arange(spec_len) ** 2

    spectrum_template = griddata(np.arange(spec_len),
                                 spectrum_template,
                                 new_bin_positions,
                                 method='cubic',
                                 fill_value=0.0)
    spectrum_template[spectrum_template < 0] = 0
    return spectrum_template[:output_len]


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

    template_probability = template / np.sum(template)
    template_poiss_sampled = np.random.poisson(total_counts *
                                               template_probability)

    return template_poiss_sampled


def apply_LLD(spectrum, LLD=10):
    """
    Applies a low level discriminator (LLD) to a channel.

    Parameters:
    -----------
        spectrum : vector
            The spectrum
        LLD : int
            The channel where the low level discriminator is applied
    Returns:
    --------
        spectrum : vector
            The spectrum with channels in the LLD set to 0
    """
    spectrum[0:LLD] = 0
    return spectrum


def construct_spectrum(spectral_template,
                       calibration=[0, 1, 0],
                       LLD=10,
                       spectrum_counts=60,
                       output_len=1024,
                       ):
    """
    This function manipulates a spectral template.

    Parameters:
    -----------
        spectral_template : vector
            Vector containing the spectral template.
        calibration : list, optional
            A list of parameters used for rebinning the data according
            to a quadratic.
            [a,b,c]; a = constant, b = linear, c = quadratic
            Default is [0, 1.0, 0].
        LLD : int, optional
            Specifies the channel number for a low level discriminator (LLD).
            Default is 10.
        spectrum_counts : int, optional
            Total expected counts for a spectrum.
        output_len : int, optional
            length of output spectrum.

    Returns:
    --------
        spectral_template : vector
            The manipulated noiseless template.
    """
    spectral_template = spectral_template.flatten()

    a = calibration[0]
    b = calibration[1]
    c = calibration[2]

    if np.count_nonzero(spectral_template) > 0:
        spectral_template = rebin_spectrum(spectral_template, a, b, c)
        spectral_template = apply_LLD(spectral_template, LLD)
        spectral_template /= np.sum(spectral_template)  # normalizes
        spectral_template *= spectrum_counts  # rescales
    else:
        spectral_template = spectral_template[:output_len]

    return spectral_template


def make_random_spectrum(source_data,
                         background_dataset,
                         background_cps=120.0,
                         integration_time=600.0,
                         signal_to_background=1.0,
                         calibration=[0, 1.0, 0],
                         LLD=10,
                         **kwargs,):
    """
    This function uses source data and background data to generate
    a random spectrum drawn from a statistical distribution.

    Parameters:
    -----------
        source_data : vector
            Vector containing the FWHM and spectrum from the main
            radiation source.
        background_dataset : dataframe
            contains the background template data
        background_cps : float, optional
            Determines the count rate for background radiation.
            Default is 120 counts per second (cps)
        integration_time : float, optional
            Sets the integration time for a simulated detector in
            seconds.
            Default is 600 seconds
        signal_to_background : float, optional
            The ratio of source signal to background signal.
        calibration : list, optional
            A list of parameters used for rebinning the data according
            to a quadratic.
            [a,b,c]; a = constant, b = linear, c = quadratic
            Default is [0, 1.0, 0].
        LLD : int, optional
            Specifies the channel number for a low level discriminator (LLD).
            Default is 10.

    Returns:
    --------
        source_spectrum : vector
            The 1024 length source spectrum
        background_spectrum : vector
            The 1024 length background spectrum
    """

    # Make source spectrum
    source_counts = background_cps * integration_time * signal_to_background
    if type(source_data) == np.ndarray:
        source_data = tf.convert_to_tensor(source_data)

    # Checks if a single spectrum.
    if tf.contrib.framework.is_tensor(source_data):
        source_spectrum = source_data
        fwhm = source_spectrum.numpy()[0]
        spectral_template = source_spectrum.numpy()[1:]
        source_spectrum = construct_spectrum(
            spectral_template=spectral_template,
            calibration=calibration,
            LLD=LLD,
            spectrum_counts=source_counts,
            )

    # if the source data is a pandas dataframe.
    else:
        for key, value in kwargs.items():
            source_data = source_data[source_data[key] == value]

        spectral_template = source_data.sample().values[0][6:]
        source_spectrum = construct_spectrum(
            spectral_template=spectral_template,
            calibration=calibration,
            LLD=LLD,
            spectrum_counts=source_counts,
            )

        if 'fwhm' in kwargs:
            fwhm = kwargs['fwhm']

    # Make background spectrum
    background_template = random_background_template_with_FWHM(
        background_dataset,
        fwhm,
        cosmic=0)
    background_counts = background_cps * integration_time
    background_spectrum = construct_spectrum(
            spectral_template=background_template,
            calibration=calibration,
            LLD=LLD,
            spectrum_counts=background_counts,
            )

    return source_spectrum, background_spectrum


def online_data_augmentation_easy():
    '''
    Returns data augmentation parameters for the easy dataset setting
    '''
    def integration_time():
        return 10**np.random.uniform(np.log10(60), np.log10(600))

    def background_cps():
        return np.random.poisson(200)

    def signal_to_background():
        return np.random.uniform(0.5, 2)

    def calibration():
        return [np.random.uniform(0, 10),
                np.random.uniform(2700 / 3000, 3300 / 3000),
                0]

    return integration_time, background_cps, signal_to_background, calibration


def online_data_augmentation_full():
    '''
    Returns data augmentation parameters for the full dataset setting
    '''
    def integration_time():
        return 10 ** np.random.uniform(np.log10(10), np.log10(3600))

    def background_cps():
        return np.random.poisson(200)

    def signal_to_background():
        return np.random.uniform(0.1, 3)

    def calibration():
        return [np.random.uniform(0, 10),
                np.random.uniform(2400 / 3000, 3600 / 3000),
                0]

    return integration_time, background_cps, signal_to_background, calibration


def online_data_augmentation_vanilla(background_dataset,
                                     background_cps,
                                     integration_time,
                                     signal_to_background,
                                     calibration,):
    """
    Uses data augmentation to generate new data from a template datasets.

    Parameters:
    -----------
    background_dataset : dataframe
        contains the background template data
    background_cps : int
        the number of counts per second due to background
        radiation.
    integration_time : float, optional
        Sets the integration time for a simulated detector in
        seconds.
    signal_to_background : float, optional
        The ratio of source signal to background signal.
    calibration : list, float
        The calibration used for quadratic rebinning.
        [a,b,c]; a = constant, b = linear, c = quadratic

    Returns:
    --------
    online_data_augmentation : function
    """
    def online_data_augmentation(input_data):
        """
        Augments data using a template dataset.

        Parameters:
        -----------
        input_data : numpy matrix
            [nxm] matrix containing all datasets.

        Returns:
        --------
        output_data : tensorflow Tensor
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
                tf.cast(source_spectrum_poiss + background_spectrum_poiss,
                        tf.double))
        return tf.convert_to_tensor(output_data)
    return online_data_augmentation


def online_data_augmentation_ae(background_dataset,
                                background_cps,
                                integration_time,
                                signal_to_background,
                                calibration,
                                background_subtracting=True,):
    """
    Augments datasets for autoencoders.

    Parameters:
    -----------
    background_dataset : dataframe
        contains the background template data
    background_cps : int
        the number of counts per second due to background
        radiation.
    integration_time : float, optional
        Sets the integration time for a simulated detector in
        seconds.
    signal_to_background : float, optional
        The ratio of source signal to background signal.
    calibration : list, float
        The calibration used for quadratic rebinning.
        [a,b,c]; a = constant, b = linear, c = quadratic
    background_subtracting : boolean, optional
        Subtracts background from signal. Default is True.


    Returns:
    --------
    online_data_augmentation : function
        can be used as input data for model.
    """
    def online_data_augmentation(input_data):
        """
        Augments data using a template dataset.

        Parameters:
        -----------
        input_data : numpy matrix
            [nxm] matrix containing all datasets.

        Returns:
        --------
        output_data : tensorflow Tensor
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
                    [tf.cast(source_spectrum_poiss + background_spectrum_poiss,
                             tf.double),
                     tf.cast(source_spectrum,
                             tf.double)])
            else:
                output_data.append(
                    [tf.cast(source_spectrum_poiss + background_spectrum_poiss,
                             tf.double),
                     tf.cast(source_spectrum + background_spectrum,
                             tf.double)])
        return tf.convert_to_tensor(output_data)
    return online_data_augmentation
