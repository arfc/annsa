from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from . import annsa as an
from scipy.interpolate import griddata

background_locations = ['albuquerque',
                        'chicago',
                        'denver',
                        'losalamos',
                        'miami',
                        'newyork',
                        'sanfrancisco']


def load_template_spectra_from_folder(parent_folder,
                                      spectrum_identifier,
                                      normalization=None):
    """
    Load template spectrum data into a dictionary. This allows templates from
    different folders to be loaded into different dictionaries.

    Parameters:
    -----------
    parent_folder : string
        Name of folder or path 
    spectrum_identifier : string
        Radioactive source identifier. Ex: '235U'
    normalization : string or boolean
        Default = None
        Accepts: 'normalheight', 'normalarea', None
        How the dataset should be normalized.
    
    Returns:
    --------
    temp_dict : Dictionary containing all template spectra from a folder.

    """

    temp_dict = {}

    def normalize_spectrum(ID):
        """
        Normalizes the spectrum data.

        Parameters:
        -----------
        ID : string
            The ID key for the radioactive source in your spectrum.

        Returns:
        --------
        temp_dict : Dictionary
            Contains all normalized datasets.
        """
        temp_spectrum = an.read_spectrum(
            parent_folder + ID + spectrum_identifier)
        if np.max(temp_spectrum) == 0:
            print(ID + ' Contains no values')
        if normalization is None:
            return temp_spectrum
        elif normalization == 'normalheight':
            return temp_spectrum / np.max(temp_spectrum)
        elif normalization == 'normalarea':
            return temp_spectrum / np.sum(temp_spectrum)

    for i in range(len(an.isotopes)-3):
        temp_dict[an.isotopes[i]] = normalize_spectrum(
            an.isotopes_sources_GADRAS_ID[i])
    return temp_dict


def load_templates(template_settings,
                   templates_folder,
                   normalization='normalarea',
                   spectrum_identifier="_10uC_spectrum.spe"
                   ): 
    """
    Loads spectrum templates from a local directory to be used to simulate
    training data.

    Parameters:
    -----------
    template_settings : 1D array of type string
        Contains information about the detector settings used in locating 
        the dataset.  
    templates_folder : string
        Name of the parent folder or path to the dataset you want. 
    normalization: type string or None
        Default = 'normalarea' 
        Accepts: 'normalheight', 'normalarea', None
        How the dataset should be normalized.

    Returns:
    --------
    """

    spectral_templates = {}

    for setting in template_settings:
        spectral_templates[setting] = load_template_spectra_from_folder(
            templates_folder + setting + "/",
            spectrum_identifier,
            normalization)

    def normalize_spectrum(location, normalization=None):
        """
        Normalizes the spectrum data.

        Parameters: 
        -----------
        location : 'string'
            Name of location for template background radiation.

        normalization : string or boolean
            Default = None
            Accepts: 'normalheight', 'normalarea', None
            How the dataset should be normalized.

        Returns:
        --------
        Normalized temp_spectrum.
        """
            temp_spectrum = an.read_spectrum(
                templates_folder +
                '/background/background-' +
                location +
                '.spe')
            if np.max(temp_spectrum) == 0:
                print(ID + ' Contains no values')
            if normalization is None:
                return temp_spectrum
            elif normalization == 'normalheight':
                return temp_spectrum/np.max(temp_spectrum)
            elif normalization == 'normalarea':
                return temp_spectrum/np.sum(temp_spectrum)

    background_locations = ['albuquerque',
                            'chicago',
                            'denver',
                            'losalamos',
                            'miami',
                            'newyork',
                            'sanfrancisco']

    spectral_templates['background'] = {}

    for location in background_locations:
        spectral_templates['background'][location] = normalize_spectrum(
            location, normalization)

    return spectral_templates


def simulate_template_dataset(isotope_list,
                              spectral_template_settings,
                              spectral_templates,
                              template_parameters,
                              output_separate_background=False):
            

    """
    Uses template to generate new training set and keys.

    Parameters: 
    -----------
    isotope_list: 1D array of type string

    spectral_template_settings : 1D array of type string

    spectral_templates : 1D array of type string

    template_parameters : 1D array of type string

    output_separate_background : boolean

    RET
    """

    integration_times = template_parameters['integration_times']
    signal_to_backgrounds = template_parameters['signal_to_backgrounds']
    calibrations = template_parameters['calibrations']

    all_source_spectra = []
    all_background_spectra = []
    all_keys = []
    # Low level discriminator set to channel 10
    LLD = 10
    # Background counts per second set to 85
    background_cps = 85.
    total_spectra = 0
    random_settings = False

    for isotope in isotope_list:
        for spectral_template_setting in spectral_template_settings:
            for integration_time in integration_times:
                for signal_to_background in signal_to_backgrounds:
                    for calibration in calibrations:

                        # Simulate source
                        if random_settings:
                            calibration = np.random.uniform(calibrations[0],
                                                            calibrations[-1])
                            signal_to_background = 10**np.random.uniform(
                                np.log10(signal_to_backgrounds[0]),
                                np.log10(signal_to_backgrounds[-1]))
                            integration_time = 10**np.random.uniform(
                                np.log10(integration_times[0]),
                                np.log10(integration_times[-1]))

                        source_template = spectral_templates[
                            spectral_template_setting][isotope]
                        source_template = griddata(range(1024),
                                                   source_template,
                                                   calibration*np.arange(1024),
                                                   method='cubic',
                                                   fill_value=0.0)
                        source_template[0:LLD] = 0
                        source_template[source_template < 0] = 0
                        source_template /= np.sum(source_template)
                        source_template *= integration_time *\
                            background_cps *\
                            signal_to_background

                        background_template = \
                            spectral_templates['background']['chicago']
                        background_template = griddata(
                            range(1024),
                            background_template,
                            calibration*np.arange(1024),
                            method='cubic',
                            fill_value=0.0)
                        background_template[0:LLD] = 0
                        background_template[background_template < 0] = 0
                        background_template /= np.sum(background_template)
                        background_template *= integration_time*background_cps

                        if not output_separate_background:
                            all_source_spectra.append(
                                source_template + background_template)
                        else:
                            all_background_spectra.append(background_template)
                            all_source_spectra.append(source_template)

                        isotope_key = isotope
                        all_keys.append(isotope_key)

                        print(('\1b[2k\r'), end=' ')
                        print(('Isotope %s, template %s,'
                              '%s total spectra simulated' % (
                                  isotope,
                                  spectral_template_setting,
                                  total_spectra)), end=' ')
    all_source_spectra = np.array(all_source_spectra)
    all_keys = np.array(all_keys)
    if not output_separate_background:
        return all_source_spectra, all_keys
    else:
        all_background_spectra = np.array(all_background_spectra)
        return all_source_spectra, all_background_spectra, all_keys


def create_template_parameters(
        integration_time_range,
        integration_time_division,
        signal_to_background_range,
        signal_to_background_division,
        calibration_range,
        calibration_division,
        print_divisions=False,
        division_offset=False): 

    """
    Generates a list of parameters for template.

    Parameters:
    -----------
    integration_time_range : list, tuple, float, int   
        A list of two floats that give the start and end points.
    integration_time_division : int
        Number of divisions for the integration time. dt.
    signal_to_background_range : list, tuple, float, int
        A list of two floats that give the start and end points.
    signal_to_background_division : int
        Number of divisions for the signal_to_background.
    calibration_range : list, tuple, float, int
        A list of two floats that give the start and end points.
    calibration_division : 
        Number of divisions for the calibration.
    print_divisions : Boolean, optional
        If true, prints the divisions. Default is false.
    division_offset : Boolean, optional
        If true, offsets all divisions by removing the last element
        of the spaces defined by the range and divisions, and 
        adding to it a list of np.diff(x)/2.0 

    RET
    """

    integration_times = np.logspace(
        np.log10(integration_time_range[0]),
        np.log10(integration_time_range[1]),
        integration_time_division)

    signal_to_backgrounds = np.logspace(
        np.log10(signal_to_background_range[0]),
        np.log10(signal_to_background_range[1]),
        signal_to_background_division)

    calibrations = np.linspace(1.19*calibration_range[0],
                               1.19*calibration_range[1],
                               calibration_division)

    if division_offset:
        integration_times = \
            integration_times[:-1]+np.diff(integration_times)/2.0
        signal_to_backgrounds = \
            signal_to_backgrounds[:-1]+np.diff(signal_to_backgrounds)/2.0
        calibrations = \
            calibrations[:-1]+np.diff(calibrations)/2.0

    if print_divisions:
        print("integration_times\n" + str(integration_times))
        print("signal_to_backgrounds\n" + str(signal_to_backgrounds))
        print("calibrations\n" + str(calibrations))

    template_parameters = {}
    template_parameters['integration_times'] = integration_times
    template_parameters['signal_to_backgrounds'] = signal_to_backgrounds
    template_parameters['calibrations'] = calibrations

    return template_parameters
