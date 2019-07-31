from __future__ import print_function
from __future__ import absolute_import
import numpy as np
# from . import annsa as an
import annsa as an

print(help(an))


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
        Radioactive source identifier. Ex: 'aluminum20pct'
    normalization : string or boolean
        Default = None
        Accepts: 'normalheight', 'normalarea', None
        How the dataset should be normalized.

    Returns:
    --------
    temp_dict : Dictionary
        Contains all template spectra from a folder.
    """

    temp_dict = {}

    def normalize_spectrum(ID):
        temp_spectrum = an.read_spectrum(parent_folder +
                                         ID +
                                         spectrum_identifier)
        if np.max(temp_spectrum) == 0:
            print(ID + ' Contains no values')
        if normalization is None:
            return temp_spectrum
        elif normalization == 'normalheight':
            return temp_spectrum / np.max(temp_spectrum)
        elif normalization == 'normalarea':
            return temp_spectrum / np.sum(temp_spectrum)
            
    for i in range(len(an.isotopes) - 3):
        temp_dict[an.isotopes[i]] = normalize_spectrum(
            an.isotopes_sources_GADRAS_ID[i])

    return temp_dict


def load_templates(normalization=None):
    """
    Automatically loads a series of templates from pre-determined directories.
    Deprecated.

    Parameters:
    -----------

    normalization : string or boolean
        Default = None
        Accepts: 'normalheight', 'normalarea', None
        How the dataset should be normalized.

    Returns:
    --------
    spectral_templates : Dictionary
        Contains all template spectra from predefined folders.
    """
    spectral_templates = {}

    spectrum_identifier = "_10uC_spectrum.spe"

    spectral_templates['noshield'] = load_template_spectra_from_folder(
        "templates/no-shielding/",
        spectrum_identifier,
        normalization)
    spectral_templates['aluminum20pct'] = load_template_spectra_from_folder(
        "templates/aluminum-20pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['aluminum40pct'] = load_template_spectra_from_folder(
        "templates/aluminum-40pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['aluminum60pct'] = load_template_spectra_from_folder(
        "templates/aluminum-60pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['aluminum80pct'] = load_template_spectra_from_folder(
        "templates/aluminum-80pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['iron20pct'] = load_template_spectra_from_folder(
        "templates/iron-20pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['iron40pct'] = load_template_spectra_from_folder(
        "templates/iron-40pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['iron60pct'] = load_template_spectra_from_folder(
        "templates/iron-60pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['iron80pct'] = load_template_spectra_from_folder(
        "templates/iron-80pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['lead20pct'] = load_template_spectra_from_folder(
        "templates/lead-20pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['lead40pct'] = load_template_spectra_from_folder(
        "templates/lead-40pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['lead60pct'] = load_template_spectra_from_folder(
        "templates/lead-60pct/",
        spectrum_identifier,
        normalization)
    spectral_templates['lead80pct'] = load_template_spectra_from_folder(
        "templates/lead-80pct/",
        spectrum_identifier,
        normalization)

    background_locations = ['albuquerque',
                            'chicago',
                            'denver',
                            'losalamos',
                            'miami',
                            'newyork',
                            'sanfrancisco']

    spectral_templates['background'] = {}

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
        temp_spectrum = an.read_spectrum('./templates/' +
                                         'background/background-' +
                                         location + '.spe')
        if np.max(temp_spectrum) == 0:
            print('spectrum contains no values')
        if normalization is None:
            return temp_spectrum
        elif normalization == 'normalheight':
            return temp_spectrum / np.max(temp_spectrum)
        elif normalization == 'normalarea':
            return temp_spectrum / np.sum(temp_spectrum)

    for location in background_locations:
        spectral_templates['background'][location] = normalize_spectrum(
            location, normalization)

    return spectral_templates

