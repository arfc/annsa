from __future__ import print_function
import numpy as np
from numpy.random import choice
from annsa.template_sampling import (apply_LLD,
                                     rebin_spectrum,)


def choose_uranium_template(uranium_dataset,
                            sourcedist,
                            sourceheight,
                            shieldingdensity,
                            fwhm,):
    '''
    Chooses a specific uranium template from a dataset.

    Inputs
        uranium_dataset : pandas dataframe
            Dataframe containing U232, U235, U238
            templates simulated in multiple conditions.
        sourcedist : int
            The source distance
        sourceheight : int
            The source height
        shieldingdensity : float
            The source density in g/cm2
        fwhm : float
            The full-width-at-half-max at 662

    Outputs
        uranium_templates : dict
            Dictionary of a single template for each isotope
            Also contains an entry for FWHM.
    '''

    uranium_templates = {}
    sourcedist_choice = sourcedist
    sourceheight_choice = sourceheight
    shieldingdensity_choice = shieldingdensity

    source_dataset_tmp = uranium_dataset[
        uranium_dataset['sourcedist'] == sourcedist_choice]
    source_dataset_tmp = source_dataset_tmp[
        source_dataset_tmp['sourceheight'] == sourceheight_choice]
    source_dataset_tmp = source_dataset_tmp[
        source_dataset_tmp['shieldingdensity'] == shieldingdensity_choice]
    source_dataset_tmp = source_dataset_tmp[
        source_dataset_tmp['fwhm'] == fwhm]

    for isotope in ['u232', 'u235', 'u238']:
        spectrum_template = source_dataset_tmp[
            source_dataset_tmp['isotope'] == isotope].values[0][6:]
        uranium_templates[isotope] = np.abs(spectrum_template)
        uranium_templates[isotope] = uranium_templates[isotope].astype(int)
    uranium_templates['fwhm'] = source_dataset_tmp['fwhm']

    return uranium_templates


def choose_random_uranium_template(uranium_dataset):
    '''
    Chooses a random uranium template from a dataset.

    Inputs
        source_dataset : pandas dataframe
            Dataframe containing U232, U235, U238
            templates simulated in multiple conditions.

    Outputs
        uranium_templates : dict
            Dictionary of a single template for each isotope.
    '''

    uranium_templates = {}

    all_sourcedist = list(set(uranium_dataset['sourcedist']))
    sourcedist_choice = choice(all_sourcedist)

    all_sourceheight = list(set(uranium_dataset['sourceheight']))
    sourceheight_choice = choice(all_sourceheight)

    all_shieldingdensity = list(set(uranium_dataset['shieldingdensity']))
    shieldingdensity_choice = choice(all_shieldingdensity)

    all_fwhm = list(set(uranium_dataset['fwhm']))
    fwhm_choice = choice(all_fwhm)

    source_dataset_tmp = uranium_dataset[
        uranium_dataset['sourcedist'] == sourcedist_choice]
    source_dataset_tmp = source_dataset_tmp[
        source_dataset_tmp['sourceheight'] == sourceheight_choice]
    source_dataset_tmp = source_dataset_tmp[
        source_dataset_tmp['shieldingdensity'] == shieldingdensity_choice]
    source_dataset_tmp = source_dataset_tmp[
        source_dataset_tmp['fwhm'] == fwhm_choice]

    for isotope in ['u232', 'u235', 'u238']:
        spectrum_template = source_dataset_tmp[
            source_dataset_tmp['isotope'] == isotope].values[0][6:]
        uranium_templates[isotope] = np.abs(spectrum_template)
        uranium_templates[isotope] = uranium_templates[isotope].astype(int)
    uranium_templates['fwhm'] = source_dataset_tmp['fwhm']

    return uranium_templates


def generate_uenriched_spectrum(uranium_templates,
                                background_dataset,
                                enrichment_level=0.93,
                                integration_time=60,
                                background_cps=200,
                                calibration=[0, 1, 0],
                                source_background_ratio=1.0,
                                ):
    '''
    Generates an enriched uranium spectrum based on .

    Inputs
        uranium_template : dict
            Dictionary of a single template for each isotope.
        background_dataset : pandas dataframe
            Dataframe of background spectra with different FWHM parameters.

    Outputs
        full_spectrum : array
            Sampled source and background spectrum
    '''

    a = calibration[0]
    b = calibration[1]
    c = calibration[2]

    template_measurment_time = 3600
    time_scaler = integration_time / template_measurment_time
    mass_fraction_u232 = choice([0,
                                 np.random.uniform(0.4, 2.0)])

    uranium_component_magnitudes = {
        'u235': time_scaler * enrichment_level,
        'u232': time_scaler * mass_fraction_u232,
        'u238': time_scaler * (1 - enrichment_level),
    }

    source_spectrum = np.zeros([1024])
    for isotope in uranium_component_magnitudes:
        source_spectrum += uranium_component_magnitudes[isotope] \ 
            * rebin_spectrum(
            uranium_templates[isotope], a, b, c)
    source_spectrum = apply_LLD(source_spectrum, 10)
    source_spectrum_sampled = np.random.poisson(source_spectrum)
    source_counts = np.sum(source_spectrum_sampled)

    background_counts = source_counts / source_background_ratio
    fwhm = uranium_templates['fwhm'].values[0]
    background_dataset = background_dataset[background_dataset['fwhm'] == fwhm]
    background_spectrum = background_dataset.sample().values[0][3:]
    background_spectrum = rebin_spectrum(background_spectrum,
                                         a, b, c)
    background_spectrum = np.array(background_spectrum, dtype='float64')
    background_spectrum = apply_LLD(background_spectrum, 10)
    background_spectrum /= np.sum(background_spectrum)
    background_spectrum_sampled = np.random.poisson(background_spectrum *
                                                    background_counts)

    full_spectrum = np.sum(
        [source_spectrum_sampled[0:1024],
         background_spectrum_sampled[0:1024]],
        axis=0,)

    return full_spectrum
