from __future__ import print_function
import numpy as np
from numpy.random import choice
from scipy.interpolate import griddata
from annsa.annsa import read_spectrum
from annsa.template_sampling import (apply_LLD,
                                     poisson_sample_template,
                                     rebin_spectrum,)


def choose_uranium_template(uranium_dataset,
                            sourcedist,
                            sourceheight,
                            shieldingdensity,):
    '''
    Chooses a specific uranium template from a dataset.

    Inputs
        uranium_dataset : pandas dataframe
            Dataframe containing U232, U235, U238, and uranium K x-ray
            templates simulated in multiple conditions.

    Outputs
        uranium_templates : dict
            Dictionary of a single template for each isotope.
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

    for isotope in ['232U', '235U', '238U', 'UXRAY']:
        spectrum_template = source_dataset_tmp[
            source_dataset_tmp['isotope'] == isotope].values[0][6:]

        template_sum = np.sum(spectrum_template)
        spectrum_template_normalized = spectrum_template / template_sum
        uranium_templates[isotope] = np.abs(spectrum_template_normalized)
        uranium_templates[isotope] = uranium_templates[isotope].astype(float)
    return uranium_templates


def choose_random_uranium_template(uranium_dataset):
    '''
    Chooses a random uranium template from a dataset.

    Inputs
        source_dataset : pandas dataframe
            Dataframe containing U232, U235, U238, and uranium K x-ray
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

    source_dataset_tmp = uranium_dataset[
        uranium_dataset['sourcedist'] == sourcedist_choice]
    source_dataset_tmp = source_dataset_tmp[
        source_dataset_tmp['sourceheight'] == sourceheight_choice]
    source_dataset_tmp = source_dataset_tmp[
        source_dataset_tmp['shieldingdensity'] == shieldingdensity_choice]

    for isotope in ['232U', '235U', '238U', 'UXRAY']:
        spectrum_template = source_dataset_tmp[
            source_dataset_tmp['isotope'] == isotope].values[0][6:]

        template_sum = np.sum(spectrum_template)
        spectrum_template_normalized = spectrum_template / template_sum
        uranium_templates[isotope] = np.abs(spectrum_template_normalized)
        uranium_templates[isotope] = uranium_templates[isotope].astype(float)
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

    for template_id in uranium_templates:
        uranium_templates[template_id] = rebin_spectrum(
            uranium_templates[template_id],
            a, b, c)
        uranium_templates[template_id] = apply_LLD(
            uranium_templates[template_id], 10)

    total_background_counts = background_cps * integration_time
    total_source_counts = total_background_counts*source_background_ratio
    background_dataset = background_dataset[background_dataset['fwhm'] == 6.5]
    background_spectrum = background_dataset.sample().values[0][3:]
    background_spectrum = np.array(background_spectrum, dtype='float64')
    background_spectrum /= np.sum(background_spectrum)
    background_spectrum_sampled = np.random.poisson(background_spectrum *
                                                    total_background_counts)

    mass_fraction_u232 = choice([0,
                                10 ** np.random.uniform(-10, -8)])
    # ph/s/gm
    u235_phsg = 207072
    u238_phsg = 3811
    u232_phsg = 1.10275e12 * mass_fraction_u232
    uxry_phsg = 43010

    # ph/s
    u235_phs = u235_phsg * enrichment_level
    u238_phs = u238_phsg * (1 - enrichment_level)
    u232_phs = u232_phsg
    uxry_phs = uxry_phsg * enrichment_level
    normalized_phs = u235_phs+u238_phs+u232_phs+uxry_phs

    # ph
    u235_ph = total_source_counts * u235_phs / normalized_phs
    u238_ph = total_source_counts * u238_phs / normalized_phs
    u232_ph = total_source_counts * u232_phs / normalized_phs
    uxry_ph = total_source_counts * uxry_phs / normalized_phs

    tmp_spectrum = poisson_sample_template(uranium_templates['235U'],
                                           u235_ph)
    tmp_spectrum += poisson_sample_template(uranium_templates['238U'],
                                            u238_ph)
    tmp_spectrum += poisson_sample_template(uranium_templates['232U'],
                                            u232_ph)
    tmp_spectrum += poisson_sample_template(uranium_templates['UXRAY'],
                                            uxry_ph)

    full_spectrum = tmp_spectrum[0:1024]+background_spectrum_sampled[0:1024]

    return full_spectrum
