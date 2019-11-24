from annsa.generate_uranium_templates import (
    choose_uranium_template,
    choose_random_uranium_template,
    generate_uenriched_spectrum,)
from numpy.testing import assert_almost_equal
import os
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def both_datasets():
    print(os.path.dirname(__file__))
    background_dataset = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        'data_folder',
        'background_template_dataset.csv',))
    uranium_dataset = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        'data_folder',
        'uranium_templates_final.csv'))
    uranium_dataset = uranium_dataset[uranium_dataset['sourcedist'] != 183]
    uranium_dataset = uranium_dataset[uranium_dataset['sourcedist'] != 250]
    uranium_dataset = uranium_dataset[uranium_dataset['sourcedist'] != 300]
    return background_dataset, uranium_dataset


def test_choose_uranium_template(both_datasets):
    """case 1: Check if right template is chosen"""
    (background_dataset, uranium_dataset) = both_datasets
    uranium_templates = choose_uranium_template(
        uranium_dataset=uranium_dataset,
        sourcedist=50,
        sourceheight=0,
        shieldingdensity=0,
        fwhm=6.5)
    assert(np.array_equal(
        uranium_templates['u232'][50:60],
        [453, 444, 440, 440, 441, 441, 441, 443, 449, 460],))


def test_generate_uenriched_spectrum(both_datasets):
    """case 1: Check if right template is generated"""
    (background_dataset, uranium_dataset) = both_datasets
    uranium_templates = choose_uranium_template(
        uranium_dataset=uranium_dataset,
        sourcedist=50,
        sourceheight=0,
        shieldingdensity=0,
        fwhm=6.5)
    spectrum = []
    for _ in range(100):
        spectrum.append(generate_uenriched_spectrum(
            uranium_templates,
            background_dataset,
            enrichment_level=0.93,
            integration_time=60,
            background_cps=200,
            calibration=[0, 1, 0],
            source_background_ratio=1.0,))
    spectrum = np.average(spectrum, axis=0)
    assert(175 < spectrum[100] < 190)
