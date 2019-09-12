from annsa.template_sampling import (rebin_spectrum,
                                     apply_LLD,
                                     construct_spectrum,)
from numpy.testing import assert_almost_equal
import numpy as np
import pytest

@pytest.fixture
def spectrum():
    # define lamba = 1000
    # define size = 1x1024(the number of channels)
    dim = 1024
    lam = 1000
    spectrum = np.random.poisson(lam=lam, size=dim)
    return spectrum

# rebinning unit test
def test_rebinning_size(spectrum):
    output_len = 512
    spectrum_rebinned = rebin_spectrum(spectrum,
                                       output_len=output_len)
    assert(len(spectrum_rebinned) == output_len)


# LLD test
def test_lld(spectrum):
    lld = 10
    spectrum_lld = apply_LLD(spectrum, LLD=lld)
    assert(np.sum(spectrum_lld[0:lld]) == 0)


# construct spectrum test
def test_construct_spectrum_test_rescale_case1(spectrum):
    """case 1: Check if rescale returns correctly scaled template"""
    spectrum_counts = 10
    spectrum_rescaled = construct_spectrum(
        spectrum,
        spectrum_counts=spectrum_counts,)
    assert_almost_equal(np.sum(spectrum_rescaled), 10.0)


def test_construct_spectrum_test_rescale_case2(spectrum):
    """case 2: Check if rescale returns values above zero"""
    spectrum_counts = 10
    spectrum_rescaled = construct_spectrum(
        spectrum,
        spectrum_counts=spectrum_counts,)
    assert(np.sum(spectrum_rescaled < 0) == 0)
