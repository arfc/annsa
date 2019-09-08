from annsa.template_sampling import (rebin_spectrum, 
                                     apply_LLD,
                                     construct_spectrum,)
import numpy as np

# define lamba = 1000
# define size = 1x1024(the number of channels)
dim = 1024
lam = 1000
random_spectrum = np.random.poisson(lam=lam, size=dim)


# rebinning unit test
def test_rebinning_size():
    output_len = 512
    random_spectrum_rebinned = rebin_spectrum(random_spectrum,
                                              output_len=output_len)
    assert(len(random_spectrum_rebinned) == output_len)


# LLD test
def test_lld():
    lld = 10
    random_spectrum_lld = apply_LLD(random_spectrum, LLD=lld)
    assert(np.sum(random_spectrum_lld[0:lld]) == 0)


# construct spectrum test
def test_construct_spectrum_test_rescale_case1():
    """case 1: Check if rescale returns correctly scaled template"""
    spectrum_counts = 10
    random_spectrum_rescaled = construct_spectrum(
        random_spectrum,
        spectrum_counts=spectrum_counts,)
    assert(np.sum(random_spectrum_rescaled) == 10.0)


def test_construct_spectrum_test_rescale_case2():
    """case 2: Check if rescale returns values above zero"""
    spectrum_counts = 10
    random_spectrum_rescaled = construct_spectrum(
        random_spectrum,
        spectrum_counts=spectrum_counts,)
    assert(np.sum(random_spectrum_rescaled < 0) == 0)
