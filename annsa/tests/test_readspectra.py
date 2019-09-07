import os
from numpy import load
from annsa.annsa import read_spectrum


def test_read_spectrum_rocky_flats():
    truedata_filename = os.path.join(os.path.dirname(__file__), './data_folder/rocky_flats.npy')
    true_spectrum = load(truedata_filename)

    annsa_data_filename = os.path.join(os.path.dirname(__file__), './data_folder/rocky_flats_spectra.spe')
    annsa_spectrum = read_spectrum(annsa_data_filename)
    assert (annsa_spectrum == true_spectrum).all()
    pass


def test_read_spectrum_gadras_template():
    truedata_filename = os.path.join(os.path.dirname(__file__), './data_folder/gadras_template.npy')
    true_spectrum = load(truedata_filename)

    annsa_data_filename = os.path.join(os.path.dirname(__file__), './data_folder/gadras_template.spe')
    annsa_spectrum = read_spectrum(annsa_data_filename, float)
    assert (annsa_spectrum == true_spectrum).all()
