from __future__ import print_function
import numpy as np


def read_spectrum(filename,
                  return_type=int):
    """
    Reads a .spe file into a numpy array.

    Parameters:
    -----------
        filename : string
            Filename with .spe extension
        return_type : int or float
            Type of number to return
    Returns:
    --------
        spectrum : vector
            The source spectrum

    """

    try:
        with open(filename, 'r') as myFile:
            filecontent = myFile.readlines()
        for index, line in enumerate(filecontent):
            if '$DATA:' in line:
                break
        spec_len_index = index + 1
        spec_index = index + 2
        spec_len = filecontent[spec_len_index]
    except:
        with open(filename, 'rb') as myFile:
            filecontent = myFile.readlines()
        for index, line in enumerate(filecontent):
            if b'$DATA:' in line:
                break
        spec_len_index = index + 1
        spec_index = index + 2
        spec_len = filecontent[spec_len_index].decode()[:-2]
    else:
        print('spe in unknown encoding')

    spec_len = int(spec_len.split(' ')[1]) + 1
    spectrum = [return_type(x) for x in filecontent[spec_index:
                                                    spec_index + spec_len]]
    return spectrum
