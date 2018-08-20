from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import annsa as an

data_path = op.join(sb.__path__[0], 'data')


def test_generate_single_source_key():
    """
    Testing to see if the generate_single_source_key function doesn't give infinite or negative values.

    """
    # Make a blank source key 
    blank_source_key = an.generate_single_source_key()
    # Check if key is blank
    inf_check = np.any(np.isinf(blank_source_key))
    npt.assert_equal(inf_check, False)
    
