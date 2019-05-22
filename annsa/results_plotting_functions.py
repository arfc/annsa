from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from annsa.template_sampling import make_random_spectrum
from sklearn.preprocessing import LabelBinarizer


def hyperparameter_efficiency_plot(accuracy):
    """
    Input: Vector of some accuracy metric. Vector length must be a power of 2.
    Output: Hyperparameter search efficiency curve plot

    This function plots the hyperparameter search efficency curve for a given
    hyperparameter search, as seen in Bergstras 2012 paper on random
    hyperparameter search.

    """
    boxplot_values = []

    # remove excess values
    accuracy = accuracy[0:np.int(2**np.floor(np.log2(len(accuracy))))]

    number_boxplots = int(np.log2(len(accuracy)))

    for plot_index in range(number_boxplots):
        n = 2**plot_index
        experiment = np.max([accuracy[i:i + n] for i in range(0,
                                                              len(accuracy),
                                                              n)], axis=1)
        boxplot_values.append(experiment)

    fig, axes = plt.subplots(figsize=(8, 4))
    _ = axes.boxplot(boxplot_values[:-2],
                     positions=np.arange(1, len(boxplot_values)-1))

    # plot last two experiments of size 4 and 2 as scatter plots
    axes.scatter([number_boxplots-1, ]*4,
                 boxplot_values[-2],
                 s=50,
                 color='r',
                 marker='+')
    axes.scatter([number_boxplots, ]*2,
                 boxplot_values[-1],
                 s=50,
                 color='r',
                 marker='+')
    axes.set_xlim(0, number_boxplots+1)
    axes.set_ylabel('Accuracy', fontsize=15)
    axes.set_xlabel('Experiment Size (number of trials)', fontsize=15)
    axes.set_xticks(np.arange(1, number_boxplots+1))
    _ = axes.set_xticklabels([2**n for n in np.arange(number_boxplots)])
    return fig, axes


def make_f1_scores(all_models, all_spectra, all_keys):
    """
    Makes a dict of f1 scores for some set of independent
    variables in a dataset

    Inputs
        all_models : dict
            Dictionary containing all models used to create F1 scores
        all_spectra : narray
            Numpy array of spectra
        all_keys : Pandas DataFrame
            Numpy array of one-hot spectral keys corresponding to all_spectra

    Outputs
        f1_scores : dict
            Dictionary containing F1 scores and model_id
    """

    # initialize empty scores
    f1_scores = {}
    for key in all_models:
        f1_scores[key] = []

    # initialize empty scores
    for key in f1_scores:
        f1_error_tmp = all_models[key].f1_error(
            all_spectra,
            all_keys)
        f1_score_tmp = 1.0-f1_error_tmp
        f1_scores[key].append(f1_score_tmp)

    return f1_scores


def make_dataset(source_dataset,
                 background_dataset,
                 sourceheight,
                 sourcedist,
                 fwhm,
                 integration_time=60,
                 signal_to_background=1.0,
                 cal_a=0.0,
                 cal_b=1.0,
                 cal_c=0.0,
                 background_cps=200,
                 total_spectra=1e1):
    """
    Makes a dataset of spectra based on template source and background spectra
    and a number of options.

    Inputs
        source_dataset : Pandas DataFrame
            DataFrame containing spectra and identifiers. Examples of
            identifiers are isotope, shielding density, source-detector
            distance
        background_dataset : Pandas DataFrame
            DataFrame containing background spectra and identifiers.
            Identifiers must include location and FWHM
        sourceheight : array, float

        sourceheight : array, float

        fwhm : array, float

        integration_time : array, float
    Outputs
        dataframe : Pandas DataFrame
            DataFrame containing the model_id, F1 score, and options from
            kwargs

    """
    dataset = {'sources': [],
               'backgrounds': [],
               'keys': []}
    calibration = [cal_a, cal_b, cal_c]
    # simulate sources
    isotopes_processed = 0
    for isotope in set(source_dataset['isotope'].values):
        for _ in np.arange(total_spectra):

            source_spectrum, background_spectrum = make_random_spectrum(
                source_dataset,
                background_dataset,
                background_cps=background_cps,
                integration_time=integration_time,
                signal_to_background=signal_to_background,
                calibration=calibration,
                isotope=isotope,
                sourceheight=sourceheight,
                sourcedist=sourcedist,
                fwhm=fwhm)
            dataset['sources'].append(source_spectrum)
            dataset['backgrounds'].append(background_spectrum)
            dataset['keys'].append(isotope)
        isotopes_processed += 1
        print(str(isotopes_processed), end="\r")

    for _ in np.arange(total_spectra):
        source_spectrum, background_spectrum = make_random_spectrum(
            source_dataset,
            background_dataset,
            background_cps=background_cps,
            integration_time=integration_time,
            signal_to_background=0.0,
            calibration=calibration,
            isotope=isotope,
            fwhm=fwhm,)
        dataset['sources'].append(source_spectrum)
        dataset['backgrounds'].append(background_spectrum)
        dataset['keys'].append('background')
    isotopes_processed += 1
    print(isotopes_processed)

    all_spectra = [x + y for x, y in zip(dataset['sources'],
                                         dataset['backgrounds'])]
    all_spectra = np.array(all_spectra)
    all_keys = np.array(dataset['keys'])

    return all_spectra, all_keys


def make_f1_scores_dataframe(models,
                             source_dataset,
                             background_dataset,
                             total_spectra=1e1,
                             **kwargs,):
    """
    Makes a dataset of F1 scores for a list of models and dataset.

    Inputs
        models : dict
            A dictionary containing all models used to create F1 scores
        source_dataset : Pandas DataFrame
            DataFrame containing spectra and identifiers. Examples of
            identifiers are isotope, shielding density, source-detector
            distance
        background_dataset : Pandas DataFrame
            DataFrame containing background spectra and identifiers.
            Identifiers
            must include location and FWHM
        total_spectra : int,float
            Total number of spectra to simulate for each setting
        kwargs :
            Choices of different parameters to simulate

    Outputs
        dataframe : Pandas DataFrame
            DataFrame containing the model_id, F1 score, and options from
            kwargs
    """
    keys = kwargs.keys()
    values = (kwargs[key] for key in keys)
    combinations = [dict(zip(keys, combination))
                    for combination in itertools.product(*values)]

    output_row = []
    f1_scores = []

    columns = ['model_id', 'f1_score']
    for key, value in kwargs.items():
        columns.append(key)

    for i, combination in enumerate(combinations):
        print('combo ' + str(i+1) + ' of ' + str(len(combinations)))
        all_spectra, all_keys = make_dataset(
            source_dataset,
            background_dataset,
            total_spectra=total_spectra,
            sourceheight=combination['sourceheight'],
            sourcedist=combination['sourcedist'],
            fwhm=combination['fwhm'],
            integration_time=combination['integration_time'],
            signal_to_background=combination['signal_to_background'],
            cal_a=combination['cal_a'],
            cal_b=combination['cal_b'],
            cal_c=combination['cal_c'],
            background_cps=combination['background_cps'],)

        all_spectra = np.random.poisson(all_spectra)
        all_keys = all_keys.reshape([all_keys.shape[0], 1])

        mlb = LabelBinarizer()
        all_keys_binarized = mlb.fit_transform(all_keys)

        f1_scores_tmp = make_f1_scores(models,
                                       all_spectra,
                                       all_keys_binarized)

        for key in models:
            output_row_tmp = []
            output_row_tmp.append(key)
            output_row_tmp.append(f1_scores_tmp[key][0])
            for column in columns[2:]:
                output_row_tmp.append(combination[column])
            output_row.append(output_row_tmp)

    dataframe = pd.DataFrame(output_row, columns=columns)

    return dataframe


def plot_f1_scores(dataframe,
                   all_models,
                   indep_variable,
                   plot_label=None,
                   **kwargs
                   ): #DOCSTRING

    f1_scores = {}
    for key, value in kwargs.items():
        dataframe = dataframe[dataframe[key] == value]

    for model in all_models:
        f1_scores_tmp = []
        for var in sorted(set(dataframe[indep_variable])):
            tmp_score = dataframe[
                (dataframe['model_id'] == model) &
                (dataframe[indep_variable] == var)]['f1_score'].values[0]
            f1_scores_tmp.append(tmp_score)

        f1_scores[model] = f1_scores_tmp

    for key, errors in f1_scores.items():
        if plot_label:
            plt.plot(errors, label=key+' '+plot_label)
        else:
            plt.plot(errors, label=key)

    plt.legend()
    plt.xlabel(indep_variable)
    plt.ylabel('F1 Score')
    plt.ylim([0, 1])
    plt.xticks(
        range(len(sorted(set(dataframe[indep_variable])))),
        [round(var, 2) for var in sorted(set(dataframe[indep_variable]))])
