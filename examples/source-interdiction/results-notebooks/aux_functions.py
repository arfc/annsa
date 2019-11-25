import numpy as np
from glob import glob
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelBinarizer
import matplotlib.colors
from annsa import read_spectrum

def ensemble_predictions(members, scaler, testX):
    # scale inputs
    testX_scaled = scaler(testX)
    # make predictions
    yhats = [model.predict(testX_scaled) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    return result


def ensemble_probas(members, scaler, testX):
    # scale inputs
    testX_scaled = scaler(testX)
    # make predictions
    yhats = [model.predict_proba(testX_scaled) for model in members]
    yhats = np.array(yhats)
    # average across ensemble members
    member_average = np.mean(yhats, axis=1)
    # average across classes
    average = np.mean(member_average, axis=0)
    return average


def shielding_predictions(dataframe_data,
                          all_models,
                          scalers,
                          isotope,
                          shielding_material,
                          shielding_amounts,
                          shielding_strings,
                          spectra_path,
                          spectra_date,):

    for shielding_index, shielding_amount in enumerate(shielding_amounts):
        print(shielding_strings[shielding_index], end='\r')
        spectra = []
        for path in glob(os.path.join('..',
                                      'training_testing_data',
                                      spectra_date,
                                      isotope,
                                      spectra_path+isotope+shielding_amount+'*.Spe')):
            spectra.append(read_spectrum(path))
        spectra_cumsum = np.cumsum(spectra, axis=0)

        for model_class in ['dnn', 'cnn', 'daednn', 'caednn',]:
            for mode in ['-easy', '-full']:
                all_output_probs = []
                model_id = model_class + mode
                for i in range(30):
                    bagged_probs = ensemble_probas(all_models[model_id], scalers[model_id], [spectra_cumsum[i]])
                    all_output_probs.append(bagged_probs)

                dataframe_data.append([model_id,
                                       shielding_material,
                                       shielding_strings[shielding_index],
                                       isotope,
                                       spectra_cumsum,
                                       all_output_probs])
        
    return dataframe_data


def plot_measured_source_shielded_results(results_dataframe,
                                          isotope,
                                          gadras_isotope,
                                          shielding_material,
                                          shielding_strings,
                                          setting,):

    plt.rcParams.update({'font.size': 20})
    gadras_index = np.argwhere(mlb.classes_ == gadras_isotope).flatten()[0]
    plt.figure(figsize=(10,5))
    for option_index, shielding_string in enumerate(shielding_strings):
        for model_idindex, model_id in enumerate(['caednn-'+setting,
                                                  'daednn-'+setting,
                                                  'dnn-'+setting,
                                                  'cnn-'+setting,]):
            results_dataframe_tmp = results_dataframe[results_dataframe['model_id'] == model_id]
            results_dataframe_tmp = results_dataframe_tmp[results_dataframe_tmp['isotope'] == isotope]
            results_dataframe_tmp = results_dataframe_tmp[results_dataframe_tmp['shielding_material'] == shielding_material]
            results_dataframe_tmp = results_dataframe_tmp[results_dataframe_tmp['shielding_strings'] == shielding_string]

            plt.plot(np.linspace(10,300,30),
                     100*np.array(results_dataframe_tmp['posterior_prob'].values[0]).reshape(30,30)[:,gadras_index],
                     linewidth=2.5,
                     linestyle=linestyles[option_index],
                     color=c1.colors[model_idindex],)
    plt.xlabel('Integration Time (seconds)')
    plt.ylabel('Posterior Probability')
    plt.ylim([0,110])


def f1_score_bagged(model_ids,
                    all_models,
                    scalers,
                    testing_spectra,
                    testing_keys_binarized,):
    '''
    Bags a specific model's f1_score from a dictionary of models.

    Inputs
        model_ids : string, list
            List of model_id strings to bag. Specific model_id examples are
            'dnn-full' or 'cae-easy'.
        all_models : dict
            Dictionary containing all models
        testing_spectra : numpy array
            Array containing multiple gamma-ray spectra
        testing_keys_binarized : numpy array
            Array containing one-hot encoded (binarized) keys corresponding to
            testing_spectra

    Outputs
        f1_scores : dict, str, float
            Dictionary indexed by model_id in model_ids, contains f1 score for
            that model
    '''
    f1_scores = {}

    for model_id in model_ids:

        predictions = ensemble_predictions(all_models[model_id], scalers[model_id], testing_spectra)
        true_labels = testing_keys_binarized.argmax(axis=1)

        f1_scores[model_id] = f1_score(true_labels,
                                       predictions,
                                       average='micro')

    return f1_scores


def plot_f1_scores_bagged(dataframe,
                          model_ids,
                          all_models,
                          scalers,
                          indep_variable,
                          plot_label,
                          linestyle,
                          color,
                          **kwargs,
                          ):
    '''
    Plots the F1 scores for model's in model_ids given some dataframe of spectra.

    Inputs
        dataframe : Pandas DataFrame
            DataFrame containing spectra, isotope names, and
            parameter options.
        model_ids : string, list
            List of model_id strings to bag. Specific model_id examples are
            'dnn-full' or 'cae-easy'.
        all_models : dict
            Dictionary containing all models
        indep_variable : str
            The key for accessing the data column that contains the independent
            variable data. This data is plotted on the x-axis.
        kwargs : list, int, float
            Choices of different parameters to simulate

    Outputs
        None
    '''
    mlb = LabelBinarizer()
    keys = list(set(dataframe['isotope']))
    mlb.fit(keys)

    plt.rcParams.update({'font.size': 20})
    f1_scores_models = {}
    for key, value in kwargs.items():
        dataframe = dataframe[dataframe[key] == value]
    for model_id in model_ids:
        tmp_f1_scores = []
        for var in sorted(set(dataframe[indep_variable])):

            subset = dataframe[indep_variable] == var
            tmp_f1_score = f1_score_bagged([model_id],
                                           all_models,
                                           scalers,
                                           np.vstack(dataframe[subset]['spectrum'].to_numpy()),
                                           mlb.transform(dataframe['isotope'])[subset],)
            tmp_f1_scores.append(tmp_f1_score[model_id])

        # f1_scores_models[model_id] = tmp_f1_scores
        if plot_label:
            plt.plot(tmp_f1_scores,
                     label=plot_label,
                     linestyle=linestyle,
                     linewidth=2.5,
                     color=color,)
        else:
            plt.plot(tmp_f1_scores,
                     label=model_id,
                     linestyle=linestyle,
                     linewidth=2.5,
                     color=color,)
#     plt.legend()
    plt.xlabel(indep_variable)
    plt.ylabel('F1 Score')
    plt.ylim([0, 1])
    plt.xticks(
        range(len(sorted(set(dataframe[indep_variable])))),
        [round(var, 2) for var in sorted(set(dataframe[indep_variable]))])


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

c1 = categorical_cmap(5,1, cmap="tab10")
plt.scatter(np.arange(5*1),np.ones(5*1)+1, c=np.arange(5*1), s=180, cmap=c1)

    


