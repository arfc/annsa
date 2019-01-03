from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt


def hyperparameter_efficiency_plot(accuracy):
    """
    Input: Vector of some accuracy metric. Vector length must be a power of 2.
    Output: Hyperparameter search efficiency curve plot

    This function plots the hyperparameter search efficency curve for a given
    hyperparameter search, as seen in Bergstras 2012 paper on random
    hyperparameter search.

    """
    boxplot_values = []

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
    _ = axes.set_xticklabels(np.arange(1, number_boxplots+1))
    return fig, axes
