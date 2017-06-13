#!/usr/bin/env python3
"""Plot grid of balanced accuracies.

Input files:
- ???

Output files:
- images/cdfs_ba_grid.pdf
- images/cdfs_ba_grid.png

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""
import itertools

import matplotlib.pyplot as plt

import configure_plotting
import pipeline

configure_plotting.configure()

titlemap = {
    'RGZ & Norris & compact': 'Compact',
    'RGZ & Norris & resolved': 'Resolved',
    'RGZ & Norris': 'All',
}

fullmap = {
    'RGZ & Norris & compact': 'RGZ & compact',
    'RGZ & Norris & resolved': 'RGZ & resolved',
    'RGZ & Norris': 'RGZ',
}

norris_labelled_sets = [
    'RGZ & Norris & compact',
    'RGZ & Norris & resolved',
    'RGZ & Norris',
]

# Load predictions.
lr_predictions = itertools.chain(
    pipeline.unserialise_predictions(
        pipeline.WORKING_DIR + 'LogisticRegression_norris_predictions'),
    pipeline.unserialise_predictions(
        pipeline.WORKING_DIR + 'LogisticRegression_rgz_predictions'))
rf_predictions = itertools.chain(
    pipeline.unserialise_predictions(
        pipeline.WORKING_DIR + 'RandomForestClassifier_norris_predictions'),
    pipeline.unserialise_predictions(
        pipeline.WORKING_DIR + 'RandomForestClassifier_rgz_predictions'))
cnn_predictions = itertools.chain(
    pipeline.unserialise_predictions(
        pipeline.WORKING_DIR + 'CNN_norris_predictions'),
    pipeline.unserialise_predictions(
        pipeline.WORKING_DIR + 'CNN_rgz_predictions'))

# Convert to the format we need. e.g. {'RGZ' -> [acc, acc, acc, acc]}
lr_norris_accuracies = {sstr: [0] * 4 for sstr in pipeline.SET_NAMES}
lr_rgz_accuracies = {sstr: [0] * 4 for sstr in pipeline.SET_NAMES}
rf_norris_accuracies = {sstr: [0] * 4 for sstr in pipeline.SET_NAMES}
rf_rgz_accuracies = {sstr: [0] * 4 for sstr in pipeline.SET_NAMES}
cnn_norris_accuracies = {sstr: [0] * 4 for sstr in pipeline.SET_NAMES}
cnn_rgz_accuracies = {sstr: [0] * 4 for sstr in pipeline.SET_NAMES}
for predictions in lr_predictions:
    dataset_name = predictions.dataset_name
    if predictions.labeller == 'norris':
        lr_norris_accuracies[dataset_name][predictions.quadrant] = predictions.balanced_accuracy
    else:
        lr_rgz_accuracies[dataset_name][predictions.quadrant] = predictions.balanced_accuracy
for predictions in rf_predictions:
    dataset_name = predictions.dataset_name
    if predictions.labeller == 'norris':
        rf_norris_accuracies[dataset_name][predictions.quadrant] = predictions.balanced_accuracy
    else:
        rf_rgz_accuracies[dataset_name][predictions.quadrant] = predictions.balanced_accuracy
for predictions in cnn_predictions:
    dataset_name = predictions.dataset_name
    if predictions.labeller == 'norris':
        cnn_norris_accuracies[dataset_name][predictions.quadrant] = predictions.balanced_accuracy
    else:
        cnn_rgz_accuracies[dataset_name][predictions.quadrant] = predictions.balanced_accuracy

colours = ['grey', 'magenta', 'blue', 'orange']
handles = {}
plt.figure(figsize=(3, 6))
for j, (classifier_name, classifier_set) in enumerate([
        ('LR', [lr_norris_accuracies, lr_rgz_accuracies]),
        ('CNN', [cnn_norris_accuracies, cnn_rgz_accuracies]),
        ('RF', [rf_norris_accuracies, rf_rgz_accuracies]),
        ]):
    for i, set_name in enumerate(norris_labelled_sets):
        ax = plt.subplot(3, 1, 1 + i)
        for k in range(4):
            handles[j] = ax.scatter([0 + (j - 1) / 5], classifier_set[0][set_name][k] * 100, color=colours[j], marker='x')
            ax.scatter([1 + (j - 1) / 5], classifier_set[1][set_name][k] * 100, color=colours[j], marker='x')
            ax.scatter([2 + (j - 1) / 5], classifier_set[1][fullmap[set_name]][k] * 100, color=colours[j], marker='x')

        ax.set_ylim((80, 100))
        ax.set_xlim((-0.5, 2.5))
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Norris',
                            'RGZ N',
                            'RGZ',
                           ], rotation='horizontal')
        if i == 2:
            plt.xlabel('Labels')
        plt.ylabel('{}\nBalanced accuracy (%)'.format(titlemap[set_name]))

        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(9)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(10)

        ax.grid(which='major', axis='y', color='#EEEEEE')

plt.figlegend([handles[j] for j in sorted(handles)], ['LR', 'CNN', 'RF'], 'lower center', ncol=3, fontsize=10)
plt.subplots_adjust(bottom=0.15, hspace=0.25)
plt.savefig('../images/cdfs_ba_grid.pdf',
            bbox_inches='tight', pad_inches=0)
plt.savefig('../images/cdfs_ba_grid.png',
            bbox_inches='tight', pad_inches=0)