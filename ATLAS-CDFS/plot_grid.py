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
lr_predictions = pipeline.unserialise_predictions(
    pipeline.WORKING_DIR + 'LogisticRegression_predictions')
rf_predictions = pipeline.unserialise_predictions(
    pipeline.WORKING_DIR + 'RandomForestClassifier_predictions')

# Convert to the format we need. e.g. {'RGZ' -> [acc, acc, acc, acc]}
lr_norris_accuracies = [None] * 4
lr_rgz_accuracies = [None] * 4
rf_norris_accuracies = [None] * 4
rf_rgz_accuracies = [None] * 4
for predictions in lr_predictions:
    if predictions.labeller == 'norris':
        lr_norris_accuracies[predictions.quadrant] = predictions.balanced_accuracy
    else:
        lr_rgz_accuracies[predictions.quadrant] = predictions.balanced_accuracy
for predictions in rf_predictions:
    if predictions.labeller == 'norris':
        rf_norris_accuracies[predictions.quadrant] = predictions.balanced_accuracy
    else:
        rf_rgz_accuracies[predictions.quadrant] = predictions.balanced_accuracy
print(lr_norris_accuracies)
raise

colours = ['blue', 'red', 'green', 'orange']
handles = []
plt.figure(figsize=(8, 5))
for j, (classifier_name, classifier_set) in enumerate([
        ('LR', [lr_norris_accuracies, lr_rgz_accuracies]),
        ('RF', [rf_norris_accuracies, rf_rgz_accuracies]),
        # ('CNN', [cnn_norris_accuracies, cnn_rgz_accuracies]),
        ]):
    for i, set_name in enumerate(norris_labelled_sets):
        ax = plt.subplot(3, 4, 1 + j + i * 4)
        for k in range(4):
            handles.append(
                ax.scatter([0], classifier_set[0][set_name][k] * 100, color=colours[k], marker='x'))
            ax.scatter([1], classifier_set[1][set_name][k] * 100, color=colours[k], marker='x')
            ax.scatter([2], classifier_set[1][fullmap[set_name]][k] * 100, color=colours[k], marker='x')

        ax.set_ylim((85, 100))
        ax.set_xlim((-0.5, 2.5))
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Norris',
                            'RGZ N',
                            'RGZ',
                           ], rotation='horizontal')
        if i == 2:
            plt.xlabel('Labels')
        if j == 0:
            plt.ylabel('{}\nBalanced accuracy (%)'.format(titlemap[set_name]))
        if i == 0:
            ax.set_title(classifier_name)

        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(10)

        ax.grid(which='major', axis='y', color='#EEEEEE')

plt.figlegend(handles, map(str, range(4)), 'lower center', ncol=4, fontsize=20)
plt.subplots_adjust(top=1, bottom=0.12, right=1, left=0, 
                    hspace=0.2, wspace=0.15)
plt.margins(0, 0)
plt.savefig('images/cdfs_ba_grid.pdf',
            bbox_inches='tight', pad_inches=0)
plt.savefig('images/cdfs_ba_grid.png',
            bbox_inches='tight', pad_inches=0)