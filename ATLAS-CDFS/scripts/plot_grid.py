#!/usr/bin/env python3
"""Plot grid of balanced accuracies.

Input files:
- ???

Output files:
- images/cdfs_ba_grid.pdf
- images/cdfs_ba_grid.png
- images/elais_ba_grid.pdf
- images/elais_ba_grid.png

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""
from collections import defaultdict
import itertools

import astropy.table
from crowdastro.crowd.util import balanced_accuracy
import matplotlib.pyplot as plt
import numpy

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

def plot_grid(field='cdfs'):
    # Load predictions.
    lr_predictions = itertools.chain(
        pipeline.unserialise_predictions(
            pipeline.WORKING_DIR + 'LogisticRegression_norris_{}_predictions'.format(field)),
        pipeline.unserialise_predictions(
            pipeline.WORKING_DIR + 'LogisticRegression_rgz_{}_predictions'.format(field)))
    rf_predictions = itertools.chain(
        pipeline.unserialise_predictions(
            pipeline.WORKING_DIR + 'RandomForestClassifier_norris_{}_predictions'.format(field)),
        pipeline.unserialise_predictions(
            pipeline.WORKING_DIR + 'RandomForestClassifier_rgz_{}_predictions'.format(field)))
    cnn_predictions = itertools.chain(
        pipeline.unserialise_predictions(
            pipeline.WORKING_DIR + 'CNN_norris_{}_predictions'.format(field)),
        pipeline.unserialise_predictions(
            pipeline.WORKING_DIR + 'CNN_rgz_{}_predictions'.format(field)))

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

    if field == 'cdfs':
        # Load RGZ cross-identifications and compute a balanced accuracy with them.
        swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False, field=field)
        swire_labels = pipeline.generate_swire_labels(swire_names, swire_coords, overwrite=False, field=field)
        (_, atlas_test_sets), (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, swire_labels, overwrite=False, field=field)
        label_rgz_accuracies = {sstr: [0] * 4 for sstr in pipeline.SET_NAMES}
        label_norris_accuracies = {sstr: [1] * 4 for sstr in pipeline.SET_NAMES}  # By definition.
        for dataset_name in pipeline.SET_NAMES:
            for quadrant in range(4):
                test_set = swire_test_sets[:, pipeline.SET_NAMES[dataset_name], quadrant]
                predictions = swire_labels[test_set, 1]
                trues = swire_labels[test_set, 0]
                ba = balanced_accuracy(trues, predictions)
                label_rgz_accuracies[dataset_name][quadrant] = ba

    colours = ['grey', 'magenta', 'blue', 'orange']
    markers = ['o', '^', 'x', 's']
    handles = {}
    plt.figure(figsize=(5, 5))

    accuracy_map = defaultdict(lambda: defaultdict(dict))  # For table output.
    output_sets = [
        ('LR', [lr_norris_accuracies, lr_rgz_accuracies]),
        ('CNN', [cnn_norris_accuracies, cnn_rgz_accuracies]),
        ('RF', [rf_norris_accuracies, rf_rgz_accuracies]),
    ]
    if field == 'cdfs':
        output_sets.append(('Labels', [label_norris_accuracies, label_rgz_accuracies]))
    for j, (classifier_name, classifier_set) in enumerate(output_sets):
        for i, set_name in enumerate(norris_labelled_sets):
            if 'compact' not in set_name:  # Skip compact.
                ax = plt.subplot(2, 1, {'RGZ & Norris & resolved': 1, 'RGZ & Norris': 2}[set_name])
                ax.set_ylim((80, 100))
                ax.set_xlim((-0.5, 1.5))
                ax.set_xticks([0, 1])#, 2])
                ax.set_xticklabels(['Norris',
                                    # 'RGZ N',
                                    'RGZ',
                                   ], rotation='horizontal')
                if i == 2:
                    plt.xlabel('Labels')
                plt.ylabel('{}\nBalanced accuracy\n(per cent)'.format(titlemap[set_name]))

                ax.title.set_fontsize(16)
                ax.xaxis.label.set_fontsize(12)
                ax.yaxis.label.set_fontsize(9)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontsize(10)

                ax.grid(which='major', axis='y', color='#EEEEEE')
            for k in range(4):
                if 'compact' in set_name:
                    continue
                if j != 3:  # !Labels
                    ax.scatter([0 + (j - 1) / 5], classifier_set[0][set_name][k] * 100,
                                color=colours[j], marker=markers[j], linewidth=1, edgecolor='k')
                rgz_offset = ((j - 1.5) / 6) if field == 'cdfs' else (j - 1) / 5
                handles[j] = ax.scatter([1 + rgz_offset],
                           classifier_set[1][fullmap[set_name]][k] * 100,
                           color=colours[j], marker=markers[j], linewidth=1, edgecolor='k')
                # ax.scatter([1 + (j - 1) / 5], classifier_set[1][set_name][k] * 100,
                #            color=colours[j], marker=markers[j], linewidth=1, edgecolor='k')
            # Compute for table.
            for labeller in ['Norris', 'RGZ N', 'RGZ']:
                if labeller == 'Norris':
                    mean = numpy.mean(classifier_set[0][set_name]) * 100
                    stdev = numpy.std(classifier_set[0][set_name]) * 100
                elif labeller == 'RGZ N':
                    continue
                    # mean = numpy.mean(classifier_set[1][set_name]) * 100
                    # stdev = numpy.std(classifier_set[1][set_name]) * 100
                elif labeller == 'RGZ':
                    mean = numpy.mean(classifier_set[1][fullmap[set_name]]) * 100
                    stdev = numpy.std(classifier_set[1][fullmap[set_name]]) * 100
                accuracy_map[labeller][classifier_name][titlemap[set_name]] = '${:.02f} \\pm {:.02f}$'.format(mean, stdev)

    # Assemble table.
    col_labeller = []
    col_classifier = []
    col_compact = []
    col_resolved = []
    col_all = []
    for labeller in ['Norris', 'RGZ N', 'RGZ']:
        if labeller == 'RGZ N':
            continue

        for classifier in ['CNN', 'LR', 'RF'] + ['Labels'] if field == 'cdfs' else []:
            col_labeller.append(labeller)
            col_classifier.append(classifier)
            col_compact.append(accuracy_map[labeller][classifier]['Compact'])
            col_resolved.append(accuracy_map[labeller][classifier]['Resolved'])
            col_all.append(accuracy_map[labeller][classifier]['All'])
    out_table = astropy.table.Table([col_labeller, col_classifier, col_compact, col_resolved, col_all],
                                    names=['Labeller', 'Classifier', "Mean `Compact' accuracy\\\\(per cent)",
                                           "Mean `Resolved' accuracy\\\\(per cent)",
                                           "Mean `All' accuracy\\\\(per cent)"])
    out_table.write('../{}_accuracy_table.tex'.format(field), format='latex')

    plt.figlegend([handles[j] for j in sorted(handles)], ['LR', 'CNN', 'RF'] + (['Labels'] if field == 'cdfs' else []), 'lower center', ncol=4, fontsize=10)
    plt.subplots_adjust(bottom=0.2, hspace=0.25)
    plt.savefig('../images/{}_ba_grid.pdf'.format(field),
                bbox_inches='tight', pad_inches=0)
    plt.savefig('../images/{}_ba_grid.png'.format(field),
                bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    plot_grid(field='cdfs')
    plot_grid(field='elais')