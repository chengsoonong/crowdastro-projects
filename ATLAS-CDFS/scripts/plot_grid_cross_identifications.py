#!/usr/bin/env python3
"""Plot grid of cross-identification accuracies and an associated table.

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
import collections
import itertools
import logging

import astropy.io.ascii
import matplotlib.pyplot as plt
import numpy

import configure_plotting
import pipeline

configure_plotting.configure()

titlemap = {
    'RGZ & Norris & compact': 'Compact',
    'RGZ & Norris & resolved': 'Resolved',
    'RGZ & Norris': 'All',
    'RGZ & compact': 'Compact',
    'RGZ & resolved': 'Resolved',
    'RGZ': 'All',
}

fullmap = {
    'RGZ & Norris & compact': 'RGZ & compact',
    'RGZ & Norris & resolved': 'RGZ & resolved',
    'RGZ & Norris': 'RGZ',
}

whatset = {
    'RGZ & Norris & compact': 'RGZ & Norris & compact',
    'RGZ & Norris & resolved': 'RGZ & Norris & resolved',
    'RGZ & Norris': 'RGZ & Norris',
    'RGZ & compact': 'RGZ & Norris & compact',
    'RGZ & resolved': 'RGZ & Norris & resolved',
    'RGZ': 'RGZ & Norris',
}

norris_labelled_sets = [
    'RGZ & Norris & compact',
    'RGZ & Norris & resolved',
    'RGZ & Norris',
]

swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False)
swire_labels = pipeline.generate_swire_labels(swire_names, swire_coords, overwrite=False)
(_, atlas_test_sets), (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, swire_labels, overwrite=False)
cids = list(pipeline.cross_identify_all(swire_names, swire_coords, swire_labels, swire_test_sets, swire_labels[:, 0]))
table = astropy.io.ascii.read(pipeline.TABLE_PATH)

atlas_to_swire_norris = {}
key_to_atlas = {}
for row in table:
    name = row['Component Name (Franzen)']
    key_to_atlas[row['Key']] = name
    swire = row['Source SWIRE (Norris)']
    if not swire or not swire.startswith('SWIRE') or not name:
        continue
    atlas_to_swire_norris[name] = swire

labeller_classifier_to_accuracies = collections.defaultdict(list)

for cid in cids:
    if cid.labeller == 'norris' and 'Norris' not in cid.dataset_name:
        continue

    if cid.classifier in {'Groundtruth', 'Random'}:
        continue

    atlas_to_swire_predictor = dict(zip(cid.radio_names, cid.ir_names))
    # For each ATLAS object in RGZ & Norris...
    atlas_keys = atlas_test_sets[:, pipeline.SET_NAMES[whatset[cid.dataset_name]], cid.quadrant].nonzero()[0]
    n_total = 0
    n_correct = 0
    n_skipped = 0
    for i in atlas_keys:
        name = key_to_atlas[i]
        if name not in atlas_to_swire_norris:
            n_skipped += 1
            continue
        if name not in atlas_to_swire_predictor:
            n_skipped += 1
            continue
        swire_norris = atlas_to_swire_norris[name]
        swire_predictor = atlas_to_swire_predictor[name]
        n_correct += swire_norris == swire_predictor
        n_total += 1
    if 'Norris' in cid.dataset_name and cid.labeller == 'rgz':
        labeller = 'RGZ N'
    elif cid.labeller == 'rgz':
        labeller = 'RGZ'
    else:
        labeller = 'Norris'
    labeller_classifier_to_accuracies[labeller, cid.classifier, titlemap[cid.dataset_name]].append(n_correct / n_total)

# Groundtruth and random classifiers exist only for the RGZ & Norris set, but we want to test on all subsets.
# This section duplicates the classifiers and evaluates them on all subsets.
for cid in cids:
    if cid.classifier not in {'Groundtruth', 'Random'}:
        continue

    for dataset_name in ['RGZ & Norris', 'RGZ & Norris & resolved', 'RGZ & Norris & compact']:
        atlas_to_swire_predictor = dict(zip(cid.radio_names, cid.ir_names))
        # For each ATLAS object in RGZ & Norris...
        atlas_keys = atlas_test_sets[:, pipeline.SET_NAMES[dataset_name], cid.quadrant].nonzero()[0]
        n_total = 0
        n_correct = 0
        n_skipped = 0
        for i in atlas_keys:
            name = key_to_atlas[i]
            if name not in atlas_to_swire_norris:
                n_skipped += 1
                continue
            if name not in atlas_to_swire_predictor:
                n_skipped += 1
                continue
            swire_norris = atlas_to_swire_norris[name]
            swire_predictor = atlas_to_swire_predictor[name]
            n_correct += swire_norris == swire_predictor
            n_total += 1
        if 'Norris' in cid.dataset_name and cid.labeller == 'rgz':
            labeller = 'RGZ N'
        elif cid.labeller == 'rgz':
            labeller = 'RGZ'
        else:
            labeller = 'Norris'
        labeller_classifier_to_accuracies[labeller, cid.classifier, titlemap[dataset_name]].append(n_correct / n_total)

labeller_classifier_to_accuracy = {}
labeller_classifier_to_stdev = {}
for key, accuracies in labeller_classifier_to_accuracies.items():
    labeller_classifier_to_accuracy[key] = numpy.mean(accuracies)
    labeller_classifier_to_stdev[key] = numpy.std(accuracies)

random_acc = {k[2]: v * 100
              for k, v in labeller_classifier_to_accuracy.items()
              if k[1] == 'Random'}
random_stdev = {k[2]: v * 100
                for k, v in labeller_classifier_to_stdev.items()
                if k[1] == 'Random'}
best_acc = {k[2]: v * 100
            for k, v in labeller_classifier_to_accuracy.items()
            if k[1] == 'Groundtruth'}
best_stdev = {k[2]: v * 100
              for k, v in labeller_classifier_to_stdev.items()
              if k[1] == 'Groundtruth'}

print('Best: {} +- {}'.format(best_acc, best_stdev))
print('Random: {} +- {}'.format(random_acc, random_stdev))

plt.figure(figsize=(3, 4))
colours = ['grey', 'magenta', 'blue', 'orange']
markers = ['o', 's', 'x', '^']
handles = {}
print('Data set & Labeller & Classifier & Mean accuracy (\\%)\\\\')
for k, set_name in enumerate(norris_labelled_sets[1:]):
    print_set_name = titlemap[set_name]
    ax = plt.subplot(2, 1, 1 + k)
    print('{} & Norris & Perfect & ${:.02f} \\pm {:.02f}$\\\\'.format(print_set_name, best_acc[titlemap[set_name]], best_stdev[titlemap[set_name]]))
    print('{} & Norris & Random & ${:.02f} \\pm {:.02f}$\\\\'.format(print_set_name, random_acc[titlemap[set_name]], random_stdev[titlemap[set_name]]))
    plt.hlines(best_acc[titlemap[set_name]], -0.5, 2.5, linestyles='solid', colors='green', linewidth=1, zorder=1, alpha=0.7)
    plt.hlines(best_acc[titlemap[set_name]] + best_stdev[titlemap[set_name]], -0.5, 2.5, linestyles='dashed', colors='green', linewidth=1, zorder=1, alpha=0.7)
    plt.hlines(best_acc[titlemap[set_name]] - best_stdev[titlemap[set_name]], -0.5, 2.5, linestyles='dashed', colors='green', linewidth=1, zorder=1, alpha=0.7)
    plt.hlines(random_acc[titlemap[set_name]], -0.5, 2.5, linestyles='solid', colors='blue', linewidth=1, zorder=1, alpha=0.7)
    plt.hlines(random_acc[titlemap[set_name]] + random_stdev[titlemap[set_name]], -0.5, 2.5, linestyles='dashed', colors='blue', linewidth=1, zorder=1, alpha=0.7)
    plt.hlines(random_acc[titlemap[set_name]] - random_stdev[titlemap[set_name]], -0.5, 2.5, linestyles='dashed', colors='blue', linewidth=1, zorder=1, alpha=0.7)
    for i, labeller in enumerate(['Norris', 'RGZ N', 'RGZ']):
        for j, classifier in enumerate(['LogisticRegression', 'CNN', 'RandomForestClassifier']):
            ys = numpy.array(labeller_classifier_to_accuracies[labeller, classifier, titlemap[set_name]]) * 100
            xs = [i + (j - 1) / 5] * len(ys)
            print('{} & {} & {} & ${:.02f} \\pm {:.02f}$\\\\'.format(print_set_name, labeller, classifier, numpy.mean(ys), numpy.std(ys)))
            ax.set_xlim((-0.5, 2.5))
            if k == 0:
                ax.set_ylim((0, 100))
            else:
                ax.set_ylim((70, 100))
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Norris', 'RGZ N', 'RGZ'])
            handles[j] = plt.scatter(xs, ys, color=colours[j], marker=markers[j], zorder=2)
        if k == 2:
            plt.xlabel('Labels')
        plt.ylabel('{}\nAccuracy (%)'.format(titlemap[set_name]))

        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(9)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(10)

        ax.grid(which='major', axis='y', color='#EEEEEE')

plt.figlegend([handles[j] for j in sorted(handles)], ['LR', 'CNN', 'RF'], 'lower center', ncol=3, fontsize=10)
plt.subplots_adjust(bottom=0.15, hspace=0.25)
plt.savefig('../images/cdfs_cross_identification_grid.pdf',
            bbox_inches='tight', pad_inches=0)
plt.savefig('../images/cdfs_cross_identification_grid.png',
            bbox_inches='tight', pad_inches=0)
