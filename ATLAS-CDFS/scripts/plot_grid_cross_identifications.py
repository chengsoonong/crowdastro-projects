#!/usr/bin/env python3
"""Plot grid of cross-identification accuracies and an associated table.

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
import collections
import itertools
import logging
import re

import astropy.io.ascii
import matplotlib.pyplot as plt
import numpy
import scipy.spatial

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

def plot(field='cdfs'):
    swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False, field=field)
    swire_labels = pipeline.generate_swire_labels(swire_names, swire_coords, overwrite=False, field=field)
    (_, atlas_test_sets), (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, swire_labels, overwrite=False, field=field)
    cids = list(pipeline.cross_identify_all(swire_names, swire_coords, swire_labels, swire_test_sets, swire_labels[:, 0], field=field))

    swire_tree = scipy.spatial.KDTree(swire_coords[swire_test_sets[:, 0, 0]])

    if field == 'cdfs':
        table = astropy.io.ascii.read(pipeline.TABLE_PATH)

        atlas_to_swire_expert = {}
        key_to_atlas = {}
        for row in table:
            name = row['Component Name (Franzen)']
            key_to_atlas[row['Key']] = name
            swire = row['Source SWIRE (Norris)']
            if not swire or not swire.startswith('SWIRE') or not name:
                continue
            atlas_to_swire_expert[name] = swire
    else:
        atlas_to_swire_expert = {}
        with astropy.io.fits.open(pipeline.MIDDELBERG_TABLE4_PATH) as elais_components_fits:
            elais_components = elais_components_fits[1].data
            atlas_cid_to_name = {}
            atlas_names = []  # Indices correspond to table 4 rows.
            for component in elais_components:
                cid = component['CID']
                name = component['ATELAIS']
                atlas_names.append(name)
                atlas_cid_to_name[cid] = name
        with open(pipeline.MIDDELBERG_TABLE5_PATH) as elais_file:
            # Took this code from pipeline.py, probably should make it a function
            lines = [line.split('|') for line in elais_file]
            for line in lines:
                if 'ATELAISJ' not in line[0]:
                    continue

                line_cids = line[1]
                if 'C0' not in line_cids and 'C1' not in line_cids:
                    continue

                line_cids = [cid.strip() for cid in line_cids.split(',')]
                swire_coord_re = re.search(r'SWIRE4J(\d\d)(\d\d)(\d\d\.\d\d)(-\d\d)(\d\d)(\d\d\.\d)', line[2])
                if not swire_coord_re:
                    continue
                swire_coord_list = swire_coord_re.groups()
                coord = astropy.coordinates.SkyCoord(
                    ra='{} {} {}'.format(*swire_coord_list[:3]),
                    dec='{} {} {}'.format(*swire_coord_list[3:]),
                    unit=('hourangle', 'deg'))
                coord = (coord.ra.deg, coord.dec.deg)
                # Nearest SWIRE...
                dist, nearest = swire_tree.query(coord)
                if dist > 5 / 60 / 60:
                    logging.warning('No SWIRE match found for Middelberg cross-identification {}'.format(line[0]))
                    continue
                name = numpy.array(swire_names)[swire_test_sets[:, 0, 0]][nearest]
                for cid in line_cids:
                    atlas_to_swire_expert[atlas_cid_to_name[cid]] = name

    labeller_classifier_to_accuracies = collections.defaultdict(list)

    for cid in cids:
        if cid.labeller == 'norris' and 'Norris' not in cid.dataset_name:
            continue

        if cid.classifier in {'Groundtruth', 'Random'}:
            # Deal with these later as they are special.
            continue

        atlas_to_swire_predictor = dict(zip(cid.radio_names, cid.ir_names))
        n_total = 0
        n_correct = 0
        n_skipped = 0
        if field == 'cdfs':
            atlas_keys = atlas_test_sets[:, pipeline.SET_NAMES[whatset[cid.dataset_name]], cid.quadrant].nonzero()[0]
            # For each ATLAS object in RGZ & Norris...
            for i in atlas_keys:
                name = key_to_atlas[i]
                if name not in atlas_to_swire_expert:
                    n_skipped += 1
                    continue
                if name not in atlas_to_swire_predictor:
                    n_skipped += 1
                    continue
                swire_norris = atlas_to_swire_expert[name]
                swire_predictor = atlas_to_swire_predictor[name]
                n_correct += swire_norris == swire_predictor
                n_total += 1
        else:
            # Only one test set for ELAIS.
            atlas_indices = atlas_test_sets[:, 0, 0].nonzero()[0]
            assert atlas_test_sets.shape[0] == len(atlas_names)
            for index in atlas_indices:
                atlas_name = atlas_names[index]
                if atlas_name not in atlas_to_swire_expert:
                    n_skipped += 1
                    continue
                if atlas_name not in atlas_to_swire_predictor:
                    n_skipped += 1
                    continue
                swire_middelberg = atlas_to_swire_expert[atlas_name]
                swire_predictor = atlas_to_swire_predictor[atlas_name]
                n_correct += swire_middelberg == swire_predictor
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
            n_total = 0
            n_correct = 0
            n_skipped = 0
            if field == 'cdfs':
                # For each ATLAS object in RGZ & Norris...
                atlas_keys = atlas_test_sets[:, pipeline.SET_NAMES[dataset_name], cid.quadrant].nonzero()[0]
                for i in atlas_keys:
                    name = key_to_atlas[i]
                    if name not in atlas_to_swire_expert:
                        n_skipped += 1
                        continue
                    if name not in atlas_to_swire_predictor:
                        n_skipped += 1
                        continue
                    swire_norris = atlas_to_swire_expert[name]
                    swire_predictor = atlas_to_swire_predictor[name]
                    n_correct += swire_norris == swire_predictor
                    n_total += 1
            else:
                atlas_indices = atlas_test_sets[:, 0, 0].nonzero()[0]
                assert atlas_test_sets.shape[0] == len(atlas_names)
                for index in atlas_indices:
                    atlas_name = atlas_names[index]
                    if atlas_name not in atlas_to_swire_expert:
                        n_skipped += 1
                        continue
                    if atlas_name not in atlas_to_swire_predictor:
                        n_skipped += 1
                        continue
                    swire_middelberg = atlas_to_swire_expert[atlas_name]
                    swire_predictor = atlas_to_swire_predictor[atlas_name]
                    n_correct += swire_middelberg == swire_predictor
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

    plt.figure(figsize=(5, 6))
    colours = ['grey', 'magenta', 'blue', 'orange']
    markers = ['o', '^', 'x', 's']
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
                handles[j] = plt.scatter(xs, ys, color=colours[j], marker=markers[j], zorder=2, edgecolor='k', linewidth=1)
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
    plt.savefig('../images/{}_cross_identification_grid.pdf'.format(field),
                bbox_inches='tight', pad_inches=0)
    plt.savefig('../images/{}_cross_identification_grid.png'.format(field),
                bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    plot(field='cdfs')
    plot(field='elais')
