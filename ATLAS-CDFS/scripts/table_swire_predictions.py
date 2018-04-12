#!/usr/bin/env python3
"""Generate a table of predicted probabilities for SWIRE objects.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import collections
import itertools

import astropy.table
import numpy

import pipeline

def print_table(field='cdfs'):
    titlemap = {
        'RGZ & Norris & compact': 'Compact',
        'RGZ & Norris & resolved': 'Resolved',
        'RGZ & Norris': 'All',
        'RGZ & compact': 'Compact',
        'RGZ & resolved': 'Resolved',
        'RGZ': 'All',
    }

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

    swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False, field=field)
    swire_labels = pipeline.generate_swire_labels(swire_names, swire_coords, overwrite=False, field=field)
    _, (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, swire_labels, overwrite=False, field=field)

    swire_names = numpy.array(swire_names)
    swire_coords = numpy.array(swire_coords)

    predictions_map = collections.defaultdict(dict) # SWIRE -> predictor -> probability
    swire_coords_map = {}
    swire_expert_map = {}
    swire_rgz_map = {}
    known_predictors = set()

    for classifier, predictions_ in [['LR', lr_predictions], ['CNN', cnn_predictions], ['RF', rf_predictions]]:
        for predictions in predictions_:
            dataset_name = predictions.dataset_name
            labeller = predictions.labeller
            if labeller == 'rgz' and 'Norris' in dataset_name:
                labeller = 'RGZ N'
                continue
            labeller = labeller.title() if labeller == 'norris' else labeller.upper()
            predictor_name = '{}({} / {})'.format(classifier, labeller, titlemap[dataset_name])
            if field == 'cdfs':
                swire_names_ = swire_names[swire_test_sets[:, pipeline.SET_NAMES['RGZ'], predictions.quadrant]]
                swire_coords_ = swire_coords[swire_test_sets[:, pipeline.SET_NAMES['RGZ'], predictions.quadrant]]
                swire_labels_ = swire_labels[swire_test_sets[:, pipeline.SET_NAMES['RGZ'], predictions.quadrant]]
            else:
                swire_names_ = swire_names[swire_test_sets[:, 0, 0]]
                swire_coords_ = swire_coords[swire_test_sets[:, 0, 0]]
                swire_labels_ = swire_labels[swire_test_sets[:, 0, 0]]
            assert predictions.probabilities.shape[0] == len(swire_names_), \
                'expected {}, got {}'.format(predictions.probabilities.shape[0], len(swire_names_))
            for name, coords, prediction, label in zip(swire_names_, swire_coords_, predictions.probabilities, swire_labels_):
                predictions_map[name][predictor_name] = prediction
                swire_coords_map[name] = coords
                swire_expert_map[name] = label[0]
                swire_rgz_map[name] = label[1]
            known_predictors.add(predictor_name)

    known_predictors = sorted(known_predictors)

    swires = sorted(predictions_map)
    ras = []
    decs = []
    is_expert_host = []
    is_rgz_host = []
    predictor_columns = collections.defaultdict(list)
    for swire in swires:
        for predictor in known_predictors:
            predictor_columns[predictor].append(predictions_map[swire].get(predictor, ''))
        ras.append(swire_coords_map[swire][0])
        decs.append(swire_coords_map[swire][1])
        is_expert_host.append(['no', 'yes'][swire_expert_map[swire]])
        is_rgz_host.append(['no', 'yes'][swire_rgz_map[swire]])

    table = astropy.table.Table(
        data=[swires, ras, decs, is_expert_host, is_rgz_host] + [predictor_columns[p] for p in known_predictors],
        names=['SWIRE', 'RA', 'Dec', 'Expert host', 'RGZ host'] + known_predictors)
    table.write('/Users/alger/data/Crowdastro/predicted_swire_table_{}_21_03_18.csv'.format(field), format='csv')
    for p in known_predictors:
        table[p].format = '{:.4f}'
    table.write('/Users/alger/data/Crowdastro/predicted_swire_table_{}_21_03_18.tex'.format(field), format='latex')


if __name__ == '__main__':
    print_table(field='cdfs')
    print_table(field='elais')
