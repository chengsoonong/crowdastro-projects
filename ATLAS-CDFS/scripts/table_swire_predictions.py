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

swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False)
_, (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, overwrite=False)

swire_names = numpy.array(swire_names)
swire_coords = numpy.array(swire_coords)

predictions_map = collections.defaultdict(dict) # SWIRE -> predictor -> probability
swire_coords_map = {}
known_predictors = set()

for classifier, predictions_ in [['LR', lr_predictions], ['CNN', cnn_predictions], ['RF', rf_predictions]]:
    for predictions in predictions_:
        dataset_name = predictions.dataset_name
        labeller = predictions.labeller
        if labeller == 'rgz' and 'Norris' in dataset_name:
            labeller = 'RGZ N'
        labeller = labeller.title() if labeller == 'norris' else labeller.upper()
        predictor_name = '{}({} / {})'.format(classifier, labeller, titlemap[dataset_name])
        swire_names_ = swire_names[swire_test_sets[:, pipeline.SET_NAMES['RGZ'], predictions.quadrant]]
        swire_coords_ = swire_coords[swire_test_sets[:, pipeline.SET_NAMES['RGZ'], predictions.quadrant]]
        assert predictions.probabilities.shape[0] == len(swire_names_), \
            'expected {}, got {}'.format(predictions.probabilities.shape[0], len(swire_names_))
        for name, coords, prediction in zip(swire_names_, swire_coords_, predictions.probabilities):
            predictions_map[name][predictor_name] = prediction
            swire_coords_map[name] = coords
        known_predictors.add(predictor_name)

known_predictors = sorted(known_predictors)

swires = sorted(predictions_map)
ras = []
decs = []
predictor_columns = collections.defaultdict(list)
for swire in swires:
    for predictor in known_predictors:
        predictor_columns[predictor].append(predictions_map[swire].get(predictor, ''))
    ras.append(swire_coords_map[swire][0])
    decs.append(swire_coords_map[swire][1])

table = astropy.table.Table(
    data=[swires, ras, decs] + [predictor_columns[p] for p in known_predictors],
    names=['SWIRE', 'RA', 'Dec'] + known_predictors)
table.write('/Users/alger/data/Crowdastro/predicted_swire_table_13_07_17.csv', format='csv')
for p in known_predictors:
    table[p].format = '{:.4f}'
table.write('/Users/alger/data/Crowdastro/predicted_swire_table_13_07_17.tex', format='latex')
