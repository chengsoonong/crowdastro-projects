#!/usr/bin/env python3
"""Generate a table of predicted probabilities for SWIRE objects.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import collections
import itertools

import astropy.io
import astropy.table
import numpy

import pipeline

def print_table(field='cdfs'):
    swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False, field=field)
    swire_labels = pipeline.generate_swire_labels(swire_names, swire_coords, overwrite=False, field=field)
    (_, atlas_test_sets), (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, swire_labels, overwrite=False, field=field)
    cids = list(pipeline.cross_identify_all(swire_names, swire_coords, swire_labels, swire_test_sets, swire_labels[:, 0], field=field))

    atlas_to_swire = collections.defaultdict(dict)  # ATLAS -> predictor -> SWIRE

    atlas_to_swire_norris = {}
    key_to_atlas = {}
    atlas_to_ras = {}
    atlas_to_decs = {}
    id_to_atlas = {}
    if field == 'cdfs':
        table = astropy.io.ascii.read(pipeline.TABLE_PATH)
        for row in table:
            name = row['Component Name (Franzen)']
            if not name:
                continue
            id_to_atlas[row['Component ID (Franzen)']] = name
            key_to_atlas[row['Key']] = name
            swire = row['Source SWIRE (Norris)']
            atlas_to_swire_norris[name] = swire
            atlas_to_ras[name] = row['Component RA (Franzen)']
            atlas_to_decs[name] = row['Component DEC (Franzen)']
    else:
        

    atlas_to_rgz = {}
    atlas_to_radio_consensus = {}
    atlas_to_ir_consensus = {}
    for row in astropy.io.ascii.read(pipeline.RGZ_PATH):
        name = id_to_atlas[row['atlas_id']]
        atlas_to_radio_consensus[name] = row['consensus.radio_level']
        atlas_to_ir_consensus[name] = row['consensus.ir_level']
        atlas_to_rgz[name] = row['SWIRE.designation']

    titlemap = {
        'RGZ & Norris & compact': 'Compact',
        'RGZ & Norris & resolved': 'Resolved',
        'RGZ & Norris': 'All',
        'RGZ & compact': 'Compact',
        'RGZ & resolved': 'Resolved',
        'RGZ': 'All',
    }

    known_predictors = set()

    for cid in cids:
        if cid.labeller == 'norris' and 'Norris' not in cid.dataset_name:
            continue

        if cid.classifier in {'Groundtruth', 'Random'}:
            continue

        atlas_keys = atlas_test_sets[:, pipeline.SET_NAMES['RGZ & Norris'], cid.quadrant].nonzero()[0]
        atlas_to_swire_predictor = dict(zip(cid.radio_names, cid.ir_names))
        n_total = 0
        n_correct = 0
        n_skipped = 0
        if 'Norris' in cid.dataset_name and cid.labeller == 'rgz':
            labeller = 'RGZ N'
            continue
        elif cid.labeller == 'rgz':
            labeller = 'RGZ'
        else:
            labeller = 'Norris'
        predictor_name = '{}({} / {})'.format(
            {'LogisticRegression': 'LR', 'CNN': 'CNN', 'RandomForestClassifier': 'RF'}[cid.classifier],
            labeller, titlemap[cid.dataset_name])
        known_predictors.add(predictor_name)
        for i in atlas_keys:
            name = key_to_atlas[i]
            swire_predictor = atlas_to_swire_predictor.get(name, '')
            atlas_to_swire[name][predictor_name] = swire_predictor

    known_predictors = sorted(known_predictors)

    atlases = sorted(atlas_to_swire)
    ras = []
    decs = []
    norrises = []
    rgzs = []
    rcs = []
    ircs = []
    predictor_columns = collections.defaultdict(list)
    for atlas in atlases:
        for predictor in known_predictors:
            predictor_columns[predictor].append(atlas_to_swire[atlas].get(predictor, ''))
        ras.append(atlas_to_ras[atlas])
        decs.append(atlas_to_decs[atlas])
        rgzs.append(atlas_to_rgz.get(atlas, ''))
        rcs.append(atlas_to_radio_consensus.get(atlas, 0.0))
        ircs.append(atlas_to_ir_consensus.get(atlas, 0.0))
        norrises.append(atlas_to_swire_norris.get(atlas, ''))

    table = astropy.table.Table(
        data=[atlases, ras, decs, norrises, rgzs, rcs, ircs] + [predictor_columns[p] for p in known_predictors],
        names=['SWIRE', 'RA', 'Dec', 'Norris', 'RGZ', 'RGZ radio consensus', 'RGZ IR consensus'] + known_predictors)
    table['RGZ radio consensus'].format = '{:.4f}'
    table['RGZ IR consensus'].format = '{:.4f}'
    table.write('/Users/alger/data/Crowdastro/predicted_cross_ids_table_19_08_17.csv', format='csv')
    table.write('/Users/alger/data/Crowdastro/predicted_cross_ids_table_19_08_17.tex', format='latex')
