#!/usr/bin/env python3
"""Generate a table of predicted probabilities for SWIRE objects.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import collections
import itertools
import re

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

    atlas_to_swire_expert = {}
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
            atlas_to_swire_expert[name] = swire
            atlas_to_ras[name] = row['Component RA (Franzen)']
            atlas_to_decs[name] = row['Component DEC (Franzen)']
    else:
        swire_scoords = astropy.coordinates.SkyCoord(ra=swire_coords[:, 0],
                                                     dec=swire_coords[:, 1],
                                                     unit='deg')
        with astropy.io.fits.open(pipeline.MIDDELBERG_TABLE4_PATH) as elais_components_fits:
            elais_components = elais_components_fits[1].data
            component_to_name = {}
            for i, component in enumerate(elais_components):
                name = component['ATELAIS']
                id_to_atlas[component['CID']] = name
                key_to_atlas[i] = name
                coord = astropy.coordinates.SkyCoord(
                    ra='{} {} {}'.format(component['RAh'], component['RAm'], component['RAs']),
                    dec='-{} {} {}'.format(component['DEd'], component['DEm'], component['DEs']),
                    unit=('hourangle', 'deg'))
                coord = (coord.ra.deg, coord.dec.deg)
                atlas_to_ras[name] = coord[0]
                atlas_to_decs[name] = coord[1]
        # Load SWIRE cross-identification from Table 5.
        with open(pipeline.MIDDELBERG_TABLE5_PATH) as elais_file:
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
                # Nearest SWIRE...
                seps = coord.separation(swire_scoords)
                nearest = numpy.argmin(seps)
                dist = seps[nearest]
                if dist.deg > 5 / 60 / 60:
                    continue
                name = swire_names[nearest]
                for cid in line_cids:
                    atlas_to_swire_expert[id_to_atlas[cid]] = name

    atlas_to_rgz = {}
    atlas_to_radio_consensus = {}
    atlas_to_ir_consensus = {}
    if field == 'cdfs':
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

        if field == 'cdfs':
            atlas_keys = atlas_test_sets[:, pipeline.SET_NAMES['RGZ & Norris'], cid.quadrant].nonzero()[0]
        else:
            atlas_keys = atlas_test_sets[:, 0, 0].nonzero()[0]

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
    expert_xids = []
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
        expert_xids.append(atlas_to_swire_expert.get(atlas, ''))

    table = astropy.table.Table(
        data=[atlases, ras, decs, expert_xids, rgzs, rcs, ircs] + [predictor_columns[p] for p in known_predictors],
        names=['ATLAS', 'RA', 'Dec', 'Norris' if field == 'cdfs' else 'Middelberg',
               'RGZ', 'RGZ radio consensus', 'RGZ IR consensus'] + known_predictors)
    table['RGZ radio consensus'].format = '{:.4f}'
    table['RGZ IR consensus'].format = '{:.4f}'
    table.write('/Users/alger/data/Crowdastro/predicted_cross_ids_table_09_10_17_{}.csv'.format(field), format='csv')
    table.write('/Users/alger/data/Crowdastro/predicted_cross_ids_table_09_10_17_{}.tex'.format(field), format='latex')


if __name__ == '__main__':
    print_table(field='elais')
