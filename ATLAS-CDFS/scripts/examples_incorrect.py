"""Find examples where classifiers are wrong.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import collections

import astropy.io.ascii
import numpy

import pipeline

def get_examples():
    titlemap = {
        'RGZ & Norris & compact': 'Compact',
        'RGZ & Norris & resolved': 'Resolved',
        'RGZ & Norris': 'All',
        'RGZ & compact': 'Compact',
        'RGZ & resolved': 'Resolved',
        'RGZ': 'All',
    }

    swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False)
    swire_labels = pipeline.generate_swire_labels(swire_names, overwrite=False)
    (_, atlas_test_sets), (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, overwrite=False)
    cids = list(pipeline.cross_identify_all(swire_names, swire_coords, swire_test_sets, swire_labels[:, 0]))
    table = astropy.io.ascii.read(pipeline.TABLE_PATH)

    atlas_to_swire_norris = {}
    key_to_atlas = {}
    atlas_to_ras = {}
    atlas_to_decs = {}
    id_to_atlas = {}
    atlas_to_zid = {}
    atlas_to_id = {}
    for row in table:
        name = row['Component Name (Franzen)']
        if not name:
            continue
        id_to_atlas[row['Component ID (Franzen)']] = name
        key_to_atlas[row['Key']] = name
        swire = row['Source SWIRE (Norris)']
        atlas_to_swire_norris[name] = swire
        atlas_to_id[name] = row['Component ID (Franzen)']
        atlas_to_ras[name] = row['Component RA (Franzen)']
        atlas_to_decs[name] = row['Component DEC (Franzen)']
        atlas_to_zid[name] = row['Component Zooniverse ID (RGZ)']

    atlas_to_rgz = {}
    atlas_to_radio_consensus = {}
    atlas_to_ir_consensus = {}
    for row in astropy.io.ascii.read(pipeline.RGZ_PATH):
        name = id_to_atlas[row['atlas_id']]
        atlas_to_radio_consensus[name] = row['consensus.radio_level']
        atlas_to_ir_consensus[name] = row['consensus.ir_level']
        atlas_to_rgz[name] = row['SWIRE.designation']

    cross_identifications = collections.defaultdict(dict)  # ATLAS -> labeller -> SWIRE

    for cid in cids:
        if cid.labeller == 'norris' and 'Norris' not in cid.dataset_name:
            continue

        atlas_to_swire_predictor = dict(zip(cid.radio_names, cid.ir_names))
        # For each ATLAS object in RGZ & Norris...
        atlas_keys = atlas_test_sets[:, pipeline.SET_NAMES['RGZ & Norris'], cid.quadrant].nonzero()[0]
        if 'Norris' in cid.dataset_name and cid.labeller == 'rgz':
            labeller = 'RGZ N'
        elif cid.labeller == 'rgz':
            labeller = 'RGZ'
        else:
            labeller = 'Norris'
        for i in atlas_keys:
            name = key_to_atlas[i]
            if name not in atlas_to_swire_norris:
                continue
            if name not in atlas_to_swire_predictor:
                continue
            cross_identifications[name]['Norris'] = atlas_to_swire_norris[name]
            cross_identifications[name]['RGZ'] = atlas_to_rgz.get(name, None)
            cross_identifications[name][labeller, cid.classifier, titlemap[cid.dataset_name]] = atlas_to_swire_predictor[name]

    """
    For each classifier, pull out examples where:
    - RGZ and Norris agree, but the classifier disagrees.

    Only include RGZ & Norris dataset ("All").
    """
    classifier_to_example = collections.defaultdict(set)
    for atlas, cids_ in cross_identifications.items():
        if cids_['Norris'] != cids_['RGZ']:
            continue

        for classifier, swire in cids_.items():
            if classifier[2] != 'All' or classifier[1] in {'Random', 'Groundtruth'}:
                continue

            if swire != cids_['Norris']:
                classifier_to_example[classifier].add((atlas, atlas_to_zid[atlas], atlas_to_id[atlas]))

    return classifier_to_example


if __name__ == '__main__':
    for classifier in get_examples():
        print(classifier)
        for example in classifier_to_example[classifier]:
            print('\t', example)
