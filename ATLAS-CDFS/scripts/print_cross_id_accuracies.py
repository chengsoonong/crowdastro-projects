#!/usr/bin/env python3
"""Output cross-identification accuracies.

These are assessed against the RGZ & Norris intersection,
on the Norris labels.

Output: stdout

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import astropy.io.ascii

import pipeline


def main():
    swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False)
    swire_labels = pipeline.generate_swire_labels(swire_names, overwrite=False)
    (_, atlas_test_sets), _ = pipeline.generate_data_sets(swire_coords, overwrite=False)
    cids = list(pipeline.cross_identify_all(swire_names, swire_coords))
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

    print('Labeller\tClassifier\tQuadrant\tDataset\tn_correct\tn_total\tn_skipped\tAccuracy')
    for cid in cids:
        atlas_to_swire_predictor = dict(zip(cid.radio_names, cid.ir_names))
        # For each ATLAS object in RGZ & Norris...
        atlas_keys = atlas_test_sets[:, pipeline.SET_NAMES['RGZ & Norris'], cid.quadrant].nonzero()[0]
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
        print(cid.labeller, cid.classifier, cid.quadrant, '{:<20}'.format(cid.dataset_name),
              n_correct, n_total, n_skipped, '{:.02%}'.format(n_correct / n_total),
              sep='\t')


if __name__ == '__main__':
    main()
