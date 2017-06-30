"""Plots classifier ambiguity against compactness.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import logging

import astropy.io.ascii
import astropy.io.fits
import astropy.visualization
import astropy.visualization.wcsaxes
import astropy.wcs
import matplotlib.pyplot as plt
import numpy
import scipy.special
from scipy.spatial import KDTree

import examples_all
import examples_incorrect
import pipeline


def get_predictions(swire_tree, swire_coords, swire_names, swire_test_sets, atlas_coords, predictor_name, radius=1 / 60):
    import pdb
    predictions_ = pipeline.unserialise_predictions(pipeline.WORKING_DIR + predictor_name + '_predictions', [0, 1, 2, 3], ['RGZ & Norris'])
    for predictions in predictions_:
        nearby = swire_tree.query_ball_point(atlas_coords, radius)  # all-SWIRE indices
        nearby_bool = numpy.zeros((swire_test_sets.shape[0],), dtype=bool)
        nearby_bool[nearby] = True
        set_ = swire_test_sets[:, pipeline.SET_NAMES['RGZ'], predictions.quadrant]  # all-SWIRE indices, mask
        if not nearby_bool[set_].any():
            # Wrong quadrant.
            continue
        # pdb.set_trace()
        nearby_predictions = predictions.probabilities[nearby_bool[set_]]  # quadrant + dataset indices
        nearby_coords = swire_coords[nearby_bool & set_]
        nearby_names = swire_names[nearby_bool & set_]
        try:
            assert len(nearby_coords) == len(nearby_predictions)
        except AssertionError:
            pdb.set_trace()
            raise
        return list(zip(nearby_names, nearby_predictions))


def main(classifier='CNN', labeller='Norris'):
    # Load SWIRE stuff.
    swire_names, swire_coords, swire_features = pipeline.generate_swire_features(overwrite=False)
    swire_labels = pipeline.generate_swire_labels(swire_names, swire_coords, overwrite=False)
    _, (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, overwrite=False)
    swire_tree = KDTree(swire_coords)
    swire_name_to_index = {n: i for i, n in enumerate(swire_names)}

    atlas_names = []
    atlas_compactnesses = []
    atlas_coords = []
    atlas_norris_swire = []

    table = astropy.io.ascii.read(pipeline.TABLE_PATH)
    for row in table:
        name = row['Component Name (Franzen)']
        if not name:
            continue

        if not row['Component Zooniverse ID (RGZ)']:
            continue

        compactness = pipeline.compactness(row)
        atlas_names.append(name)
        atlas_compactnesses.append(compactness)
        atlas_coords.append((row['Component RA (Franzen)'], row['Component DEC (Franzen)']))
        atlas_norris_swire.append(row['Source SWIRE (Norris)'])

    ys = []
    xs_entropy = []
    xs_margin = []
    no_groundtruth = []
    correct = []

    for name, compactness, coords, swire in zip(atlas_names, atlas_compactnesses, atlas_coords, atlas_norris_swire):
        predictor_name = '{}_{}'.format(classifier, labeller)
        predictions = get_predictions(swire_tree, swire_coords, numpy.array(swire_names), swire_test_sets, coords, predictor_name)

        if not predictions:
            print('No predictions for {}'.format(name))
            continue

        chosen_swire = predictions[numpy.argmax([p for _, p in predictions])][0]
        predictions = [p for _, p in predictions]

        predictions_softmax = [numpy.exp(p) / sum(numpy.exp(p) for p in predictions) for p in predictions]
        if len(predictions_softmax) == 1:
            entropy_ambiguity = 0
            margin_ambiguity = 0
        else:
            entropy_ambiguity = -sum(p * numpy.log(p) for p in predictions_softmax if p)
            predictions.sort()
            margin_ambiguity = 1 - (predictions[-1] - predictions[-2])

        ys.append(compactness)
        xs_entropy.append(entropy_ambiguity)
        xs_margin.append(margin_ambiguity)
        no_groundtruth.append(not swire or not swire.startswith('SWIRE'))
        correct.append(swire == chosen_swire)

    ys = numpy.array(ys)
    xs_margin = numpy.array(xs_margin)
    xs_entropy = numpy.array(xs_entropy)
    no_groundtruth = numpy.array(no_groundtruth, dtype=bool)
    correct = numpy.array(correct, dtype=bool)

    print(sum(1 for y in ys if y <= 1))

    plt.subplot(1, 2, 1)
    plt.scatter(xs_margin[no_groundtruth], ys[no_groundtruth], marker='x', color='black', alpha=0.05)
    plt.scatter(xs_margin[~no_groundtruth & correct], ys[~no_groundtruth & correct], marker='x', color='blue', alpha=0.7)
    plt.scatter(xs_margin[~no_groundtruth & ~correct], ys[~no_groundtruth & ~correct], marker='x', color='magenta', alpha=0.7)
    plt.title('Margin')
    plt.xlabel('1 - margin')
    plt.ylabel('$1.3 SNR S / 10 S_p$')
    plt.yscale('log')
    plt.axhline(1, min(xs_margin), max(xs_margin))
    plt.subplot(1, 2, 2)
    plt.scatter(xs_entropy[no_groundtruth], ys[no_groundtruth], marker='x', color='black', alpha=0.05)
    plt.scatter(xs_entropy[~no_groundtruth & correct], ys[~no_groundtruth & correct], marker='x', color='blue', alpha=0.7)
    plt.scatter(xs_entropy[~no_groundtruth & ~correct], ys[~no_groundtruth & ~correct], marker='x', color='magenta', alpha=0.7)
    plt.title('Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('$1.3 SNR S / 10 S_p$')
    plt.yscale('log')
    plt.axhline(1, min(xs_entropy), max(xs_entropy), zorder=-100, linestyle='--', color='black')
    plt.show()

if __name__ == '__main__':
    main()
