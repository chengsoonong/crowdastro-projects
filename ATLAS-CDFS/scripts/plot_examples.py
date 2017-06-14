"""Plots examples of where classifiers are wrong.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import astropy.io.ascii
import astropy.io.fits
import astropy.visualization
import astropy.visualization.wcsaxes
import astropy.wcs
import matplotlib.pyplot as plt
import numpy
import scipy.special
from scipy.spatial import KDTree

import examples_incorrect
import pipeline


CDFS_PATH = '/Users/alger/data/RGZ/cdfs/2x2/'

# divide by this to convert mad into sigma in radio image noise
# calculation - from Enno Middelberg
mad2sigma = numpy.sqrt(2) * scipy.special.erfinv(2 * 0.75 - 1)

# number of radio image sigmas for lowest contour.
nsig = 3.0

# factor by which to increase contours
sigmult = 1.5


def get_predictions(swire_tree, swire_coords, swire_test_sets, atlas_coords, predictor_name, radius=1 / 60):
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
        try:
            assert len(nearby_coords) == len(nearby_predictions)
        except AssertionError:
            pdb.set_trace()
            raise
        return list(zip(nearby_coords, nearby_predictions))


def main(classifier='CNN', labeller='Norris'):
    # Load SWIRE stuff.
    swire_names, swire_coords, swire_features = pipeline.generate_swire_features(overwrite=False)
    swire_labels = pipeline.generate_swire_labels(swire_names, overwrite=False)
    _, (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, overwrite=False)
    swire_tree = KDTree(swire_coords)
    swire_name_to_index = {n: i for i, n in enumerate(swire_names)}
    # Load ATLAS coords.
    table = astropy.io.ascii.read(pipeline.TABLE_PATH)
    atlas_to_coords = {}
    atlas_to_swire_coords = {}
    for row in table:
        name = row['Component Name (Franzen)']
        if not name:
            continue

        atlas_to_coords[name] = row['Component RA (Franzen)'], row['Component DEC (Franzen)']
        index = swire_name_to_index.get(row['Source SWIRE (Norris)'] or '')
        if index:
            atlas_to_swire_coords[name] = swire_coords[index]

    examples = examples_incorrect.get_examples()
    ir_stretch = astropy.visualization.LogStretch(0.001)
    for example in examples[labeller, classifier, 'All']:
        predictor_name = '{}_{}'.format(classifier, labeller)
        cid = example[2]
        # Load FITS stuff.
        radio_fits = astropy.io.fits.open(CDFS_PATH + cid + '_radio.fits')
        ir_fits = astropy.io.fits.open(CDFS_PATH + cid + '_ir.fits')
        wcs = astropy.wcs.WCS(radio_fits[0].header)
        # Compute info for contour levels. (also from Enno Middelberg)
        median = numpy.median(radio_fits[0].data)
        mad = numpy.median(numpy.abs(radio_fits[0].data - median))
        sigma = mad / mad2sigma
        # Set up the plot.
        fig = plt.figure()
        ax = astropy.visualization.wcsaxes.WCSAxes(
            fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs)
        fig.add_axes(ax)
        ax.set_title('{}'.format(example[0], example[1]))
        # Show the infrared.
        ax.imshow(ir_stretch(ir_fits[0].data), cmap='cubehelix_r',
                  origin='lower')
        # Show the radio.
        ax.contour(radio_fits[0].data, colors='black',
                    levels=[nsig * sigma * sigmult ** i for i in range(15)],
                    linewidths=1, origin='lower', zorder=1)
        # Plot predictions.
        predictions = get_predictions(swire_tree, swire_coords, swire_test_sets, atlas_to_coords[example[0]], predictor_name)
        coords = [p[0] for p in predictions]
        probabilities = [p[1] for p in predictions]
        coords = wcs.all_world2pix(coords, 1)
        ax.scatter(coords[:, 0], coords[:, 1], s=numpy.sqrt(numpy.array(probabilities)) * 200, color='white', edgecolor='black', linewidth=1, alpha=0.9, marker='o', zorder=2)
        choice = numpy.argmax(probabilities)
        ax.scatter(coords[choice, 0], coords[choice, 1], s=200 / numpy.sqrt(2), color='blue', marker='x', zorder=2.5)
        norris_coords, = wcs.all_world2pix([atlas_to_swire_coords[example[0]]], 1)
        ax.scatter(norris_coords[0], norris_coords[1], marker='+', s=200, zorder=3, color='green')
        lon, lat = ax.coords
        lon.set_major_formatter('hh:mm:ss')
        lon.set_axislabel('Right Ascension')
        lat.set_axislabel('Declination')
        fn = '{}_{}_{}'.format(classifier, labeller, example[0])
        plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/examples/' + fn + '.png',
            bbox_inches='tight', pad_inches=0)
        plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/examples/' + fn + '.pdf',
            bbox_inches='tight', pad_inches=0)
        plt.clf()


if __name__ == '__main__':
    for classifier in ['CNN', 'LogisticRegression', 'RandomForestClassifier']:
        for labeller in ['Norris', 'RGZ']:
            main(classifier, labeller)
