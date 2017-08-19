"""Plot a colour-colour diagram for SWIRE.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import h5py
import matplotlib.pyplot as plt
import numpy

import configure_plotting
import pipeline

configure_plotting.configure()

CROWDASTRO_PATH = '/Users/alger/data/Crowdastro/crowdastro-swire.h5'
SWIRE_PATH = '/Users/alger/data/SWIRE/SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl'

def plot_basic(show=False):
    """Plot basic colour-colour diagram."""
    data = []
    with open(SWIRE_PATH) as f:
        # Skip 5 rows.
        for i in range(5): next(f)
        # Read names.
        names = [i.strip() for i in next(f).split('|')]
        # Skip types.
        for i in range(3): next(f)
        for row in f:
            data.append(dict(zip(names[1:], row.split())))

    f_36 = []
    f_45 = []
    f_58 = []
    f_80 = []

    for row in data:
        f_36.append(float(row['flux_ap2_36']))
        f_45.append(float(row['flux_ap2_45']))
        try:
            f_58.append(float(row['flux_ap2_58']))
        except ValueError:
            f_58.append(-99)  # Already used for some nulls.
        try:
            f_80.append(float(row['flux_ap2_80']))
        except ValueError:
            f_80.append(-99)
    f_36 = numpy.array(f_36)
    f_45 = numpy.array(f_45)
    f_58 = numpy.array(f_58)
    f_80 = numpy.array(f_80)

    # We need a detection in all bands for the diagram.
    detection_58 = (f_58 != -99)
    detection_80 = (f_80 != -99)
    # 3.6/4.5 guaranteed for SWIRE.
    detection_all = detection_58 & detection_80

    ratio_58_36 = numpy.log10(f_58[detection_all] / f_36[detection_all])
    ratio_80_45 = numpy.log10(f_80[detection_all] / f_45[detection_all])
    plt.scatter(ratio_58_36, ratio_80_45, s=0.2, c='k', linewidth=0)
    if show:
        # plt.plot([-0.2, 1.3], [-0.2, -0.2], '--', color='grey', linewidth=1)
        # plt.plot([-0.2, -0.2], [-0.2, 0.3], '--', color='grey', linewidth=1)
        # plt.plot([-0.2, 1.0], [0.3, 1.3], '--', color='grey', linewidth=1)
        plt.xlim((-0.75, 1.0))
        plt.ylim((-0.75, 1.0))
        plt.xlabel('$\\log_{10}(S_{5.8}/S_{3.6})$')
        plt.ylabel('$\\log_{10}(S_{8.0}/S_{4.5})$')
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
        plt.show()


def plot_predictions(cut=0.95, labeller='norris', dataset_name=None, classifier=None):
    """Plot colour-colour diagram for predicted host galaxies.

    labeller in {'norris', 'rgz'}
    dataset_name in {'RGZ & Norris', ...}
    """

    with h5py.File(CROWDASTRO_PATH, 'r') as f:
        swire_numeric_cdfs = f['/swire/cdfs/numeric'][:, 2:2 + 4]

    f_36 = swire_numeric_cdfs[:, 0]
    f_45 = swire_numeric_cdfs[:, 1]
    f_58 = swire_numeric_cdfs[:, 2]
    f_80 = swire_numeric_cdfs[:, 3]
    detection_58 = (f_58 != -99)
    detection_80 = (f_80 != -99)

    p = pipeline.unserialise_predictions(
            pipeline.WORKING_DIR + '{}_{}_cdfs_predictions'.format(classifier, labeller))
    predictions = {}
    for i in p:
        predictions[i.dataset_name, i.quadrant] = i


    swire_names, swire_coords, _ = pipeline.generate_swire_features(overwrite=False, field='cdfs')
    swire_labels = pipeline.generate_swire_labels(swire_names, swire_coords, overwrite=False, field='cdfs')
    _, (_, swire_test_sets) = pipeline.generate_data_sets(swire_coords, swire_labels, overwrite=False, field='cdfs')

    xs = []
    ys = []
    colours = []
    for q in range(4):
        swire_set = swire_test_sets[:, pipeline.SET_NAMES['RGZ'], q]
        if labeller == 'norris' and not dataset_name:
            # predictions_set = predictions['RGZ & Norris', q].probabilities > cut
            f_36_ = f_36[swire_set & swire_labels[:, 0]]#[predictions_set]
            f_45_ = f_45[swire_set & swire_labels[:, 0]]#[predictions_set]
            f_58_ = f_58[swire_set & swire_labels[:, 0]]#[predictions_set]
            f_80_ = f_80[swire_set & swire_labels[:, 0]]#[predictions_set]
        elif labeller == 'rgz' and not dataset_name:
            f_36_ = f_36[swire_set & swire_labels[:, 1]]
            f_45_ = f_45[swire_set & swire_labels[:, 1]]
            f_58_ = f_58[swire_set & swire_labels[:, 1]]
            f_80_ = f_80[swire_set & swire_labels[:, 1]]
        if labeller == 'norris' and dataset_name:
            predictions_set = predictions[dataset_name, q].probabilities > cut
            f_36_ = f_36[swire_set][predictions_set]
            f_45_ = f_45[swire_set][predictions_set]
            f_58_ = f_58[swire_set][predictions_set]
            f_80_ = f_80[swire_set][predictions_set]
            probabilities = predictions[dataset_name, q].probabilities[predictions_set]
        detection_58_ = (f_58_ != -99)
        detection_80_ = (f_80_ != -99)
        detection_all_ = detection_58_ & detection_80_

        ratio_58_36 = numpy.log10(f_58_[detection_all_] / f_36_[detection_all_])
        ratio_80_45 = numpy.log10(f_80_[detection_all_] / f_45_[detection_all_])
        probabilities = probabilities[detection_all_]
        xs.extend(ratio_58_36)
        ys.extend(ratio_80_45)
        colours.extend(probabilities)

    assert len(xs) == len(ys)
    assert len(xs) == len(colours)

    plot_basic()
    if dataset_name:
        plt.scatter(xs, ys, s=20, marker='^', linewidth=0, alpha=0.5, c=numpy.array(colours), cmap='winter')
    else:
        plt.scatter(xs, ys, s=25, c='r', marker='^', linewidth=0)
    plt.xlim((-0.75, 1.0))
    plt.ylim((-0.75, 1.0))
    plt.xlabel('$\\log_{10}(S_{5.8}/S_{3.6})$')
    plt.ylabel('$\\log_{10}(S_{8.0}/S_{4.5})$')
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
    plt.colorbar()
    plt.show()



if __name__ == '__main__':
    plt.figure(figsize=(4, 3))
    # plot_basic(show=True)
    # plot_predictions()
    plot_predictions(dataset_name='RGZ & Norris', classifier='LogisticRegression')
