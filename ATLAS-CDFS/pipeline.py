#!/usr/bin/env python3
"""
Apply binary classification to cross-identify radio emission with
the associated host galaxy in ATLAS-EMU.

Input files:
- one-table-to-rule-them-all.tbl - contains Norris et al. (2006) table 6,
    Franzen et al. (2015) 11JAN14 prerelease DR3, and Wong et al. (2017)
    prerelease DR1.
- crowdastro-swire.h5 - output from Alger (2016) crowdastro.import_data.
- static_rgz_host_full.csv - output from Wong et al. (2017) Radio Galaxy Zoo
    static_catalog.py.
- SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl - Fall '05 CDFS SWIRE catalogue.

References:
Alger, *Learning from Crowd Labels to find Black Holes*. 2016.
Alger, M. J., et al., *Radio Galaxy Zoo: crowdsourced labels for training machine
    learning methods for radio host cross-identification*. 2017, in preparation.
Banfield, J. K., et al., *Radio Galaxy Zoo: host galaxies and radio
    morphologies derived from visual inspection*. 2015.
Franzen, T. M. O., et al., *ATLAS - I. Third Release of 1.4 GHz Mosaics and
    Component Catalogues*. 2015.
Lonsdale, C. J., et al., *SWIRE: The SIRTF Wide-Area Infrared Extragalactic
    Survey*. 2003.
Norris, R. P., et al., *Deep ATLAS radio observations of the CDFS-SWIRE field*.
    2006.
Wong, O. I., et al.,
    *Radio Galaxy Zoo Data Release 1: morphological classifications of 100,000
    FIRST radio sources*. 2017, in preparation.

Requirements:
    astropy
    attrs
    h5py
    matplotlib
    numpy
    scikit-learn
    scipy
    seaborn

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import collections
import errno
import logging
import os
from typing import List, Sequence, Union, Dict, Set, Any

import astropy.io.ascii
import h5py
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy
from scipy.spatial import KDTree
import seaborn
from sklearn.base import ClassifierMixin as Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CROWDASTRO_PATH = '/Users/alger/data/Crowdastro/crowdastro-swire.h5'
RGZ_PATH = '/Users/alger/data/RGZ/dr1_weighted/static_rgz_host_full.csv'
SWIRE_PATH = '/Users/alger/data/SWIRE/SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl'
TABLE_PATH = '/Users/alger/data/Crowdastro/one-table-to-rule-them-all.tbl'
WORKING_DIR = '/tmp/atlas-ml/'
IMAGE_SIZE = 1024

# Sensitivities of Spitzer (in µJy).
SPITZER_SENSITIVITIES = {
    36: 7.3,
    45: 9.7,
    58: 27.5,
    80: 32.5,
    24: 450,
}

# Names of datasets and corresponding indices.
SET_NAMES = {
    'RGZ & Norris & compact': 0,
    'RGZ & Norris & resolved': 1,
    'RGZ & Norris': 2,
    'RGZ & compact': 3,
    'RGZ & resolved': 4,
    'RGZ': 5,
}

N = 'N'  # For type annotations.
D = 'D'
Figure = matplotlib.figure.Figure

# Create working directory.
try:
    os.makedirs(WORKING_DIR)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

# Setup plotting style.
seaborn.set_style('white')

# Setup logging.
log = logging.getLogger(__name__)


class NDArray:
    """Represent NumPy arrays."""

    def __init__(self, *shape: Union[int, str]) -> None:
        self.shape = shape

    def __getitem__(self, item: type) -> 'NDArray':
        self.item = item
        return self

    def __repr__(self) -> str:
        return 'NDArray({})[{}]'.format(', '.join(str(i) for i in self.shape),
                                        self.item)


def generate_swire_features(
        ) -> (List[str], NDArray(N, 2)[float], NDArray(N, D)[float]):
    """Generate features for SWIRE objects.

    Source: 102_classification.ipynb
    
    Returns
    -------
    (SWIRE names,
     SWIRE RA/dec coordinates,
     SWIRE features)
    """
    try:
        with h5py.File(os.path.join(WORKING_DIR, 'swire.h5'), 'r') as f:
            log.info('Reading features from swire.h5')
            names = [name.decode('ascii') for name in f['names']]
            coords = f['coords'].value
            features = f['features'].value
            return (names, coords, features)
    except OSError as e:
        pass
        # I'd love to check the errno here, but h5py hides it...

    with h5py.File(CROWDASTRO_PATH, 'r') as crowdastro_f:
        # Load coordinates of SWIRE objects.
        swire_coords = crowdastro_f['/swire/cdfs/numeric'][:, :2]
        n_swire = len(swire_coords)
        # Initialise features array.
        swire_features = numpy.zeros((n_swire,
                                      6 +  # Magnitude differences
                                      1 +  # S_3.6
                                      2 +  # Stellarities
                                      1 +  # Distances
                                      32 * 32  # Image
                                     ))
        # Load radio images of SWIRE objects.
        swire_features[:, -IMAGE_SIZE:] = \
            crowdastro_f['/swire/cdfs/numeric'][:, -IMAGE_SIZE:]
        # asinh stretch the images.
        swire_features[:, -IMAGE_SIZE:] = numpy.arcsinh(
            swire_features[:, -IMAGE_SIZE:] / 0.1) / numpy.arcsinh(1 / 0.1)
        # Load minimum distances to ATLAS objects.
        distances = crowdastro_f['/atlas/cdfs/numeric'][:, -n_swire:].min(
            axis=0)
        assert distances.shape == (n_swire,)
        # Load names of SWIRE objects.
        swire_names = [
            name.decode('ascii')
            for name in crowdastro_f['/swire/cdfs/string'].value]
        crowdastro_swire_names = {name: index 
                                  for index, name in enumerate(swire_names)}
    
    # Load features from SWIRE catalogue.
    # The catalogue is too big for AstroPy, so we parse it ourselves.
    headers = []
    for row_num, line in enumerate(open(SWIRE_PATH)):
        if line.startswith('\\'):
            continue
        
        if line.startswith('|') and not headers:
            headers.extend(map(str.strip, line.split('|')[1:-1]))
            lengths = list(map(len, headers))
            continue
        
        if line.startswith('|'):
            continue
        
        line = dict(zip(headers, line.split()))
        
        name = line['object']
        if name not in crowdastro_swire_names:
            continue  # Skip non-crowdastro SWIRE.

        crowdastro_index = crowdastro_swire_names[name]

        fluxes = []
        for s in [36, 45, 58, 80]:
            v = line['flux_ap2_{}'.format(s)]
            try:
                v = float(v)
            except:
                # This is the "default" value in the SWIRE catalogue.
                v = -99.0

            if v == -99.0:
                # 5 sigma is an upper-bound for flux in each band.
                v = SPITZER_SENSITIVITIES[s]

            fluxes.append(v)
            
        mags = [numpy.log10(s) for s in fluxes]
        mag_diffs = [mags[0] - mags[1], mags[0] - mags[2], mags[0] - mags[3],
                     mags[1] - mags[2], mags[1] - mags[3],
                     mags[2] - mags[3]]
        # Guaranteed a stellarity in the first two bands only.
        stellarities_ = [line['stell_{}'.format(s)] for s in [36, 45]]
        stellarities = []
        for s in stellarities_:
            if s != 'null' and s != '-9.00':
                stellarities.append(float(s))
            else:
                stellarities.append(float('nan'))
        # We will have nan stellarities, but we will replace those with the
        # mean later.
        non_image_features = numpy.concatenate([
            mag_diffs,
            mags[:1],
            stellarities,
            [distances[crowdastro_index]],
        ])
        swire_features[crowdastro_index, :-IMAGE_SIZE] = non_image_features

    # Set nans to the mean.
    for feature in range(swire_features.shape[1]):
        nan = numpy.isnan(swire_features[:, feature])
        swire_features[:, feature][nan] = \
            swire_features[:, feature][~nan].mean()
    assert not numpy.any(numpy.isnan(swire_features))

    # Normalise and centre the non-image features.
    swire_features[:, :-IMAGE_SIZE] -= \
        swire_features[:, :-IMAGE_SIZE].mean(axis=0)
    swire_features[:, :-IMAGE_SIZE] /= \
        swire_features[:, :-IMAGE_SIZE].std(axis=0)

    # Write back to a file.
    log.info('Creating new file swire.h5')
    with h5py.File(os.path.join(WORKING_DIR, 'swire.h5'), 'w') as f:
        names = numpy.array(swire_names, dtype='<S{}'.format(
            max(len(i) for i in swire_names)))
        f.create_dataset('names', data=names)
        f.create_dataset('coords', data=swire_coords)
        f.create_dataset('features', data=swire_features)

    return swire_names, swire_coords, swire_features


def generate_swire_labels(swire_names: List[str]) -> NDArray(N, 2)[bool]:
    """Generate Norris and RGZ SWIRE labels.

    Source: 102_classification.ipynb
    """
    try:
        with h5py.File(os.path.join(WORKING_DIR, 'swire_labels.h5'),
                       'r') as f_h5:
            log.info('Reading labels from swire_labels.h5')
            return f_h5['labels'].value
    except OSError:
        pass

    table = astropy.io.ascii.read(TABLE_PATH)
    swire_name_to_index = {name: index
                           for index, name in enumerate(swire_names)}
    n_swire = len(swire_names)
    swire_labels = numpy.zeros((n_swire, 2), dtype=bool)
    # Load Norris labels.
    for row in table:
        norris = row['Source SWIRE (Norris)']
        if norris and norris in swire_name_to_index:
            swire_labels[swire_name_to_index[norris], 0] = True
    # Load RGZ labels.
    rgz_catalogue = astropy.io.ascii.read(RGZ_PATH)
    for row in rgz_catalogue:
        rgz = row['SWIRE.designation']
        if rgz in swire_name_to_index:
            swire_labels[swire_name_to_index[rgz], 1] = True

    assert numpy.any(swire_labels, axis=0).all()

    log.info('Creating new file swire_labels.h5')
    with h5py.File(os.path.join(WORKING_DIR, 'swire_labels.h5'), 'w') as f_h5:
        f_h5.create_dataset('labels', data=swire_labels)

    return swire_labels


_ats_swire_tree = None  # Cache the SWIRE KDTree because generating it is slow.
_ats_table = None  # Cache the table because reading it is slow.

def atlas_to_swire(
        swire_coords: NDArray(N, 2)[float],
        atlas: List[int],
        radius: float=1 / 60) -> List[int]:
    """Convert ATLAS objects to nearby SWIRE objects.

    Parameters
    ----------
    swire_coords
        Array of SWIRE RA/dec.
    atlas
        List of table Keys.
    radius
        Radius to search for SWIRE objects.

    Returns
    -------
    List of SWIRE objects.
    """
    global _ats_swire_tree, _ats_table
    if _ats_swire_tree is None:
        _ats_swire_tree = KDTree(swire_coords)
    if _ats_table is None:
        _ats_table = astropy.io.ascii.read(TABLE_PATH)
    swire_tree = _ats_swire_tree
    table = _ats_table

    # Look up the coordinates of the ATLAS objects.
    atlas = set(atlas)
    ras = [r['Component RA (Franzen)'] for r in table if r['Key'] in atlas]
    decs = [r['Component DEC (Franzen)'] for r in table if r['Key'] in atlas]
    coords = numpy.vstack([ras, decs]).T
    nearby = sorted({int(i)
                     for i in numpy.concatenate(
                        swire_tree.query_ball_point(coords, radius))})
    return nearby


def compact_test(r: Dict[str, float]) -> bool:
    """Check if a row represents a compact component."""
    if not r['Component S (Franzen)']:  # Why does this happen?
        return True

    R = numpy.log(r['Component S (Franzen)'] / r['Component Sp (Franzen)'])
    R_err = numpy.sqrt(
        (r['Component S_ERR (Franzen)'] / r['Component S (Franzen)']) ** 2 +
        (r['Component Sp_ERR (Franzen)'] / r['Component Sp (Franzen)']) ** 2)
    return R < 2 * R_err


_fs_table = None  # Cache the table because reading it is slow.

def filter_subset(subset: Set[int], q: int) -> Set[int]:
    """Filter subset to just include indices of ATLAS objects in a quadrant."""
    global _fs_table
    if _fs_table is None:
        _fs_table = astropy.io.ascii.read(TABLE_PATH)
    table = _fs_table

    # Where we'll split our quadrants. RA/dec.
    middle = (52.8, -28.1)

    subset_ = set()
    for s in subset:
        row = table[table['Key'] == s][0]
        coords = row['Component RA (Franzen)'], row['Component DEC (Franzen)']

        if (
              (q == 0 and coords[0] >= middle[0] and coords[1] >= middle[1]) or
              (q == 1 and coords[0] < middle[0] and coords[1] >= middle[1]) or
              (q == 2 and coords[0] < middle[0] and coords[1] < middle[1]) or
              (q == 3 and coords[0] >= middle[0] and coords[1] < middle[1])):
            subset_.add(s)
    return subset_


def generate_data_sets(
        swire_coords: NDArray(N, 2)
    ) -> (
        NDArray(N, 6, 4)[bool],
        NDArray(N, 6, 4)[bool]):
    """Generate training/testing sets.

    Sets generated:
    - RGZ & Norris & compact
    - RGZ & Norris & resolved
    - RGZ & Norris
    - RGZ & compact
    - RGZ & resolved
    - RGZ

    Source: 104_one_notebook_to_train_them_all.ipynb

    Returns
    -------
    Two boolean arrays. Each indicates which set each SWIRE object is in. The
    first array is training, the second array is testing.

    First index: SWIRE index.
    Second index: Set from above list.
    Third index: Quadrant.
    """
    try:
        with h5py.File(os.path.join(WORKING_DIR, 'swire_sets.h5'),
                       'r') as f_h5:
            log.info('Reading sets from swire_sets.h5')
            return f_h5['train'].value, f_h5['test'].value
    except OSError:
        pass

    table = astropy.io.ascii.read(TABLE_PATH)
    n_swire = len(swire_coords)
    # Generate the base ATLAS sets.
    rgz = {r['Key'] for r in table
           if r['Component Zooniverse ID (RGZ)'] and
           r['Component ID (Franzen)'] == r['Primary Component ID (RGZ)'] and
           r['Component ID (Franzen)']}
    norris = {r['Key'] for r in table if r['Component # (Norris)'] and
                                         r['Component ID (Franzen)']}
    compact = {r['Key'] for r in table if r['Component ID (Franzen)'] and
                                          compact_test(r)}
    subsets = [
        ('RGZ & Norris & compact', rgz & norris & compact),
        ('RGZ & Norris & resolved', rgz & norris - compact),
        ('RGZ & Norris', rgz & norris),
        ('RGZ & compact', rgz & compact),
        ('RGZ & resolved', rgz - compact),
        ('RGZ', rgz),
    ]
    # Check these are in the right order...
    for i, (ss, _) in enumerate(subsets):
        assert SET_NAMES[ss] == i
    assert len(SET_NAMES) == len(subsets)
    training_testing_atlas_sets = {s:[] for s, _ in subsets}
    for subset_str, subset_set in subsets:
        log.debug('Filtering ATLAS/{}'.format(subset_str))
        for q in range(4):  # Quadrants.
            test = filter_subset(subset_set, q)
            train = {i for i in subset_set if i not in test}
            training_testing_atlas_sets[subset_str].append((train, test))
    training_testing_swire_sets = {s:[] for s, _ in subsets}
    for subset_str, subset_set in subsets:
        log.debug('Filtering SWIRE/{}'.format(subset_str))
        for train, test in training_testing_atlas_sets[subset_str]:
            train = atlas_to_swire(swire_coords, train)
            test = atlas_to_swire(swire_coords, test)
            log.debug('{} {} {} {} {}'.format(
                subset_str,
                len(set(train) & set(test)), 'out of',
                len(set(test)), 'overlap'))
            train = sorted(set(train) - set(test))
            training_testing_swire_sets[subset_str].append((train, test))

    # Convert sets to output format.
    # Two arrays (train/test) of size N x 6 x 4.
    log.debug('Converting SWIRE set format')
    swire_sets_test = numpy.zeros((n_swire, 6, 4), dtype=bool)
    swire_sets_train = numpy.zeros((n_swire, 6, 4), dtype=bool)
    for s, (subset_str, subset_set) in enumerate(subsets):
        for q in range(4):
            for n in training_testing_swire_sets[subset_str][q][0]:
                swire_sets_train[n, s, q] = True
            for n in training_testing_swire_sets[subset_str][q][1]:
                swire_sets_test[n, s, q] = True

    log.info('Creating new file swire_sets.h5')
    with h5py.File(os.path.join(WORKING_DIR, 'swire_sets.h5'), 'w') as f_h5:
        f_h5.create_dataset('train', data=swire_sets_train)
        f_h5.create_dataset('test', data=swire_sets_test)

    return swire_sets_train, swire_sets_test

def plot_distributions(swire_features: NDArray(N, D)[float]) -> Figure:
    """Plot feature distributions.

    Source: 107_features.ipynb
    """
    fig = plt.figure(figsize=(5, 10))
    xlabels = ['$\log_{10}(S_{3.6}/S_{4.5})$', '$\log_{10}(S_{3.6}/S_{5.8})$',
               '$\log_{10}(S_{3.6}/S_{8.0})$',
               '$\log_{10}(S_{4.5}/S_{5.8})$', '$\log_{10}(S_{4.5}/S_{8.0})$',
               '$\log_{10}(S_{5.8}/S_{8.0})$', '$\log_{10} S_{3.6}$',
               'Stellarity$_{3.6}$', 'Stellarity$_{4.5}$', 'Distance']
    for i in range(10):
        ax = fig.add_subplot(5, 2, i + 1)
        ax.set_xlim((-2.5, 2.5))
        if i != 5:
            ax.set_ylim((0, 1))
        seaborn.distplot(swire_features[:, i][
            numpy.logical_and(swire_features[:, i] < 2.5,
                              swire_features[:, i] > -2.5)],
            kde_kws={'alpha': 0.0},  # Disabling the KDE breaks things...
            hist_kws={'facecolor': 'black', 'alpha': 1.0})
        ax.set_xlabel(xlabels[i])
    fig.subplots_adjust(hspace=0.5)


def train_classifier(
        classifier: type,
        swire_features: NDArray(N, D)[float],
        swire_labels: NDArray(N, 2)[bool],
        swire_train_sets: NDArray(N, 6, 4)[bool],
        labeller: str,
        dataset_name: str,
        quadrant: int,
        **kwargs: Dict[str, Any]) -> Classifier:
    """Train a classifier using the scikit-learn API.

    Parameters
    ----------
    classifier
        scikit-learn classifier class.
    swire_features
        SWIRE object features.
    swire_labels
        Norris, RGZ labels for each SWIRE object.
    swire_train_sets
        Output of generate_data_sets.
    labeller
        'norris' or 'rgz'.
    dataset_name
        'RGZ & Norris & compact' or
        'RGZ & Norris & resolved' or
        'RGZ & Norris' or
        'RGZ & compact' or
        'RGZ & resolved' or
        'RGZ'.
    quadrant
        int in [0, 4). Quadrant of CDFS to train on.
    kwargs
        Keyword arguments for the classifier.

    Returns
    -------
    Classifier
        scikit-learn classifier.
    """
    classifier = Classifier(class_weight='balanced', **kwargs)
    train = swire_train_sets[:, SET_NAMES[dataset_name], quadrant]
    features = swire_features[train]
    # norris -> 0, rgz -> 1
    assert labeller in {'norris', 'rgz'}
    labels = swire_labels[train, int(labeller != 'norris')]
    classifier.fit(features, labels)
    return classifier


def main():
    swire_names, swire_coords, swire_features = generate_swire_features()
    swire_labels = generate_swire_labels(swire_names)
    generate_data_sets(swire_coords)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()