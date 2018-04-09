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
    click
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
from pprint import pprint
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Union

import astropy.io.ascii
import astropy.io.fits
import astropy.coordinates
import attr
import click
import h5py
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy
from scipy.spatial import KDTree
import scipy.special
import scipy.stats
import seaborn
from sklearn.base import ClassifierMixin as Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics

CROWDASTRO_PATH = '/Users/alger/data/Crowdastro/crowdastro-swire.h5'
RGZ_PATH = '/Users/alger/data/RGZ/dr1_weighted_old/static_rgz_host_full.csv'
MIDDELBERG_TABLE5_PATH = '/Users/alger/data/SWIRE/middelberg_2008_table5.dat'
MIDDELBERG_TABLE4_PATH = '/Users/alger/data/SWIRE/middelberg_2008_table4.fits'
FRANZEN_PATH = '/Users/alger/data/SWIRE/middelberg_2008_table5.dat'
SWIRE_CDFS_PATH = '/Users/alger/data/SWIRE/SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl'
SWIRE_ELAIS_PATH = '/Users/alger/data/SWIRE/SWIRE3_ELAIS_cat_IRAC24_21Dec05.tbl'
TABLE_PATH = '/Users/alger/data/Crowdastro/one-table-to-rule-them-all.tbl'
WORKING_DIR = '/Users/alger/data/Crowdastro/atlas-ml/'
IMAGE_SIZE = 1024

# Spread of half-Gaussian prediction falloff (in degrees).
FALLOFF_SIGMA = 1 / 120

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
M = 'M'
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
        overwrite: bool=False,
        field: str='cdfs',
        ) -> (List[str], NDArray(N, 2)[float], NDArray(N, D)[float]):
    """Generate features for SWIRE objects.

    Source: 102_classification.ipynb

    Parameters
    ----------
    overwrite
        Overwrite existing results.
    field
        'cdfs' or 'elais'
    
    Returns
    -------
    (SWIRE names,
     SWIRE RA/dec coordinates,
     SWIRE features)
    """
    if not overwrite:
        try:
            with h5py.File(os.path.join(WORKING_DIR, 'swire_{}.h5'.format(field)), 'r') as f:
                log.info('Reading features from swire_{}.h5'.format(field))
                names = [name.decode('ascii') for name in f['names']]
                coords = f['coords'].value
                features = f['features'].value
                return (names, coords, features)
        except OSError as e:
            pass
            # I'd love to check the errno here, but h5py hides it...

    with h5py.File(CROWDASTRO_PATH, 'r') as crowdastro_f:
        # Load coordinates of SWIRE objects.
        swire_coords = crowdastro_f['/swire/{}/numeric'.format(field)][:, :2]
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
            crowdastro_f['/swire/{}/numeric'.format(field)][:, -IMAGE_SIZE:]
        # asinh stretch the images.
        swire_features[:, -IMAGE_SIZE:] = numpy.arcsinh(
            swire_features[:, -IMAGE_SIZE:] / 0.1) / numpy.arcsinh(1 / 0.1)
        # Load minimum distances to ATLAS objects.
        distances = crowdastro_f['/atlas/{}/numeric'.format(field)][:, -n_swire:].min(
            axis=0)
        assert distances.shape == (n_swire,)
        # Load names of SWIRE objects.
        swire_names = [
            name.decode('ascii')
            for name in crowdastro_f['/swire/{}/string'.format(field)].value]
        crowdastro_swire_names = {name: index 
                                  for index, name in enumerate(swire_names)}
    
    # Load features from SWIRE catalogue.
    # The catalogue is too big for AstroPy, so we parse it ourselves.
    headers = []
    path = SWIRE_CDFS_PATH if field == 'cdfs' else SWIRE_ELAIS_PATH
    for row_num, line in enumerate(open(path)):
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

    # Normalise and centre the features.
    swire_features = process(swire_features)

    # Write back to a file.
    log.info('Creating new file swire.h5')
    with h5py.File(os.path.join(WORKING_DIR, 'swire_{}.h5'.format(field)), 'w') as f:
        names = numpy.array(swire_names, dtype='<S{}'.format(
            max(len(i) for i in swire_names)))
        f.create_dataset('names', data=names)
        f.create_dataset('coords', data=swire_coords)
        f.create_dataset('features', data=swire_features)

    return swire_names, swire_coords, swire_features


def generate_swire_labels(
        swire_names: List[str],
        swire_coords: NDArray(N, 2)[float],
        overwrite: bool=False,
        field: str='cdfs') -> NDArray(N, 2)[bool]:
    """Generate Norris and RGZ SWIRE labels.

    Source: 102_classification.ipynb

    Returns
    -------
    N x 2 NDArray; first column Norris label, second column RGZ label.
    """
    if not overwrite:
        try:
            with h5py.File(os.path.join(WORKING_DIR, 'swire_labels_{}.h5'.format(field)),
                           'r') as f_h5:
                log.info('Reading labels from swire_labels_{}.h5'.format(field))
                return f_h5['labels'].value
        except OSError:
            pass

    table = astropy.io.ascii.read(TABLE_PATH)
    swire_name_to_index = {name: index
                           for index, name in enumerate(swire_names)}
    n_swire = len(swire_names)
    swire_labels = numpy.zeros((n_swire, 2), dtype=bool)
    if field == 'cdfs':
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
    else:
        with open(MIDDELBERG_TABLE5_PATH) as f_middelberg:
            swires = re.findall(r'SWIRE4J(\d\d)(\d\d)(\d\d\.\d\d)(-\d\d)(\d\d)(\d\d\.\d)', f_middelberg.read())
            swire_tree = KDTree(swire_coords)
            for swire in swires:
                coord = astropy.coordinates.SkyCoord(ra='{} {} {}'.format(*swire[:3]), dec='{} {} {}'.format(*swire[3:6]),
                                                     unit=('hourangle', 'deg'))
                distance, nearest = swire_tree.query([coord.ra.deg, coord.dec.deg])
                if distance > 5 / 60 / 60:  # 5" tolerance:
                    log.warning('No match within 5" for SWIRE4J{}{}{}{}{}{} in SWIRE3, nearest is {} deg'.format(*swire, distance))
                    continue
                swire = swire_names[nearest]
                swire_labels[swire_name_to_index[swire], 0] = True

    assert numpy.any(swire_labels)

    log.info('Creating new file swire_labels_{}.h5'.format(field))
    with h5py.File(os.path.join(WORKING_DIR, 'swire_labels_{}.h5'.format(field)), 'w') as f_h5:
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


def compactness(r: Dict[str, float]) -> float:
    """Check how compact a component is. Higher = less compact."""
    aa = 10.0
    bb = 1.3
    s = r['Component S (Franzen)'] if r['Component S (Franzen)'] != 0.0 else r['Component Sp (Franzen)']
    ssp = s / r['Component Sp (Franzen)']
    aasnrbb = aa / (r['Component SNR (Franzen)'] ** bb)
    return ssp / aasnrbb



def compact_test(r: Dict[str, float]) -> bool:
    """Check if a row represents a compact component."""
    S = r['Component S (Franzen)']
    Sp = r['Component Sp (Franzen)']
    S_ERR = r['Component S_ERR (Franzen)']
    Sp_ERR = r['Component Sp_ERR (Franzen)']
    if not S:
        S = Sp
        S_ERR = Sp_ERR

    R = numpy.log(S / Sp)
    R_err = numpy.sqrt(
        (S_ERR / S) ** 2 +
        (Sp_ERR / Sp) ** 2)
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
        swire_coords: NDArray(N, 2)[float],
        swire_labels: NDArray(N, 2)[bool],
        overwrite: bool=False,
        field: str='cdfs',
    ) -> (
        (NDArray(N, 6, 4)[bool],
         NDArray(N, 6, 4)[bool]),
        (NDArray(M, 6, 4)[bool],
         NDArray(M, 6, 4)[bool])):
    """Generate training/testing sets.

    Sets generated for CDFS:
    - RGZ & Norris & compact
    - RGZ & Norris & resolved
    - RGZ & Norris
    - RGZ & compact
    - RGZ & resolved
    - RGZ

    Only one set is generated for ELAIS-S1.

    Source: 104_one_notebook_to_train_them_all.ipynb

    Returns
    -------
    Two tuples of two boolean arrays.

    For the first tuple: Each indicates which set each ATLAS object is in. The
    first array is training, the second array is testing.
    For the second tuple: Each indicates which set each SWIRE object is in.
    The first array is training, the second array is testing.

    First index: ATLAS key/SWIRE index.
    Second index: Set from above list.
    Third index: Quadrant.
    """
    if not overwrite:
        try:
            with h5py.File(os.path.join(WORKING_DIR, 'swire_sets_{}.h5'.format(field)),
                           'r') as f_h5:
                log.info('Reading sets from swire_sets_{}.h5'.format(field))
                swire_train_sets = f_h5['train'].value
                swire_test_sets = f_h5['test'].value
            with h5py.File(os.path.join(WORKING_DIR, 'atlas_sets_{}.h5'.format(field)),
                           'r') as f_h5:
                log.info('Reading sets from atlas_sets_{}.h5'.format(field))
                atlas_train_sets = f_h5['train'].value
                atlas_test_sets = f_h5['test'].value
            return ((atlas_train_sets, atlas_test_sets),
                    (swire_train_sets, swire_test_sets))

        except OSError:
            pass

    if field == 'cdfs':
        table = astropy.io.ascii.read(TABLE_PATH)
        n_swire = len(swire_coords)
        # Generate the base ATLAS sets.
        rgz = {r['Key'] for r in table
               if r['Component Zooniverse ID (RGZ)'] and
               r['Component ID (Franzen)'] == r['Primary Component ID (RGZ)'] and
               r['Component ID (Franzen)']}
        norris = {r['Key'] for r in table if r['Component # (Norris)'] and
                                             r['Component ID (Franzen)'] and
                                             r['Source SWIRE (Norris)'] and
                                             r['Source SWIRE (Norris)'].startswith(
                                                'SWIRE')}
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
        n_atlas = max(table['Key']) + 1
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
        log.debug('Converting ATLAS set format')
        atlas_sets_test = numpy.zeros((n_atlas, 6, 4), dtype=bool)
        atlas_sets_train = numpy.zeros((n_atlas, 6, 4), dtype=bool)
        for s, (subset_str, subset_set) in enumerate(subsets):
            for q in range(4):
                for n in training_testing_atlas_sets[subset_str][q][0]:
                    atlas_sets_train[n, s, q] = True
                for n in training_testing_atlas_sets[subset_str][q][1]:
                    atlas_sets_test[n, s, q] = True

    elif field == 'elais':
        # Don't worry about making different sets for ELAIS, since we will only use it for testing.
        # We will choose only ATLAS/SWIRE objects within 1' of a positive Middelberg identification
        # that appears in SWIRE3. This will necessarily exclude some hard sources like IR-faint
        # objects, but this is by *far* the easiest way to filter the set.

        with astropy.io.fits.open(MIDDELBERG_TABLE4_PATH) as elais_components_fits:
            elais_components = elais_components_fits[1].data
            elais_coords = []
            n_elais = len(elais_components)
            n_swire = len(swire_coords)
            for component in elais_components:
                coord = astropy.coordinates.SkyCoord(
                    ra='{} {} {}'.format(component['RAh'], component['RAm'], component['RAs']),
                    dec='-{} {} {}'.format(component['DEd'], component['DEm'], component['DEs']),
                    unit=('hourangle', 'deg'))
                coord = (coord.ra.deg, coord.dec.deg)
                elais_coords.append(coord)
            elais_coords = numpy.array(elais_coords)
            positive_swire_coords = swire_coords[swire_labels[:, 0]]
            elais_tree = KDTree(elais_coords)
            nearby_elais = sorted(set(numpy.concatenate(elais_tree.query_ball_point(positive_swire_coords, 1 / 60))))
            atlas_sets_train = numpy.zeros((n_elais, 1, 1), dtype=bool)
            atlas_sets_test = numpy.zeros((n_elais, 1, 1), dtype=bool)
            atlas_sets_test[nearby_elais] = True
            # Convert the ATLAS set into a SWIRE set.
            swire_tree = KDTree(swire_coords)
            nearby_swire = sorted(set(numpy.concatenate(swire_tree.query_ball_point(elais_coords[atlas_sets_test[:, 0, 0]], 1 / 60))))
            swire_sets_train = numpy.zeros((n_swire, 1, 1), dtype=bool)
            swire_sets_test = numpy.zeros((n_swire, 1, 1), dtype=bool)
            swire_sets_test[nearby_swire] = True


    log.info('Creating new file swire_sets_{}.h5'.format(field))
    with h5py.File(os.path.join(WORKING_DIR, 'swire_sets_{}.h5'.format(field)), 'w') as f_h5:
        f_h5.create_dataset('train', data=swire_sets_train)
        f_h5.create_dataset('test', data=swire_sets_test)
    log.info('Creating new file atlas_sets_{}.h5'.format(field))
    with h5py.File(os.path.join(WORKING_DIR, 'atlas_sets_{}.h5'.format(field)), 'w') as f_h5:
        f_h5.create_dataset('train', data=atlas_sets_train)
        f_h5.create_dataset('test', data=atlas_sets_test)

    return ((atlas_sets_train, atlas_sets_test),
            (swire_sets_train, swire_sets_test))

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


class CNN:
    """Modified convolutional neural network"""

    def __init__(self, classifier: Classifier):
        self._cnn = classifier

    def _transform(self, features: NDArray(N, M)[float]) -> NDArray(N, M)[float]:
        as_features = features[:, :-IMAGE_SIZE]
        im_features = features[:, -IMAGE_SIZE:].reshape((-1, 1, 32, 32))
        assert im_features.shape[0] == as_features.shape[0]
        return [as_features, im_features]

    def fit(self, features: NDArray(N, M)[float], labels: NDArray(N)[bool]):
        pass

    def predict(self, features: NDArray(N, M)[float]) -> NDArray(N)[bool]:
        return self.predict_proba(features) > 0.5

    def predict_proba(self, features: NDArray(N, M)[float]) -> NDArray(N)[float]:
        features = self._transform(features)
        return self._cnn.predict(features).ravel()

    def get_params(self):
        return {}


class CNNLR(CNN):
    """Modified convolutional neural network with logistic regression on the end for fine-tuning."""

    def __init__(self, classifier: Classifier):
        super().__init__(classifier)

        # Take the CNN as a feature extractor.
        import keras.backend # :(
        self._extractor = keras.backend.function(
            [self._cnn.layers[0].input],  # Input 1
            [self._cnn.layers[4].output])  # Pool 2
        # 1 x 32 x 32 -> 32 x 5 x 5
        self._lr = LogisticRegression(
            class_weight='balanced',
            C=100000.0)

    def _transform(self, features: NDArray(N, M)[float]) -> NDArray(N, M)[float]:
        as_features, im_features = super()._transform(features)
        im_features = self._extractor([im_features])[0].reshape((-1, 32 * 5 * 5))
        features = numpy.concatenate([as_features, im_features], axis=1)
        return features

    def fit(self, features: NDArray(N, M)[float], labels: NDArray(N)[bool]):
        # Extract features.
        features = self._transform(features)
        self._lr.fit(features, labels)

    def predict(self, features: NDArray(N, M)[float]) -> NDArray(N)[bool]:
        features = self._transform(features)
        return self._lr.predict(features)

    def predict_proba(self, features: NDArray(N, M)[float]) -> NDArray(N)[float]:
        features = self._transform(features)
        return self._lr.predict_proba(features)


def process(features: NDArray(N, M)[float]) -> NDArray(N, M)[float]:
    """Normalise and centre non-image features."""
    features = features.copy()
    features[:, :-IMAGE_SIZE] -= features[:, :-IMAGE_SIZE].mean(axis=0)
    features[:, :-IMAGE_SIZE] /= features[:, :-IMAGE_SIZE].std(axis=0)
    return features


def train_classifier(
        Classifier: type,
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
        int in [0, 4). Held-out quadrant of CDFS.
    kwargs
        Keyword arguments for the classifier.

    Returns
    -------
    Classifier
        scikit-learn classifier.
    """
    train = swire_train_sets[:, SET_NAMES[dataset_name], quadrant]
    features = swire_features[train]
    # norris -> 0, rgz -> 1
    assert labeller in {'norris', 'rgz'}
    labels = swire_labels[train, int(labeller != 'norris')]

    if Classifier == CNN:
        import keras.models
        with open('/Users/alger/data/Crowdastro/model_03_06_17.json') as f:
            classifier = keras.models.model_from_json(f.read())
        weights_map = {
            ('norris', 'RGZ & Norris & compact'): 'weights_{}_norris_compact.h5'.format(quadrant),
            ('norris', 'RGZ & Norris & resolved'): 'weights_{}_norris_resolved.h5'.format(quadrant),
            ('norris', 'RGZ & Norris'): 'weights_{}_norris.h5'.format(quadrant),
            ('rgz', 'RGZ & Norris & compact'): 'weights_{}_rgz_compact.h5'.format(quadrant),
            ('rgz', 'RGZ & Norris & resolved'): 'weights_{}_rgz_resolved.h5'.format(quadrant),
            ('rgz', 'RGZ & Norris'): 'weights_{}_rgz.h5'.format(quadrant),
            ('rgz', 'RGZ & compact'): 'weights_{}_rgz_full_compact.h5'.format(quadrant),
            ('rgz', 'RGZ & resolved'): 'weights_{}_rgz_full_resolved.h5'.format(quadrant),
            ('rgz', 'RGZ'): 'weights_{}_rgz_full.h5'.format(quadrant),
        }
        if (labeller, dataset_name) in weights_map:
            weights_file = weights_map[labeller, dataset_name]
            path = '/Users/alger/data/Crowdastro/weights_03_06_17/{}'.format(weights_file)
            log.debug('Loading weights {}'.format(path))
            classifier.load_weights(path)
        else:
            log.warning('Missing weights for {}, {}'.format(labeller, dataset_name))
        classifier = Classifier(classifier)
    else:
        classifier = Classifier(class_weight='balanced', **kwargs)

    classifier.fit(features, labels)

    # A quick accuracy check.
    predictions = classifier.predict(features)
    log.debug('Balanced accuracy on {}/{}/{}/{} training set: {:.02%}'.format(
        classifier, labeller, dataset_name, quadrant,
        balanced_accuracy(labels, predictions)))

    return classifier


def balanced_accuracy(
        y_true: NDArray(N)[bool],
        y_pred: NDArray(N)[bool]) -> float:
    """Computes the balanced accuracy of a predictor.

    Source: crowdastro.crowd.util.balanced_accuracy

    Parameters
    ----------
    y_true
        Array of true labels.
    y_pred
        (Masked) array of predicted labels.

    Returns
    -------
    float
        balanced accuracy or None if the balanced accuracy isn't defined.
    """
    if hasattr(y_pred, 'mask') and not isinstance(y_pred.mask, bool):
        cm = sklearn.metrics.confusion_matrix(
                y_true[~y_pred.mask], y_pred[~y_pred.mask]).astype(float)
    else:
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred).astype(float)

    tp = cm[1, 1]
    n, p = cm.sum(axis=1)
    tn = cm[0, 0]
    if not n or not p:
        return None

    ba = (tp / p + tn / n) / 2
    return ba


@attr.s
class Predictions:
    """Represent classifier outputs."""
    probabilities = attr.ib()  # type: NDArray(N)[float]
    labels = attr.ib()  # type: NDArray(N)[bool]
    balanced_accuracy = attr.ib()  # type: float
    dataset_name = attr.ib()  # type: str
    quadrant = attr.ib()  # type: int
    params = attr.ib()  # type: Dict[str, Any]
    labeller = attr.ib()  # type: str
    classifier = attr.ib()  # type: str

    def to_hdf5(self: 'Predictions', path: str) -> None:
        """Serialise predictions as HDF5."""
        with h5py.File(path, 'w') as f_h5:
            f_h5.create_dataset('probabilities', data=self.probabilities)
            f_h5.create_dataset('labels', data=self.labels)
            f_h5.attrs['balanced_accuracy'] = self.balanced_accuracy
            f_h5.attrs['dataset_name'] = self.dataset_name
            f_h5.attrs['quadrant'] = self.quadrant
            f_h5.attrs['labeller'] = self.labeller
            f_h5.attrs['classifier'] = self.classifier
            for param, value in self.params.items():
                if value is None:
                    value = '__builtins__.None'
                f_h5.attrs['param_{}'.format(param)] = value

    @classmethod
    def from_hdf5(cls: type, path: str) -> 'Predictions':
        with h5py.File(path, 'r') as f_h5:
            probabilities = f_h5['probabilities'].value
            labels = f_h5['labels'].value
            balanced_accuracy = f_h5.attrs['balanced_accuracy']
            dataset_name = f_h5.attrs['dataset_name']
            quadrant = f_h5.attrs['quadrant']
            labeller = f_h5.attrs['labeller']
            classifier = f_h5.attrs['classifier']
            params = {}
            for attr in f_h5.attrs:
                if attr.startswith('param_'):
                    param = attr[6:]
                    value = f_h5.attrs[attr]
                    if value == '__builtins__.None':
                        value = None
                    params[param] = value
        return Predictions(probabilities=probabilities,
                           labels=labels,
                           balanced_accuracy=balanced_accuracy,
                           dataset_name=dataset_name,
                           quadrant=quadrant,
                           params=params,
                           labeller=labeller,
                           classifier=classifier)


@attr.s
class CrossIdentifications:
    """Represent cross-identifications."""
    radio_names = attr.ib()  # type: List[str]
    ir_names = attr.ib()  # type: List[str]
    accuracy = attr.ib()  # type: float
    dataset_name = attr.ib()  # type: str
    quadrant = attr.ib()  # type: int
    labeller = attr.ib()  # type: str
    classifier = attr.ib()  # type: str
    params = attr.ib()  # type: Dict[str, Any]

    def to_hdf5(self: 'CrossIdentifications', path: str) -> None:
        """Serialise predictions as HDF5."""
        with h5py.File(path, 'w') as f_h5:
            radio_dtype = '<S{}'.format(max(len(i) for i in self.radio_names) + 10)
            ir_dtype = '<S{}'.format(max(len(i) for i in self.ir_names) + 10)
            f_h5.create_dataset('radio_names', data=numpy.array(self.radio_names, dtype=radio_dtype))
            f_h5.create_dataset('ir_names', data=numpy.array(self.ir_names, dtype=ir_dtype))
            f_h5.attrs['accuracy'] = self.accuracy
            f_h5.attrs['dataset_name'] = self.dataset_name
            f_h5.attrs['quadrant'] = self.quadrant
            f_h5.attrs['labeller'] = self.labeller
            f_h5.attrs['classifier'] = self.classifier
            for param, value in self.params.items():
                if value is None:
                    value = '__builtins__.None'
                f_h5.attrs['param_{}'.format(param)] = value

    @classmethod
    def from_hdf5(cls: type, path: str) -> 'CrossIdentifications':
        with h5py.File(path, 'r') as f_h5:
            radio_names = [i.decode('ascii')
                             for i in f_h5['radio_names'].value]
            ir_names = [i.decode('ascii')
                             for i in f_h5['ir_names'].value]
            accuracy = f_h5.attrs['accuracy']
            dataset_name = f_h5.attrs['dataset_name']
            quadrant = f_h5.attrs['quadrant']
            labeller = f_h5.attrs['labeller']
            classifier = f_h5.attrs['classifier']
            params = {}
            for attr in f_h5.attrs:
                if attr.startswith('param_'):
                    param = attr[6:]
                    value = f_h5.attrs[attr]
                    if value == '__builtins__.None':
                        value = None
                    params[param] = value
        return cls(radio_names=radio_names,
                   ir_names=ir_names,
                   accuracy=accuracy,
                   dataset_name=dataset_name,
                   quadrant=quadrant,
                   params=params,
                   labeller=labeller,
                   classifier=classifier)


def predict(
        classifier: Classifier,
        swire_features: NDArray(N, D)[float],
        swire_labels: NDArray(N, 2)[bool],
        swire_test_sets: NDArray(N, 6, 4)[bool],
        dataset_name: str,
        quadrant: int,
        labeller: str,
        field: str='cdfs') -> Predictions:
    """Predict labels using a classifier.

    Note that predictions will be made for all SWIRE objects in RGZ,
    regardless of training dataset.

    Note that testing will be performed against Norris or Middelberg labels.

    Parameters
    ----------
    classifier
        scikit-learn classifier.
    swire_features
        SWIRE object features.
    swire_labels
        Norris, RGZ labels for each SWIRE object.
    swire_test_sets
        Output of generate_data_sets.
    dataset_name
        'RGZ & Norris & compact' or
        'RGZ & Norris & resolved' or
        'RGZ & Norris' or
        'RGZ & compact' or
        'RGZ & resolved' or
        'RGZ'.
    quadrant
        int in [0, 4). Quadrant of CDFS to test on.
    labeller
        Labeller that the classifier was trained against. 'norris' or 'rgz'.
    field
        'cdfs' or 'elais'.

    Returns
    -------
    Predictions
        Predictions of the classifier on the specified data.
    """
    if field == 'cdfs':
        test = swire_test_sets[:, SET_NAMES['RGZ'], quadrant]
    else:
        test = swire_test_sets[:, 0, 0]
    features = swire_features[test]
    assert labeller in {'norris', 'rgz'}
    labels = swire_labels[test, 0]  # Test on Norris/Middelberg.
    predicted_labels = classifier.predict(features)
    predicted_probabilities = classifier.predict_proba(features)
    if len(predicted_probabilities.shape) > 1 and \
            len(predicted_probabilities.shape) == 2:
        # Probability of the positive class.
        predicted_probabilities = predicted_probabilities[:, 1]
    assert predicted_labels.shape == (test.sum(),)
    assert predicted_probabilities.shape == (test.sum(),)
    ba = balanced_accuracy(labels, predicted_labels)
    return Predictions(
        probabilities=predicted_probabilities,
        labels=predicted_labels,
        balanced_accuracy=ba,
        dataset_name=dataset_name,
        quadrant=quadrant,
        params=classifier.get_params(),
        labeller=labeller,
        classifier=Classifier.__name__)


def train_all_quadrants(
        Classifier: type,
        swire_features: NDArray(N, D)[float],
        swire_labels: NDArray(N, 2)[bool],
        swire_train_sets: NDArray(N, 6, 4)[bool],
        labeller: str,
        dataset_name: str,
        **kwargs: Dict[str, Any]) -> List[Classifier]:
    """Train a classifier using the scikit-learn API across all quadrants.

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
    kwargs
        Keyword arguments for the classifier.

    Returns
    -------
    List[Classifier]
        List of scikit-learn classifiers (one for each quadrant).
    """
    return [train_classifier(
                Classifier,
                swire_features,
                swire_labels,
                swire_train_sets,
                labeller,
                dataset_name,
                q, **kwargs) for q in range(4)]


def predict_all_quadrants(
        classifiers: List[Classifier],
        swire_features: NDArray(N, D)[float],
        swire_labels: NDArray(N, 2)[bool],
        swire_test_sets: NDArray(N, 6, 4)[bool],
        dataset_name: str,
        labeller: str) -> List[Predictions]:
    """Predict labels using classifiers across all quadrants.

    Note that predictions will be made for all SWIRE objects in RGZ,
    regardless of training dataset.

    Parameters
    ----------
    classifiers
        List of scikit-learn classifiers (one per quadrant).
    swire_features
        SWIRE object features.
    swire_labels
        Norris, RGZ labels for each SWIRE object.
    swire_test_sets
        Output of generate_data_sets.
    dataset_name
        'RGZ & Norris & compact' or
        'RGZ & Norris & resolved' or
        'RGZ & Norris' or
        'RGZ & compact' or
        'RGZ & resolved' or
        'RGZ'.
    labeller
        Labeller the classifier was trained against. 'norris' or 'rgz'.

    Returns
    -------
    List[Predictions]
        List of predictions of the classifier on the specified data.
    """
    return [predict(
                classifier,
                swire_features,
                swire_labels,
                swire_test_sets,
                dataset_name,
                q,
                labeller=labeller) for q, classifier in enumerate(classifiers)]


def train_all(
        Classifier: type,
        swire_features: NDArray(N, D)[float],
        swire_labels: NDArray(N, 2)[bool],
        swire_train_sets: NDArray(N, 6, 4)[bool],
        labeller: str,
        **kwargs: Dict[str, Any]) -> List[Classifier]:
    """Train a classifier across all quadrants and datasets.

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
    kwargs
        Keyword arguments for the classifier.

    Returns
    -------
    Dict[str, List[Classifier]]
        dict mapping dataset name to a list of scikit-learn classifiers
        (one for each quadrant).
    """
    return {dataset_name: [train_classifier(
                Classifier,
                swire_features,
                swire_labels,
                swire_train_sets,
                labeller,
                dataset_name,
                q, **kwargs)
            for q in range(4)] for dataset_name in sorted(SET_NAMES)}


def predict_all(
        classifiers: Dict[str, List[Classifier]],
        swire_features: NDArray(N, D)[float],
        swire_labels: NDArray(N, 2)[bool],
        swire_test_sets: NDArray(N, 6, 4)[bool],
        labeller: str,
        field: str='cdfs') -> List[Predictions]:
    """Predict labels using classifiers across all quadrants.

    For CDFS:
        Note that predictions will be made for all SWIRE objects in RGZ,
        regardless of training dataset.
    For ELAIS-S1:
        Predictions will be made for all objects in ELAIS-S1, using all
        classifiers from CDFS.

    Parameters
    ----------
    classifiers
        dict mapping dataset names to a list of classifiers (one per quadrant).
    swire_features
        SWIRE object features.
    swire_labels
        Norris, RGZ labels for each SWIRE object.
    swire_test_sets
        Output of generate_data_sets.
    labeller
        Labeller the classifier was trained against. 'norris' or 'rgz'.
    field
        'cdfs' or 'elais'.

    Returns
    -------
    Dict[str, List[Predictions]]
        dict mapping dataset names to lists of predictions of the classifier on
        each quadrant.
    """
    return {dataset_name: [predict(
                classifier,
                swire_features,
                swire_labels,
                swire_test_sets,
                dataset_name,
                q,
                labeller=labeller,
                field=field) for q, classifier in enumerate(classifiers)]
            for dataset_name, classifiers in classifiers.items()}


def serialise_predictions(
        predictions: List[Predictions], base_path: str) -> None:
    """Serialise a list of predictions to files.

    base_path will be prepended to the filename, e.g. base_path = /tmp/run
    will give files such as /tmp/run_0_RGZ & Norris.h5.
    """
    for prediction in predictions:
        filename = '{}_{}_{}.h5'.format(base_path, prediction.quadrant,
                                        prediction.dataset_name)
        log.debug('Writing predictions to {}'.format(filename))
        prediction.to_hdf5(filename)


def serialise_cross_identifications(
        cross_identifications: List[CrossIdentifications],
        base_path: str) -> None:
    """Serialise a list of cross-identifications to files.

    base_path will be prepended to the filename, e.g. base_path = /tmp/run
    will give files such as /tmp/run_0_RGZ & Norris.h5.
    """
    for cid in cross_identifications:
        filename = '{}_{}_{}.h5'.format(base_path, cid.quadrant,
                                        cid.dataset_name)
        log.debug('Writing cross-identifications to {}'.format(filename))
        cid.to_hdf5(filename)


def unserialise_cross_identifications(
        base_path: str,
        quadrants: List[int]=None,
        dataset_names: List[str]=None) -> Iterable[Predictions]:
    """Unserialise a list of predictions."""
    if quadrants is None:
        quadrants = [0, 1, 2, 3]
    if dataset_names is None:
        dataset_names = [n for n in sorted(SET_NAMES)]
    for quadrant in quadrants:
        for dataset_name in dataset_names:
            filename = '{}_{}_{}.h5'.format(base_path, quadrant, dataset_name)
            log.debug('Reading cross-identifications from {}'.format(filename))
            yield CrossIdentifications.from_hdf5(filename)


def unserialise_predictions(
        base_path: str,
        quadrants: List[int]=None,
        dataset_names: List[str]=None) -> Iterable[Predictions]:
    """Unserialise a list of predictions."""
    if quadrants is None:
        quadrants = [0, 1, 2, 3]
    if dataset_names is None:
        dataset_names = [n for n in sorted(SET_NAMES)]
    for quadrant in quadrants:
        for dataset_name in dataset_names:
            filename = '{}_{}_{}.h5'.format(base_path, quadrant, dataset_name)
            log.debug('Reading predictions from {}'.format(filename))
            yield Predictions.from_hdf5(filename)


def train_and_predict(
        Classifier: type,
        swire_features: NDArray(N, D)[float],
        swire_labels: NDArray(N, 2)[bool],
        swire_train_sets: NDArray(N, 6, 4)[bool],
        swire_test_sets: NDArray(N, 6, 4)[bool],
        labeller: str,
        overwrite: bool=False,
        **kwargs: Dict[str, Any]) -> List[Predictions]:
    if not overwrite:
        try:
            return list(
                unserialise_predictions(WORKING_DIR + Classifier.__name__ +
                                        '_' + labeller + '_' + 'cdfs' +
                                        '_predictions'))
        except OSError:
            pass

    log.debug('Training all.')
    classifiers = train_all(
        Classifier,
        swire_features,
        swire_labels,
        swire_train_sets,
        labeller,
        **kwargs)
    predictions = predict_all(
        classifiers,
        swire_features,
        swire_labels,
        swire_test_sets,
        labeller,
        field='cdfs')
    predictions = [i
                   for quadrant_preds in predictions.values()
                   for i in quadrant_preds]
    serialise_predictions(predictions, WORKING_DIR + Classifier.__name__ +
                                       '_' + labeller + '_' + 'cdfs' +
                                       '_predictions')
    return predictions


def train_and_predict_elais(
        Classifier: type,
        swire_features_cdfs: NDArray(N, D)[float],
        swire_labels_cdfs: NDArray(N, 2)[bool],
        swire_train_sets_cdfs: NDArray(N, 6, 4)[bool],
        swire_test_sets_cdfs: NDArray(N, 6, 4)[bool],
        labeller: str,
        swire_features_elais: NDArray(N, D)[float],
        swire_labels_elais: NDArray(N, 2)[bool],
        swire_sets_elais: NDArray(N, 6, 4)[bool],
        overwrite: bool=False,
        **kwargs: Dict[str, Any]) -> List[Predictions]:
    if not overwrite:
        try:
            return list(
                unserialise_predictions(WORKING_DIR + Classifier.__name__ +
                                        '_' + labeller + '_' + 'elais' +
                                        '_predictions'))
        except OSError:
            pass

    log.debug('Training all.')
    classifiers = train_all(
        Classifier,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        labeller,
        **kwargs)
    log.debug('Predicting on ELAIS-S1.')
    predictions = predict_all(
        classifiers,
        swire_features_elais,
        swire_labels_elais,
        swire_sets_elais,
        labeller,
        field='elais')
    predictions = [i
                   for quadrant_preds in predictions.values()
                   for i in quadrant_preds]
    serialise_predictions(predictions, WORKING_DIR + Classifier.__name__ +
                                       '_' + labeller + '_' + 'elais' +
                                       '_predictions')
    return predictions


def cross_identify_all(
        swire_names: List[str],
        swire_coords: NDArray(N, 2)[float],
        swire_labels: NDArray(N, 2)[bool],
        swire_sets: NDArray(N, 6, 4)[bool],
        norris_labels: NDArray(N)[bool],
        field: str='cdfs') -> Iterable[CrossIdentifications]:
    """Cross-identify with all predictors.

    ELAIS is cross-identified using CDFS predictors.
    """
    for classifier in ['RandomForestClassifier', 'LogisticRegression', 'CNN']:
        for labeller in ['norris', 'rgz']:
            # Cross-identification of all ATLAS sources begins here. We
            # ideally get non-overlapping quadrants, so each
            # classifier+labeller pair is fully described by four predictors,
            # one per quadrant.
            path = WORKING_DIR + classifier + '_' + labeller + '_' + field
            try:
                all_cids = list(unserialise_cross_identifications(
                    path + '_cross_ids'))
                for cid in all_cids:
                    cid.classifier = classifier
                    cid.labeller = labeller
                log.debug('Loaded {}/{} cross-identifications'.format(classifier, labeller))
            except OSError:
                log.debug('Generating {}/{} cross-identifications'.format(classifier, labeller))
                predictions = unserialise_predictions(path + '_predictions')
                all_cids = []
                for pred in predictions:
                    log.debug('Cross-identifying quadrant {} with {}'.format(
                        pred.quadrant, pred.dataset_name))
                    cids = cross_identify(swire_names, swire_coords, swire_labels, pred, field=field)
                    cids.classifier = classifier  # scikit-learn __name__ :(
                    assert cids.labeller == labeller
                    all_cids.append(cids)
                serialise_cross_identifications(all_cids, path + '_cross_ids')
            yield from all_cids

    # "Perfect" classifier (reads groundtruth).
    try:
        if field == 'cdfs':
            all_cids = list(unserialise_cross_identifications(
                WORKING_DIR + 'groundtruth_norris_cross_ids',
                dataset_names=['RGZ & Norris']))
        else:
            all_cids = list(unserialise_cross_identifications(
                WORKING_DIR + 'groundtruth_middelberg_cross_ids',
                dataset_names=['RGZ & Norris']))
        log.debug('Loaded groundtruth cross-identifications')
    except OSError:
        log.debug('Generating groundtruth cross-identifications')
        all_cids = []
        for q in range(4):
            if field == 'cdfs':
                probabilities = norris_labels[swire_sets[:, SET_NAMES['RGZ'], q]].astype(float)
                # # Noise the probabilities *ever* so slightly to fix non-determinism.
                # # This isn't needed with the Gaussian multiplier.
                # probabilities += numpy.random.normal(
                #     size=probabilities.shape,
                #     scale=0.1) ** 2
                labeller = 'norris'
            else:
                probabilities = swire_labels[swire_sets[:, 0, 0], 0].astype(float)
                # # Noise the probabilities *ever* so slightly to fix non-determinism.
                # # This isn't needed with the Gaussian multiplier.
                # probabilities += numpy.random.normal(
                #     size=probabilities.shape,
                #     scale=0.1) ** 2
                labeller = 'middelberg'
            labels = probabilities > 0.5
            predictions = Predictions(
                probabilities=probabilities,
                labels=labels,
                balanced_accuracy=1.0,
                dataset_name='RGZ & Norris',
                quadrant=q,
                params={},
                labeller=labeller,
                classifier='Groundtruth')
            cids = cross_identify(swire_names, swire_coords, swire_labels, predictions, field=field)
            all_cids.append(cids)
        if field == 'cdfs':
            serialise_cross_identifications(all_cids, WORKING_DIR + 'groundtruth_norris_cross_ids')
        else:
            serialise_cross_identifications(all_cids, WORKING_DIR + 'groundtruth_middelberg_cross_ids')
    yield from all_cids

    # Random classifier.
    numpy.random.seed(0)
    for trial in range(25):
        try:
            all_cids = list(unserialise_cross_identifications(
                WORKING_DIR + 'random_{}_{}_cross_ids'.format(trial, field),
                dataset_names=['RGZ & Norris']))
            log.debug('Loaded random cross-identifications {}'.format(trial))
        except OSError:
            log.debug('Generating random cross-identifications {}'.format(trial))
            all_cids = []
            for q in range(4):
                if field == 'cdfs':
                    pshape = norris_labels[swire_sets[:, SET_NAMES['RGZ'], q]].shape
                else:
                    pshape = swire_labels[swire_sets[:, 0, 0], 0].shape
                random_probabilities = numpy.random.uniform(size=pshape)
                predictions = Predictions(
                    probabilities=random_probabilities,
                    labels=numpy.zeros(pshape, dtype=bool),
                    balanced_accuracy=1.0,
                    dataset_name='RGZ & Norris',
                    quadrant=q,
                    params={},
                    labeller='norris',  # Just for consistency.
                    classifier='Random')
                cids = cross_identify(swire_names, swire_coords, swire_labels, predictions, field=field)
                all_cids.append(cids)
            serialise_cross_identifications(all_cids, WORKING_DIR + 'random_{}_{}_cross_ids'.format(trial, field))
        yield from all_cids


def cross_identify(
        swire_names: List[str],
        swire_coords: NDArray(N, 2)[float],
        swire_labels: NDArray(N, 2)[bool],
        predictions: Predictions,
        radius: float=1 / 60,
        compact_split: bool=True,
        field: str='cdfs') -> CrossIdentifications:
    """Cross-identify radio objects in a quadrant."""
    (_, atlas_test_sets), (_, swire_test_sets) = generate_data_sets(
        swire_coords, swire_labels, overwrite=False, field=field)
    if field == 'cdfs':
        # Read ATLAS info from the big table.
        # Outputs: atlas_names, atlas_coords, atlas_name_to_compact
        atlas_names = []
        table = astropy.io.ascii.read(TABLE_PATH)
        atlas_coords = []
        norris_truth = []
        atlas_name_to_compact = {}
        for i, row in enumerate(table):
            assert i == row['Key']
            name = row['Component Name (Franzen)']
            atlas_names.append(name)
            atlas_coords.append((row['Component RA (Franzen)'], row['Component DEC (Franzen)']))
            norris_truth.append((row['Source SWIRE (Norris)']))
            if name:
                atlas_name_to_compact[name] = compact_test(row)
        atlas_coords = numpy.array(atlas_coords)
    else:
        # Read ATLAS component info from Middelberg table.
        with astropy.io.fits.open(MIDDELBERG_TABLE4_PATH) as elais_components_fits:
            elais_components = elais_components_fits[1].data
            atlas_coords = []
            atlas_names = []
            atlas_name_to_compact = {}
            atlas_cid_to_name = {}
            for component in elais_components:
                coord = astropy.coordinates.SkyCoord(
                    ra='{} {} {}'.format(component['RAh'], component['RAm'], component['RAs']),
                    dec='-{} {} {}'.format(component['DEd'], component['DEm'], component['DEs']),
                    unit=('hourangle', 'deg'))
                coord = (coord.ra.deg, coord.dec.deg)
                cid = component['CID']
                name = component['ATELAIS']
                atlas_coords.append(coord)
                atlas_names.append(name)
                atlas_cid_to_name[cid] = name
                row = {'Component S (Franzen)': component['Sint'],  # Fitting in with the CDFS API...
                       'Component S_ERR (Franzen)': component['e_Sint'],
                       'Component Sp (Franzen)': component['Sp'],
                       'Component Sp_ERR (Franzen)': component['e_Sp']}
                atlas_name_to_compact[name] = compact_test(row)
            atlas_coords = numpy.array(atlas_coords)

    quadrant = predictions.quadrant
    dataset_name = predictions.dataset_name
    labeller = predictions.labeller
    classifier = predictions.classifier
    params = predictions.params
    if field == 'cdfs':
        atlas_set = atlas_test_sets[:, SET_NAMES['RGZ'], quadrant].nonzero()[0]  # Cross-identify against *everything*.
        swire_set = swire_test_sets[:, SET_NAMES['RGZ'], quadrant].nonzero()[0]  # Note that this is independent from dataset_name.
    else:
        atlas_set = atlas_test_sets[:, 0, 0].nonzero()[0]
        swire_set = swire_test_sets[:, 0, 0].nonzero()[0]

    swire_tree = KDTree(swire_coords[swire_set])

    radio_names_ = []
    ir_names_ = []

    assert len(predictions.probabilities) == len(swire_set)
    radio_to_ir = {}

    no_matches = set()

    for atlas_i in atlas_set:
        coords = atlas_coords[atlas_i]
        radio_name = atlas_names[atlas_i]
        if compact_split and atlas_name_to_compact[radio_name] and 'compact' not in dataset_name:
            # Compact, so find nearest neighbour.
            # Note that we skip compact pipeline for cross-identifying the compact set.
            distance, nearest = swire_tree.query(coords)
            if distance > 5 / 60 / 60:  # 5 arcsec
                log.debug('No SWIRE host found for compact {}'.format(radio_name))
                no_matches.add(radio_name)
                continue
            ir_name = swire_names[swire_set[nearest]]
        else:
            # Resolved - ML pipeline.
            nearby = swire_tree.query_ball_point(coords, radius)  # indices of swire_set
            nearby_predictions = predictions.probabilities[nearby]  # probabilities matches swire_set indices
            if len(nearby_predictions) == 0:
                log.warning('No nearby SWIRE found for {}'.format(radio_name))
                no_matches.add(radio_name)
                continue
            # Multiply predictions by a Gaussian of location distances.
            scoords = astropy.coordinates.SkyCoord(ra=coords[0], dec=coords[1], unit='deg')
            nearby_scoords = astropy.coordinates.SkyCoord(ra=swire_coords[swire_set][nearby, 0], dec=swire_coords[swire_set][nearby, 1], unit='deg')
            separations = numpy.array(scoords.separation(nearby_scoords).deg)
            gaussians = scipy.stats.norm.pdf(separations, scale=FALLOFF_SIGMA)
            assert gaussians.shape == nearby_predictions.shape
            nearby_predictions *= gaussians
            argmax = numpy.argmax(nearby_predictions)  # index of nearby_predictions
            ir_name = swire_names[swire_set[nearby[argmax]]]
        radio_names_.append(radio_name)
        ir_names_.append(ir_name)
        radio_to_ir[radio_name] = ir_name

    # Compute accuracy.
    n_correct = 0
    n_total = 0
    if field == 'cdfs':
        for row in table:
            if not (row['Key'] in atlas_set and
                    row['Source SWIRE (Norris)'] and
                    row['Source SWIRE (Norris)'].startswith('SWIRE')):
                continue

            assert row['Component Name (Franzen)'].startswith('ATLAS')
            if row['Component Name (Franzen)'] in no_matches:
                continue

            n_correct += row['Source SWIRE (Norris)'] == radio_to_ir[row['Component Name (Franzen)']]
            n_total += 1
    else:
        # Read correct cross-identifications from Middelberg table.
        middelberg_cross_ids = {}
        swire_names = numpy.array(swire_names)
        with open(MIDDELBERG_TABLE5_PATH) as elais_file:
            # The FITS version of this table is corrupt, so we need to parse this ourselves.
            # We'll opt for the easy approach and use regex.
            # Here's a sample line:
            # S336   ATELAISJ003836.72-440617.8|C0336, C0336.1, C0336.2                         |00 38 36.727 -44 06 17.82SWIRE4J003836.72-440617.8|
            # We can ignore the source ID and name, but we will have to pull out the CID list and the SWIRE name.
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
                coord = (coord.ra.deg, coord.dec.deg)
                # Nearest SWIRE...
                dist, nearest = swire_tree.query(coord)
                if dist > 5 / 60 / 60:
                    log.warning('No SWIRE match found for Middelberg cross-identification {}'.format(line[0]))
                    continue
                name = swire_names[swire_set][nearest]
                for cid in line_cids:
                    if atlas_cid_to_name[cid] in radio_to_ir:
                        n_correct += radio_to_ir[atlas_cid_to_name[cid]] == name
                        n_total += 1

    accuracy = n_correct / n_total
    log.debug('Accuracy: {:.02%}'.format(accuracy))

    return CrossIdentifications(
        radio_names=radio_names_,
        ir_names=ir_names_,
        quadrant=quadrant,
        dataset_name=dataset_name,
        labeller=labeller,
        classifier=classifier,
        params=params,
        accuracy=accuracy)


@click.command()
@click.option('-f', '--overwrite-predictions', is_flag=True,
              help='Overwrite output predictions')
@click.option('-F', '--overwrite-all', is_flag=True,
              help='Overwrite all predictions and features')
@click.option('--jobs', type=int, default=-1, help='Number of parallel jobs')
def main(overwrite_predictions: bool=False,
         overwrite_all: bool=False,
         jobs: int=-1):
    # Generate SWIRE info.
    swire_names_cdfs, swire_coords_cdfs, swire_features_cdfs = generate_swire_features(overwrite=overwrite_all, field='cdfs')
    swire_names_elais, swire_coords_elais, swire_features_elais = generate_swire_features(overwrite=overwrite_all, field='elais')
    swire_labels_cdfs = generate_swire_labels(swire_names_cdfs, swire_coords_cdfs, overwrite=overwrite_all, field='cdfs')
    swire_labels_elais = generate_swire_labels(swire_names_elais, swire_coords_elais, overwrite=overwrite_all, field='elais')
    _, (swire_train_sets_cdfs, swire_test_sets_cdfs) = generate_data_sets(swire_coords_cdfs, swire_labels_cdfs, overwrite=overwrite_all, field='cdfs')
    _, (swire_train_sets_elais, swire_test_sets_elais) = generate_data_sets(swire_coords_elais, swire_labels_elais, overwrite=overwrite_all, field='elais')
    # Predict for LR, RF.
    lr_norris_pred_cdfs = train_and_predict(
        LogisticRegression,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'norris',
        overwrite=overwrite_predictions or overwrite_all,
        n_jobs=jobs,
        C=100000.0)
    lr_norris_pred_elais = train_and_predict_elais(
        LogisticRegression,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'norris',
        swire_features_elais,
        swire_labels_elais,
        swire_test_sets_elais,
        overwrite=overwrite_predictions or overwrite_all,
        n_jobs=jobs,
        C=100000.0)
    rf_norris_pred = train_and_predict(
        RandomForestClassifier,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'norris',
        overwrite=overwrite_predictions or overwrite_all,
        n_jobs=jobs,
        min_samples_leaf=45,
        criterion='entropy')
    rf_norris_pred_elais = train_and_predict_elais(
        RandomForestClassifier,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'norris',
        swire_features_elais,
        swire_labels_elais,
        swire_test_sets_elais,
        overwrite=overwrite_predictions or overwrite_all,
        n_jobs=jobs,
        min_samples_leaf=45,
        criterion='entropy')
    cnn_norris_pred = train_and_predict(
        CNN,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'norris',
        overwrite=overwrite_predictions or overwrite_all)
    cnn_norris_pred_elais = train_and_predict_elais(
        CNN,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'norris',
        swire_features_elais,
        swire_labels_elais,
        swire_test_sets_elais,
        overwrite=overwrite_predictions or overwrite_all)
    lr_rgz_pred = train_and_predict(
        LogisticRegression,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'rgz',
        overwrite=overwrite_predictions or overwrite_all,
        n_jobs=jobs,
        C=100000.0)
    rf_rgz_pred = train_and_predict(
        RandomForestClassifier,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'rgz',
        overwrite=overwrite_predictions or overwrite_all,
        n_jobs=jobs,
        min_samples_leaf=45,
        criterion='entropy')
    cnn_rgz_pred = train_and_predict(
        CNN,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'rgz',
        overwrite=overwrite_predictions or overwrite_all)
    lr_rgz_pred_elais = train_and_predict_elais(
        LogisticRegression,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'rgz',
        swire_features_elais,
        swire_labels_elais,
        swire_test_sets_elais,
        overwrite=overwrite_predictions or overwrite_all,
        n_jobs=jobs,
        C=100000.0)
    rf_rgz_pred_elais = train_and_predict_elais(
        RandomForestClassifier,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'rgz',
        swire_features_elais,
        swire_labels_elais,
        swire_test_sets_elais,
        overwrite=overwrite_predictions or overwrite_all,
        n_jobs=jobs,
        min_samples_leaf=45,
        criterion='entropy')
    cnn_rgz_pred_elais = train_and_predict_elais(
        CNN,
        swire_features_cdfs,
        swire_labels_cdfs,
        swire_train_sets_cdfs,
        swire_test_sets_cdfs,
        'rgz',
        swire_features_elais,
        swire_labels_elais,
        swire_test_sets_elais,
        overwrite=overwrite_predictions or overwrite_all)
    cids = list(cross_identify_all(swire_names_cdfs, swire_coords_cdfs, swire_labels_cdfs, swire_test_sets_cdfs, swire_labels_cdfs[:, 0], field='cdfs'))
    cids = list(cross_identify_all(swire_names_elais, swire_coords_elais, swire_labels_elais, swire_test_sets_elais, swire_labels_elais[:, 0], field='elais'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
