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

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import errno
import os
from typing import List, Sequence, Union

import h5py
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy
import seaborn

CROWDASTRO_PATH = '/Users/alger/data/Crowdastro/crowdastro-swire.h5'
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

    Returns
    -------
    (list of SWIRE names,
     N x 2 array of SWIRE RA/dec coordinates,
     N x D array of SWIRE features)
    """
    try:
        with h5py.File(os.path.join(WORKING_DIR, 'swire.h5'), 'r') as f:
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
    with h5py.File(os.path.join(WORKING_DIR, 'swire.h5'), 'w') as f:
        names = numpy.array(swire_names, dtype='<S{}'.format(
            max(len(i) for i in swire_names)))
        f.create_dataset('names', data=names)
        f.create_dataset('coords', data=swire_coords)
        f.create_dataset('features', data=swire_features)

    return swire_names, swire_coords, swire_features


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


def main():
    swire_names, swire_coords, swire_features = generate_swire_features()
    plot_distributions(swire_features)


if __name__ == '__main__':
    main()