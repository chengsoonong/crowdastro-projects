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

from typing import List, Sequence, Union

import h5py
import numpy

CROWDASTRO_PATH = '/Users/alger/data/Crowdastro/crowdastro-swire.h5'
SWIRE_PATH = '/Users/alger/data/SWIRE/SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl'
TABLE_PATH = '/Users/alger/data/Crowdastro/one-table-to-rule-them-all.tbl'
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
    with h5py.File(CROWDASTRO_PATH, 'r') as crowdastro_f:
        # Load coordinates of SWIRE objects.
        swire_coords = crowdastro_f['/swire/cdfs/numeric'][:, :2]
        # Initialise features array.
        swire_features = numpy.zeros((len(swire_coords),
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
        # Load names of SWIRE objects.
        swire_names = crowdastro_f['/swire/cdfs/string'].value
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
        features = numpy.concatenate([
            mag_diffs,
            mags[:1],
            stellarities,
            [swire_distances[crowdastro_index]],
            swire_images[crowdastro_index],
        ])
        swire_features[crowdastro_index] = features

    return swire_names, swire_coords, swire_features


def main():
    swire_names, swire_coords, swire_features = generate_swire_features()


if __name__ == '__main__':
    main()