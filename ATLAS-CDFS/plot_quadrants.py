#!/usr/bin/env python3
"""Plot quadrants of CDFS.

Input files:
- CDFSmosaic_allch_8March2015.fits - ATLAS CDFS mosaic

Output files:
- images/quadrants.pdf
- images/quadrants.png

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import astropy.io.fits
import astropy.visualization
import astropy.visualization.wcsaxes
import astropy.wcs
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker

import configure_plotting
configure_plotting.configure()

MOSAIC_PATH = '/Users/alger/data/ATLAS/CDFSmosaic_allch_8March2015.fits'

# Open mosaic.
atlas = astropy.io.fits.open(MOSAIC_PATH)[0]
wcs = astropy.wcs.WCS(atlas.header).dropaxis(3).dropaxis(2)

# Set up plotting.
fig = plt.figure(figsize=(8, 5))
ax = astropy.visualization.wcsaxes.WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs, facecolor='white')
fig.add_axes(ax)
stretch = astropy.visualization.ManualInterval(-100e-6, 300e-6)
ax.imshow(stretch(atlas.data[0, 0]), cmap='Greys')
ax.set_xlim((1500, 8500))
ax.set_ylim((1700, 7300))
middle, = wcs.all_world2pix([(52.8, -28.1)], 1)
ax.scatter([middle[0]], [middle[1]], s=100, c='black', marker='o')
ax.axvline(middle[0], color='black')
ax.axhline(middle[1], color='black')
plt.text((1500 + middle[0]) / 2,
         (7300 + middle[1]) / 2,
         '0', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
plt.text((8500 + middle[0]) / 2,
         (7300 + middle[1]) / 2,
         '1', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
plt.text((8500 + middle[0]) / 2,
         (1700 + middle[1]) / 2,
         '2', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
plt.text((1500 + middle[0]) / 2,
         (1700 + middle[1]) / 2,
         '3', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
lon, lat = ax.coords
lon.set_ticklabel(size=12)
lat.set_ticklabel(size=12)
lon.set_major_formatter('hh:mm')
lon.set_axislabel('Right Ascension', size=12)
lat.set_axislabel('Declination', size=12)
plt.savefig('images/quadrants.pdf')
plt.savefig('images/quadrants.png')
