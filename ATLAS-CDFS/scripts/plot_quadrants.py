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

import aplpy
import astropy.io.fits
import astropy.visualization
import astropy.visualization.wcsaxes
import astropy.wcs
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy

INCHES_PER_PT = 1.0 / 72.27
COLUMN_WIDTH_PT = 240.0
FONT_SIZE_PT = 8.0

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": FONT_SIZE_PT,
    "font.size": FONT_SIZE_PT,
    "legend.fontsize": FONT_SIZE_PT,
    "xtick.labelsize": FONT_SIZE_PT,
    "ytick.labelsize": FONT_SIZE_PT,
    "figure.figsize": (COLUMN_WIDTH_PT * INCHES_PER_PT, 0.8 * COLUMN_WIDTH_PT * INCHES_PER_PT),
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

MOSAIC_PATH = '/Users/alger/data/ATLAS/CDFSmosaic_allch_8March2015.fits'

# Open mosaic.


fig = plt.figure()
fig = aplpy.FITSFigure(MOSAIC_PATH, figure=fig)
fig.set_theme('publication')
with astropy.io.fits.open(MOSAIC_PATH) as f:
    fig.show_grayscale(vmin=-100e-6, vmax=300e-6)
fig.axis_labels.set_xtext('Right Ascension (J2000)')
fig.axis_labels.set_ytext('Declination (J2000)')
fig.set_tick_labels_format(xformat='hh:mm:ss',yformat='dd:mm:ss')
plt.subplots_adjust(top=1, right=0.95, left=0.3)
fig.show_markers([52.8], [-28.1], color='red', edgecolor='red', marker='o', s=25)
fig.add_label((3 + 34 / 60) * 360 / 24, -27 - 32 / 60, '0', color='black', fontweight='bold', fontsize=30, ha='center', va='center')
fig.add_label((3 + 34 / 60) * 360 / 24, -28 - 36 / 60, '3', color='black', fontweight='bold', fontsize=30, ha='center', va='center')
fig.add_label(52.0, -27 - 32 / 60, '1', color='black', fontweight='bold', fontsize=30, ha='center', va='center')
fig.add_label(52.0, -28 - 36 / 60, '2', color='black', fontweight='bold', fontsize=30, ha='center', va='center')
fig.show_lines([
                            # The linspace gives us some resolution so we don't just whack a straight line on the curved sky.
                            # Does aplpy handle this automatically? Not sure.
                            (numpy.array([(52.8, -28.1)]).T + numpy.stack([numpy.linspace(-2, 0, 10), numpy.zeros((10,))])),
                            (numpy.array([(52.8, -28.1)]).T + numpy.stack([numpy.linspace(0, 2, 10), numpy.zeros((10,))])),
                            (numpy.array([(52.8, -28.1)]).T + numpy.stack([numpy.zeros((10,)), numpy.linspace(-2, 0, 10)])),
                            (numpy.array([(52.8, -28.1)]).T + numpy.stack([numpy.zeros((10,)), numpy.linspace(0, 2, 10)])),
               ], color='red')
plt.gca().set_xlim((1500, 8500))
plt.gca().set_ylim((1700, 7300))
# plt.text((1500 + middle[0]) / 2,
#          (7300 + middle[1]) / 2,
#          '0', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
# plt.text((8500 + middle[0]) / 2,
#          (7300 + middle[1]) / 2,
#          '1', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
# plt.text((8500 + middle[0]) / 2,
#          (1700 + middle[1]) / 2,
#          '2', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
# plt.text((1500 + middle[0]) / 2,
#          (1700 + middle[1]) / 2,
#          '3', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
# plt.show()

# # Set up plotting.
# fig = plt.figure(figsize=(8, 5))
# ax = astropy.visualization.wcsaxes.WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs, facecolor='white')
# fig.add_axes(ax)
# stretch = astropy.visualization.ManualInterval(-100e-6, 300e-6)
# ax.imshow(stretch(atlas.data[0, 0]), cmap='Greys')
# ax.set_xlim((1500, 8500))
# ax.set_ylim((1700, 7300))
# middle, = wcs.all_world2pix([(52.8, -28.1)], 1)
# ax.scatter([middle[0]], [middle[1]], s=100, c='black', marker='o')
# ax.axvline(middle[0], color='black')
# ax.axhline(middle[1], color='black')
# plt.text((1500 + middle[0]) / 2,
#          (7300 + middle[1]) / 2,
#          '0', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
# plt.text((8500 + middle[0]) / 2,
#          (7300 + middle[1]) / 2,
#          '1', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
# plt.text((8500 + middle[0]) / 2,
#          (1700 + middle[1]) / 2,
#          '2', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
# plt.text((1500 + middle[0]) / 2,
#          (1700 + middle[1]) / 2,
#          '3', {'fontsize': 25, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
# lon, lat = ax.coords
# lon.set_ticklabel(size=12)
# lat.set_ticklabel(size=12)
# lon.set_major_formatter('hh:mm')
# lon.set_axislabel('Right Ascension', size=12)
# lat.set_axislabel('Declination', size=12)
plt.savefig('../images/quadrants.pdf')
# plt.savefig('images/quadrants.png')
