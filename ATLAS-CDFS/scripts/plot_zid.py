"""Plot a Zooniverse subject.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import astropy.io.fits
import astropy.visualization
import astropy.wcs
import matplotlib.pyplot as plt
import matplotlib.patches

import configure_plotting
configure_plotting.configure()


def plot(path, h=0.1):
    fig = aplpy.FITSFigure(path)
    fig.show_grayscale()
    # with astropy.io.fits.open(path) as fits:
    #     image = fits[0].data
    #     image -= image.min()
    #     image /= image.max()
    #     wcs = astropy.wcs.WCS(fits[0].header)
    #     try:
    #         plt.subplot(projection=wcs)
    #     except ValueError:
    #         # 4 axes instead of 2.
    #         wcs = wcs.dropaxis(3).dropaxis(2)
    #         plt.subplot(projection=wcs)
    #     stretch = astropy.visualization.AsinhStretch(h)
    #     lon, lat = plt.gca().coords
    #     lon.set_ticklabel(size=16)
    #     lat.set_ticklabel(size=16)
    #     lon.set_major_formatter('hh:mm:ss')
    #     lon.set_axislabel('Right Ascension', size=16)
    #     lat.set_axislabel('Declination', size=16)
    #     plt.imshow(stretch(image), cmap='Greys_r')
    #     plt.subplots_adjust(bottom=0.15, top=1.0, right=1.0)

def plot_box():
    rect = matplotlib.patches.Rectangle((267 / 2 - 267 / 8 * 3 / 2, 267 / 2 - 267 / 8 * 3 / 2), 267 / 8 * 3, 267 / 8 * 3, facecolor='None', edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)

if __name__ == '__main__':
    plot('/Users/alger/data/RGZ/cdfs/2x2/CI0077C1_radio.fits')
    # plot("J:\\repos\\crowdastro-projects\\ATLAS-CDFS\\images\\FIRSTJ151227.2+454026_3.fits", h=0.01)
    # plot("J:\\repos\\crowdastro-projects\\ATLAS-CDFS\\images\\FIRSTJ151227.2+454026_8.fits", h=0.01)
    # plot_box()
    plt.show()
