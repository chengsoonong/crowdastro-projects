"""Plot a Zooniverse subject.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import aplpy
import astropy.coordinates
import astropy.io.ascii
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy

import configure_plotting
configure_plotting.configure()


def plot(path, plot_atlas_hosts=False, v=None):
    fig = aplpy.FITSFigure(path, slices=[0, 1])
    if not v:
        fig.show_grayscale(stretch='arcsinh', invert=True)
    else:
        fig.show_grayscale(stretch='arcsinh', vmin=v[0], vmax=v[1], invert=True)
    if plot_atlas_hosts:
        table = astropy.io.ascii.read(
            '/Users/alger/data/RGZ/dr1_weighted_old/static_rgz_host_full.csv')
        ras = table['SWIRE.ra']
        decs = table['SWIRE.dec']
        fig.show_markers(ras, decs, marker='x', s=200, c='red')

def plot_box_FIRST():
    rect = matplotlib.patches.Rectangle((267 / 2 - 267 / 8 * 3 / 2, 267 / 2 - 267 / 8 * 3 / 2), 267 / 8 * 3, 267 / 8 * 3, facecolor='None', edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)

def plot_box_ATLAS():
    rect = matplotlib.patches.Rectangle((100 - 35, 100 - 35), 70, 70, facecolor='None', edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)

if __name__ == '__main__':
    # plot('/Users/alger/data/RGZ/cdfs/2x2/CI0077C1_radio.fits', plot_atlas_hosts=True)
    # plot("J:\\repos\\crowdastro-projects\\ATLAS-CDFS\\images\\FIRSTJ151227.2+454026_3.fits")
    # plot("/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/FIRSTJ151227.2+454026_8.fits", v=(-5.472e-04, 3.887e-02))
    plt.subplots_adjust(left=0.2)
    plot("/Users/alger/data/RGZ/cdfs/2x2/CI2363_radio.fits")
    plot_box_ATLAS()
    plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/CI2363_fig.pdf')
    # plot_box_FIRST()
    # plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/FIRSTJ151227_fig.pdf')
    # plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/CI0077C1_fig.pdf')
    # plt.show()
