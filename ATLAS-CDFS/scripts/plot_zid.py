"""Plot a Zooniverse subject.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import aplpy
import astropy.coordinates
import astropy.io.ascii
import astropy.io.fits
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy


def plot(path, plot_atlas_hosts=False, v=None, centrebox=False,
         centreboxwidth=None, width_in_px=True):
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

    if centrebox:
        with astropy.io.fits.open(path) as f:
            centre = numpy.array(f[0].data.shape) // 2
            cdelt1 = f[0].header['CDELT1']
            cdelt2 = f[0].header['CDELT2']
            ra, dec = fig.pixel2world(centre[0], centre[1])
        if width_in_px:
            width = -cdelt1 * centreboxwidth
            height = cdelt2 * centreboxwidth
        else:
            width = centreboxwidth
            height = centreboxwidth
        fig.show_rectangles([ra], [dec], width, height,
                            cdelt2 * centreboxwidth, color='r', linewidth=3)

    fig.ticks.set_linewidth(2)
    fig.ticks.set_color('black')
    fig.tick_labels.set_font(size='xx-large', weight='medium', \
                             stretch='normal', family='sans-serif', \
                             style='normal', variant='normal')
    fig.axis_labels.set_font(size='xx-large', weight='medium', \
                             stretch='normal', family='sans-serif', \
                             style='normal', variant='normal')
    fig.set_tick_labels_format(xformat='hh:mm:ss',yformat='dd:mm:ss')

    return fig

def plot_box_FIRST(fig, path):
    fig.show_rectangles([])
    # rect = matplotlib.patches.Rectangle((267 / 2 - 267 / 8 * 3 / 2, 267 / 2 - 267 / 8 * 3 / 2), 267 / 8 * 3, 267 / 8 * 3, facecolor='None', edgecolor='red', linewidth=2)
    # plt.gca().add_patch(rect)

# def plot_box_ATLAS(fig, path):
    # rect = matplotlib.patches.Rectangle((100 - 35, 100 - 35), 70, 70, facecolor='None', edgecolor='red', linewidth=2)
    # plt.gca().add_patch(rect)

if __name__ == '__main__':
    path = "/Users/alger/data/RGZ/cdfs/2x2/CI2363_radio.fits"
    fig = plot(path, plot_atlas_hosts=False, centrebox=True, centreboxwidth=32)
    plt.subplots_adjust(left=0.2)
    plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/CI2363_fig.pdf')

    path = "/Users/alger/data/RGZ/cdfs/2x2/CI0077C1_radio.fits"
    fig = plot(path, plot_atlas_hosts=True, centrebox=False)
    plt.subplots_adjust(left=0.2)
    plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/CI0077C1_fig.pdf')

    path = "/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/FIRSTJ151227.2+454026_8.fits"
    fig = plot(path, plot_atlas_hosts=False, centrebox=True, centreboxwidth=3 / 60, width_in_px=False)
    plt.subplots_adjust(left=0.2)
    plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/FIRSTJ151227_fig.pdf')

    # plot('/Users/alger/data/RGZ/cdfs/2x2/CI0077C1_radio.fits', plot_atlas_hosts=True)
    # plot("J:\\repos\\crowdastro-projects\\ATLAS-CDFS\\images\\FIRSTJ151227.2+454026_3.fits")
    # plot("/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/FIRSTJ151227.2+454026_8.fits", v=(-5.472e-04, 3.887e-02))
    # plot_box_ATLAS(fig, path)
    # plot_box_FIRST()
    # plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/FIRSTJ151227_fig.pdf')
    # plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/CI0077C1_fig.pdf')
    # plt.show()
