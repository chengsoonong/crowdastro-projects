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
import matplotlib
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


def plot(radio_path, ir_path, plot_atlas_hosts=False, vmax=99.5, centrebox=False,
         centreboxwidth=None, width_in_px=True, stretch='arcsinh', fig=None,
         first=False):
    fig = aplpy.FITSFigure(ir_path, slices=[0, 1], figure=fig)
    fig.set_theme('publication')
    # if not v:
        # fig.show_grayscale(stretch=stretch, invert=True)
    # else:
    #     fig.show_grayscale(stretch=stretch, vmin=v[0], vmax=v[1], invert=True)
    with astropy.io.fits.open(ir_path) as f:
        fig.show_colorscale(cmap='cubehelix_r', vmin=f[0].data.min(), vmax=numpy.percentile(f[0].data, vmax))
    if plot_atlas_hosts:
        table = astropy.io.ascii.read(
            '/Users/alger/data/RGZ/dr1_weighted_old/static_rgz_host_full.csv')
        ras = table['SWIRE.ra']
        decs = table['SWIRE.dec']
        fig.show_markers(ras, decs, marker='x', s=50, c='red')

    if centrebox:
        with astropy.io.fits.open(radio_path) as f, astropy.io.fits.open(ir_path) as g:
            if first:
                contours = numpy.array([4, 8, 16, 32, 64, 128, 256]) * 0.14e-3
                fig.show_contour(f, levels=contours, colors='black', linewidths=0.75, zorder=2, slices=[2, 3])
            else:
                contours = [4, 8, 16, 32, 64, 128, 256]
                fig.show_contour(f, levels=contours, colors='black', linewidths=0.75, zorder=2)
            centre = numpy.array(g[0].data.shape) // 2
            if not first:
                cdelt1 = f[0].header['CDELT1']
                cdelt2 = f[0].header['CDELT2']
                ra, dec = fig.pixel2world(centre[0], centre[1])
            else:
                cdelt1 = f[0].header['CDELT3']
                cdelt2 = f[0].header['CDELT4']
                ra, dec = fig.pixel2world(centre[0], centre[1])
        if width_in_px:
            width = -cdelt1 * centreboxwidth
            height = cdelt2 * centreboxwidth
        else:
            width = centreboxwidth
            height = centreboxwidth
        fig.show_rectangles([ra], [dec], width, height, color='r', linewidth=1)
    else:
        with astropy.io.fits.open(radio_path) as f:
            if first:
                contours = numpy.array([4, 8, 16, 32, 64]) * 0.14e-3
                fig.show_contour(f, levels=contours, colors='black', linewidths=0.75, zorder=2, slices=[2, 3])
            else:
                contours = [4, 8, 16, 32, 64, 128, 256]
                fig.show_contour(f, levels=contours, colors='black', linewidths=0.75, zorder=2)
    fig.axis_labels.set_xtext('Right Ascension (J2000)')
    fig.axis_labels.set_ytext('Declination (J2000)')

    # fig.ticks.set_linewidth(2)
    # fig.ticks.set_color('black')
    # fig.tick_labels.set_font(size='xx-large', weight='medium', \
    #                          stretch='normal', family='sans-serif', \
    #                          style='normal', variant='normal')
    # fig.axis_labels.set_font(size='xx-large', weight='medium', \
    #                          stretch='normal', family='sans-serif', \
    #                          style='normal', variant='normal')
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
    # radio_path = "/Users/alger/data/RGZ/cdfs/2x2/CI2363_radio.fits"
    # ir_path = "/Users/alger/data/RGZ/cdfs/2x2/CI2363_ir.fits"
    # fig = plt.figure()
    # fig = plot(radio_path, ir_path, plot_atlas_hosts=False, centrebox=True, centreboxwidth=48 / 60 / 60, width_in_px=False, fig=fig)
    # plt.subplots_adjust(top=1, right=0.95, left=0.3)
    # plt.show()
    # plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/CI2363_fig.pdf')

    # radio_path = "/Users/alger/data/RGZ/cdfs/2x2/CI0077C1_radio.fits"
    # ir_path = "/Users/alger/data/RGZ/cdfs/2x2/CI0077C1_ir.fits"
    # fig = plt.figure()
    # fig = plot(radio_path, ir_path, plot_atlas_hosts=True, centrebox=False, fig=fig, vmax=99.9)
    # plt.subplots_adjust(top=1, right=0.95, left=0.3)
    # plt.show()
    # plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/CI0077C1_fig.pdf')

    # radio_path = "/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/FIRSTJ151227.2+454026_8.fits"
    # ir_path = "/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/2279p454_ac51-w1-int-3_ra228.11333333333332_dec45.67388888888889_asec480.000.fits"
    # fig = plt.figure()
    # fig = plot(radio_path, ir_path, plot_atlas_hosts=False, centrebox=True, centreboxwidth=3 / 60, width_in_px=False, stretch='linear', fig=fig, first=True)
    # plt.subplots_adjust(top=1, right=0.95, left=0.3)
    # plt.show()
    # plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/FIRSTJ151227_fig.pdf')
