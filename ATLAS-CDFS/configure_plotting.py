"""Configures matplotlib plots.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import matplotlib
import matplotlib.pyplot


def configure():
    font = {'family' : 'serif',
            'size'   : 16}

    matplotlib.rc('font', **font)

    matplotlib.pyplot.rcParams['axes.facecolor'] = 'white'
    matplotlib.pyplot.rcParams['savefig.facecolor'] = 'white'
