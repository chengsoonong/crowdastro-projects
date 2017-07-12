"""Find examples.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import collections

import astropy.io.ascii
import numpy

import pipeline

def get_examples(field='cdfs'):
    if field == 'cdfs':
        table = astropy.io.ascii.read(pipeline.TABLE_PATH)
        for row in table:
            name = row['Component Name (Franzen)']
            if not name:
                continue

            yield (name, row['Component Zooniverse ID (RGZ)'], row['Component ID (Franzen)'])
    else:
        

