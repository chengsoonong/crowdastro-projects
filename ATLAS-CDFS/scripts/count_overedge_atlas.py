"""Count the number of overedge sources from the Norris et al. (2006)
source catalogue.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import itertools

import astropy.coordinates
import astropy.io.ascii
import matplotlib.pyplot as plt
import numpy

components_to_coords = {}

for row in astropy.io.ascii.read('J:/repos/crowdastro/data/one-table-to-rule-them-all.tbl'):
    if not row['Component Radio RA (Norris)']:
        continue

    coord = astropy.coordinates.SkyCoord(ra=row['Component Radio RA (Norris)'],
                                         dec=row['Component Radio dec (Norris)'],
                                         unit=('hourangle', 'deg'))
    coord = coord.ra.deg, coord.dec.deg
    components_to_coords[row['Component # (Norris)']] = coord

dists = []
sources = []

for row in astropy.io.ascii.read('J:/repos/crowdastro/data/one-table-to-rule-them-all.tbl'):
    if not row['Source # (Norris)']:
        continue
    components = row['Source Component (Norris)'].split(',')
    coords = [components_to_coords[component.strip()] for component in components]
    if len(coords) < 2:
        continue
    max_dist = 0
    for i, j in itertools.combinations(coords, 2):
        dist = numpy.hypot(i[0] - j[0], i[1] - j[1])
        if dist > max_dist:
            max_dist = dist
    dists.append(max_dist)
    sources.append(row['Source # (Norris)'])

overedge = {j: i for j, i in zip(sources, dists) if i > 1 / 60}
print(len(overedge), 'overedge')
hist_dists = list(overedge.values())
print(overedge.keys())
plt.hist(hist_dists)
plt.show()
