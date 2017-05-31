#!/usr/bin/env python3
"""Output cross-identification accuracies.

These are assessed against the RGZ & Norris intersection,
on the Norris labels.

Output files:
- images/cdfs_ba_grid.pdf
- images/cdfs_ba_grid.png

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import pipeline


def main():
    swire_names, swire_coords, _ = generate_swire_features(overwrite=False)
    swire_labels = generate_swire_labels(swire_names, overwrite=overwrite_all)
    cids = list(cross_identify_all(swire_names, swire_coords))


if __name__ == '__main__':
    main()