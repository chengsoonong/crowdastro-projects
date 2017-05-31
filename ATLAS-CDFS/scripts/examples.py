#!/usr/bin/env python3
"""Find good/bad examples of cross-identification.

Input files:
- ???

Output files:
- images/cdfs_ba_grid.pdf
- images/cdfs_ba_grid.png

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""
import matplotlib.pyplot as plt

import configure_plotting
import pipeline

configure_plotting.configure()

# Load predictions.
lr_predictions = pipeline.unserialise_predictions(
    pipeline.WORKING_DIR + 'LogisticRegression_predictions')
rf_predictions = pipeline.unserialise_predictions(
    pipeline.WORKING_DIR + 'RandomForestClassifier_predictions')