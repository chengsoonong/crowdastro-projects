"""Plot the example figure for object localisation.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import aplpy
import astropy.io.fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches

radio_path = '/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/EI0093C1_radio.fits'
fig = aplpy.FITSFigure(radio_path)
fig.show_grayscale(stretch='arcsinh')
# fig = plt.figure(figsize=(10, 10), dpi=50)
# ax = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)
# ax.imshow(radio)
ax = plt.gca()
rect = patches.Rectangle((201-75, 10), 71, 71, edgecolor='red', linewidth=5, fill=None)
ax.add_patch(rect)
rect = patches.Rectangle((100-71/2, 100-71/2), 71, 71, edgecolor='red', linewidth=5, fill=None)
ax.add_patch(rect)
rect = patches.Rectangle((5, 50), 71, 71, edgecolor='red', linewidth=5, fill=None)
ax.add_patch(rect)
# plt.axis('off')
# plt.title('a)')
# ax = plt.subplot2grid((3, 3), (2, 0))
# ax.imshow(radio[10:10+87, 5:5+87], cmap='Greys', vmin=radio.min(), vmax=radio.max())
# plt.title('b)')
# plt.text(44, 110, '$p = 0.01$', fontsize=35, horizontalalignment='center')
# for axis in ['top','bottom','left','right']:
#     plt.gca().spines[axis].set_linewidth(5)
#     plt.gca().spines[axis].set_edgecolor('red')
# plt.tick_params(
#     axis='both',
#     which='both',
#     bottom='off',
#     top='off',
#     left='off',
#     right='off',
#     labelbottom='off',
#     labelleft='off')
# ax = plt.subplot2grid((3, 3), (2, 1))
# plt.title('c)')
# ax.imshow(radio[170:170+87, 30:30+87], cmap='Greys', vmin=radio.min(), vmax=radio.max())
# plt.text(44, 110, '$p = 0.48$', fontsize=35, horizontalalignment='center')
# for axis in ['top','bottom','left','right']:
#     plt.gca().spines[axis].set_linewidth(5)
#     plt.gca().spines[axis].set_edgecolor('red')
# plt.tick_params(
#     axis='both',
#     which='both',
#     bottom='off',
#     top='off',
#     left='off',
#     right='off',
#     labelbottom='off',
#     labelleft='off')
# ax = plt.subplot2grid((3, 3), (2, 2))
# plt.title('d)')
# ax.imshow(radio[137:137+87, 137:137+87], cmap='Greys', vmin=radio.min(), vmax=radio.max())
# plt.text(44, 110, '$p = 0.99$', fontsize=35, horizontalalignment='center')
# for axis in ['top','bottom','left','right']:
#     plt.gca().spines[axis].set_linewidth(5)
#     plt.gca().spines[axis].set_edgecolor('red')
# plt.tick_params(
#     axis='both',
#     which='both',
#     bottom='off',
#     top='off',
#     left='off',
#     right='off',
#     labelbottom='off',
#     labelleft='off')
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.2, bottom=0.1)
# plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/windows.pdf')
# plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/windows.eps')
plt.show()