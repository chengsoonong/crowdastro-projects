import astropy.io.ascii, numpy
import matplotlib.pyplot as plt
import configure_plotting
configure_plotting.configure()
table = astropy.io.ascii.read('/Users/alger/data/Crowdastro/one-table-to-rule-them-all.tbl')
norris_fluxes = []
rgz_fluxes = []
for row in table:
    if row['Source SWIRE (Norris)'] and row['Source SWIRE (Norris)'].startswith('SWIRE') and row['Component ID (Franzen)']:
        # This is in the Norris training set.
        norris_fluxes.append(row['Component S (Franzen)'])
    if row['Component Zooniverse ID (RGZ)']:
        # This is in the RGZ training set.
        rgz_fluxes.append(row['Component S (Franzen)'])
norris_fluxes = numpy.array(norris_fluxes)
rgz_fluxes = numpy.array(rgz_fluxes)
plt.hist([norris_fluxes, rgz_fluxes], bins=numpy.logspace(-1, 2, 20), normed=True)
plt.xscale('log')
plt.xlabel('Integrated flux (mJy)')
plt.ylabel('Normalised number of sources')
plt.legend(['Norris', 'RGZ'])
plt.subplots_adjust(bottom=0.15)
plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/flux_histogram.pdf')
plt.savefig('/Users/alger/repos/crowdastro-projects/ATLAS-CDFS/images/flux_histogram.png')