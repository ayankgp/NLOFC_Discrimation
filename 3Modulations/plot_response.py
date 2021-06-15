import numpy as np
import matplotlib.pyplot as plt
from functions import render_axis
import pickle
import matplotlib.gridspec as gs

with open('Pickle/P3_plot.pickle', 'rb') as f_P3:
    data_P3 = pickle.load(f_P3)

with open('Pickle/S3_plot.pickle', 'rb') as f_S3:
    data_S3 = pickle.load(f_S3)

frequency = data_P3['freq']

het_P3_real = data_P3['het_real']
het_P3_real /= het_P3_real.max()
het_P3_imag = data_P3['het_imag']
het_P3_imag /= het_P3_imag.max()
het_S3_real = data_S3['het_real']
het_S3_real /= het_S3_real.max()
het_S3_imag = data_S3['het_imag']
het_S3_imag /= het_S3_imag.max()

P3 = data_P3['pol3']
S3 = data_S3['pol3']

molNUM = 3

colors = ['r', 'k', 'b']
alphas = [0.3, 0.6, 0.9]

fig2 = plt.figure(constrained_layout=True)
spec2 = gs.GridSpec(ncols=2, nrows=2, figure=fig2)
f2_ax1 = fig2.add_subplot(spec2[0, 0])
f2_ax2 = fig2.add_subplot(spec2[0, 1])
f2_ax3 = fig2.add_subplot(spec2[1, 0])
f2_ax4 = fig2.add_subplot(spec2[1, 1])



# fig, ax = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=True, figsize=(11, 11))
# for i in range(3):
#     ax[i, 0].plot(frequency, P3.real)
#     ax[i, 1].plot(frequency, P3.imag)
#
#     ax[i][0].set_ylabel("$Re[P^{(3)}(\omega)]$ \n Mol " + str(i + 1), fontsize='x-large', fontweight="bold")
#     ax[i][1].set_ylabel("$Im[P^{(3)}(\omega)]$ \n Mol " + str(i + 1), fontsize='x-large', fontweight="bold")
#
#     render_axis(ax[i][0], labelSIZE='xx-large')
#     render_axis(ax[i][1], labelSIZE='xx-large')
#
# ax[2][0].set_xlabel("Frequency (in THz)", fontsize='x-large', fontweight="bold")
# ax[2][1].set_xlabel("Frequency (in Thz)", fontsize='x-large', fontweight="bold")
#
# for i in range(molNUM):
#     ax[i][0].plot(frequency, het_P3_real[i], colors[i], alpha=alphas[i])
#     ax[i][1].plot(frequency, het_P3_imag[i], colors[i], alpha=alphas[i])
#
#     ax[i][0].set_ylabel("$Re[E_{het}(\omega)]$ \n Mol " + str(i + 1), fontsize='x-large', fontweight="bold")
#     ax[i][1].set_ylabel("$Im[E_{het}(\omega)]$ \n Mol " + str(i + 1), fontsize='x-large', fontweight="bold")
#
#     render_axis(ax[i][0], labelSIZE='xx-large')
#     render_axis(ax[i][1], labelSIZE='xx-large')
#
# ax[2][0].set_xlabel("Frequency (in THz)", fontsize='x-large', fontweight="bold")
# ax[2][1].set_xlabel("Frequency (in Thz)", fontsize='x-large', fontweight="bold")

plt.show()