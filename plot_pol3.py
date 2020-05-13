import pickle
import numpy as np
import matplotlib.pyplot as plt
from functions import render_axis

with open('pol3.pickle', 'rb') as f:
    data = pickle.load(f)

molNUM = 3
timeFACTOR = 2.418884e-5
polarizationTOTALEMPTY = data['pol3empty']
polarizationTOTALFIELD = data['pol3field']
field1FREQ = data['field1FREQ']
field2FREQ = data['field2FREQ']
frequency = data['frequency']
field1 = data['field1']
field2 = data['field2']


fig_pol3, ax_pol3 = plt.subplots(nrows=molNUM, ncols=2, sharex=True, figsize=(11, 8))
fig_pol3.suptitle("NLOFC response in 3 similar molecules")
polMAX = [max(polarizationTOTALEMPTY[_].real.max(), polarizationTOTALEMPTY[_].imag.max(),
              polarizationTOTALFIELD[_].real.max(), polarizationTOTALFIELD[_].imag.max()) for _ in range(molNUM)]

if molNUM > 1:
    for molINDX in range(molNUM):
        ax_pol3[molINDX, 0].plot(field1FREQ / (timeFACTOR * 2 * np.pi), field1 * polMAX[molINDX] / field1.max(), 'b', alpha=0.4, label='field-1')
        ax_pol3[molINDX, 0].plot(field2FREQ / (timeFACTOR * 2 * np.pi), field2 * polMAX[molINDX] / field1.max(), 'r', alpha=0.4, label='field-2')
        ax_pol3[molINDX, 1].plot(field1FREQ / (timeFACTOR * 2 * np.pi), field1 * polMAX[molINDX] / field1.max(), 'b', alpha=0.4)
        ax_pol3[molINDX, 1].plot(field2FREQ / (timeFACTOR * 2 * np.pi), field2 * polMAX[molINDX] / field1.max(), 'r', alpha=0.4)
        ax_pol3[molINDX, 0].plot(frequency / (timeFACTOR * 2 * np.pi), polarizationTOTALFIELD[molINDX].real, 'b', linewidth=1., alpha=0.7)
        ax_pol3[molINDX, 1].plot(frequency / (timeFACTOR * 2 * np.pi), polarizationTOTALFIELD[molINDX].imag, 'b', linewidth=1., alpha=0.7)
        ax_pol3[molINDX, 0].plot(frequency / (timeFACTOR * 2 * np.pi), polarizationTOTALEMPTY[molINDX].real, 'k', linewidth=1.5)
        ax_pol3[molINDX, 1].plot(frequency / (timeFACTOR * 2 * np.pi), polarizationTOTALEMPTY[molINDX].imag, 'k', linewidth=1.5, label=f'$P_{molINDX + 1}'+'^{(3)}(\\omega)$')
        render_axis(ax_pol3[molINDX, 0], 'x-large')
        render_axis(ax_pol3[molINDX, 1], 'x-large')
        ax_pol3[molINDX, 0].set_ylabel(f'Re[$P_{molINDX + 1}'+'^{(3)}(\\omega)$]')
        ax_pol3[molINDX, 1].set_ylabel(f'Im[$P_{molINDX + 1}'+'^{(3)}(\\omega)$]')
        ax_pol3[molNUM - 1, 0].set_xlabel('Frequency (in THz)')
        ax_pol3[molNUM - 1, 1].set_xlabel('Frequency (in THz)')
        ax_pol3[molINDX, 1].yaxis.set_label_position("right")
        ax_pol3[molINDX, 1].yaxis.tick_right()
else:
    ax_pol3[0].plot(frequency / (timeFACTOR * 2 * np.pi), polarizationTOTALEMPTY[0].real, 'r', linewidth=1., alpha=0.7)
    ax_pol3[1].plot(frequency / (timeFACTOR * 2 * np.pi), polarizationTOTALEMPTY[0].imag, 'b', linewidth=1., alpha=0.7)

fig_pol3.subplots_adjust(hspace=0.1, wspace=0.02)
plt.show()
