#!/usr/bin/env python

"""
QRdecomposition.py:

Python class performing basis change and QR decomposition of transformed polarization to find heterodyne fields.
"""

__author__ = "Ayan Chattopadhyay"
__affiliation__ = "Princeton University"

# ---------------------------------------------------------------------------- #
#                      LOADING PYTHON LIBRARIES AND FILES                      #
# ---------------------------------------------------------------------------- #

from types import MethodType, FunctionType
import numpy as np
from FP_QRdiscrimination import ADict
from functions import render_axis


class QRD:
    """
    Calculates the QR decomposition to calculate orthogonal heterodyne fields in OFC experiment
    """

    def __init__(self, params, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
         :type SystemVars: object
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.molNUM = params.molNUM
        self.combNUM = params.combNUM
        self.freqNUM = params.freqNUM
        self.resolutionNUM = params.resolutionNUM
        self.omegaM1 = params.omegaM1
        self.omegaM2 = params.omegaM2
        self.combGAMMA = params.combGAMMA
        self.freqDEL = params.freqDEL
        self.termsNUM = params.termsNUM
        self.frequency = params.frequency
        self.field1FREQ = params.field1FREQ
        self.field2FREQ = params.field2FREQ
        self.field1 = params.field1 / params.field1.max()
        self.field2 = params.field2 / params.field2.max()
        self.round = params.round
        self.basisNUM_FB = params.basisNUM_FB
        self.basiswidth_FB = params.basiswidth_FB
        self.pol3_EMPTY.real /= np.abs(self.pol3_EMPTY.real).max()
        self.pol3_EMPTY.imag /= np.abs(self.pol3_EMPTY.imag).max()
        self.pol3basisMATRIX = np.empty((self.molNUM, self.basisNUM_FB))
        self.freq2basisMATRIX = np.zeros((self.basisNUM_FB, self.freqNUM))
        self.rangeFREQ = params.rangeFREQ


    def basis_transform(self):
        # ------------------------------------------------------------------------------------------------------------ #
        #             BASIS TRANSFORMATION MATRICES: F->frequency C->comb B->basis (any newly devised basis            #
        # ------------------------------------------------------------------------------------------------------------ #

        arrayFREQ_FB = self.frequency[:, np.newaxis, np.newaxis]
        arrayBASIS_FB = np.linspace(self.rangeFREQ[0] * self.combNUM, self.rangeFREQ[1] * self.combNUM,
                                    self.basisNUM_FB, endpoint=False)[np.newaxis, :, np.newaxis] * self.freqDEL
        BASIS_bw = int((arrayBASIS_FB[0, 1, 0] - arrayBASIS_FB[0, 0, 0]) / self.freqDEL + 0.5)
        arrayCOMB_FB1 = np.linspace(0., BASIS_bw, self.basiswidth_FB, endpoint=False)[np.newaxis,  np.newaxis, :] * self.freqDEL
        arrayCOMB_FB2 = np.linspace(0., BASIS_bw, self.basiswidth_FB, endpoint=False)[np.newaxis,  np.newaxis, :] * self.freqDEL

        plt.figure()
        arrayFB1 = (arrayBASIS_FB + arrayCOMB_FB1)
        arrayFB2 = (arrayBASIS_FB + arrayCOMB_FB2)

        self.freq2basisMATRIX = self.combGAMMA / ((arrayFREQ_FB - self.omegaM2 * 2 + self.omegaM1 - arrayFB1) ** 2 + self.combGAMMA ** 2) \
                             + self.combGAMMA / ((arrayFREQ_FB - self.omegaM1 * 2 + self.omegaM2 - arrayFB2) ** 2 + self.combGAMMA ** 2)

        colors = ['r', 'b', 'k']

        self.freq2basisMATRIX = self.freq2basisMATRIX.sum(axis=2)
        plt.plot(self.frequency / self.freqDEL, self.freq2basisMATRIX)
        print(self.freq2basisMATRIX.shape)
        plt.figure()
        plt.imshow(self.freq2basisMATRIX.T.dot(self.freq2basisMATRIX).real)
        plt.colorbar()

        print(self.pol3_EMPTY.shape)
        plt.figure()
        self.pol3basisMATRIX = self.pol3_EMPTY.dot(self.freq2basisMATRIX)
        for i in range(self.molNUM):
            plt.plot(self.pol3basisMATRIX[i].real, color=colors[i])

        fig, ax = plt.subplots(nrows=3, ncols=2)
        for i in range(3):
            ax[i, 0].plot(self.pol3_EMPTY[i].real, color=colors[i])
            ax[i, 1].plot(self.pol3_EMPTY[i].imag, color=colors[i])
            ax[i][0].set_ylabel("$Re[P^{(3)}(\omega)]$ -- Mol " + str(i+1), fontsize='xx-large')
            ax[i][1].set_ylabel("$Im[P^{(3)}(\omega)]$ -- Mol " + str(i+1), fontsize='xx-large')
        ax[2][0].set_xlabel("Frequency (in THz)")
        ax[2][1].set_xlabel("Frequency (in Thz)")
        return

    def calculate_heterodyne(self):
        np.set_printoptions(precision=3, suppress=True)
        Q_mat = np.empty((self.molNUM, self.basisNUM_FB, self.basisNUM_FB), dtype=np.complex)
        heterodyne = np.empty((self.molNUM, self.basisNUM_FB), dtype=np.complex)
        R_mat = np.empty((self.molNUM, self.basisNUM_FB, self.molNUM - 1), dtype=np.complex)
        ImatBASIS = np.empty((self.molNUM, self.molNUM), dtype=np.complex)
        ImatFREQ = np.empty((self.molNUM, self.molNUM), dtype=np.complex)

        basisAXIS = np.arange(self.basisNUM_FB)
        envelopeBASIS = (np.exp(-(basisAXIS - 0.5 * self.basisNUM_FB)**2 / (2 * (self.basisNUM_FB * 2) ** 2)))

        for molINDX in range(self.molNUM):
            Q_mat[molINDX], R_mat[molINDX] = np.linalg.qr(np.delete(self.pol3basisMATRIX.T, molINDX, 1), mode='complete')
            heterodyne[molINDX] = sum(q * np.vdot(q, envelopeBASIS) for q in Q_mat[molINDX, :, self.molNUM - 1:].T)

            for j in range(self.molNUM):
                ImatBASIS[molINDX, j] = np.vdot(heterodyne[molINDX], self.pol3basisMATRIX[j])
                ImatFREQ[molINDX, j] = np.vdot(heterodyne[molINDX].dot(self.freq2basisMATRIX.T), self.pol3_EMPTY[j])

        print(ImatBASIS)

        for i in range(molNUM):
            ImatFREQ[i] /= np.abs(ImatFREQ[i]).max()

        print(np.abs(ImatFREQ))


        fig2, axes2 = plt.subplots(nrows=1, ncols=2, sharex=True)
        for molINDX in range(self.molNUM):
            axes2[0].plot(heterodyne[molINDX].real, '-')
            axes2[1].plot(heterodyne[molINDX].imag, '-')

        print(self.freq2basisMATRIX.shape)
        print(heterodyne.shape)

        colors = ['r', 'b', 'k']

        shaped_het = heterodyne.dot(self.freq2basisMATRIX.T)
        fig, ax = plt.subplots(nrows=3, ncols=2)
        for i in range(molNUM):
            ax[i][0].plot(self.frequency / self.freqDEL, shaped_het[i].real / shaped_het.real.max(), colors[i])
            ax[i][1].plot(self.frequency / self.freqDEL, shaped_het[i].imag / shaped_het.imag.max(), colors[i])
            ax[i][0].set_ylabel("$Re[E_{het}(\omega)]$ -- Mol " + str(i+1), fontsize='xx-large')
            ax[i][1].set_ylabel("$Im[E_{het}(\omega)]$ -- Mol " + str(i+1), fontsize='xx-large')
            render_axis(ax[i][0])
            render_axis(ax[i][1])
        ax[2][0].set_xlabel("Frequency (in THz)")
        ax[2][1].set_xlabel("Frequency (in Thz)")

        return


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    import time

    with open('pol3.pickle', 'rb') as f:
        data = pickle.load(f)

    molNUM = 3
    timeFACTOR = 2.418884e-5
    polarizationTOTALEMPTY = data['pol3empty']
    polarizationTOTALFIELD = data['pol3field']
    field1FREQ = data['field1FREQ']
    field2FREQ = data['field2FREQ']
    frequency = data['frequency']
    freqNUM = frequency.size
    field1 = data['field1']
    field2 = data['field2']

    with open('pol3_args.pickle', 'rb') as f_args:
        data = pickle.load(f_args)

    combNUM = data['combNUM']
    resolutionNUM = data['resolutionNUM']
    omegaM1 = data['omegaM1']
    omegaM2 = data['omegaM2']
    freqDEL = data['freqDEL']
    combGAMMA = data['combGAMMA']
    termsNUM = data['termsNUM']
    envelopeWIDTH = data['envelopeWIDTH']
    envelopeCENTER = data['envelopeCENTER']
    chiNUM = data['chiNUM']
    rangeFREQ = data['rangeFREQ']
    basisNUM_FB = 50

    SystemVars = ADict(
        molNUM=molNUM,
        combNUM=combNUM,
        freqNUM=freqNUM,
        resolutionNUM=resolutionNUM,
        omegaM1=omegaM1,
        omegaM2=omegaM2,
        combGAMMA=combGAMMA,
        freqDEL=freqDEL,
        termsNUM=termsNUM,
        frequency=frequency,
        field1FREQ=field1FREQ,
        field2FREQ=field2FREQ,
        field1=field1,
        field2=field2,
        round=1,
        basisNUM_FB=basisNUM_FB,
        basiswidth_FB=int(combNUM/basisNUM_FB),
        rangeFREQ=rangeFREQ
    )

    SystemArgs = dict(
        pol3_EMPTY=polarizationTOTALEMPTY,
        pol3_FIELD=polarizationTOTALFIELD
    )

    start = time.time()
    system = QRD(SystemVars, **SystemArgs)
    system.basis_transform()
    system.calculate_heterodyne()
    print("Time elapsed: ", time.time() - start)

    plt.show()
