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
        self.basisNUM_CB = params.basisNUM_CB
        self.basisNUM_FB = params.basisNUM_FB
        self.basisENVwidth_CB = params.basisENVwidth_CB
        self.basisENVwidth_FB = params.basisENVwidth_FB
        self.pol3_EMPTY.real /= np.abs(self.pol3_EMPTY.real).max()
        self.pol3_EMPTY.imag /= np.abs(self.pol3_EMPTY.imag).max()
        self.pol3basisMATRIX = np.empty((self.basisNUM_FB, self.freqNUM))
        self.pol3combMATRIX = np.empty((self.basisNUM_CB, self.combNUM))
        self.freq2combMATRIX = np.empty((self.freqNUM, self.combNUM))
        self.freq2basisMATRIX = np.zeros((self.basisNUM_FB, self.freqNUM))
        self.comb2basisMATRIX = np.zeros((self.combNUM, self.basisNUM_CB))


    def basis_transform(self):
        # ------------------------------------------------------------------------------------------------------------ #
        #             BASIS TRANSFORMATION MATRICES: F->frequency C->comb B->basis (any newly devised basis            #
        # ------------------------------------------------------------------------------------------------------------ #

        arrayFREQ_FC = self.frequency[:, np.newaxis]
        arrayCOMB_FC = np.linspace(-1. * self.combNUM * self.freqDEL, 1.4 * self.combNUM * self.freqDEL, self.combNUM + 1)[np.newaxis, :]

        freq2combMATRIX_1 = self.combGAMMA / ((arrayFREQ_FC - 2 * self.omegaM2 + self.omegaM1 - arrayCOMB_FC) ** 2 + self.combGAMMA ** 2)
        freq2combMATRIX_2 = self.combGAMMA / ((arrayFREQ_FC - 2 * self.omegaM1 + self.omegaM2 - arrayCOMB_FC) ** 2 + self.combGAMMA ** 2)

        self.freq2combMATRIX = freq2combMATRIX_1 + freq2combMATRIX_2

        del freq2combMATRIX_1
        del freq2combMATRIX_2

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(self.pol3_EMPTY.dot(self.freq2combMATRIX).real.T)
        ax[1].plot(self.pol3_EMPTY.real.T)

        return


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    import time

    with open('pol30.pickle', 'rb') as f:
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

    with open('pol3_args0.pickle', 'rb') as f_args:
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

    SystemVars = ADict(
        molNUM=molNUM,
        combNUM=5000,
        freqNUM=freqNUM,
        resolutionNUM=3,
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
        basisNUM_CB=50,
        basisENVwidth_CB=500,
        basisNUM_FB=10,
        basisENVwidth_FB=1000
    )

    SystemArgs = dict(
        pol3_EMPTY=polarizationTOTALEMPTY,
        pol3_FIELD=polarizationTOTALFIELD
    )

    start = time.time()
    system = QRD(SystemVars, **SystemArgs)
    system.basis_transform()
    print("Time elapsed: ", time.time() - start)

    plt.show()
