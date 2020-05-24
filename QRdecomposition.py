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
        self.basisNUM_FB = params.basisNUM_FB
        self.basiswidth_FB = params.basiswidth_FB
        self.pol3_EMPTY.real /= np.abs(self.pol3_EMPTY.real).max()
        self.pol3_EMPTY.imag /= np.abs(self.pol3_EMPTY.imag).max()
        self.pol3basisMATRIX = np.empty((self.basisNUM_FB, self.freqNUM))
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
        print(BASIS_bw, self.basiswidth_FB)
        arrayCOMB_FB1 = np.linspace(0., BASIS_bw, self.basiswidth_FB, endpoint=False)[np.newaxis,  np.newaxis, :] * self.freqDEL
        print(arrayCOMB_FB1 / self.freqDEL)
        arrayCOMB_FB2 = np.linspace(0., BASIS_bw, self.basiswidth_FB, endpoint=False)[np.newaxis,  np.newaxis, :] * self.freqDEL

        plt.figure()
        arrayFB1 = (arrayBASIS_FB + arrayCOMB_FB1)
        arrayFB2 = (arrayBASIS_FB + arrayCOMB_FB2)

        print(arrayFB1 / self.freqDEL)
        # print((arrayFREQ_FB.sum(axis=(1,2)) + self.omegaM2 * 2 - self.omegaM1) / self.freqDEL)
        print((arrayFREQ_FB.sum(axis=(1,2)) - self.omegaM2 * 2 + self.omegaM1) / self.freqDEL)

        freq2basisMATRIX_1 = self.combGAMMA / ((arrayFREQ_FB - self.omegaM2 * 2 + self.omegaM1 - arrayFB1) ** 2 + self.combGAMMA ** 2) \
                             + self.combGAMMA / ((arrayFREQ_FB - self.omegaM1 * 2 + self.omegaM2 - arrayFB2) ** 2 + self.combGAMMA ** 2)
        # freq2basisMATRIX = (freq2basisMATRIX_1 + freq2basisMATRIX_2).sum(axis=2)
        plt.plot(self.frequency / self.freqDEL, freq2basisMATRIX_1.sum(axis=2))
        # plt.plot(self.frequency / self.freqDEL, freq2basisMATRIX_2.sum(axis=2))

        # pointsFREQpolarization = np.linspace(self.rangeFREQ[0] * self.combNUM * self.freqDEL,
        #                                      self.rangeFREQ[1] * self.combNUM * self.freqDEL,
        #                                      self.combNUM + 1)[:, np.newaxis]
        # resolution = np.linspace(-0.02, 0.02, self.resolutionNUM) * self.freqDEL

        # frequency_12 = (2 * self.omegaM2 - self.omegaM1 + pointsFREQpolarization + resolution) / self.freqDEL
        # frequency_21 = (2 * self.omegaM1 - self.omegaM2 + pointsFREQpolarization + resolution) / self.freqDEL
        # frequency_1 = (self.omegaM1 + pointsFREQpolarization + resolution) / self.freqDEL
        # frequency_2 = (self.omegaM2 + pointsFREQpolarization + resolution) / self.freqDEL
        # freq = (pointsFREQpolarization + resolution) / self.freqDEL
        #
        # plt.plot(frequency_12, np.zeros_like(frequency_12), 'r*-')
        # plt.plot(frequency_21, np.zeros_like(frequency_21), 'b*-')
        # plt.plot(frequency_1, np.zeros_like(frequency_1), 'y*-')
        # plt.plot(frequency_2, np.zeros_like(frequency_2), 'g*-')
        # plt.plot(freq, np.zeros_like(freq), 'k*-')
        # plt.plot(freq2basisMATRIX)
        # plt.imshow(freq2basisMATRIX.T.dot(freq2basisMATRIX))
        plt.show()
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
    basisNUM_FB = 10

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
    print("Time elapsed: ", time.time() - start)

    # plt.show()
