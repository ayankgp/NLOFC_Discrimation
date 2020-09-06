#!/usr/bin/env python

"""
CalculateSpectra.py:

Class containing C calls for spectra calculation and discriminating OFC-pulse generation.
Plots results obtained from C calls.
"""

__author__ = "Ayan Chattopadhyay"
__affiliation__ = "Princeton University"


# ---------------------------------------------------------------------------- #
#                      LOADING PYTHON LIBRARIES AND FILES                      #
# ---------------------------------------------------------------------------- #

from multiprocessing import cpu_count
from types import MethodType, FunctionType
import time
from itertools import product, permutations
from functions import *
from wrapper import *
import pickle


class ADict(dict):
    """
    Appended Dictionary: where keys can be accessed as attributes: A['*'] --> A.*
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class Spectra:
    """
    Calculates the linear absorption spectra and fits molecular parameters in the process
    """

    def __init__(self, spectra_variables, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.rho_0 = np.ascontiguousarray(spectra_variables.rho_0)
        self.rho = np.ascontiguousarray([spectra_variables.rho_0.copy() for _ in range(spectra_variables.molNUM)])
        self.spectra_time = np.linspace(0, spectra_variables.spectra_timeAMP, spectra_variables.spectra_timeDIM)
        self.spectra_field = np.zeros(spectra_variables.spectra_timeDIM, dtype=np.complex)
        self.gammaMATRIXpopd = np.ascontiguousarray(self.gammaMATRIXpopd)
        self.gammaMATRIXdephasing = np.ascontiguousarray(self.gammaMATRIXdephasing)
        self.muMATRIX = np.ascontiguousarray(self.muMATRIX)
        self.energies = np.ascontiguousarray(self.energies)
        self.levelsNUM = spectra_variables.levelsNUM
        self.spectra_absTOTAL = np.ascontiguousarray(np.zeros((spectra_variables.molNUM, len(self.spectra_frequencies[0]))))
        self.spectra_absDIST = np.ascontiguousarray(np.empty((spectra_variables.molNUM, spectra_variables.ensembleNUM, len(self.spectra_frequencies[0]))))
        # ------------------------------------------------------------------------------------------------------------ #
        #                       DECLARE NEW SET OF VARIABLES FOR ALL N MOLECULES IN ENSEMBLE                           #
        # ------------------------------------------------------------------------------------------------------------ #

    def create_molecule(self, spectra_molecule, indices):
        spectra_molecule.levelsNUM = self.levelsNUM
        spectra_molecule.energies = self.energies[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.gammaMATRIXpopd = self.gammaMATRIXpopd[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.gammaMATRIXdephasing = self.gammaMATRIXdephasing[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_frequencies = self.spectra_frequencies[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_freqDIM = len(self.spectra_frequencies[indices])
        spectra_molecule.muMATRIX = self.muMATRIX[indices].ctypes.data_as(POINTER(c_complex))
        spectra_molecule.spectra_field = self.spectra_field.ctypes.data_as(POINTER(c_complex))
        spectra_molecule.rho = self.rho[indices].ctypes.data_as(POINTER(c_complex))
        spectra_molecule.spectra_absTOTAL = self.spectra_absTOTAL[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_absDIST = self.spectra_absDIST[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_absREF = self.spectra_absREF[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.levelsVIBR = self.levelsVIBR[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.levels = self.levels[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.probabilities = self.probabilities[indices].ctypes.data_as(POINTER(c_double))
        return

    def create_parameters(self, spectra_parameters, variables):
        spectra_parameters.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        spectra_parameters.levelsNUM = self.levelsNUM
        spectra_parameters.excitedNUM = variables.excitedNUM
        spectra_parameters.spectra_time = self.spectra_time.ctypes.data_as(POINTER(c_double))
        spectra_parameters.spectra_timeAMP = variables.spectra_timeAMP
        spectra_parameters.spectra_timeDIM = len(self.spectra_time)
        spectra_parameters.spectra_fieldAMP = variables.spectra_fieldAMP
        spectra_parameters.threadNUM = variables.threadNUM
        spectra_parameters.ensembleNUM = variables.ensembleNUM
        spectra_parameters.guessLOWER = variables.guessLOWER.ctypes.data_as(POINTER(c_double))
        spectra_parameters.guessUPPER = variables.guessUPPER.ctypes.data_as(POINTER(c_double))
        spectra_parameters.iterMAX = variables.iterMAX
        return

    def fit_spectra(self, variables):
        spectra_parameters = SpectraParameters()
        self.create_parameters(spectra_parameters, variables)

        molENSEMBLE = [SpectraMolecule() for _ in range(variables.molNUM)]
        for molINDX in range(variables.molNUM):
            self.create_molecule(molENSEMBLE[molINDX], molINDX)
            CalculateSpectra(molENSEMBLE[molINDX], spectra_parameters)


class OFC:
    """
    Calculates the ofc response of the molecule
    """

    def __init__(self, ofc_variables, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.muMATRIX = np.ascontiguousarray(self.muMATRIX)

        self.energies = np.ascontiguousarray(self.energies)
        self.levelsNUM = ofc_variables.levelsNUM
        self.frequency, self.freq12, self.field1FREQ, self.field2FREQ, self.field3FREQ = nonuniform_frequency_range_3(ofc_variables)
        self.omega_chi = np.linspace(0.49 * ofc_variables.freqDEL * ofc_variables.combNUM, 0.85 * ofc_variables.freqDEL * ofc_variables.combNUM, ofc_variables.chiNUM)
        self.polarizationEMPTY = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationFIELD = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationINDEX = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationMOLECULE = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALEMPTY = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALFIELD = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALEMPTY_DIST = np.zeros((ofc_variables.basisNUM, ofc_variables.basisNUM,
                                                     ofc_variables.basisNUM, ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALFIELD_DIST = np.zeros((ofc_variables.basisNUM, ofc_variables.basisNUM,
                                                     ofc_variables.basisNUM, ofc_variables.molNUM, self.frequency.size),
                                                    dtype=np.complex)
        self.chi1DIST = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.omega_chi.size), dtype=np.complex)
        self.chi3DIST = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.omega_chi.size), dtype=np.complex)
        self.chi1INDEX = np.zeros((ofc_variables.molNUM, self.omega_chi.size), dtype=np.complex)
        self.chi3INDEX = np.zeros((ofc_variables.molNUM, self.omega_chi.size), dtype=np.complex)
        self.polINDX = np.empty((ofc_variables.basisNUM, ofc_variables.basisNUM, ofc_variables.basisNUM))
        self.basisINDX = np.empty(3, dtype=int)
        self.indices = np.empty(3, dtype=int)

        self.chi3MATRIX = np.empty((ofc_variables.molNUM, self.omega_chi.size, self.omega_chi.size), dtype=np.complex)

        # ------------------------------------------------------------------------------------------------------------ #
        #                       DECLARE NEW SET OF VARIABLES FOR ALL N MOLECULES IN ENSEMBLE                           #
        # ------------------------------------------------------------------------------------------------------------ #


    def create_ofc_molecule(self, ofc_molecule, indices):
        ofc_molecule.levelsNUM = self.levelsNUM
        ofc_molecule.energies = self.energies[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.levelsVIBR = self.levelsVIBR[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.levels = self.levels[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.gammaMATRIX = np.ascontiguousarray(self.gammaMATRIX[indices]).ctypes.data_as(POINTER(c_double))
        ofc_molecule.muMATRIX = self.muMATRIX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationINDEX = self.polarizationINDEX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationMOLECULE = self.polarizationMOLECULE[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.chi1DIST = self.chi1DIST[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.chi3DIST = self.chi3DIST[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.chi1INDEX = self.chi1INDEX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.chi3INDEX = self.chi3INDEX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.probabilities = self.probabilities[indices].ctypes.data_as(POINTER(c_double))
        return

    def create_ofc_parameters(self, ofc_parameters, ofc_variables):
        ofc_parameters.excitedNUM = ofc_variables.excitedNUM
        ofc_parameters.ensembleNUM = ofc_variables.ensembleNUM
        ofc_parameters.freqNUM = len(self.frequency)
        ofc_parameters.chiNUM = ofc_variables.chiNUM
        ofc_parameters.combNUM = ofc_variables.combNUM
        ofc_parameters.resolutionNUM = ofc_variables.resolutionNUM
        ofc_parameters.basisNUM = ofc_variables.basisNUM
        ofc_parameters.frequency = self.frequency.ctypes.data_as(POINTER(c_double))
        ofc_parameters.omega_chi = self.omega_chi.ctypes.data_as(POINTER(c_double))
        ofc_parameters.combGAMMA = ofc_variables.combGAMMA
        ofc_parameters.freqDEL = ofc_variables.freqDEL
        ofc_parameters.termsNUM = ofc_variables.termsNUM
        ofc_parameters.indices = self.indices.ctypes.data_as(POINTER(c_long))
        ofc_parameters.basisINDX = self.basisINDX.ctypes.data_as(POINTER(c_long))
        ofc_parameters.modulations = np.zeros(3, dtype=int).ctypes.data_as(POINTER(c_double))
        ofc_parameters.envelopeWIDTH = ofc_variables.envelopeWIDTH
        ofc_parameters.envelopeCENTER = ofc_variables.envelopeCENTER
        ofc_parameters.frequencyMC = ofc_variables.frequencyMC.ctypes.data_as(POINTER(c_double))
        return

    def calculate_ofc_system(self, ofc_variables):
        ofc_parameters = OFCParameters()
        self.create_ofc_parameters(ofc_parameters, ofc_variables)
        molENSEMBLE = [OFCMolecule() for _ in range(ofc_variables.molNUM)]

        basisRNG = int((ofc_parameters.basisNUM - (ofc_parameters.basisNUM % 2)) / 2)
        print(basisRNG)
        for I_ in range(-basisRNG, basisRNG + (ofc_parameters.basisNUM % 2)):
            ofc_parameters.basisINDX[0] = I_
            self.polINDX[I_] = I_
            for J_ in range(-basisRNG, basisRNG + (ofc_parameters.basisNUM % 2)):
                ofc_parameters.basisINDX[1] = J_
                self.polINDX[J_] = J_
                for K_ in range(-basisRNG, basisRNG + (ofc_parameters.basisNUM % 2)):
                    ofc_parameters.basisINDX[2] = K_
                    self.polINDX[K_] = K_
                    print(I_, J_, K_)
                    fig, ax = plt.subplots(nrows=ofc_variables.molNUM, ncols=2, sharex=True, sharey=True, figsize=(22, 11))
                    for molINDX in range(ofc_variables.molNUM):
                        self.create_ofc_molecule(molENSEMBLE[molINDX], molINDX)
                        for i, modulations in enumerate(list(product(*(3 * [[ofc_variables.omegaM1, ofc_variables.omegaM2, ofc_variables.omegaM3]])))):
                            if (i == 5) or (i == 11):
                                for mINDX, nINDX, vINDX in ofc_variables.modulationINDXlist:
                                    ofc_parameters.indices[0] = mINDX
                                    ofc_parameters.indices[1] = nINDX
                                    ofc_parameters.indices[2] = vINDX
                                    ofc_parameters.modulations = np.asarray(modulations).ctypes.data_as(POINTER(c_double))
                                    mu_product = self.muMATRIX[molINDX][0, mINDX] * self.muMATRIX[molINDX][mINDX, nINDX] * \
                                                 self.muMATRIX[molINDX][nINDX, vINDX] * self.muMATRIX[molINDX][vINDX, 0]
                                    self.polarizationMOLECULE[molINDX][:] = 0.
                                    CalculateOFCResponse(molENSEMBLE[molINDX], ofc_parameters)
                                    self.polarizationMOLECULE[molINDX] *= mu_product
                                    for ensembleINDX in range(ofc_variables.ensembleNUM):
                                        if (i == 5) or (i == 11):
                                            self.polarizationEMPTY[molINDX][ensembleINDX] += self.polarizationMOLECULE[molINDX][
                                                ensembleINDX]
                                        else:
                                            self.polarizationFIELD[molINDX][ensembleINDX] += self.polarizationMOLECULE[molINDX][
                                                ensembleINDX]

                        for ensembleINDX in range(ofc_variables.ensembleNUM):
                            self.polarizationTOTALEMPTY[molINDX] += (self.polarizationEMPTY[molINDX])[ensembleINDX]*self.probabilities[molINDX][ensembleINDX]
                            self.polarizationTOTALFIELD[molINDX] += (self.polarizationFIELD[molINDX])[ensembleINDX]*self.probabilities[molINDX][ensembleINDX]

                        self.polarizationTOTALEMPTY_DIST[I_][J_][K_][molINDX] = self.polarizationTOTALEMPTY[molINDX]
                        self.polarizationTOTALFIELD_DIST[I_][J_][K_][molINDX] = self.polarizationTOTALFIELD[molINDX]

                        if ofc_variables.molNUM > 1:
                            ax[molINDX, 0].plot(self.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALEMPTY[molINDX].real, 'r')
                            ax[molINDX, 0].plot(self.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALFIELD[molINDX].real, 'k')
                            ax[molINDX, 1].plot(self.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALEMPTY[molINDX].imag, 'r')
                            ax[molINDX, 1].plot(self.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALFIELD[molINDX].imag, 'k')
                            ax[molINDX, 0].grid()
                            ax[molINDX, 1].grid()
                        else:
                            ax[0].plot(self.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALEMPTY[molINDX].real, 'r')
                            ax[0].plot(self.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALFIELD[molINDX].real, 'k')
                            ax[1].plot(self.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALEMPTY[molINDX].imag, 'r')
                            ax[1].plot(self.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALFIELD[molINDX].imag, 'k')
                            ax[0].grid()
                            ax[1].grid()

                        # if np.abs(self.polarizationTOTALEMPTY).max() > 1e1:
                        #     plt.savefig("Plots4/" + str(I_) + str(J_) + str(K_) + "_500lines.png", dpi=fig.dpi)

                        self.polarizationFIELD *= 0.
                        self.polarizationEMPTY *= 0.
                    plt.close(fig)

    def calculate_susceptibilities(self, ofc_variables):
        ofc_parameters = OFCParameters()
        self.create_ofc_parameters(ofc_parameters, ofc_variables)
        molENSEMBLE = [OFCMolecule() for _ in range(ofc_variables.molNUM)]

        for molINDX in range(ofc_variables.molNUM):
            self.create_ofc_molecule(molENSEMBLE[molINDX], molINDX)
            CalculateChi(molENSEMBLE[molINDX], ofc_parameters)


if __name__ == '__main__':


    # --------------------------------------------------------- #
    #                       LIST OF CONSTANTS                   #
    # --------------------------------------------------------- #

    energyFACTOR = 1./27.211385
    timeFACTOR = 2.418884e-5
    wavelength2freqFACTOR = 1239.84
    cm_inv2evFACTOR = 1.23984e-4

    # ------------------------------------------------------------------------------------------ #
    #                       MOLECULAR CONSTANTS, VARIABLES, VECTORS & MATRICES                   #
    # ------------------------------------------------------------------------------------------ #

    molNUM = 3
    levelsNUM = 4
    ensembleNUM = 44
    groundNUM = 2
    excitedNUM = levelsNUM - groundNUM

    # ------------------ MOLECULAR ENERGY LEVEL STRUCTURE ------------------ #

    energies = np.empty((molNUM, levelsNUM))
    # levelMIN = [555, 558, 558]
    # levelMAX = [725, 735, 785]
    # wavelengthMIN = [539, 551, 551]
    # wavelengthMAX = [772, 819, 799]

    levelMIN = [412, 415, 415]
    levelMAX = [607, 619, 625]
    wavelengthMIN = [385, 401, 401]
    wavelengthMAX = [631, 633, 649]

    levels = [
        np.asarray(wavelength2freqFACTOR * energyFACTOR / np.linspace(levelMIN[i], levelMAX[i], excitedNUM * ensembleNUM)[::-1])
        for i in range(molNUM)
    ]

    vibrations = [1600, 1610, 1590, 1605, 1595]
    levelsVIBR = [np.asarray([0, vibrations[i]]) * energyFACTOR * cm_inv2evFACTOR for i in range(molNUM)]

    # ------------------------ INITIAL DENSITY MATRIX ---------------------- #

    rho_0 = np.zeros((levelsNUM, levelsNUM), dtype=np.complex)
    rho_0[0, 0] = 1 + 0j

    # ------------------ TRANSITION DIPOLE MOMENT AND DECAY PARAMETERS ------------------ #

    # MU = [2.2, 2.1, 2.15, 2.1, 2.25]
    # MUvibr = [0.12, 0.11, 0.10, 0.13, 0.09]
    #
    # gammaPOPD = [2.418884e-8, 2.518884e-8, 2.618884e-8, 2.718884e-8, 2.818884e-8]
    # gammaVIBR = [1.e-6, 1.2e-6, 1.5e-6, 1.15e-6, 1.45e-6]
    # gammaELEC = [2.1 * 2.518884e-4, 2.1 * 2.518884e-4, 2.1 * 2.618884e-4, 2.3 * 2.518884e-4, 2.15 * 2.618884e-4]

    MU = [2.0, 2.0, 2.0]
    MUvibr = [0.125, 0.115, 0.12]

    gammaPOPD = [2.418884e-8, 2.518884e-8, 2.618884e-8]
    gammaVIBR = [2.418884e-6, 2.518884e-6, 2.618884e-6]
    gammaELEC = [2.5 * 2.418884e-4, 2.5 * 2.518884e-4, 2.5 * 2.218884e-4]

    muMATRIX = [MUvibr[i] * np.ones((levelsNUM, levelsNUM), dtype=np.complex) for i in range(molNUM)]
    [np.fill_diagonal(muMATRIX[i], 0j) for i in range(molNUM)]
    for i in range(molNUM):
        np.fill_diagonal(muMATRIX[i], 0j)
        for j in range(groundNUM):
            for k in range(groundNUM, levelsNUM):
                muMATRIX[i][j, k] = MU[i]
                muMATRIX[i][k, j] = MU[i]

    gammaMATRIXpopd = [np.ones((levelsNUM, levelsNUM), dtype=np.float64) * gammaPOPD[i] for i in range(molNUM)]
    gammaMATRIXdephasing = [np.ones((levelsNUM, levelsNUM), dtype=np.float64) * gammaVIBR[i] for i in range(molNUM)]
    for i in range(molNUM):
        np.fill_diagonal(gammaMATRIXpopd[i], 0.0)
        gammaMATRIXpopd[i] = np.tril(gammaMATRIXpopd[i]).T
        np.fill_diagonal(gammaMATRIXdephasing[i], 0.0)
        for j in range(groundNUM):
            for k in range(groundNUM, levelsNUM):
                gammaMATRIXdephasing[i][j, k] = gammaELEC[i]
                gammaMATRIXdephasing[i][k, j] = gammaELEC[i]
    gammaMATRIX = gammaMATRIXdephasing[:]
    for k in range(molNUM):
        for n in range(levelsNUM):
            for m in range(levelsNUM):
                for i in range(levelsNUM):
                    gammaMATRIX[k][n][m] += 0.5 * (gammaMATRIXpopd[k][n][i] + gammaMATRIXpopd[k][m][i])
        np.fill_diagonal(gammaMATRIX[k], 0.0)

    for k in range(molNUM):
        gammaMATRIX[k][1:,1:] *= (1 + k * 0.5)

    # ------------------ SPECTRA FITTING PROBABILITIES  ------------------ #

    probabilities = np.empty((molNUM, ensembleNUM))
    probabilities = np.asarray(
        [
            [0.000235464, 0.0182477, 0.0445466, 0.0743497, 0.200194, 0.275919, 0.994308, 0.997843, 0.999919, 0.999657,
             0.999959, 0.800436, 0.660872, 0.591294, 0.5679, 0.541256, 0.515917, 0.494946, 0.460524, 0.39882, 0.268642,
             0.252768, 0.244494, 0.211459, 0.194288, 0.1699828, 0.158349, 0.1444209, 0.12989, 0.100931, 0.0811531,
             0.068017, 0.0476712, 0.0398955, 0.025205, 0.0222609, 0.0202314, 0.0179509, 0.0159138, 0.0140906, 0.0127108,
             0.0111656, 0.0102622, 0.0088874],
            [0.000159222, 0.0184829, 0.0775252, 0.0812718, 0.284982, 0.445154, 0.738436, 0.997581, 0.999909, 0.99964,
             0.999588, 0.835973, 0.702272, 0.605137, 0.552009, 0.551918, 0.5559, 0.538397, 0.554753, 0.453616, 0.413365,
             0.307081, 0.260385, 0.233048, 0.196534, 0.1895426, 0.158587, 0.1232462, 0.0930239, 0.0746957, 0.0458443,
             0.0333323, 0.0223485, 0.0233593, 0.01514, 0.0236354, 0.0114312, 0.0558953, 0.0244014, 0.0318158, 0.0259083,
             0.0167475, 0.0203759, 0.0156996],
            [0.00015465, 0.0187052, 0.0747483, 0.0788058, 0.434415, 0.449858, 0.971228, 0.997552, 0.999914, 0.998754,
             0.998763, 0.647207, 0.611017, 0.617428, 0.619939, 0.607858, 0.588308, 0.562265, 0.524471, 0.435686,
             0.395995, 0.312465, 0.25622, 0.237325, 0.212928, 0.2045875, 0.180568, 0.1288332, 0.104029, 0.0863363,
             0.0628899, 0.0482691, 0.0392197, 0.0228059, 0.0151043, 0.0137039, 0.0114739, 0.0556456, 0.0444313,
             0.031833, 0.0254178, 0.0183261, 0.0195045, 0.0157216]
        ]
    )

    # for i in range(molNUM):
    #     probabilities[i] = probabilities[2]

    # poly = np.polyfit(np.arange(ensembleNUM), probabilities[0], 5)
    # probabilities[0] = np.poly1d(poly)(np.arange(ensembleNUM))
    #
    # for i in range(molNUM):
    #     probabilities[i] = probabilities[0] / (1.05 ** i)
    #
    # for i in range(len(probabilities[0])):
    #     for k in range(molNUM):
    #         probabilities[k][i] *= np.random.normal(1, .025)

    guessLOWER = np.zeros(ensembleNUM)
    guessUPPER = np.ones(ensembleNUM)

    # ---------------------------------------------------------------------------------------------------------------- #
    #              READ csv-DATA FILES INTO WAVELENGTH & ABSORPTION MATRICES: (SIZE) N x wavelengthNUM                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    data_protein_files_list = ['DataFP/mSCARLET.csv', 'DataFP/FusionRED.csv', 'DataFP/mCHERRY.csv', 'DataFP/mCHERRY.csv', 'DataFP/mCHERRY.csv']
    protein_plot_colors = ['r', 'b', 'k']

    wavelengthNUM = 100
    wavelengths = np.empty([molNUM, wavelengthNUM])
    absorptions = np.empty_like(wavelengths)
    frequencies = np.empty_like(wavelengths)
    # plt.figure()
    for i in range(molNUM):
        wavelengths[i], absorptions[i] = get_experimental_spectra(data_protein_files_list[i], wavelengthMIN[i], wavelengthMAX[i], wavelengthNUM)
        frequencies[i] = wavelength2freqFACTOR * energyFACTOR / wavelengths[i]
        # plt.plot(wavelengths[i], absorptions[i])


    # -------------------------------------------#
    #              OFC PARAMETERS                #
    # -------------------------------------------#

    combNUM = 3000
    resolutionNUM = 3
    omegaM1 = 0.45 * timeFACTOR * 10 / 5.
    omegaM2 = 0.73 * timeFACTOR * 10 / 5.
    omegaM3 = 0.08 * timeFACTOR * 10 / 5.
    freqDEL = 1.10 * timeFACTOR * 10 / 5.
    combGAMMA = 1e-10 * timeFACTOR
    termsNUM = 3
    envelopeWIDTH = 5000
    envelopeCENTER = 2500
    chiNUM = 10000

    rangeFREQ = np.asarray([0.35, 0.65])

    SystemArgs = dict(
        gammaMATRIXpopd=gammaMATRIXpopd,
        gammaMATRIXdephasing=gammaMATRIXdephasing,
        gammaMATRIX=gammaMATRIX,
        muMATRIX=muMATRIX,
        energies=energies,
        levelsVIBR=levelsVIBR,
        levels=levels,
        probabilities=probabilities,
        spectra_wavelengths=np.ascontiguousarray(wavelengths),
        spectra_frequencies=np.ascontiguousarray(frequencies),
        spectra_absREF=np.ascontiguousarray(absorptions),
    )

    SystemVars = ADict(
        molNUM=molNUM,
        levelsNUM=levelsNUM,
        excitedNUM=excitedNUM,
        ensembleNUM=ensembleNUM,
        threadNUM=cpu_count(),
        rho_0=rho_0,
        spectra_timeAMP=5000,
        spectra_timeDIM=1000,
        spectra_fieldAMP=8e-6,
        guessLOWER=guessLOWER,
        guessUPPER=guessUPPER,
        iterMAX=1,
        combNUM=combNUM,
        basisNUM=1,
        resolutionNUM=resolutionNUM,
        omegaM1=omegaM1,
        omegaM2=omegaM2,
        omegaM3=omegaM3,
        combGAMMA=combGAMMA,
        freqDEL=freqDEL,
        termsNUM=termsNUM,
        envelopeWIDTH=envelopeWIDTH,
        envelopeCENTER=envelopeCENTER,
        # modulationINDXlist=[(3, 1, 2)],
        modulationINDXlist=[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1),],
        chiNUM=chiNUM,
        frequencyMC=np.random.uniform(400., 650., (chiNUM, 3)),
        rangeFREQ=rangeFREQ
    )

    # start = time.time()

    # system = Spectra(SystemVars, **SystemArgs)
    # system.fit_spectra(SystemVars)
    # fig, ax = plt.subplots(nrows=molNUM, ncols=1, figsize=(6, 6), sharex=True)
    # for i in range(molNUM):
        # ax[i].plot(system.spectra_wavelengths[i], system.spectra_absTOTAL[i], protein_plot_colors[i], linestyle='--', label="theoretical fit")

    # print('TIME ELAPSED FOR SPECTRA CALCULATION:', time.time() - start, 'seconds')

    # ----------------------------------------------------------------------------------------------------------------#
    #                                             PLOT ABSORPTION SPECTRA FIT                                         #
    # ----------------------------------------------------------------------------------------------------------------#

    # for i in range(molNUM):
    #     ax[i].plot(wavelengths[i], absorptions[i], protein_plot_colors[i],
    #                label=data_protein_files_list[i][7:-4] + " spectra")

    start = time.time()

    # ---------------------------------------------------------------------------------------------------------------- #
    #                  MONTE-CARLO DETERMINATION OF CHI(1) AND CHI(3) CORRELATIONS BETWEEN MOLECULES                   #
    # ---------------------------------------------------------------------------------------------------------------- #
    if True:
        system = OFC(SystemVars, **SystemArgs)
        system.calculate_susceptibilities(SystemVars)
        fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
        abs = np.empty((molNUM, len(system.omega_chi)))
        for i in range(molNUM):
            abs[i] = (system.probabilities[i][:ensembleNUM].T.dot(system.chi1DIST[i])).imag
        max_abs = abs.max()
        for i in range(molNUM):
            abs[i] /= max_abs / 100.
            ax.plot(energyFACTOR * wavelength2freqFACTOR / system.omega_chi, abs[i], protein_plot_colors[i],
                       linestyle='-', label="Mol " + str(i + 1))
            render_axis(ax, gridLINE='-')
            ax.legend()
            ax.set_ylabel('Normalised \n absorption', fontsize='x-large')

        plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=0.0)
        ax.set_xlabel('Wavelength (in $nm$)', fontsize='x-large')
        del system

    if True:
        start = time.time()
        omegaM1array = np.asarray([omegaM1 / timeFACTOR]) * timeFACTOR
        omegaM2array = np.asarray([omegaM2 / timeFACTOR]) * timeFACTOR

        for i in range(len(omegaM1array)):
            for j in range(len(omegaM2array)):
                SystemVars.omegaM1 = omegaM1array[i]
                SystemVars.omegaM2 = omegaM2array[j]
                SystemVars.omegaM3 = SystemVars.omegaM1 + SystemVars.omegaM2 - SystemVars.freqDEL

                data = np.around(np.asarray([SystemVars.omegaM1, SystemVars.omegaM2, SystemVars.omegaM3, SystemVars.freqDEL]) / timeFACTOR, 2)
                print(data)
                system = OFC(SystemVars, **SystemArgs)
                system.calculate_ofc_system(SystemVars)

                fig_pol3, ax_pol3 = plt.subplots(nrows=molNUM, ncols=2, sharex=True, sharey=True)
                field1, field2, field3 = plot_field_pol_params(system, SystemVars, rangeFREQ)

                print(time.time() - start)

                for molINDX in range(molNUM):
                    system.polarizationTOTALEMPTY[molINDX] *= MU[molINDX]*MU[molINDX]*MUvibr[molINDX]*MUvibr[molINDX]
                    system.polarizationTOTALFIELD[molINDX] *= MU[molINDX]*MU[molINDX]*MUvibr[molINDX]*MUvibr[molINDX]

                polMAX = [max(system.polarizationTOTALEMPTY[_].real.max(),
                              system.polarizationTOTALEMPTY[_].imag.max()) for _ in range(molNUM)]

                alphas = [0.45, 0.6, 0.75]
                if molNUM > 1:
                    for molINDX in range(molNUM):
                        for axINDX in range(2):
                            ax_pol3[molINDX, axINDX].plot(system.field1FREQ / (timeFACTOR * 2 * np.pi), field1 * 100 / field1.max(), 'g')
                            ax_pol3[molINDX, axINDX].plot(system.field2FREQ / (timeFACTOR * 2 * np.pi), field2 * 100 / field2.max(), 'y')
                            ax_pol3[molINDX, axINDX].plot(system.field3FREQ / (timeFACTOR * 2 * np.pi), field3 * 100 / field3.max(), 'm', alpha=0.4)
                            render_axis(ax_pol3[molINDX, axINDX], labelSIZE='xx-large')
                        ax_pol3[molINDX, 0].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALEMPTY[molINDX].real * 100 / max(np.abs(polMAX)), 'k', linewidth=1., alpha=alphas[molINDX])
                        ax_pol3[molINDX, 1].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALEMPTY[molINDX].imag * 100 / max(np.abs(polMAX)), 'k', linewidth=1., alpha=alphas[molINDX])
                        # ax_pol3[molINDX, 0].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALFIELD[molINDX].real, 'b', linewidth=1., alpha=0.7)
                        # ax_pol3[molINDX, 1].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALFIELD[molINDX].imag, 'b', linewidth=1., alpha=0.7)
                        ax_pol3[molINDX][0].set_ylabel("$Re[P^{(3)}(\omega)]$ -- Mol " + str(molINDX + 1), fontsize='x-large')
                        ax_pol3[molINDX][1].set_ylabel("$Im[P^{(3)}(\omega)]$ -- Mol " + str(molINDX + 1), fontsize='x-large')
                    ax_pol3[2, 0].set_xlabel("Frequency (in THz)")
                    ax_pol3[2, 1].set_xlabel("Frequency (in THz)")
                else:
                    ax_pol3[0].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALEMPTY[0].real, 'r', linewidth=1., alpha=0.7)
                    ax_pol3[1].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALEMPTY[0].imag, 'b', linewidth=1., alpha=0.7)

                plt.savefig('Plots/' + str(round(SystemVars.omegaM1 / timeFACTOR, 2)) +
                            '_' + str(round(SystemVars.omegaM2 / timeFACTOR, 2)) +
                            '_' + str(round(SystemVars.omegaM3 / timeFACTOR, 2)) + '_2.png')


        with open('Pickle/P3.pickle', 'wb') as f:
            pickle.dump(
                {
                    "pol3field": system.polarizationTOTALFIELD,
                    "pol3empty": system.polarizationTOTALEMPTY,
                    "field1FREQ": system.field1FREQ,
                    "field2FREQ": system.field2FREQ,
                    "field3FREQ": system.field3FREQ,
                    "frequency": system.frequency,
                    "field1": field1,
                    "field2": field2,
                    "field3": field3
                },
                f)

        with open('Pickle/pol3_args.pickle', 'wb') as f:
            pickle.dump(
                {
                    "combNUM": combNUM,
                    "resolutionNUM": resolutionNUM,
                    "omegaM1": omegaM1,
                    "omegaM2": omegaM2,
                    "omegaM3": omegaM3,
                    "freqDEL": freqDEL,
                    "combGAMMA": combGAMMA,
                    "termsNUM": termsNUM,
                    "envelopeWIDTH": envelopeWIDTH,
                    "envelopeCENTER": envelopeCENTER,
                    "chiNUM": chiNUM,
                    "rangeFREQ": rangeFREQ
                },
                f)
    plt.show()
