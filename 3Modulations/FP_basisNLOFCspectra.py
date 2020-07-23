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
        self.omega_chi = np.linspace(0.425 * ofc_variables.freqDEL * ofc_variables.combNUM, 0.675 * ofc_variables.freqDEL * ofc_variables.combNUM, ofc_variables.chiNUM)
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
        ofc_molecule.gammaMATRIX = np.ascontiguousarray(self.gammaMATRIX).ctypes.data_as(POINTER(c_double))
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
    levelMIN = [555, 575, 595, 615, 635]
    levelMAX = [725, 735, 745, 755, 765]
    wavelengthMIN = [539, 551, 551, 551, 551]
    wavelengthMAX = [772, 819, 799, 799, 799]
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

    MU = [2.2, 2.1, 2.15, 2.1, 2.25]
    MUvibr = [0.12, 0.11, 0.10, 0.13, 0.09]

    gammaPOPD = [2.418884e-8, 2.518884e-8, 2.618884e-8, 2.718884e-8, 2.818884e-8]
    gammaVIBR = [1.e-6, 1.2e-6, 1.5e-6, 1.15e-6, 1.45e-6]
    gammaELEC = [2.1 * 2.518884e-4, 2.25 * 2.518884e-4, 2.4 * 2.618884e-4, 2.3 * 2.518884e-4, 2.15 * 2.618884e-4]
    muMATRIX = [MUvibr[i] * np.ones((levelsNUM, levelsNUM), dtype=np.complex) for i in range(molNUM)]
    [np.fill_diagonal(muMATRIX[i], 0j) for i in range(molNUM)]
    for i in range(molNUM):
        np.fill_diagonal(muMATRIX[i], 0j)
        for j in range(groundNUM):
            for k in range(groundNUM, levelsNUM):
                muMATRIX[i][j, k] = MU[i]
                muMATRIX[i][k, j] = MU[i]

    gammaMATRIXpopd = [np.ones((levelsNUM, levelsNUM)) * gammaPOPD[i] for i in range(molNUM)]
    gammaMATRIXdephasing = [np.ones((levelsNUM, levelsNUM)) * gammaVIBR[i] for i in range(molNUM)]
    for i in range(molNUM):
        np.fill_diagonal(gammaMATRIXpopd[i], 0.0)
        gammaMATRIXpopd[i] = np.tril(gammaMATRIXpopd[i]).T
        np.fill_diagonal(gammaMATRIXdephasing[i], 0.0)
        for j in range(groundNUM):
            for k in range(groundNUM, levelsNUM):
                gammaMATRIXdephasing[i][j, k] = gammaELEC[i]
                gammaMATRIXdephasing[i][k, j] = gammaELEC[i]
    gammaMATRIX = gammaMATRIXdephasing
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
    probabilities[0] = np.asarray(
        [0.0118298, 0.0720250, 0.1321300, 0.2457230, 0.3622290, 0.6249910, 0.9031510, 0.9662930, 0.9996000, 0.9999550,
          0.9999910, 0.9982230, 0.9798070, 0.8255240, 0.7010270, 0.6480470, 0.6335650, 0.5958800, 0.5436750, 0.5033890,
          0.4499950, 0.4540130, 0.3828920, 0.3446280, 0.3148250, 0.2910750, 0.2724920, 0.2350670, 0.1673730, 0.1316270,
          0.1646670, 0.1737700, 0.1258730, 0.0889188, 0.0899126, 0.0872398, 0.0561983, 0.0698249, 0.0540158, 0.0505829,
          0.0502181, 0.0496727, 0.0405529, 0.0559383]
    )

    # poly = np.polyfit(np.arange(ensembleNUM), probabilities[0], 5)
    # probabilities[0] = np.poly1d(poly)(np.arange(ensembleNUM))
    #
    for i in range(molNUM):
        probabilities[i] = probabilities[0] / (1.05 ** i)

    for i in range(len(probabilities[0])):
        for k in range(molNUM):
            probabilities[k][i] *= np.random.normal(1, .025)

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
    for i in range(molNUM):
        wavelengths[i], absorptions[i] = get_experimental_spectra(data_protein_files_list[i], wavelengthMIN[i], wavelengthMAX[i], wavelengthNUM)
        frequencies[i] = wavelength2freqFACTOR * energyFACTOR / wavelengths[i]

    # -------------------------------------------#
    #              OFC PARAMETERS                #
    # -------------------------------------------#

    combNUM = 5000
    resolutionNUM = 3
    omegaM1 = 0.59 * timeFACTOR * 1.05
    omegaM2 = 0.83 * timeFACTOR * 1.05
    omegaM3 = 0.32 * timeFACTOR * 1.05
    freqDEL = 1.10 * timeFACTOR * 1.05
    combGAMMA = 1e-10 * timeFACTOR
    termsNUM = 3
    envelopeWIDTH = 100000
    envelopeCENTER = 0
    chiNUM = 10000

    rangeFREQ = np.asarray([-1., 1.4])

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
        basisNUM=2,
        resolutionNUM=resolutionNUM,
        omegaM1=omegaM1,
        omegaM2=omegaM2,
        omegaM3=omegaM3,
        combGAMMA=combGAMMA,
        freqDEL=freqDEL,
        termsNUM=termsNUM,
        envelopeWIDTH=envelopeWIDTH,
        envelopeCENTER=envelopeCENTER,
        modulationINDXlist=[(3, 1, 2)],
        # modulationINDXlist=[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1),],
        chiNUM=chiNUM,
        frequencyMC=np.random.uniform(550., 750., (chiNUM, 3)),
        rangeFREQ=rangeFREQ
    )

    start = time.time()

    # ---------------------------------------------------------------------------------------------------------------- #
    #                  MONTE-CARLO DETERMINATION OF CHI(1) AND CHI(3) CORRELATIONS BETWEEN MOLECULES                   #
    # ---------------------------------------------------------------------------------------------------------------- #
    if True:
        system = OFC(SystemVars, **SystemArgs)
        system.calculate_susceptibilities(SystemVars)
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.plot(energyFACTOR * wavelength2freqFACTOR / system.omega_chi, np.asarray([np.abs(system.probabilities[i].T.dot(system.chi1DIST[i]))
                                                                                     for i in range(molNUM)]).real.T, linestyle='-.')
        # ax.set_xlim(600, 800)
        render_axis(ax, labelSIZE='xx-large')
        for i in range(3):
            system.chi3DIST[i] *= MU[i] * MU[i] * MUvibr[i] * MUvibr[i]
        np.set_printoptions(precision=3)
        for i in range(molNUM):
            print(system.gammaMATRIX[i], '\n \n')
        print(np.corrcoef(np.asarray([np.abs(system.probabilities[i].T.dot(system.chi1DIST[i])) for i in range(molNUM)])), '\n \n')
        print(np.corrcoef(np.asarray([np.abs(system.probabilities[i].T.dot(system.chi3DIST[i])) for i in range(molNUM)])), '\n \n')
        print(time.time() - start)
        del system

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

            fig_pol3, ax_pol3 = plt.subplots(nrows=molNUM, ncols=2, sharex=True)
            fig_pol3.suptitle(f"{data}")
            field1, field2, field3 = plot_field_pol_params(system, SystemVars, rangeFREQ)

            print(time.time() - start)

            for molINDX in range(molNUM):
                system.polarizationTOTALEMPTY[molINDX] *= MU[molINDX]*MU[molINDX]*MUvibr[molINDX]*MUvibr[molINDX]
                system.polarizationTOTALFIELD[molINDX] *= MU[molINDX]*MU[molINDX]*MUvibr[molINDX]*MUvibr[molINDX]

            polMAX = [max(system.polarizationTOTALEMPTY[_].real.max(),
                          system.polarizationTOTALEMPTY[_].imag.max()) for _ in range(molNUM)]

            if molNUM > 1:
                for molINDX in range(molNUM):
                    for axINDX in range(2):
                        ax_pol3[molINDX, axINDX].plot(system.field1FREQ / (timeFACTOR * 2 * np.pi), field1 * max(polMAX) / field1.max(), 'g')
                        ax_pol3[molINDX, axINDX].plot(system.field2FREQ / (timeFACTOR * 2 * np.pi), field2 * max(polMAX) / field2.max(), 'y')
                        ax_pol3[molINDX, axINDX].plot(system.field3FREQ / (timeFACTOR * 2 * np.pi), field3 * max(polMAX) / field3.max(), 'm', alpha=0.4)
                    ax_pol3[molINDX, 0].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALEMPTY[molINDX].real, 'k', linewidth=1., alpha=0.7)
                    ax_pol3[molINDX, 1].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALEMPTY[molINDX].imag, 'k', linewidth=1., alpha=0.7)
                    # ax_pol3[molINDX, 0].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALFIELD[molINDX].real, 'b', linewidth=1., alpha=0.7)
                    # ax_pol3[molINDX, 1].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALFIELD[molINDX].imag, 'b', linewidth=1., alpha=0.7)

            else:
                ax_pol3[0].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALEMPTY[0].real, 'r', linewidth=1., alpha=0.7)
                ax_pol3[1].plot(system.frequency / (timeFACTOR * 2 * np.pi), system.polarizationTOTALEMPTY[0].imag, 'b', linewidth=1., alpha=0.7)

            plt.savefig('Plots/' + str(round(SystemVars.omegaM1 / timeFACTOR, 2)) +
                        '_' + str(round(SystemVars.omegaM2 / timeFACTOR, 2)) +
                        '_' + str(round(SystemVars.omegaM3 / timeFACTOR, 2)) + '_2.png')


    with open('pol3.pickle', 'wb') as f:
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

    with open('pol3_args.pickle', 'wb') as f:
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
