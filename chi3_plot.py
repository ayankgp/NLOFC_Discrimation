import numpy as np
import matplotlib.pyplot as plt


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
resolutionNUM = 3

combNUM = 500
freqDEL = 1.00 * timeFACTOR * 1.05

omegaM1 = 0.59 * timeFACTOR * 1.05
omegaM2 = 0.83 * timeFACTOR * 1.05
omegaM3 = 0.42 * timeFACTOR * 1.05

# ------------------ MOLECULAR ENERGY LEVEL STRUCTURE ------------------ #

energies = np.empty((molNUM, levelsNUM))
levelMIN = [555, 575, 595, 615, 635]
levelMAX = [725, 735, 745, 755, 765]
wavelengthMIN = [539, 551, 551, 551, 551]
wavelengthMAX = [772, 819, 799, 799, 799]
levels = [
    np.asarray(
        wavelength2freqFACTOR * energyFACTOR / np.linspace(levelMIN[i], levelMAX[i], excitedNUM * ensembleNUM)[::-1])
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
    gammaMATRIX[k][1:, 1:] *= (1 + k * 0.5)

def nonuniform_frequency_range_3():

    pointsFREQpolarization = np.linspace(combNUM * freqDEL, combNUM * freqDEL, combNUM, endpoint=False)[:, np.newaxis]
    pointsFREQcomb = np.linspace(combNUM * freqDEL, combNUM * freqDEL, combNUM, endpoint=False)[:, np.newaxis]
    resolution = np.linspace(-0.02 * freqDEL, 0.02 * freqDEL, resolutionNUM)

    frequency_123 = params.omegaM1 + params.omegaM2 - params.omegaM3 + pointsFREQpolarization + resolution
    # frequency_21 = 2 * params.omegaM1 - params.omegaM2 + pointsFREQpolarization + resolution

    field1FREQ = params.omegaM1 + pointsFREQcomb + resolution
    field2FREQ = params.omegaM2 + pointsFREQcomb + resolution

    frequency = np.sort(np.hstack([frequency_123.flatten()]))
    # frequency = np.sort(np.hstack([frequency_12.flatten(), frequency_21.flatten()]))
    field1FREQ = np.ascontiguousarray(field1FREQ.flatten())
    field2FREQ = np.ascontiguousarray(field2FREQ.flatten())

    return frequency, frequency_123.flatten()


omega1 =
def chi3(w1, w2, w3):

