import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

N_comb = 60
N_sys = 3
f_comb = np.arange(N_comb)
z = np.random.uniform(0, .25, (N_sys, N_comb))
r = np.random.uniform(0., 1., N_comb)
weights = np.asarray([0.2, 0.5, 0.3])

for i in range(N_sys):
    z[i, :] += r
z_new = np.empty_like(z)

for i in range(N_sys):
    z_new[i] = z[i] * np.exp(-((f_comb - int(N_comb / 2))**2 / (2 * (N_comb / 8) ** 2)))

fig, ax = plt.subplots(nrows=N_sys + 1, ncols=1, sharex=True, sharey=True)

for j in range(N_sys):
    for i in range(N_comb):
        ax[j].vlines(i, ymin=0, ymax=z_new[j][i], color='b')

    f = interpolate.interp1d(f_comb, z_new[j], kind='cubic')
    x_new = np.linspace(0., N_comb-1, 1000)
    y_new = f(x_new)

    ax[j].plot(x_new, y_new, color='k', linewidth=0.7)

    ax[j].plot(f_comb, np.zeros_like(z_new[j]), color='k', linewidth=0.5)
    ax[j].axis('off')

z_new = z_new * weights[:, np.newaxis]
for i in range(N_comb):
    ax[N_sys].vlines(i, ymin=0, ymax=z_new.sum(axis=0)[i], color='r')

f = interpolate.interp1d(f_comb, z_new.sum(axis=0), kind='cubic')
x_new = np.linspace(0., N_comb-1, 1000)
y_new = f(x_new)

ax[N_sys].plot(x_new, y_new, color='k', linewidth=1.25)
# ax[N_sys].fill(x_new, y_new, color='r', linewidth=0.9, alpha=0.3)

ax[N_sys].plot(f_comb, np.zeros_like(z_new.sum(axis=0)), color='k', linewidth=1.25)
ax[N_sys].axis('off')

plt.show()