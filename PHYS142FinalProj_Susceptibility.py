from multiprocessing import Pool, cpu_count
import itertools
import numpy as np
import matplotlib.pyplot as plt
# import tqdm


N = 100
J = 1
KB = 1

n_temp, n_field = 51, 2
temps = np.linspace(0.01,5.01,n_temp)
B_s = np.linspace(-0.05,0.05,n_field)

lattice_spins = np.ones((N, N))
burnin_up = 100_000
steps = 500_000

def MCMC(lattice_spins, temp, steps, B):
    m_values = []
    t = 0
    while t < steps:
        i, j = np.random.randint(N), np.random.randint(N)
        # we only need to consider the neighbors of 
        # (i, j) to calculate the change in energy
        delta_energy = 0
        for k, l in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            i_neigh = i + k if i + k < N else 0
            j_neigh = j + l if j + l < N else 0
            delta_energy += ( (-J * -2 * lattice_spins[i, j] * lattice_spins[i_neigh, j_neigh]) - (B * lattice_spins[i, j]) )
        if delta_energy <= 0:
            lattice_spins[i, j] *= -1
        elif delta_energy > 0:
            prob = np.exp(-delta_energy / (KB * temp))
            if np.random.random() < prob:
                lattice_spins[i, j] *= -1
        else: 
            continue
        m_values.append(np.mean(lattice_spins))
        t += 1
    return m_values

def for_multi_processing(item):
    m_values = np.array(MCMC(lattice_spins, item[0], steps, item[1]))
    m_mean = np.mean(m_values[burnin_up:])
    return m_mean

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        m_mean_list = pool.map(for_multi_processing, itertools.product(temps, B_s), 1)

    m_mean_list = np.array(m_mean_list).reshape(n_temp, n_field)
    chi_list = (m_mean_list[:,1]-m_mean_list[:,0])/0.1
    print("Plotting...")

    # fig, ax = plt.subplots()
    # img = ax.imshow(chi_list,cmap="YlGnBu")
    # plt.colorbar(img, label=r"$\chi$")
    # ax.invert_yaxis()
    # ax.set_xlabel(r"$B$")
    # ax.set_ylabel(r"$T$")
    # plt.hlines(2.269*5, 0, 20, colors="tab:orange",linestyles=":")
    # plt.vlines(10,0, 20, colors="tab:orange",linestyles="--")
    # plt.text(0.2, 10., "T = 2.269", color="tab:orange")
    # plt.text(6.8, 0.2, "B = 0", color="tab:orange")
    # ax.set_xticks(np.arange(0, n_field,5), np.round(B_s,1)[::5])
    # ax.set_yticks(np.arange(0, n_temp,5), np.round(temps,1)[::5])
    # plt.title(r"Susceptibility in $B$ vs $T$")
    # # plt.savefig("phase_diagram", dpi=600)
    plt.plot(temps,-chi_list)
    plt.show()
