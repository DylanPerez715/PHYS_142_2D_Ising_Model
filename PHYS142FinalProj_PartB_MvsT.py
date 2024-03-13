from multiprocessing import Pool, cpu_count
import itertools
import numpy as np
import matplotlib.pyplot as plt
# import tqdm


N = 100
J = 1
KB = 1

n_temp, n_field = 21,3
temps = np.linspace(0.01,4.01,n_temp)
B_s = [-1, 0, 1]

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
    m_values = MCMC(lattice_spins, item[0], steps, item[1])
    m_mean = np.mean(m_values[burnin_up:])
    m_std = np.std(m_values[burnin_up:])
    return m_mean, m_std

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        results = pool.map(for_multi_processing, itertools.product(temps, B_s), 1)

    stacked_results = np.stack(results)
    m_means = stacked_results[:,0].reshape((n_temp,n_field))
    m_stds = stacked_results[:,1].reshape((n_temp,n_field))
    
    print("Plotting...")

    fig, ax = plt.subplots()
    for ind,b in enumerate(B_s):
        ax.plot(temps, m_means[:,ind], label=f'B = {b}')
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$M$")
    plt.title('1D Scan of Magnetization (M) versus Temperature (T) for different B values')
    plt.legend()
    plt.savefig("MvsT",dpi=600)
    plt.show()
