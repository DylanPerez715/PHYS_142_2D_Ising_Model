#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ## Part A

# In[2]:


STEPS = 200_000
N = 100
J = 1
KB = 1
T = 2

#Configure as either all spin up or all random
lattice_spins = np.ones((N, N))
#lattice_spin_random = 2 * (np.random.randint(2, size=(N, N)) - 0.5)



        
plt.figure()
plt.imshow(lattice_spins, cmap="YlGnBu")
for i in range(N):
    plt.axhline(i + 0.5, color="black", lw=0.1)
    plt.axvline(i + 0.5, color="black", lw=0.1)
    
plt.colorbar()
plt.clim(-1, 1)
plt.show()


# In[3]:


num_accept=0

def Hastings (B, T ,lattice, steps,shape):
    
    m_values=[]
    num_accept=0
    for t in tqdm.tqdm(range(steps)):
        i, j = np.random.randint(shape), np.random.randint(shape)
   
        delta_energy = 0
        for k, l in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            i_neigh = i + k if i + k < shape else 0
            j_neigh = j + l if j + l < shape else 0
            delta_energy += ((-J * -2 * lattice[i, j] * lattice[i_neigh, j_neigh]) - (B * -2 * lattice[i,j]))
        if delta_energy <= 0:
            lattice[i, j] *= -1
            num_accept += 1
        elif delta_energy > 0:
            prob = np.exp(-delta_energy / (T))
            if np.random.random() < prob:
                lattice[i, j] *= -1
                num_accept += 1
        m_values.append(np.mean(lattice))
        
    return lattice, m_values



# In[4]:


#Example of algorithm and graph of lattice
temp_1, mean_1 =Hastings(-1,1,lattice_spins, 200_000, 100)


plt.figure()
plt.imshow(temp_1, cmap="YlGnBu")
plt.colorbar()
plt.clim(-1, 1)

for i in range(N):
    plt.axhline(i + 0.5, color="black", lw=0.1)
    plt.axvline(i + 0.5, color="black", lw=0.1)
plt.show()


# In[5]:


#Example of algorithm and graph of lattice
temp_6, mean_6 =Hastings(-1,1.5,lattice_spins, 200_000, 100)


plt.figure()
plt.imshow(temp_6, cmap="YlGnBu")
plt.colorbar()
plt.clim(-1, 1)

for i in range(N):
    plt.axhline(i + 0.5, color="black", lw=0.1)
    plt.axvline(i + 0.5, color="black", lw=0.1)
plt.show()


# In[6]:


#Example of algorithm and graph of lattice
temp_5, mean_5 =Hastings(-1,2,lattice_spins, 200_000, 100)


plt.figure()
plt.imshow(temp_5, cmap="YlGnBu")
plt.colorbar()
plt.clim(-1, 1)

for i in range(N):
    plt.axhline(i + 0.5, color="black", lw=0.1)
    plt.axvline(i + 0.5, color="black", lw=0.1)
plt.show()



# In[7]:


#Example of algorithm and graph of lattice
temp_2, mean_2=Hastings(-1,4,lattice_spins,200_000,100)

plt.figure()
plt.imshow(temp_2, cmap="YlGnBu")
plt.colorbar()
plt.clim(-1, 1)

for i in range(N):
    plt.axhline(i + 0.5, color="black", lw=0.1)
    plt.axvline(i + 0.5, color="black", lw=0.1)
plt.show()


# In[ ]:





# In[8]:


#Example of algorithm and graph of lattice
temp_3, mean_3=Hastings(1,6,lattice_spins,200_000,100)

plt.figure()
plt.imshow(temp_3, cmap="YlGnBu")
plt.colorbar()
plt.clim(-1, 1)

for i in range(N):
    plt.axhline(i + 0.5, color="black", lw=0.1)
    plt.axvline(i + 0.5, color="black", lw=0.1)
plt.show()


# In[9]:


#Example of algorithm and graph of lattice
temp_4, mean_4=Hastings(1,15,lattice_spins,200_000,100)

plt.figure()
plt.imshow(temp_4, cmap="YlGnBu")
plt.colorbar()
plt.clim(-1, 1)
plt.title(f"B=1, T=15")
for i in range(N):
    plt.axhline(i + 0.5, color="black", lw=0.1)
    plt.axvline(i + 0.5, color="black", lw=0.1)
plt.show()


# In[ ]:




