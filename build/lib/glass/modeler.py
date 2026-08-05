import numpy as np

### SIS model functions
def phi_m_sis(y):
    return y + 1/2

def psi_sis(x):
    return x

### PML model functions
def phi_m_pml(y):
    x_m = (y + np.sqrt(y**2 + 4))/2
    return 1/2 * (x_m - y)**2  - np.log(x_m)

def psi_pml(x):
    return np.log(x)

### Asymmetric SIS model functions
def psi_sis_asym(x, theta, epsilon=0.1):
    return np.sqrt((x)**2) * (1 + epsilon * np.cos(theta))