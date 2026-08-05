import numpy as np

"""
Nr = 256
Ntheta = 512
r_max = 3.0
r = np.linspace(0, r_max, Nr)
theta = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)
R, Theta = np.meshgrid(r, theta, indexing="ij")

"""


def reconstruct_circ_hm(M, r, theta, f):
    
    Nr = len(r)
    Ntheta = len(theta)
    
    # Decompose in circular harmonics up to order M
    # store coefficients a_m(r) for each m and r
    a_m = {m: np.zeros(Nr, dtype=complex) for m in range(-M, M + 1)}
    for i in range(Nr):
        fi = f[i, :]  
        for m in range(-M, M + 1):
            a_m[m][i] = np.mean(fi * np.exp(-1j * m * theta))

    # Reconstruct the field from coefficients
    f_rec = np.zeros_like(f, dtype=complex)
    for m in range(-M, M + 1):
        f_rec += a_m[m][:, None] * np.exp(1j * m * theta[None, :])

    f_rec = f_rec.real  # original field is real
    
    return f_rec