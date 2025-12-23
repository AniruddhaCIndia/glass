import numpy as np
from scipy.integrate import quad
from scipy.special import j0 

def f_z(z, w, y, phi_model=None, psi_model=None):
    if psi_model is None:
        psi_model = lambda x: 0.0
    if phi_model is None:
        phi_model = lambda y: 0.0
        
    x = np.sqrt(2.0 * z)
    phase_prefactor = np.exp(1j * w * (0.5 * y**2 + phi_model(y)))
    bessel_part = j0(w * y * x)
    lens_phase = np.exp(-1j * w * psi_model(x))
    return -1j * w * phase_prefactor * bessel_part * lens_phase

def F_segment_0_to_b(w, y, b, limit=1000, z_epsabs=1e-6, z_epsrel=1e-6, phi_model=None, psi_model=None):
    integrand_real = lambda z: np.real(f_z(z, w, y, phi_model, psi_model) * np.exp(1j * w * z))
    integrand_imag = lambda z: np.imag(f_z(z, w, y, phi_model, psi_model) * np.exp(1j * w * z))

    real_part, _ = quad(integrand_real, 0.0, b,
                        limit=limit, epsabs=z_epsabs, epsrel=z_epsrel)
    imag_part, _ = quad(integrand_imag, 0.0, b,
                        limit=limit, epsabs=z_epsabs, epsrel=z_epsrel)

    return real_part + 1j * imag_part

def tail_asymptotic_1term(w, y, b, **kwargs):
    fb = f_z(b, w, y, **kwargs)
    return - fb * np.exp(1j * w * b) / (1j * w)

def fprime_z(z, w, y, h=5e-3, **kwargs):
    return (f_z(z + h, w, y, **kwargs) - f_z(z - h, w, y, **kwargs)) / (2*h)

def tail_asymptotic_2term(w, y, b, h= 5e-3, **kwargs):
    fb = f_z(b, w, y, **kwargs)
    fpb = fprime_z(b, w, y, h, **kwargs)
    phase = np.exp(1j * w * b)
    return (-fb / (1j * w) + fpb / (1j * w)**2) * phase

def amplification_F_t1(w, y, **kwargs):
    b = max(100.0, 3.0 * w)  # example heuristic
    F_main = F_segment_0_to_b(w, y, b, **kwargs)
    F_tail1 = tail_asymptotic_1term(w, y, b, **kwargs)
    return F_main + F_tail1

def amplification_F_t2(w, y, **kwargs):
    b = max(100.0, 3.0 * w)  # example heuristic
    F_main = F_segment_0_to_b(w, y, b, **kwargs)
    F_tail2 = tail_asymptotic_2term(w, y, b, **kwargs)
    return F_main + F_tail2