import numpy as np
from scipy.integrate import quad


def quad_complex(fun, a, b, **kwargs):
    real_part = quad(lambda t: np.real(fun(t)), a, b, **kwargs)[0]
    imag_part = quad(lambda t: np.imag(fun(t)), a, b, **kwargs)[0]
    return real_part + 1j*imag_part

def f_z_asymmetric(z, w, y, psi_model= None , phi_model= None , **kwargs):
    
    if psi_model is None:
        psi_model = lambda x, theta: 0.0
    if phi_model is None:
        phi_model = lambda y: 0.0
        
    if z <= 0:
        return 0.0 + 0.0j
    
    x = np.sqrt(2.0 * z)
    prefactor = (w / (2.0 * np.pi * 1j)) * np.exp(1j * w * (0.5 * y**2 + phi_model(y)))

    def theta_integrand(theta):
        return np.exp(-1j * w * (y * x * np.cos(theta) + psi_model(x, theta)))

    theta_integral = quad_complex(theta_integrand, 0.0, 2.0 * np.pi,
                                  epsabs=kwargs.get('theta_epsabs', 1e-6), 
                                  epsrel=kwargs.get('theta_epsrel', 1e-6), 
                                  limit=kwargs.get('limit_theta', 1000))
    return prefactor * theta_integral

# from functools import lru_cache
# @lru_cache(None)
# def f_cached(z, w, y, **kwargs):
#     return f_z_asymmetric(z, w, y, psi_model= None , phi_model= None , **kwargs)

def F_segment_0_to_b(w, y, b, **kwargs):
    
    def z_integrand(z):
        return f_z_asymmetric(z, w, y, **kwargs) * np.exp(1j * w * z)

    return quad_complex(z_integrand, 0.0, b, epsabs=kwargs.get('z_epsabs', 1e-6), 
                        epsrel=kwargs.get('z_epsrel', 1e-6), 
                        limit=kwargs.get('limit_z', 1000))

def tail_asymptotic_1term(w, y, b, **kwargs):
    
    fb = f_z_asymmetric(b, w, y, **kwargs)
    return - fb * np.exp(1j * w * b) / (1j * w)

def tail_asymptotic_2term(w, y, b, h=1e-10, **kwargs):  ### h = 5e-3
    
    fb = f_z_asymmetric(b, w, y, **kwargs)
    
    fpb = (f_z_asymmetric(b + h, w, y, **kwargs) - 
           f_z_asymmetric(b - h, w, y, **kwargs)) / (2.0 * h)
    
    #fpb = np.imag(f_z_asymmetric(b + 1j*h, w, y, **kwargs)) / h
    
    phase = np.exp(1j * w * b)
    series = ( -1j * w * fb + fpb ) / (1j * w)**2
    return phase * series


def amplification_asym_1term(w, y, b, **kwargs):
    F0b_sample = F_segment_0_to_b(w, y, b, **kwargs)
    F_tail_1 = tail_asymptotic_1term(w, y, b, **kwargs)
    
    return F0b_sample + F_tail_1

def amplification_asym_2term(w, y, b, h=1e-10, **kwargs):  ### h = 5e-3
    F0b_sample = F_segment_0_to_b(w, y, b, **kwargs)
    F_tail_2 = tail_asymptotic_2term(w, y, b, h = h, **kwargs)
    
    return F0b_sample + F_tail_2

