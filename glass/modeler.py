import numpy as np
from scipy.optimize import minimize, differential_evolution

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

### SIE model functions

def psi_SIE(x, theta, b, q, phi_PA=0.0, q_circular_tol=1e-6): 
    """
    Dimensionless lensing potential of a Singular Isothermal Ellipsoid (SIE),
    Kormann, Schneider & Bartelmann (1994) convention.

    kappa(x1, x2) = b / (2 * xi),   xi = sqrt(q^2 x1^2 + x2^2)

    psi = x1*alpha1 + x2*alpha2   (exact identity: kappa is homogeneous of
    degree -1 in x, so psi is homogeneous of degree +1 -> Euler's theorem
    gives psi = x . grad(psi) = x . alpha, with NO extra additive/log term
    needed for the singular, non-cored SIE.)

    Parameters
    ----------
    x : float
        Radial coordinate in the lens plane (dimensionless, x = theta_sky/theta_0).
        Scalar -- matches how f_z_asymmetric calls psi(x, theta) at fixed z/x.
    theta : float or ndarray
        Azimuthal angle(s) in the SKY frame (same frame as source_angle),
        radians. Must broadcast, since quad_vec batches evaluation points.
    b : float
        SIE normalization (lensing strength). For q=1 this reduces to the SIS
        Einstein radius: psi_SIS = b * x.
    q : float
        Projected axis ratio, 0 < q <= 1 (q=1 -> circular / SIS).
    phi_PA : float
        Position angle of the lens major axis in the SAME frame as `theta`
        and `source_angle`, radians. This is NOT something to rotate away --
        it is the physical orientation of the elliptical lens relative to
        the fixed source direction.
    q_circular_tol : float
        Below this value of sqrt(1-q^2), fall back to the exact SIS formula
        to avoid 0/0 numerical noise (the qp -> 0 limit is smooth
        analytically, but arctan/qp and arctanh/qp lose precision near qp=0).

    Returns
    -------
    psi : ndarray (or float), same shape as theta
    """
    theta = np.asarray(theta, dtype=float)

    if x == 0.0:
        # xi = 0 at the origin; psi(0, theta) = 0 identically regardless of
        # theta (elliptical radius vanishes), avoids 0/0 in alpha below.
        return np.zeros_like(theta)

    theta_l = theta - phi_PA  # rotate into the lens (major-axis) frame

    x1 = x * np.cos(theta_l)
    x2 = x * np.sin(theta_l)
    xi = np.sqrt(q**2 * x1**2 + x2**2)

    qp = np.sqrt(max(1.0 - q**2, 0.0))

    if qp < q_circular_tol:
        # SIS limit (q -> 1): alpha_i = b * x_i / xi
        alpha1 = b * x1 / xi
        alpha2 = b * x2 / xi
    else:
        alpha1 = (b / qp) * np.arctan(qp * x1 / xi)
        alpha2 = (b / qp) * np.arctanh(np.clip(qp * x2 / xi, -1.0 + 1e-14, 1.0 - 1e-14))

    psi = x1 * alpha1 + x2 * alpha2
    return psi


def fermat_potential_cartesian(xvec, y, source_angle, psi, psi_kwargs=None):
    """
    T(x, y) = 1/2 |x - y|^2 - psi(x, theta), evaluated in Cartesian
    coordinates so the optimizer never has to deal with the polar
    coordinate singularity at the origin.

    xvec : (x1, x2) Cartesian lens-plane position (what the optimizer varies)
    y, source_angle : polar source position, same convention as
        f_z_asymmetric(z, w, y, phi_m, psi, source_angle=...)
    psi : callable psi(x, theta, **psi_kwargs) -- your existing convention
          (x = radius, theta = angle; must return psi(0, theta) sanely,
          e.g. 0, for any lens potential regular enough at the origin)
    """
    psi_kwargs = psi_kwargs or {}
    x1, x2 = xvec
    y1 = y * np.cos(source_angle)
    y2 = y * np.sin(source_angle)

    x = float(np.hypot(x1, x2))
    if x < 1e-12:
        theta = 0.0
    else:
        theta = float(np.arctan2(x2, x1))

    psi_val = float(np.asarray(psi(x, theta, **psi_kwargs)))
    return 0.5 * ((x1 - y1) ** 2 + (x2 - y2) ** 2) - psi_val


def find_phi_m(
    y,
    psi,
    source_angle=0.0,
    psi_kwargs=None,
    search_radius=5.0,
    polish_method="Nelder-Mead",
    de_kwargs=None,
    return_diagnostics=False,
    n_diagnostic_starts=60,
    dedupe_tol=1e-4,
    seed=0,
):
    """
    Numerically find phi_m(y) = -T_min(y) for a general lens potential
    psi(x, theta), where T_min is the Fermat potential at the global
    minimum-time image.

    Strategy
    --------
    1. Global search over the lens plane with differential_evolution
       (gradient-free -- robust even if psi comes from an FFT/interpolated
       map rather than a smooth analytic formula, and doesn't assume the
       image configuration is known in advance).
    2. Local polish of the best point found, for numerical precision.
    3. Optionally, a multi-start battery of local minimizations purely as
       a diagnostic, to sanity-check the image count / catch cases where
       the global search missed a deeper basin (increase search_radius
       or n_diagnostic_starts if this disagrees with step 1-2).

    Returns
    -------
    dict with keys:
        'phi_m'   : the value to pass as phi_m(y) in your pipeline
        'T_min'   : Fermat potential at the minimum-time image
        'x_min'   : (x1, x2) position of that image
        'diagnostics' : (if requested) all distinct local minima found,
                         sorted by T value -- image 0 should match x_min above
    """
    psi_kwargs = psi_kwargs or {}
    de_kwargs = de_kwargs or {}

    def T(v):
        return fermat_potential_cartesian(v, y, source_angle, psi, psi_kwargs)

    bounds = [(-search_radius, search_radius)] * 2

    de_defaults = dict(seed=seed, tol=1e-10, polish=False, maxiter=300, popsize=20)
    de_defaults.update(de_kwargs)
    de_res = differential_evolution(T, bounds, **de_defaults)

    polish_opts = (
        {"xatol": 1e-11, "fatol": 1e-13, "maxiter": 5000}
        if polish_method == "Nelder-Mead"
        else {}
    )
    polish_res = minimize(T, de_res.x, method=polish_method, options=polish_opts)

    result = {
        "phi_m": -polish_res.fun,
        "T_min": polish_res.fun,
        "x_min": tuple(polish_res.x),
    }

    if return_diagnostics:
        rng = np.random.default_rng(seed)
        starts = rng.uniform(-search_radius, search_radius, size=(n_diagnostic_starts, 2))
        found = []
        for s in starts:
            r = minimize(T, s, method=polish_method, options=polish_opts)
            if not r.success:
                continue
            pt = r.x
            if not any(np.hypot(pt[0] - f[0][0], pt[1] - f[0][1]) < dedupe_tol for f in found):
                found.append((pt, r.fun))
        found.sort(key=lambda t: t[1])
        result["diagnostics"] = found

    return result