import numpy as np
from astropy.constants import G, c, M_sun
from astropy import units as u
from astropy.cosmology import Planck18


def point_lens_times_and_magnifications(M_lz_solar, y):
    """
    Compute image arrival times and magnifications for a point-mass lens.

    This function evaluates the two geometric-optics images produced by a
    point gravitational lens using the standard dimensionless lens equation.

    Parameters
    ----------
    M_lz_solar : float
        Redshifted lens mass in units of solar masses, i.e.
        M_lz = M_l * (1 + z_l), expressed as a scalar multiple of M_sun.

    y : float
        Dimensionless source position (impact parameter) in units of the
        Einstein angle. Can be positive or negative; the image positions and
        magnifications depend only on |y|.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``x_plus`` : float
            Dimensionless position of the positive-parity image.
        - ``x_minus`` : float
            Dimensionless position of the negative-parity image.
        - ``t_plus`` : `~astropy.units.Quantity`
            Arrival time of the positive-parity image, shifted so that the
            earliest image has time 0. Units: seconds.
        - ``t_minus`` : `~astropy.units.Quantity`
            Arrival time of the negative-parity image, shifted so that the
            earliest image has time 0. Units: seconds.
        - ``mu_plus`` : float
            Absolute magnification of the positive-parity image.
        - ``mu_minus`` : float
            Absolute magnification of the negative-parity image.
        - ``mu_total`` : float
            Total magnification, ``mu_plus + mu_minus``.
        - ``delta_t`` : `~astropy.units.Quantity`
            Differential time delay between the two images:
            ``t_later - t_earlier``. Always non-negative. Units: seconds.

    Notes
    -----
    The dimensionless image positions for a point lens are

    .. math::

        x_\\pm = \\frac{1}{2}\\left(y \\pm \\sqrt{y^2 + 4}\\right)

    where this implementation uses ``|y|`` since the observable magnifications
    and delays depend only on the magnitude of the source offset.

    The Fermat potential (dimensionless time-delay function) is

    .. math::

        T(x, y) = \\frac{1}{2}(x - y)^2 - \\ln |x|.

    The physical time delay is

    .. math::

        t = \\frac{4 G M_{lz}}{c^3} T(x, y),

    where ``M_lz`` is the redshifted lens mass.

    Important
    ---------
    At ``y = 0`` the point-source magnification diverges formally, so this
    function raises a ``ValueError`` for that case.

    Examples
    --------
    >>> out = point_lens_times_and_magnifications(100.0, 1.0)
    >>> out["mu_total"]
    1.3416407864998738
    >>> out["delta_t"].to(u.ms)
    <Quantity ... ms>
    """
    # -------------------------
    # Input validation
    # -------------------------
    if M_lz_solar <= 0:
        raise ValueError("M_lz_solar must be positive.")

    if y == 0:
        raise ValueError(
            "y = 0 gives formally divergent point-source magnifications "
            "for a point lens. Use a nonzero y."
        )

    y_abs = float(abs(y))
    sqrt_term = np.sqrt(y_abs**2 + 4.0)

    # Image positions (dimensionless)
    x_plus = 0.5 * (y_abs + sqrt_term)
    x_minus = 0.5 * (y_abs - sqrt_term)

    # Signed magnifications of the two images
    mu_plus_signed = 0.5 + (y_abs**2 + 2.0) / (2.0 * y_abs * sqrt_term)
    mu_minus_signed = 0.5 - (y_abs**2 + 2.0) / (2.0 * y_abs * sqrt_term)

    # Observable magnifications are absolute values
    mu_plus = abs(mu_plus_signed)
    mu_minus = abs(mu_minus_signed)
    mu_total = mu_plus + mu_minus

    # Redshifted lens mass as an astropy Quantity
    M_lz = M_lz_solar * M_sun

    # Characteristic time scale: 4GM/c^3
    t0 = (4.0 * G * M_lz / c**3).to(u.s)

    def T(x, y0):
        """
        Dimensionless Fermat potential for a point lens.
        """
        return 0.5 * (x - y0)**2 - np.log(np.abs(x))

    # Raw arrival times as astropy Quantities
    t_plus_raw = t0 * T(x_plus, y_abs)
    t_minus_raw = t0 * T(x_minus, y_abs)

    # Shift so that the earliest image arrives at t = 0
    t_earliest = min(t_plus_raw, t_minus_raw)
    t_plus = (t_plus_raw - t_earliest).to(u.s)
    t_minus = (t_minus_raw - t_earliest).to(u.s)

    # Positive differential delay between the two images
    delta_t = np.abs(t_minus_raw - t_plus_raw).to(u.s)

    return {
        "x_plus": x_plus,
        "x_minus": x_minus,
        "t_plus": t_plus,
        "t_minus": t_minus,
        "mu_plus": mu_plus,
        "mu_minus": mu_minus,
        "mu_total": mu_total,
        "delta_t": delta_t,
    }



def sis_lens_times_and_magnifications(sigma_v, y, z_l, z_s, cosmology=Planck18):
    """
    Compute image arrival times and magnifications for a Singular Isothermal
    Sphere (SIS) gravitational lens.

    This function returns the image positions, absolute magnifications, and
    relative arrival times for an SIS lens in the geometric-optics limit.

    Parameters
    ----------
    sigma_v : `~astropy.units.Quantity`
        One-dimensional velocity dispersion of the SIS lens.
        Must have velocity units, e.g. ``200 * u.km / u.s``.

    y : float
        Dimensionless source position in units of the Einstein angle.
        Can be positive or negative; the observable magnifications and time
        delays depend only on ``|y|``.

    z_l : float
        Lens redshift.

    z_s : float
        Source redshift. Must satisfy ``z_s > z_l``.

    cosmology : `~astropy.cosmology.Cosmology`, optional
        Cosmology object used to compute angular-diameter distances.
        Default is ``astropy.cosmology.Planck18``.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``x_plus`` : float
            Dimensionless position of the outer image.
        - ``x_minus`` : float or None
            Dimensionless position of the inner image if it exists,
            otherwise ``None``.
        - ``t_plus`` : `~astropy.units.Quantity`
            Arrival time of the outer image, shifted so that the earliest
            image has time 0. Units: seconds.
        - ``t_minus`` : `~astropy.units.Quantity` or None
            Arrival time of the inner image if it exists, shifted so that
            the earliest image has time 0. Units: seconds. ``None`` if there
            is no second image.
        - ``mu_plus`` : float
            Absolute magnification of the outer image.
        - ``mu_minus`` : float
            Absolute magnification of the inner image if it exists,
            otherwise 0.
        - ``mu_total`` : float
            Total magnification.
        - ``delta_t`` : `~astropy.units.Quantity`
            Differential time delay between the two images:
            ``t_later - t_earlier``. Zero if only one image exists.
        - ``theta_E`` : `~astropy.units.Quantity`
            Einstein angle in radians.
        - ``n_images`` : int
            Number of images (1 or 2).
        - ``D_l`` : `~astropy.units.Quantity`
            Angular-diameter distance to the lens.
        - ``D_s`` : `~astropy.units.Quantity`
            Angular-diameter distance to the source.
        - ``D_ls`` : `~astropy.units.Quantity`
            Angular-diameter distance from lens to source.

    Notes
    -----
    For an SIS lens, the lens equation in dimensionless variables is

    .. math::

        y = x - \\operatorname{sgn}(x),

    where ``x = theta/theta_E`` and ``y = beta/theta_E``.

    The Einstein angle is

    .. math::

        \\theta_E = 4\\pi \\left(\\frac{\\sigma_v}{c}\\right)^2
        \\frac{D_{LS}}{D_S}.

    The physical arrival time is

    .. math::

        t(\\theta) = \\frac{1+z_L}{c}\\frac{D_L D_S}{D_{LS}}
        \\left[ \\frac{1}{2}(\\theta-\\beta)^2 - \\psi(\\theta) \\right].

    Writing ``theta = x theta_E`` and ``beta = y theta_E``, we get

    .. math::

        t(x) = t_0 \\, T(x,y),

    where

    .. math::

        t_0 = \\frac{1+z_L}{c}\\frac{D_L D_S}{D_{LS}}\\theta_E^2

    and for the SIS lens

    .. math::

        T(x,y) = \\frac{1}{2}(x-y)^2 - |x|.

    Image structure
    ---------------
    - If ``|y| < 1``: two images
    - If ``|y| >= 1``: one image

    For ``|y| < 1`` the image positions are

    .. math::

        x_+ = y + 1, \\qquad x_- = y - 1.

    Important
    ---------
    At ``y = 0`` the point-source SIS magnification diverges formally
    (Einstein ring), so this function raises a ``ValueError``.

    Examples
    --------
    >>> from astropy import units as u
    >>> out = sis_lens_times_and_magnifications(
    ...     sigma_v=220 * u.km / u.s,
    ...     y=0.3,
    ...     z_l=0.5,
    ...     z_s=2.0
    ... )
    >>> out["mu_total"]
    6.666666666666667
    >>> out["delta_t"].to(u.day)
    <Quantity ... d>
    """
    # -------------------------
    # Input validation
    # -------------------------
    if not isinstance(sigma_v, u.Quantity):
        raise TypeError("sigma_v must be an astropy Quantity with velocity units.")
    if not sigma_v.unit.is_equivalent(u.m / u.s):
        raise u.UnitConversionError("sigma_v must have velocity units.")
    if sigma_v <= 0 * sigma_v.unit:
        raise ValueError("sigma_v must be positive.")

    if z_l < 0 or z_s < 0:
        raise ValueError("z_l and z_s must be non-negative.")
    if z_s <= z_l:
        raise ValueError("Source redshift must satisfy z_s > z_l.")

    if y == 0:
        raise ValueError(
            "y = 0 gives formally divergent point-source magnification "
            "for an SIS lens (Einstein ring). Use a nonzero y."
        )

    y_abs = float(abs(y))

    # -------------------------
    # Angular-diameter distances
    # -------------------------
    D_l = cosmology.angular_diameter_distance(z_l)
    D_s = cosmology.angular_diameter_distance(z_s)
    D_ls = cosmology.angular_diameter_distance_z1z2(z_l, z_s)

    # Einstein angle
    theta_E = (
        4.0
        * np.pi
        * (sigma_v / c) ** 2
        * (D_ls / D_s)
    ).to(u.dimensionless_unscaled) * u.rad

    # Overall time scale
    t0 = (((1.0 + z_l) / c) * (D_l * D_s / D_ls) * theta_E**2).to(u.s)

    def T(x, y0):
        """
        Dimensionless Fermat potential for an SIS lens.
        """
        return 0.5 * (x - y0) ** 2 - np.abs(x)

    # -------------------------
    # Image positions and magnifications
    # -------------------------
    # Outer image always exists
    x_plus = y_abs + 1.0

    if y_abs < 1.0:
        # Two-image regime
        x_minus = y_abs - 1.0

        # Signed magnifications (for y > 0 convention)
        mu_plus_signed = 1.0 + 1.0 / y_abs
        mu_minus_signed = 1.0 - 1.0 / y_abs

        mu_plus = abs(mu_plus_signed)
        mu_minus = abs(mu_minus_signed)
        mu_total = mu_plus + mu_minus

        # Raw arrival times
        t_plus_raw = t0 * T(x_plus, y_abs)
        t_minus_raw = t0 * T(x_minus, y_abs)

        # Shift so earliest image arrives at t = 0
        t_earliest = min(t_plus_raw, t_minus_raw)
        t_plus = (t_plus_raw - t_earliest).to(u.s)
        t_minus = (t_minus_raw - t_earliest).to(u.s)

        # Physical differential delay
        delta_t = np.abs(t_minus_raw - t_plus_raw).to(u.s)

        n_images = 2

    else:
        # One-image regime
        x_minus = None

        mu_plus_signed = 1.0 + 1.0 / y_abs
        mu_plus = abs(mu_plus_signed)
        mu_minus = 0.0
        mu_total = mu_plus

        t_plus = 0.0 * u.s
        t_minus = None
        delta_t = 0.0 * u.s

        n_images = 1

    return {
        "x_plus": x_plus,
        "x_minus": x_minus,
        "t_plus": t_plus,
        "t_minus": t_minus,
        "mu_plus": mu_plus,
        "mu_minus": mu_minus,
        "mu_total": mu_total,
        "delta_t": delta_t,
    }