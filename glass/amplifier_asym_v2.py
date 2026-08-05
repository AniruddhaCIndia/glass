import math
import numpy as np
from scipy.integrate import quad_vec
import warnings

def quad_complex_vec(fun, a, b, return_error=False, **kwargs):
    """
    Integrate a complex scalar function by passing its real and imaginary
    components together to scipy.integrate.quad_vec.
    """
    def vector_fun(t):
        value = fun(t)
        return np.array([value.real, value.imag])

    result, error = quad_vec(vector_fun, a, b, **kwargs)
    value = result[0] + 1j * result[1]

    if return_error:
        return value, error
    return value

def f_z_asymmetric(z, w, y, source_angle=0.0, theta_epsabs=1e-8, theta_epsrel=1e-8):
    if z < 0.0:
        return 0.0 + 0.0j

    x = math.sqrt(2.0 * z)

    prefactor = (
        w / (2.0 * math.pi * 1j)
        * np.exp(1j * w * (0.5 * y**2 + phi_m(y)))
    )

    def theta_integrand(theta):
        return np.exp(
            -1j * w * (y * x * np.cos(theta-source_angle) + psi(x, theta))
        )

    theta_integral = quad_complex_vec(
        theta_integrand,
        0.0,
        2.0 * math.pi,
        epsabs=theta_epsabs,
        epsrel=theta_epsrel,
        limit=1000,
    )

    return prefactor * theta_integral

def F_segment_0_to_b(
    w,
    y,
    b,
    z_epsabs=1e-8,
    z_epsrel=1e-8,
):
    def integrand(z):
        return f_z_asymmetric(z, w, y) * np.exp(1j * w * z)

    return quad_complex_vec(
        integrand,
        0.0,
        b,
        epsabs=z_epsabs,
        epsrel=z_epsrel,
        limit=1000,
    )
    
def f_prime_fd_5pt(b, w, y, h=5e-3):
    """
    Five-point central finite-difference approximation to df/dz at b.

    Error is O(h^4), assuming f is sufficiently smooth near b.
    """
    if b - 2.0 * h < 0.0:
        raise ValueError("Need b > 2*h for the five-point stencil.")

    return (
        f_z_asymmetric(b - 2.0 * h, w, y)
        - 8.0 * f_z_asymmetric(b - h, w, y)
        + 8.0 * f_z_asymmetric(b + h, w, y)
        - f_z_asymmetric(b + 2.0 * h, w, y)
    ) / (12.0 * h)
    
def f_second_fd_5pt(b, w, y, h=5e-3):
    return (
        -f_z_asymmetric(b - 2*h, w, y)
        + 16.0 * f_z_asymmetric(b - h, w, y)
        - 30.0 * f_z_asymmetric(b, w, y)
        + 16.0 * f_z_asymmetric(b + h, w, y)
        - f_z_asymmetric(b + 2*h, w, y)
    ) / (12.0 * h**2)
    
def f_prime_fd_7pt(b, w, y, h=5e-3):
    return (
        -f_z_asymmetric(b - 3*h, w, y)
        + 9*f_z_asymmetric(b - 2*h, w, y)
        - 45*f_z_asymmetric(b - h, w, y)
        + 45*f_z_asymmetric(b + h, w, y)
        - 9*f_z_asymmetric(b + 2*h, w, y)
        + f_z_asymmetric(b + 3*h, w, y)
    ) / (60*h)
    
def f_second_fd_7pt(b, w, y, h=5e-3):
    return (
        2*f_z_asymmetric(b - 3*h, w, y)
        - 27*f_z_asymmetric(b - 2*h, w, y)
        + 270*f_z_asymmetric(b - h, w, y)
        - 490*f_z_asymmetric(b, w, y)
        + 270*f_z_asymmetric(b + h, w, y)
        - 27*f_z_asymmetric(b + 2*h, w, y)
        + 2*f_z_asymmetric(b + 3*h, w, y)
    ) / (180*h**2)
    
def tail_terms_2term(w, y, b, h=5e-3):
    """
    Returns:
        term_1 = -exp(iwb) f(b)/(iw)
        term_2 =  exp(iwb) f'(b)/(iw)^2
    """
    f_b = f_z_asymmetric(b, w, y)
    fp_b = f_prime_fd_5pt(b, w, y, h=h)

    phase = np.exp(1j * w * b)

    term_1 = phase * (-f_b / (1j * w))
    term_2 = phase * (fp_b / (1j * w)**2)

    return term_1, term_2

def tail_asymptotic_1term(w, y, b):
    f_b = f_z_asymmetric(b, w, y)
    return np.exp(1j * w * b) * (-f_b / (1j * w))


def tail_asymptotic_2term(w, y, b, h=5e-3):
    term_1, term_2 = tail_terms_2term(w, y, b, h=h)
    return term_1 + term_2

def tail_asymptotic_3term(w, y, b, h=5e-3):
    f_b = f_z_asymmetric(b, w, y)
    fp_b = f_prime_fd_5pt(b, w, y, h=h)
    fpp_b = f_second_fd_5pt(b, w, y, h=h)

    phase = np.exp(1j * w * b)

    return phase * (
        -f_b / (1j * w)
        + fp_b / (1j * w)**2
        - fpp_b / (1j * w)**3
    )

def converged_F(
    w,
    y,
    b_values,
    h=5e-3,
    rtol=1e-4,
    atol=1e-8,
):
    """
    Compute one- and two-term-tail results over several cutoffs.

    A cutoff b is accepted when:
      1. The two-term result is stable against the next larger cutoff.
      2. The one- and two-term results agree within tolerance.

    Returns a dictionary containing the selected result and diagnostics.
    """
    b_values = np.asarray(b_values, dtype=float)

    if len(b_values) < 2:
        raise ValueError("Provide at least two increasing b values.")

    if np.any(b_values <= 0.0):
        raise ValueError("All b values must be positive.")

    if np.any(np.diff(b_values) <= 0.0):
        raise ValueError("b_values must be strictly increasing.")

    results = []

    for b in b_values:
        F_segment = F_segment_0_to_b(w, y, b)

        tail_1, tail_2_correction = tail_terms_2term(w, y, b, h=h)

        F_1term = F_segment + tail_1
        F_2term = F_segment + tail_1 + tail_2_correction

        results.append({
            "b": b,
            "F_segment": F_segment,
            "F_1term": F_1term,
            "F_2term": F_2term,
            "tail_1": tail_1,
            "tail_2_correction": tail_2_correction,
        })

    diagnostics = []

    for i in range(len(results) - 1):
        current = results[i]
        next_result = results[i + 1]

        F1 = current["F_1term"]
        F2 = current["F_2term"]
        F2_next = next_result["F_2term"]

        cutoff_difference = abs(F2_next - F2)
        tail_order_difference = abs(F2 - F1)

        scale = max(abs(F2), abs(F2_next))
        tolerance = atol + rtol * scale

        cutoff_converged = cutoff_difference <= tolerance
        order_converged = tail_order_difference <= tolerance

        diagnostic = {
            "b": current["b"],
            "next_b": next_result["b"],
            "F_1term": F1,
            "F_2term": F2,
            "cutoff_difference": cutoff_difference,
            "tail_order_difference": tail_order_difference,
            "tolerance": tolerance,
            "cutoff_converged": cutoff_converged,
            "order_converged": order_converged,
        }

        diagnostics.append(diagnostic)

        if cutoff_converged and order_converged:
            return {
                "converged": True,
                "b_selected": current["b"],
                "F": F2,
                "absF": abs(F2),
                "diagnostics": diagnostics,
                "all_results": results,
            }

    return {
        "converged": False,
        "b_selected": None,
        "F": None,
        "absF": None,
        "diagnostics": diagnostics,
        "all_results": results,
    }
    
def converged_F_v2(
    w,
    y,
    b_values,
    h=5e-3,
    rtol=1e-4,
    atol=1e-8,
):
    """
    Compute F using one- and two-term tail corrections for increasing b.

    Convergence requires:
      1. The two-term F is stable between adjacent b values.
      2. The second tail term is smaller than the first.

    Parameters
    ----------
    w, y : float
        Frequency and source position.
    b_values : sequence of float
        Strictly increasing radial cutoffs; doubling is recommended.
    h : float
        Step size for the five-point derivative.
    rtol, atol : float
        Relative and absolute cutoff-convergence tolerances.
    """
    b_values = np.asarray(b_values, dtype=float)

    if len(b_values) < 2:
        raise ValueError("Provide at least two increasing b values.")

    if np.any(b_values <= 0.0):
        raise ValueError("All b values must be positive.")

    if np.any(np.diff(b_values) <= 0.0):
        raise ValueError("b_values must be strictly increasing.")

    results = []

    for b in b_values:
        if b <= 2.0 * h:
            raise ValueError(
                f"b={b} is too small for the five-point derivative with h={h}."
            )

        F_segment = F_segment_0_to_b(w, y, b)

        tail_1, tail_2_correction = tail_terms_2term(
            w, y, b, h=h
        )

        F_1term = F_segment + tail_1
        F_2term = F_1term + tail_2_correction

        results.append({
            "b": b,
            "F_segment": F_segment,
            "tail_1": tail_1,
            "tail_2_correction": tail_2_correction,
            "F_1term": F_1term,
            "F_2term": F_2term,
        })

    diagnostics = []

    for i in range(len(results) - 1):
        current = results[i]
        next_result = results[i + 1]

        F2 = current["F_2term"]
        F2_next = next_result["F_2term"]

        tail_1_size = abs(current["tail_1"])
        tail_2_size = abs(current["tail_2_correction"])

        cutoff_difference = abs(F2_next - F2)
        scale = max(abs(F2), abs(F2_next))
        tolerance = atol + rtol * scale

        cutoff_converged = cutoff_difference <= tolerance
        tail_hierarchy_ok = tail_2_size < tail_1_size

        diagnostic = {
            "b": current["b"],
            "next_b": next_result["b"],
            "F_1term": current["F_1term"],
            "F_2term": F2,
            "cutoff_difference": cutoff_difference,
            "tolerance": tolerance,
            "tail_1_size": tail_1_size,
            "tail_2_size": tail_2_size,
            "tail_ratio": tail_2_size / max(tail_1_size, 1e-30),
            "cutoff_converged": cutoff_converged,
            "tail_hierarchy_ok": tail_hierarchy_ok,
        }

        diagnostics.append(diagnostic)

        if cutoff_converged and tail_hierarchy_ok:
            return {
                "converged": True,
                "b_selected": current["b"],
                "F": F2,
                "absF": abs(F2),
                "diagnostics": diagnostics,
                "all_results": results,
            }

    return {
        "converged": False,
        "b_selected": None,
        "F": None,
        "absF": None,
        "diagnostics": diagnostics,
        "all_results": results,
    }
      
def check():
    check = converged_F(
        w=1.0,
        y=1.0,
        b_values=[50.0, 100.0, 200.0, 400.0],
        h=5e-3,
        rtol=1e-4,
    )

    if check["converged"]:
        print("Converged cutoff:", check["b_selected"])
        print("F =", check["F"])
        print("|F| =", check["absF"])
    else:
        print("Not converged; extend b_values.")
        
    b = 100.0
    w = 1.0
    y = 1.0

    fp_h = f_prime_fd_5pt(b, w, y, h=5e-3)
    fp_h2 = f_prime_fd_5pt(b, w, y, h=2.5e-3)

    relative_derivative_change = abs(fp_h - fp_h2) / max(abs(fp_h2), 1e-10)
    print("Derivative stability:", relative_derivative_change)  
      
def check_v2():

    check = converged_F_v2(
        w=5,
        y=1.0,
        b_values=[100.0, 200.0, 500.0, 1000.0],
        h=1e-3,
        rtol=1e-3,
    )

    if check["converged"]:
        print("Selected b:", check["b_selected"])
        print("F:", check["F"])
        print("|F|:", check["absF"])
    else:
        print("Not converged. Diagnostics:")
        warnings.warn(
            f"Tail did not converge for w={w}; increase b_values "
            "or use a different low-frequency treatment."
        )

    for row in check["diagnostics"]:
        print(
            f"b={row['b']:.0f} -> {row['next_b']:.0f}, "
            f"|ΔF|={row['cutoff_difference']:.3e}, "
            f"tol={row['tolerance']:.3e}, "
            f"|T2|/|T1|={row['tail_ratio']:.3e}, "
            f"cutoff_ok={row['cutoff_converged']}, "
            f"tail_ok={row['tail_hierarchy_ok']}"
        )