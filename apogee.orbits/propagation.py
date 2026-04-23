"""
apogee.orbits.propagation
-------------------------
Time evolution of orbits under two-body Keplerian dynamics.

In two-body motion, orbital shape and orientation remain constant.
Only the satellite’s position along the orbit evolves with time.

We propagate using mean anomaly (uniform in time), solving Kepler’s
equation numerically to recover the true anomaly.
"""

import math
import numpy as np

from .constants import MU_EARTH, DEG_2_RAD, RAD_2_DEG
from .elements import rv_to_elements
from .conversion import elements_to_rv


# Kepler equation utilities

def true_to_eccentric(nu_deg: float, e: float) -> float:
    """True anomaly (deg) → eccentric anomaly (rad)."""
    nu = nu_deg * DEG_2_RAD
    E = 2 * math.atan2(
        math.sqrt(1 - e) * math.sin(nu / 2),
        math.sqrt(1 + e) * math.cos(nu / 2)
    )
    return E % (2 * math.pi)


def eccentric_to_mean(E_rad: float, e: float) -> float:
    """Eccentric anomaly → mean anomaly."""
    return (E_rad - e * math.sin(E_rad)) % (2 * math.pi)


def solve_kepler(M_rad: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """
    Solve Kepler’s equation M = E - e*sin(E) using Newton-Raphson.
    """
    # Better initial guess
    E = M_rad if e < 0.8 else math.pi

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M_rad
        df = 1 - e * math.cos(E)

        if abs(df) < 1e-12:
            break  # avoid division instability

        dE = f / df
        E -= dE

        if abs(dE) < tol:
            return E % (2 * math.pi)

    raise RuntimeError(
        f"Kepler solve failed: M={math.degrees(M_rad):.4f}°, e={e:.6f}"
    )


def eccentric_to_true(E_rad: float, e: float) -> float:
    """Eccentric anomaly → true anomaly (deg)."""
    nu = 2 * math.atan2(
        math.sqrt(1 + e) * math.sin(E_rad / 2),
        math.sqrt(1 - e) * math.cos(E_rad / 2)
    )
    return (nu % (2 * math.pi)) * RAD_2_DEG


# Main propagation

def propagate_kepler(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate state using two-body Keplerian motion.

    Steps:
        rv → elements → advance mean anomaly → solve Kepler →
        recover true anomaly → elements → rv
    """

    r_vec = np.array(r_vec, dtype=float)
    v_vec = np.array(v_vec, dtype=float)

    # Extract orbital elements
    el = rv_to_elements(r_vec, v_vec)

    a = el["a_km"]
    e = el["eccentricity"]
    i = el["inclination_deg"]
    raan = el["raan_deg"]
    arg_p = el["arg_periapsis_deg"]
    nu = el["true_anomaly_deg"]

    # Mean motion
    n = math.sqrt(MU_EARTH / a**3)

    # Current anomalies
    E0 = true_to_eccentric(nu, e)
    M0 = eccentric_to_mean(E0, e)

    # Advance mean anomaly
    M_new = (M0 + n * dt) % (2 * math.pi)

    # Solve Kepler
    E_new = solve_kepler(M_new, e)

    # Back to true anomaly
    nu_new = eccentric_to_true(E_new, e)

    # Reconstruct state
    return elements_to_rv(a, e, i, raan, arg_p, nu_new)