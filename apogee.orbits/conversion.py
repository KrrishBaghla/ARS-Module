"""
apogee.orbits.conversion

Bidirectional conversion between state vectors and classical orbital elements.
`elements.py` provides rv → elements. This module completes the inverse:
elements → rv.

Together, they form a near-closed transformation (subject to floating-point
precision and known singularities such as circular or equatorial orbits).

Method

1. Construct position and velocity in the perifocal (PQW) frame.
2. Rotate PQW → ECI using a 3-1-3 Euler rotation (arg_p, inclination, RAAN).
"""

import numpy as np
from .constants import MU_EARTH, DEG_2_RAD


def elements_to_rv(
    a_km: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    arg_p_deg: float,
    nu_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert classical orbital elements → Cartesian state vectors (ECI).

    Parameters
    -
    a_km : float
        Semi-major axis (km), must be > 0.
    e : float
        Eccentricity (0 ≤ e < 1 for elliptical orbits).
    i_deg : float
        Inclination (degrees).
    raan_deg : float
        Right Ascension of Ascending Node (degrees).
    arg_p_deg : float
        Argument of periapsis (degrees).
    nu_deg : float
        True anomaly (degrees).

    Returns
    -
    r_vec : np.ndarray
        Position vector in ECI frame (km).
    v_vec : np.ndarray
        Velocity vector in ECI frame (km/s).

    Notes
    --
    - Singularities exist:
        * e ≈ 0 → argument of periapsis undefined
        * i ≈ 0 → RAAN undefined
    - Inputs are assumed physically consistent.

    Raises
    
    ValueError
        If inputs are outside valid physical ranges.
    """

    # Validation 
    if not (0 <= e < 1):
        raise ValueError(f"Eccentricity must be in [0, 1). Got e={e}.")
    if a_km <= 0:
        raise ValueError(f"Semi-major axis must be positive. Got a={a_km}.")

    # Convert angles to radians 
    i = i_deg * DEG_2_RAD
    raan = raan_deg * DEG_2_RAD
    arg_p = arg_p_deg * DEG_2_RAD
    nu = nu_deg * DEG_2_RAD

    # Perifocal frame (PQW) 
    p = a_km * (1 - e**2)  # semi-latus rectum
    r_mag = p / (1 + e * np.cos(nu))

    r_pqw = np.array([
        r_mag * np.cos(nu),
        r_mag * np.sin(nu),
        0.0
    ], dtype=float)

    v_pqw = np.sqrt(MU_EARTH / p) * np.array([
        -np.sin(nu),
        e + np.cos(nu),
        0.0
    ], dtype=float)

    # Rotation: PQW → ECI (3-1-3 Euler) 
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(arg_p), np.sin(arg_p)

    R = np.array([
        [
            cos_raan * cos_w - sin_raan * sin_w * cos_i,
           -cos_raan * sin_w - sin_raan * cos_w * cos_i,
            sin_raan * sin_i
        ],
        [
            sin_raan * cos_w + cos_raan * sin_w * cos_i,
           -sin_raan * sin_w + cos_raan * cos_w * cos_i,
           -cos_raan * sin_i
        ],
        [
            sin_w * sin_i,
            cos_w * sin_i,
            cos_i
        ]
    ], dtype=float)

    r_vec = R @ r_pqw
    v_vec = R @ v_pqw

    return r_vec, v_vec