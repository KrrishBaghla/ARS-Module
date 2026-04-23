"""
TECHNICAL DETAILS & MATHEMATICAL FOUNDATION

Definition:
This module converts Cartesian State Vectors (Position r and Velocity v) into 
the six Classical Orbital Elements (COE). It transforms 3D coordinates into 
parameters that describe the size, shape, and orientation of an orbit.

Fundamental Formulas (Standard Two-Body Physics):
1. Specific Mechanical Energy (E): E = (v^2 / 2) - (mu / r)
2. Semi-major axis (a): a = -mu / (2 * E)
3. Specific Angular Momentum (h): h_vector = r_vector cross v_vector
4. Eccentricity (e): Magnitude of the vector e = ((v^2 - mu/r) * r - (r dot v) * v) / mu
5. Inclination (i): Angle between h_vector and the Z-axis.

The Six Orbital Elements:
- Semi-major axis (a): Distance from the center to the furthest point of the ellipse (km).
- Eccentricity (e): How circular the orbit is (0 = circle, 0 to 1 = ellipse).
- Inclination (i): The tilt of the orbit relative to the Earth's equator (degrees).
- RAAN (Omega): The angle where the orbit crosses the equator heading north.
- Argument of Periapsis (omega): The angle from the node to the orbit's closest point.
- True Anomaly (nu): The satellite's current angular position in its orbit.

Use Case:
Essential for mission planning, ground station tracking, and 3D visualization.

Example:
>>> pos = [6524.8, 6862.5, 6448.1]
>>> vel = [4.9, -3.5, 2.1]
>>> elements = rv_to_elements(pos, vel)
"""

import numpy as np
from .constants import MU_EARTH, RAD_2_DEG

def rv_to_elements(r_vec, v_vec):
    # Convert inputs to numpy arrays for vector math
    r_vec = np.array(r_vec)
    v_vec = np.array(v_vec)
    
    # 1. Magnitudes
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # 2. Specific Angular Momentum (h)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # 3. Semi-major Axis (a) 
    # Derived from Total Energy: Energy = (v^2 / 2) - (mu / r)
    energy = (v**2 / 2) - (MU_EARTH / r)
    a = -MU_EARTH / (2 * energy)
    
    # 4. Eccentricity (e)
    # The eccentricity vector points toward the periapsis
    e_vec = ((v**2 - MU_EARTH / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU_EARTH
    e = np.linalg.norm(e_vec)
    
    # 5. Inclination (i)
    # The angle between the momentum vector and the North Pole (Z-axis)
    i = np.arccos(h_vec[2] / h)
    
    # 6. RAAN (Omega - Longitude of Ascending Node)
    # n_vec points toward the point where the satellite crosses the equator
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)
    
    if n != 0:
        raan = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0 # Case for equatorial orbits
        
    # 7. Argument of Periapsis (omega)
    # The angle from the ascending node to the eccentricity vector
    if n != 0:
        arg_p = np.arccos(np.dot(n_vec, e_vec) / (n * e))
        if e_vec[2] < 0:
            arg_p = 2 * np.pi - arg_p
    else:
        arg_p = 0
        
    # 8. True Anomaly (nu)
    # The current position of the satellite relative to periapsis
    nu = np.arccos(np.dot(e_vec, r_vec) / (e * r))
    if np.dot(r_vec, v_vec) < 0:
        nu = 2 * np.pi - nu

    # Return results in a human-readable dictionary
    return {
        "a_km": a,
        "eccentricity": e,
        "inclination_deg": i * RAD_2_DEG,
        "raan_deg": raan * RAD_2_DEG,
        "arg_periapsis_deg": arg_p * RAD_2_DEG,
        "true_anomaly_deg": nu * RAD_2_DEG
    }