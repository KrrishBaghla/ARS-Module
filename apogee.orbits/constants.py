"""
APOGEE Space Research Software: apogee.orbits
Standardized constants for orbital mechanics simulations.
"""

# --- Earth Model Constants ---
# Gravitational parameter (mu) in km^3/s^2 [cite: 29, 38]
# Product of G and M from fundamental equations
MU_EARTH = 398600.4418 

# Mean Radius of Earth (R_earth) in kilometers [cite: 29, 38]
# Required for altitude and visibility logic [cite: 11]
RADIUS_EARTH = 6378.137

# --- Physical Constants ---
# Universal Gravitational Constant (G) in m^3 kg^-1 s^-2
# Used in the force equation F = G*M*m / r^2
G_CONSTANT = 6.67430e-11

# Standard Earth Mass (M) in kilograms
# Used for energy (E) and angular momentum (L) calculations
MASS_EARTH = 5.9722e24

# --- Conversion Factors ---
# Essential for converting between raw vectors and Keplerian elements [cite: 40, 47]
DEG_2_RAD = 0.017453292519943295
RAD_2_DEG = 57.29577951308232