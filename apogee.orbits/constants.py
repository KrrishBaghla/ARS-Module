"""
APOGEE Space Research Software Suite: apogee.orbits
====================================================
Standardized physical constants for orbital mechanics computations.

All values are sourced from internationally recognized standards:
- Gravitational parameter: IERS Conventions 2010
- Earth radius: WGS84 reference ellipsoid
- G and M_earth: CODATA 2018 / IAU 2015

Units note
----------
This module uses a mixed unit system consistent with astrodynamics convention:
- Distances in kilometers (km)
- Time in seconds (s)
- Mass in kilograms (kg)
- Angles in radians internally, converted to degrees at output

Do not mix SI (meters) with these constants without explicit conversion.
"""

# Earth Model Constants

MU_EARTH: float = 398600.4418  # km^3 / s^2
"""Earth's standard gravitational parameter

Defined as the product G*M. Always prefer this over computing G * MASS_EARTH,
which introduces unnecessary error.

Value source: IERS Conventions 2010, Table 1.1
"""

RADIUS_EARTH: float = 6378.137  # km
"""Mean equatorial radius of the Earth.

This is the WGS84 semi-major axis. Used for converting between altitude
above the surface and distance from Earth's center:
    r = RADIUS_EARTH + altitude

Value source: WGS84 reference ellipsoid
"""

# Fundamental Physical Constants


G_CONSTANT: float = 6.67430e-11  # m^3/ (kg s^2)
"""Universal gravitational constant.

Note the SI units (meters, not kilometers). Convert before combining
with km-based constants.

Value source: CODATA 2018 recommended value
"""

MASS_EARTH: float = 5.9722e24  # kg
"""Standard gravitational mass of the Earth.

Used in energy and angular momentum calculations. Note that
MU_EARTH = G_CONSTANT * MASS_EARTH holds approximately but not exactly,
due to independent measurement uncertainties.

Value source: IAU 2015 nominal solar and planetary parameters
"""

# Conversion Factors

DEG_2_RAD: float = 0.017453292519943295  # rad / deg
"""Multiply by this to convert degrees to radians.

Equivalent to pi / 180. Stored as a constant to avoid repeated
computation and to make conversion intent explicit in code.
"""

RAD_2_DEG: float = 57.29577951308232  # deg / rad
"""Multiply by this to convert radians to degrees.

Equivalent to 180 / pi. The reciprocal of DEG_2_RAD.
"""