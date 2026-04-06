"""
APOGEE Space Research Software: apogee.orbits

Provides tools for orbital element computation, propagation, and analysis.
"""

from .constants import MU_EARTH, RADIUS_EARTH, G_CONSTANT, MASS_EARTH
from .elements import rv_to_elements
from .visualize import plot_orbit_3d,plot_ground_track,plot_orbit_comparison,plot_orbital_elements,_plot_earth_sphere
__version__ = "0.1.0"
__all__ = ["rv_to_elements", "MU_EARTH", "RADIUS_EARTH", "G_CONSTANT", "MASS_EARTH","plot_orbit_3d","plot_ground_track","plot_orbit_comparison","plot_orbital_elements","_plot_earth_sphere"]