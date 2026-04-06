"""
APOGEE Space Research Software: apogee.orbits.visualize
Visualization utilities for orbital mechanics and space bodies.

Provides functions to:
- Plot 3D orbital trajectories
- Display ground tracks on Earth's surface
- Visualize orbital elements
- Animate satellite motion
- Multi-body visualization

Dependencies:
- matplotlib (3D plotting)
- numpy (numerical computation)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from .constants import RADIUS_EARTH, MU_EARTH, DEG_2_RAD, RAD_2_DEG


def plot_orbit_3d(
    positions,
    earth_radius=RADIUS_EARTH,
    title="3D Orbital Trajectory",
    show_earth=True,
    show_legend=True,
    figsize=(12, 10)
):
    """
    Plot a 3D orbital trajectory for one or more space bodies.
    
    Parameters
    ----------
    positions : dict or list of np.ndarray
        If dict: {"body_name": np.ndarray of shape (N, 3), ...}
        If list: [np.ndarray of shape (N, 3), ...]
        Position vectors in km for each time step.
        
    earth_radius : float
        Earth's radius in km (default: 6378.137)
        
    title : str
        Plot title
        
    show_earth : bool
        Whether to show Earth as a sphere at origin
        
    show_legend : bool
        Whether to show legend
        
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
    
    Examples
    --------
    >>> import numpy as np
    >>> from visualize import plot_orbit_3d
    >>> # Single orbit
    >>> positions = np.random.randn(100, 3) * 1000 + 7000
    >>> fig, ax = plot_orbit_3d(positions)
    >>> plt.show()
    
    >>> # Multiple orbits
    >>> orbits = {
    ...     "Satellite 1": np.random.randn(100, 3) * 1000 + 7000,
    ...     "Satellite 2": np.random.randn(100, 3) * 800 + 8000,
    ... }
    >>> fig, ax = plot_orbit_3d(orbits)
    >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize input to dict format
    if isinstance(positions, np.ndarray):
        positions = {"Orbit": positions}
    elif isinstance(positions, list):
        positions = {f"Orbit {i}": pos for i, pos in enumerate(positions)}
    
    # Plot Earth
    if show_earth:
        _plot_earth_sphere(ax, earth_radius)
    
    # Plot trajectories
    colors = plt.cm.hsv(np.linspace(0, 1, len(positions)))
    for (body_name, pos_array), color in zip(positions.items(), colors):
        pos_array = np.asarray(pos_array)
        ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 
                label=body_name, linewidth=2, color=color)
        
        # Mark start and end points
        ax.scatter(*pos_array[0], s=100, marker='o', color=color, 
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.scatter(*pos_array[-1], s=100, marker='s', color=color, 
                   edgecolors='black', linewidths=1.5, zorder=5)
    
    # Labels and formatting
    ax.set_xlabel("X (km)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Y (km)", fontsize=11, fontweight='bold')
    ax.set_zlabel("Z (km)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Equal aspect ratio
    max_range = np.array([
        np.asarray(pos_array).max(axis=0) - np.asarray(pos_array).min(axis=0) 
        for pos_array in positions.values()
    ]).max() / 2.0
    
    mid_x = np.mean([np.asarray(pos_array)[:, 0].min() + 
                     np.asarray(pos_array)[:, 0].max() 
                     for pos_array in positions.values()])
    mid_y = np.mean([np.asarray(pos_array)[:, 1].min() + 
                     np.asarray(pos_array)[:, 1].max() 
                     for pos_array in positions.values()])
    mid_z = np.mean([np.asarray(pos_array)[:, 2].min() + 
                     np.asarray(pos_array)[:, 2].max() 
                     for pos_array in positions.values()])
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if show_legend and len(positions) > 0:
        ax.legend(loc='upper left', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_ground_track(
    positions,
    earth_radius=RADIUS_EARTH,
    title="Ground Track (Latitude vs Longitude)",
    figsize=(14, 8)
):
    """
    Plot the ground track (latitude/longitude) of one or more space bodies.
    
    Parameters
    ----------
    positions : dict or list of np.ndarray
        Position vectors in km (same format as plot_orbit_3d).
        
    earth_radius : float
        Earth's radius in km
        
    title : str
        Plot title
        
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    
    Examples
    --------
    >>> from visualize import plot_ground_track
    >>> positions = {"Satellite": np.random.randn(100, 3) * 1000 + 7000}
    >>> fig, ax = plot_ground_track(positions)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize input
    if isinstance(positions, np.ndarray):
        positions = {"Track": positions}
    elif isinstance(positions, list):
        positions = {f"Track {i}": pos for i, pos in enumerate(positions)}
    
    # Convert to lat/lon
    colors = plt.cm.hsv(np.linspace(0, 1, len(positions)))
    for (body_name, pos_array), color in zip(positions.items(), colors):
        pos_array = np.asarray(pos_array)
        lats, lons = _xyz_to_lat_lon(pos_array)
        
        ax.plot(lons, lats, label=body_name, linewidth=2, color=color, alpha=0.8)
        ax.scatter(lons[0], lats[0], s=100, marker='o', color=color,
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.scatter(lons[-1], lats[-1], s=100, marker='s', color=color,
                   edgecolors='black', linewidths=1.5, zorder=5)
    
    # World map grid
    ax.set_xlabel("Longitude (degrees)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Latitude (degrees)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add world map outline
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
    
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    return fig, ax


def plot_orbital_elements(
    orbital_elements,
    body_name="Space Body",
    figsize=(10, 8)
):
    """
    Display orbital elements in a formatted text panel.
    
    Parameters
    ----------
    orbital_elements : dict
        Dictionary with keys: 'a_km', 'eccentricity', 'inclination_deg',
        'raan_deg', 'arg_periapsis_deg', 'true_anomaly_deg'
        (from elements.rv_to_elements)
        
    body_name : str
        Name of the space body
        
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    
    Examples
    --------
    >>> from elements import rv_to_elements
    >>> from visualize import plot_orbital_elements
    >>> r = [6524.8, 6862.5, 6448.1]
    >>> v = [4.9, -3.5, 2.1]
    >>> elements = rv_to_elements(r, v)
    >>> fig, ax = plot_orbital_elements(elements, "ISS")
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Extract and format elements
    text_lines = [
        f"{'ORBITAL ELEMENTS':^50}",
        f"{'Body: ' + body_name:^50}",
        "=" * 50,
        "",
        f"Semi-major Axis (a):        {orbital_elements.get('a_km', 0):>12.3f} km",
        f"Eccentricity (e):           {orbital_elements.get('eccentricity', 0):>12.6f}",
        f"Inclination (i):            {orbital_elements.get('inclination_deg', 0):>12.3f}°",
        f"RAAN (Ω):                   {orbital_elements.get('raan_deg', 0):>12.3f}°",
        f"Argument of Periapsis (ω):  {orbital_elements.get('arg_periapsis_deg', 0):>12.3f}°",
        f"True Anomaly (ν):           {orbital_elements.get('true_anomaly_deg', 0):>12.3f}°",
        "",
        "=" * 50,
    ]
    
    # Calculate derived quantities
    a = orbital_elements.get('a_km', 0)
    e = orbital_elements.get('eccentricity', 0)
    
    if a > 0:
        periapsis = a * (1 - e)
        apoapsis = a * (1 + e)
        period_sec = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        period_hours = period_sec / 3600
        
        text_lines.extend([
            "DERIVED QUANTITIES:",
            f"Periapsis Altitude:         {periapsis - RADIUS_EARTH:>12.3f} km",
            f"Apoapsis Altitude:          {apoapsis - RADIUS_EARTH:>12.3f} km",
            f"Orbital Period:             {period_hours:>12.3f} hours",
        ])
    
    text_str = "\n".join(text_lines)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig, ax


def plot_orbit_comparison(
    orbital_elements_list,
    body_names=None,
    figsize=(14, 10)
):
    """
    Compare orbital elements of multiple space bodies in a grid layout.
    
    Parameters
    ----------
    orbital_elements_list : list of dict
        List of orbital element dictionaries
        
    body_names : list of str, optional
        Names for each body (default: Body 0, Body 1, ...)
        
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    if body_names is None:
        body_names = [f"Body {i}" for i in range(len(orbital_elements_list))]
    
    n_bodies = len(orbital_elements_list)
    n_cols = min(3, n_bodies)
    n_rows = (n_bodies + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_bodies == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (elements, name) in enumerate(zip(orbital_elements_list, body_names)):
        ax = axes[idx]
        ax.axis('off')
        
        text_lines = [
            f"{name:^40}",
            "=" * 40,
            f"a: {elements.get('a_km', 0):>10.1f} km",
            f"e: {elements.get('eccentricity', 0):>10.6f}",
            f"i: {elements.get('inclination_deg', 0):>10.2f}°",
            f"Ω: {elements.get('raan_deg', 0):>10.2f}°",
            f"ω: {elements.get('arg_periapsis_deg', 0):>10.2f}°",
            f"ν: {elements.get('true_anomaly_deg', 0):>10.2f}°",
        ]
        
        text_str = "\n".join(text_lines)
        ax.text(0.1, 0.9, text_str, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    # Hide unused subplots
    for idx in range(n_bodies, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig, axes


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _plot_earth_sphere(ax, radius, color='blue', alpha=0.3):
    """
    Plot Earth as a sphere at the origin.
    
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D matplotlib axis
        
    radius : float
        Earth radius in km
        
    color : str
        Color of the sphere
        
    alpha : float
        Transparency (0-1)
    """
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')


def _xyz_to_lat_lon(positions):
    """
    Convert XYZ Cartesian coordinates to latitude and longitude.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with [x, y, z] in km
        
    Returns
    -------
    lats : np.ndarray
        Latitudes in degrees, shape (N,)
        
    lons : np.ndarray
        Longitudes in degrees, shape (N,)
    """
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    lons = np.arctan2(y, x) * RAD_2_DEG
    lats = np.arctan2(z, np.sqrt(x**2 + y**2)) * RAD_2_DEG
    
    return lats, lons
