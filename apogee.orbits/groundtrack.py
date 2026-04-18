"""
apogee.orbits.groundtrack
-------------------------
Ground track computation and ground station access windows.
"""

import math
import numpy as np

from .constants import DEG_2_RAD
from .propagation import propagate_kepler


# WGS84 parameters

_WGS84_A  = 6378.137
_WGS84_F  = 1 / 298.257223563
_WGS84_B  = _WGS84_A * (1 - _WGS84_F)
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2

_OMEGA_EARTH = 7.2921150e-5


# Helpers

def _normalize_lon(lon_deg: float) -> float:
    return (lon_deg + 180.0) % 360.0 - 180.0


def _eci_to_ecef(r_eci: np.ndarray, gmst: float) -> np.ndarray:
    cos_g = math.cos(gmst)
    sin_g = math.sin(gmst)

    return np.array([
        [ cos_g,  sin_g, 0.0],
        [-sin_g,  cos_g, 0.0],
        [  0.0,    0.0,  1.0]
    ]) @ r_eci


def _ecef_to_geodetic(r_ecef: np.ndarray):
    x, y, z = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])

    lon = math.atan2(y, x)
    p = math.sqrt(x**2 + y**2)

    if p < 1e-10:
        lat = math.copysign(math.pi / 2, z)
        alt = abs(z) - _WGS84_B
        return math.degrees(lat), _normalize_lon(math.degrees(lon)), alt

    lat = math.atan2(z, p * (1.0 - _WGS84_E2))

    for _ in range(10):
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
        lat_new = math.atan2(z + _WGS84_E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat**2)

    alt = p / cos_lat - N if abs(cos_lat) > 1e-8 else abs(z) / abs(sin_lat) - N * (1.0 - _WGS84_E2)

    return math.degrees(lat), _normalize_lon(math.degrees(lon)), alt


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    lat = lat_deg * DEG_2_RAD
    lon = lon_deg * DEG_2_RAD

    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat**2)

    return np.array([
        (N + alt_km) * cos_lat * cos_lon,
        (N + alt_km) * cos_lat * sin_lon,
        (N * (1.0 - _WGS84_E2) + alt_km) * sin_lat
    ])


def _enu_basis(lat_deg: float, lon_deg: float):
    lat = lat_deg * DEG_2_RAD
    lon = lon_deg * DEG_2_RAD

    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    east  = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up    = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    return east, north, up


def _elevation_azimuth(r_sat_ecef, r_gs_ecef, lat_deg, lon_deg):
    rho = r_sat_ecef - r_gs_ecef
    rng = float(np.linalg.norm(rho))

    if rng < 1e-10:
        return 90.0, 0.0, 0.0

    east, north, up = _enu_basis(lat_deg, lon_deg)

    e = np.dot(rho, east)
    n = np.dot(rho, north)
    u = np.dot(rho, up)

    el = math.degrees(math.atan2(u, math.sqrt(e**2 + n**2)))
    az = math.degrees(math.atan2(e, n)) % 360.0

    return el, az, rng


# Public API

def compute_groundtrack(r_vec, v_vec, duration_s, dt_s=60.0, t0_gmst_deg=0.0):
    if duration_s <= 0 or dt_s <= 0:
        raise ValueError("duration_s and dt_s must be positive")

    r_vec = np.array(r_vec, dtype=float)
    v_vec = np.array(v_vec, dtype=float)

    t0 = t0_gmst_deg * DEG_2_RAD

    results = []
    r, v = r_vec.copy(), v_vec.copy()
    t = 0.0

    while t <= duration_s:
        gmst = (t0 + _OMEGA_EARTH * t) % (2 * math.pi)

        r_ecef = _eci_to_ecef(r, gmst)
        lat, lon, alt = _ecef_to_geodetic(r_ecef)

        results.append({
            "t_s": t,
            "lat_deg": lat,
            "lon_deg": lon,
            "alt_km": alt,
        })

        if t >= duration_s:
            break

        step = min(dt_s, duration_s - t)
        r, v = propagate_kepler(r, v, step)
        t += step

    return results


def access_windows(
    r_vec,
    v_vec,
    station_lat_deg,
    station_lon_deg,
    station_alt_km=0.0,
    duration_s=86400.0,
    dt_s=30.0,
    t0_gmst_deg=0.0,
    min_elevation_deg=5.0,
):
    if duration_s <= 0 or dt_s <= 0:
        raise ValueError("duration_s and dt_s must be positive")

    if not (-90 <= station_lat_deg <= 90):
        raise ValueError("Invalid latitude")
    if not (-180 <= station_lon_deg <= 180):
        raise ValueError("Invalid longitude")

    r_vec = np.array(r_vec, dtype=float)
    v_vec = np.array(v_vec, dtype=float)

    r_gs = _geodetic_to_ecef(station_lat_deg, station_lon_deg, station_alt_km)

    t0 = t0_gmst_deg * DEG_2_RAD
    r, v = r_vec.copy(), v_vec.copy()
    t = 0.0

    windows = []
    in_pass = False

    while t <= duration_s:
        gmst = (t0 + _OMEGA_EARTH * t) % (2 * math.pi)
        r_ecef = _eci_to_ecef(r, gmst)

        el, az, rng = _elevation_azimuth(r_ecef, r_gs, station_lat_deg, station_lon_deg)
        visible = el >= min_elevation_deg

        if visible and not in_pass:
            in_pass = True
            start_t = t
            start_az = az
            max_el, max_t, max_az, max_rng = el, t, az, rng

        elif visible:
            if el > max_el:
                max_el, max_t, max_az, max_rng = el, t, az, rng

        elif in_pass:
            in_pass = False
            windows.append({
                "rise_time_s": start_t,
                "set_time_s": t,
                "duration_s": t - start_t,
                "max_elevation_deg": max_el,
                "max_el_time_s": max_t,
                "max_el_azimuth_deg": max_az,
                "rise_azimuth_deg": start_az,
                "set_azimuth_deg": az,
                "range_at_max_el_km": max_rng,
            })

        if t >= duration_s:
            break

        step = min(dt_s, duration_s - t)
        r, v = propagate_kepler(r, v, step)
        t += step

    return windows