"""
Checks the correctness & consistency of SciTec ECEF conversions using:
1) Recompute-and-compare (LLA to ECEF vs saved XYZ)
2) Independent inverse (ECEF to LLA via pyproj)
3) Physical sanity checks (Earth radius scale, NaNs, speed)
4) Velocity consistency (finite-difference of XYZ vs saved Vx/Vy/Vz)
5) Known-point unit test
"""

from __future__ import annotations
import sys
from pathlib import Path
import math
from typing import NoReturn
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer, Geod

#import functions from the original converter for consistency
from scitec_lla_to_ecef import lla_to_ecef


# WGS84 parameters from LLAtoECEF.pdf reference
WGS84_A = 6378137.0 #semi major axis in meters (Earth's equatorial radius)
WGS84_B = 6356752.314245  # semi-minor axis in meters (Earth's polar radius)
WGS84_F = 1 / 298.257223563 #flattening ratio
WGS84_E2 = WGS84_F * (2 - WGS84_F) #square of first eccentricity ≈ 0.0066943799901413165
MARGIN = 20000.0  # 20 km safety margin for radius checks

# Define CRS explicitly (module-level is ideal)
CRS_ECEF  = CRS.from_epsg(4978)  # WGS84 Geocentric (ECEF) XYZ in meters
CRS_GEO3D = CRS.from_epsg(4979)  # WGS84 Geographic 3D (lon, lat, h) in deg/m
TO_GEO    = Transformer.from_crs(CRS_ECEF, CRS_GEO3D, always_xy=True)

# Geodesic calculator on the WGS84 ellipsoid
GEOD_WGS84 = Geod(ellps="WGS84")

# Tolerances for various checks
TOL = {
        # LLA to ECEF recompute vs saved
        "recompute_max_pos_err_m": 1e-4,
        "recompute_mean_pos_err_m": 1e-6,

        # ECEF to LLA inverse (pyproj) vs original LLA
        "inv_latlon_max_deg": 1e-7, # ~1 cm horizontal error at Earth's surface
        "inv_alt_max_m": 0.05, # 5 cm altitude error tolerance

        # Velocity consistency tolerance (finite-difference vs saved)
        "vel_max_diff_mps": 1e-3, 

        # Physical sanity: generous maximum speed cap (m/s)
        "speed_max_mps": 2.0e4,
    }

def finite_diff_velocity(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Reproduce the simple forward-difference velocity used by LLA to ECEF pipeline:
      v[0] = 0; for i>=1 and dt>0, v[i] = (r[i] - r[i-1]) / (t[i] - t[i-1]).
    """
    v = np.zeros_like(r, dtype=np.float64)
    dt = np.diff(t)
    dr = np.diff(r, axis=0)
    valid = dt > 0
    v[1:][valid] = dr[valid] / dt[valid, None]
    return v

#  Fail-fast assert helper functions
def die(msg: str, code: int = 1) -> NoReturn:
    """
    Stop the program right now and show a clear message.

    - msg:  The text to display so the user knows what went wrong.
    - code: The exit status sent back to the OS (1 means "error"; 0 would mean "success").
    - NoReturn: A type hint that says this function never comes back (it always exits).
    """
    sys.exit(f"[FATAL] {msg}")

def assert_true(cond: bool, msg: str) -> None:
    """Fail fast if a condition is false."""
    if not cond:
        die(msg)

# File I/O helper # 1
def load_lla(path: Path) -> pd.DataFrame:
    """
    Check that 'cond' is True; if it's False, stop the program with a helpful message.

    - cond: A condition you expect to be True (e.g., file exists, arrays same length).
    - msg:  What to show the user if the expectation is not met.
    """
    # Ensure the input LLA CSV file exists
    if not path.exists():
        die(f"LLA file not found: {path}")

    # Load the input CSV file into a pandas DataFrame (no header row)
    df = pd.read_csv(path, header=None)

    # Basic structure checks - the file must have at least 4 columns
    if df.shape[1] < 4:
        die("LLA CSV must have 4 columns (time_s, lat_deg, lon_deg, alt_km).")

    # Keep only the first 4 columns and assign names
    # time in seconds, latitude in degrees, longitude in degrees, altitude in kilometers
    df = df.iloc[:, :4].copy()
    df.columns = ["time_s", "lat_deg", "lon_deg", "alt_km"]

    # Drop any rows that contain missing values (NaNs)
    # Reset index after dropping rows
    df = df.dropna().reset_index(drop=True)
    assert_true(len(df) > 0, "LLA CSV is empty after dropping NaNs.")

    # Strong dtypes
    df["time_s"]  = df["time_s"].astype("float64")
    df["lat_deg"] = df["lat_deg"].astype("float64")
    df["lon_deg"] = df["lon_deg"].astype("float64")
    df["alt_km"]  = df["alt_km"].astype("float64")

    # Sort + basic sanity on time
    df = df.sort_values("time_s").reset_index(drop=True)

    # If nothing is left, fail fast with a helpful message.
    assert_true(np.all(np.diff(df["time_s"]) >= 0), "LLA times are not non-decreasing.")

    return df

# File I/O helper # 2
def load_ecef(path: Path) -> pd.DataFrame:
    """
    Load the ECEF CSV written by converter:
      required: time_s, X_m, Y_m, Z_m
      optional: Vx_mps, Vy_mps, Vz_mps

    - Enforces types, sorts by time, checks monotonicity & NaNs.
    - Fails fast if missing/invalid
    """
    # Ensure the input ECEF CSV file exists
    if not path.exists():
        die(f"ECEF file not found: {path}")
    
    # Load the input CSV file into a pandas DataFrame
    df = pd.read_csv(path)

    # Required columns
    required = ["time_s", "X_m", "Y_m", "Z_m"]

    # Check for required columns
    missing = [c for c in required if c not in df.columns]

    # Fail fast if any required columns are missing
    assert_true(not missing, f"ECEF CSV missing required columns: {missing}")

    # Enforce dtypes
    for c in required:
        df[c] = df[c].astype("float64")
    for c in ["Vx_mps", "Vy_mps", "Vz_mps"]:
        if c in df.columns:
            df[c] = df[c].astype("float64")
    
    # Drop rows with NaNs in required columns
    df = df.dropna(subset=required).reset_index(drop=True)

    # Ensure we have data left
    assert_true(len(df) > 0, "ECEF CSV is empty or has NaNs in required columns.")

    # Sort time and check monotonicity
    df = df.sort_values("time_s").reset_index(drop=True)
    assert_true(np.all(np.diff(df["time_s"]) >= 0), "ECEF times are not non-decreasing.")

    return df

# Core validation function #1
def check_times_align(lla: pd.DataFrame, ecef: pd.DataFrame) -> None:
    """
    Ensure that the converter produced a one-to-one mapping for timestamps:
      - Every LLA time appears exactly once in ECEF CSV
      - Same counts and identical sorted arrays
    """
    # Extract time arrays
    t_lla  = lla["time_s"].to_numpy(np.float64)
    t_ecef = ecef["time_s"].to_numpy(np.float64)

    # Check lengths match
    assert_true(len(t_lla) == len(t_ecef),
                f"Row count mismatch: LLA has {len(t_lla)} rows, ECEF has {len(t_ecef)}.")
    
    # Check sorted arrays match exactly
    assert_true(np.allclose(t_lla, t_ecef, rtol=0, atol=0),
                "Timestamps do not match exactly between LLA and ECEF CSVs (check sorting/dedup).")

# Core validation function #2
def check_recompute_xyz(lla: pd.DataFrame, ecef: pd.DataFrame) -> None:
    """
    Recompute ECEF (LLA -> ECEF) from the original LLA data and compare it to the
    XYZ values already saved in the ECEF CSV. If the differences are too large,
    stop the program with a clear error. This ensures the saved XYZ was produced
    correctly by the LLA->ECEF converter.
    """
    # Pull latitude (deg), longitude (deg), and altitude (km) from the LLA table
    # and turn them into high-precision NumPy arrays
    lat = lla["lat_deg"].to_numpy(np.float64)
    lon = lla["lon_deg"].to_numpy(np.float64)

    # Convert altitude from kilometers to meters to match the ECEF XYZ units (meters)
    alt_m = lla["alt_km"].to_numpy(np.float64) * 1000.0

    # Recompute ECEF XYZ from LLA using the same formula/function as the converter.
    # xyz_calc has shape (N, 3) with columns [X, Y, Z] in meters
    xyz_calc = lla_to_ecef(lat, lon, alt_m)

    # Extract the already-saved XYZ columns from the ECEF CSV as float64 arrays
    X, Y, Z = [ecef[c].to_numpy(np.float64) for c in ("X_m", "Y_m", "Z_m")]

    # Compute per-row differences between recomputed XYZ and saved XYZ
    # diff is an (N, 3) array of [delta X, delta Y, delta Z] for each timestamp
    diff = np.column_stack([xyz_calc[:, 0] - X, xyz_calc[:, 1] - Y, xyz_calc[:, 2] - Z])

    # Turn each row's deltas into a single distance Euclidean norm in meters
    # err is a length-N array of position errors, one for each sample.
    err = np.linalg.norm(diff, axis=1)
    
    # Check the worst-case (maximum) position error against a strict tolerance
    # If any sample is too far off, fail fast with a descriptive message
    assert_true(err.max() <= TOL["recompute_max_pos_err_m"],
                f"LLA->ECEF recompute failed: max position error {err.max():.3e} m "
                f"(limit {TOL['recompute_max_pos_err_m']:.1e})")
    
    # Check the average (mean) position error across all samples for overall quality
    assert_true(err.mean() <= TOL["recompute_mean_pos_err_m"],
                f"LLA->ECEF recompute failed: mean position error {err.mean():.3e} m "
                f"(limit {TOL['recompute_mean_pos_err_m']:.1e})")

# Core validation function #3
def check_inverse_pyproj(lla: pd.DataFrame, ecef: pd.DataFrame) -> None:
    """
    Do an independent "inverse" check using a trusted library:
      1) Convert the saved ECEF XYZ back to geodetic LLA (lon, lat, height) with a
         CRS-aware transformer (EPSG:4978 -> EPSG:4979).
      2) Measure horizontal error in METERS using geodesic distance on WGS84
         (this compares only lon/lat on the ellipsoid surface).
      3) Measure vertical (altitude) error in METERS.
      4) Fail fast if either error exceeds the allowed tolerances.
    """

    # 1) Inverse transform: (X, Y, Z) in meters  ->  (lon_deg, lat_deg, h_m)
    # Extract saved ECEF coordinates from the output CSV as float64 arrays.
    x = ecef["X_m"].to_numpy(np.float64)
    y = ecef["Y_m"].to_numpy(np.float64)
    z = ecef["Z_m"].to_numpy(np.float64)

    # TO_GEO is a pre-built transformer with explicit CRS and axis order
    # It returns longitude (deg), latitude (deg), and ellipsoidal height (meters)
    lon_calc, lat_calc, h_calc = TO_GEO.transform(x, y, z)

    # 2) Reference geodetic coordinates from the original LLA CSV 
    # the input CSV stores altitude in KILOMETERS; convert to METERS to match h_calc.
    lat_ref = lla["lat_deg"].to_numpy(np.float64)
    lon_ref = lla["lon_deg"].to_numpy(np.float64)
    h_ref_m = (lla["alt_km"].to_numpy(np.float64) * 1000.0)

    # 3) Horizontal error (meters) using geodesic distance on WGS84
    # GEOD_WGS84.inv(lon1, lat1, lon2, lat2) returns (fwd_azimuth, back_azimuth, distance_m).
    # We only need the surface distance in meters between the two lon/lat points.
    _, _, dist_m = GEOD_WGS84.inv(lon_ref, lat_ref, lon_calc, lat_calc)

    # 4) Altitude absolute error (meters)
    alt_err = np.abs(h_calc - h_ref_m)

    # 5) Tolerances (prefer meter-based horizontal tolerance)
    #  If TOL dict has 'inv_horiz_max_m', use it; otherwise default to 0.05 m (5 cm).
    horiz_max_m = TOL.get("inv_horiz_max_m", 0.05)
    alt_max_m   = TOL["inv_alt_max_m"]  # keep your existing altitude limit
    
    # Take the worst (max) error across all samples for a strict check
    max_horiz = float(np.nanmax(dist_m))
    max_alt   = float(np.nanmax(alt_err))
    
    # If either horizontal or vertical error is too large, stop with a clear message
    assert_true(
        max_horiz <= horiz_max_m,
        f"Inverse check failed: horizontal max error {max_horiz:.3f} m "
        f"(limit {horiz_max_m:.3f} m)"
    )
    assert_true(
        max_alt <= alt_max_m,
        f"Inverse check failed: alt max error {max_alt:.3f} m "
        f"(limit {alt_max_m:.1e} m)"
    )

# Core validation function #4
def check_physical_sanity(ecef: pd.DataFrame) -> None:
    """
    Make sure the saved ECEF data looks physically reasonable.

    We check:
      - The distance from Earth's center (|r|) stays within realistic bounds (in METERS).
      - No NaNs in XYZ (already checked earlier, but we recompute radius here).
      - If velocity columns exist, their values are finite and the max speed is capped.
    """
    # Position sanity: radius from Earth's center 
    # Pull ECEF positions (meters) as float64 arrays
    X, Y, Z = [ecef[c].to_numpy(np.float64) for c in ("X_m", "Y_m", "Z_m")]

    # Compute the radius |r| = sqrt(X^2 + Y^2 + Z^2) for each row
    # This should be roughly Earth's radius (~6.37e6 m) +/- altitude
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Get min and max radius over the whole trajectory for strict bounds checks
    rmin, rmax = float(r.min()), float(r.max())

    # Enforce lower and upper bounds for |r| using tolerances in meters
    # Example: you might set radius_min_m ≈ 6.3e6 and radius_max_m ≈ 6.5e6 for LEO-like data,
    # or wider for other scenarios
    assert_true(rmin >= TOL["radius_min_m"],
                f"Radius too small: min |r|={rmin:.1f} m (limit {TOL['radius_min_m']:.0f})")
    assert_true(rmax <= TOL["radius_max_m"],
                f"Radius too large: max |r|={rmax:.1f} m (limit {TOL['radius_max_m']:.0f})")
    
    # Velocity sanity
    # Only run these checks if velocity columns are present in the file.
    has_v = all(c in ecef.columns for c in ("Vx_mps", "Vy_mps", "Vz_mps"))
    if has_v:
        # Load velocities (meters/second) as float64 arrays
        Vx, Vy, Vz = [ecef[c].to_numpy(np.float64) for c in ("Vx_mps", "Vy_mps", "Vz_mps")]
        # Ensure there are no missing values in any velocity component
        assert_true(not (np.isnan(Vx).any() or np.isnan(Vy).any() or np.isnan(Vz).any()),
                    "NaNs detected in velocity columns.")
        
        # Compute speed magnitude |v| = sqrt(Vx^2 + Vy^2 + Vz^2) for each sample,
        # then check that the maximum speed is below threshold
        speed = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        smax = float(speed.max())
        assert_true(smax <= TOL["speed_max_mps"],
                    f"Speed too large: max |v|={smax:.2f} m/s (limit {TOL['speed_max_mps']:.0f})")

# Core validation function #5
def check_velocity_consistency(ecef: pd.DataFrame) -> None:
    """
    Compare saved velocities to velocities re-computed from positions.

    Idea:
      - If the file contains velocity columns (Vx_mps, Vy_mps, Vz_mps),
        we recompute velocity from the saved XYZ positions using
        finite differences (v ≈ delta r / delta t).
      - If the largest difference between "saved" and "recomputed"
        speeds is too big, we stop with a clear error.
    """
    # The velocity columns we expect to find (meters per second)
    needed = ("Vx_mps", "Vy_mps", "Vz_mps")

    # If any of these columns are missing, there is nothing to validate—just return
    if not all(c in ecef.columns for c in needed):
        return  # No velocity data to validate; skip.
    
    # Pull time (seconds) and position (meters) as high-precision NumPy arrays
    t = ecef["time_s"].to_numpy(np.float64)
    r = ecef[["X_m", "Y_m", "Z_m"]].to_numpy(np.float64)
    
    # Recompute velocity from positions with a standard forward finite difference:
    # v_fd[i] approx (r[i] - r[i-1]) / (t[i] - t[i-1])  (with safe handling of the first row)
    v_fd = finite_diff_velocity(t, r)  # matches the common forward-diff behavior
    
    # Load the velocity that was already saved in the CSV
    v_saved = ecef[list(needed)].to_numpy(np.float64)
    
    # Compute the per-row vector difference between recomputed and saved velocities,
    # then take the Euclidean norm (length) to get an error magnitude in m/s
    diff = np.linalg.norm(v_fd - v_saved, axis=1)

    # Take the worst (maximum) error over the whole series for a strict check
    dmax = float(diff.max())

    # Enforce the tolerance (in m/s). If exceeded, fail fast with a descriptive message
    assert_true(dmax <= TOL["vel_max_diff_mps"],
                f"Velocity consistency failed: max |Δv|={dmax:.3e} m/s "
                f"(limit {TOL['vel_max_diff_mps']:.1e})")

def known_point_unit_test() -> None:
    """
    Quick "does this still work?" test for the LLA to ECEF converter.

    Idea:
      - Use a point we know the exact answer for: (lat, lon, alt) = (0°, 0°, 0 m).
      - On the equator at the prime meridian and sea level, the ECEF result should be:
          X = a (Earth's WGS84 equatorial radius),
          Y = 0,
          Z = 0.
      - If the function returns anything different (beyond tiny rounding noise),
        we fail immediately with a clear message.
    """
    # Run the converter on the known point.
    # We pass arrays to match the function's vectorized (array) interface
    xyz = lla_to_ecef(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0]

    # Unpack X, Y, Z as plain floats for easy checks
    X0, Y0, Z0 = float(xyz[0]), float(xyz[1]), float(xyz[2])

    # Very tight (~1e-6 meters) absolute tolerances (numerical noise only)
    assert_true(abs(X0 - WGS84_A) <= 1e-6, f"Known-point X failed: {X0} vs {WGS84_A}")
    assert_true(abs(Y0) <= 1e-6, f"Known-point Y failed: {Y0}")
    assert_true(abs(Z0) <= 1e-6, f"Known-point Z failed: {Z0}")

# Main orchestration
def main(argv: list[str]) -> int:
    """
    Orchestrate all checks in a fail-fast manner.

    Inputs:
      - argv[1]: LLA CSV path (headerless: time_s, lat_deg, lon_deg, alt_km)
      - argv[2]: ECEF CSV path; defaults to <lla_stem>_ecef.csv in the same dir.
      - If no args are given, defaults to SciTec filenames in cwd.

    Returns:
      0 on success, non-zero on failure (exits early on first failure).
    """
    if len(argv) != 3:
        print(
            "ERROR: Please provide two CSV arguments:\n"
            "  1) LLA coordinates CSV (e.g., SciTec_code_problem_data.csv)\n"
            "  2) ECEF coordinates CSV (e.g., SciTec_code_problem_data_ecef.csv)\n\n"
            "Usage:\n"
            "python verify_ecef.py SciTec_code_problem_data.csv SciTec_code_problem_data_ecef.csv",
            file=sys.stderr,
        )
        sys.exit(2)
    
    # Parse input paths
    lla_path = Path(argv[1])
    ecef_path = Path(argv[2])
   
    # 0) Load inputs (these functions already fail fast on problems).
    lla  = load_lla(lla_path)
    ecef = load_ecef(ecef_path)
    
    # Diagnostic: report approximate radius and altitude stats
    R = np.linalg.norm(ecef[["X_m","Y_m","Z_m"]].to_numpy(np.float64), axis=1)
    approx_alt_m = R - WGS84_A
    print(
        f"[diag] ECEF |r| stats (m): min={R.min():.1f}, max={R.max():.1f}, "
        f"approx h (m): min={approx_alt_m.min():.1f}, max={approx_alt_m.max():.1f}"
    )

    # Reference altitude in meters based on original LLA CSV
    h_ref_m = lla["alt_km"].to_numpy(np.float64) * 1000.0

    # Add data specific tolerances for altitude-based radius checks based on input data
    TOL["radius_min_m"] = max(0.0, WGS84_B + np.nanmin(h_ref_m) - MARGIN)
    TOL["radius_max_m"] = WGS84_A + np.nanmax(h_ref_m) + MARGIN

    # 1) Sanity-test the forward converter implementation (constants & formulas).
    known_point_unit_test()

    # 2) Confirm timestamps align exactly between input and output CSVs.
    check_times_align(lla, ecef)

    # 3) Recompute ECEF positions from LLA and enforce tight error bounds vs saved XYZ.
    check_recompute_xyz(lla, ecef)

    # 4) Independent inverse check (ECEF->LLA via pyproj) against original LLA.
    check_inverse_pyproj(lla, ecef)

    # 5) Physical sanity checks (radius scale; optional speed cap if velocities exist).
    check_physical_sanity(ecef)

    # 6) Velocity consistency (if Vx/Vy/Vz are present in output CSV).
    check_velocity_consistency(ecef)

    # If we got here, all checks passed with strict tolerances.
    print("\n All verification checks PASSED. "
          "LLA→ECEF conversions, inverse checks, and physical sanity tests succeeded.\n")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))