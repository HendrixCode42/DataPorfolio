from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

# WGS84 parameters from LLAtoECEF.pdf reference
WGS84_A = 6378137.0 #semi major axis in meters (Earth's equatorial radius)
WGS84_F = 1 / 298.257223563 #flattening ratio
WGS84_E2 = WGS84_F * (2 - WGS84_F) #square of first eccentricity ≈ 0.0066943799901413165

def lla_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion from LLA (Latitude, Longitude, Altitude)
    to ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates, in meters.

    Inputs (all same length arrays):
      - lat_deg : latitude in degrees   (−90 to +90)
      - lon_deg : longitude in degrees  (−180 to +180)
      - alt_m   : altitude in meters above mean sea level

    Output:
      - Nx3 NumPy array with columns [X, Y, Z] in meters.
    """
    # Trig functions in NumPy expect radians
    # Convert latitude and longitude from degrees to radians.
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    # Precompute sines and cosines of latitude and longitude for reuse.
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    # N is the "prime vertical radius of curvature" at each latitude:
    #                      a
    #   N = ---------------------------
    #        sqrt(1 - e^2 * sin^2(lat))
    # Because Earth is an ellipsoid the distance from Earth's center
    # to the surface depends on latitude.
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    # Compute ECEF coordinates.
    # X and Y lie in the equatorial plane; Z points toward the North Pole
    # (N + alt_m) and (N*(1 - e^2) + alt_m) adjust for altitude above sea level
    X = (N + alt_m) * cos_lat * cos_lon
    Y = (N + alt_m) * cos_lat * sin_lon
    Z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat

    # Stack the three 1D arrays into a single Nx3 array: columns [X, Y, Z].
    # Returns a "point per row" format.
    return np.column_stack((X, Y, Z))

@dataclass
class ECEFVelocityInterpolator:
    """
    A small container (data class) that holds:
      - t : 1D array of times (seconds, strictly increasing, no duplicates after cleaning)
      - r : Nx3 array of ECEF positions (meters)
      - v : Nx3 array of ECEF velocities (meters/second), aligned with t

    It also provides:
      - from_lla_csv(...) : a classmethod to build the object from a CSV of LLA data
      - velocity_at(time) : linearly interpolates velocity at an arbitrary query time
    """
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray

    @classmethod
    def from_lla_csv(cls, csv_path: Path) -> "ECEFVelocityInterpolator":
        """
        Builds an instance from a CSV whose columns are:
          time_s, lat_deg, lon_deg, alt_km
        
        Steps:
          1) Read CSV and grab columns as float64 arrays
          2) Sort rows by time (so time increases)
          3) Drop duplicate time stamps (keep first occurrence)
          4) Convert LLA -> ECEF positions
          5) Estimate velocities by finite difference: v[i] ≈ (r[i]-r[i-1]) / (t[i]-t[i-1])
          6) Return a fully populated ECEFVelocityInterpolator
        """
        # Step 1: Read CSV (no header in file, so assign names explicitly)
        df = pd.read_csv(csv_path, header=None, names=["time_s", "lat_deg", "lon_deg", "alt_km"])
        
        # Drop any rows that contain NaN or missing values in any column
        df = df.dropna(subset=["time_s", "lat_deg", "lon_deg", "alt_km"]).reset_index(drop=True)

        # Step 2: Extract columns as NumPy arrays with a consistent dtype
        t = df["time_s"].to_numpy(dtype=np.float64) # time in seconds
        lat = df["lat_deg"].to_numpy(dtype=np.float64) # latitude in degrees
        lon = df["lon_deg"].to_numpy(dtype=np.float64) # longitude in degrees
        alt_m = df["alt_km"].to_numpy(dtype=np.float64) * 1000.0 # altitude in meters

        # Step 3: Sort by time so everything is in chronological order and drop duplicates
        order = np.argsort(t) # indices sort t (ascending)
        t = t[order]
        lat = lat[order]
        lon = lon[order]
        alt_m = alt_m[order]
        
        # Step 4: Remove duplicate timestamps (keep first). 
        # This avoids divide-by-zero in velocity.
        # unique_mask[i] is True if t[i] != t[i-1] (for i>0); first element is always True.
        unique_mask = np.ones_like(t, dtype=bool) # initialize all True
        unique_mask[1:] = t[1:] != t[:-1] # mark duplicates as False
        # Apply mask to all arrays
        t = t[unique_mask]
        lat = lat[unique_mask]
        lon = lon[unique_mask]
        alt_m = alt_m[unique_mask]

        # Step 5: Convert geodetic LLA to ECEF Cartesian coordinates (meters)
        #  r has shape (N, 3) with columns [X, Y, Z]
        r = lla_to_ecef(lat, lon, alt_m)

        # Step 6: Estimate velocities
        # Estimate velocities by finite differences of position over time.
        #  For N time samples, we get N-1 differences.
        N = r.shape[0] # number of samples after cleaning
        v = np.zeros((N, 3), dtype=np.float64) # initialize velocities to zero

        dt = np.diff(t) # computes the difference between consecutive times: shape (N-1,)
        dr = np.diff(r, axis=0) # computes the difference between consecutive position rows:shape (N-1, 3) position differences

        # Only compute v where dt > 0 (protect against any pathological zero gaps)
        valid = dt > 0 # boolean mask of valid time gaps

        # v[1:] corresponds to the difference between sample i and i-1
        # broadcasting dt[valid, None] so we divide each vector by the scalar time gap
        v[1:][valid] = dr[valid] / dt[valid, None]

        # Return the fully constructed instance
        return cls(t=t, r=r, v=v)

    def velocity_at(self, query_time_s: float) -> np.ndarray:
        """
        Return the velocity vector at an arbitrary query time by exact match or linear interpolation.

        Assumes:
        - self.t : 1D np.ndarray of monotonically increasing times (seconds), length N
        - self.v : 2D np.ndarray of velocities (N, 3) aligned with self.t (m/s)
        """
        # Local aliases for readability
        t = self.t; v = self.v

        # Guard: reject queries outside the sampled time span
        if query_time_s < t[0] or query_time_s > t[-1]:
            raise ValueError(f"query time {query_time_s} outside [{t[0]}, {t[-1]}]")
        
        # Find insertion index where query_time_s would be placed to keep t sorted
        # idx is the first index such that t[idx] >= query_time_s
        idx = np.searchsorted(t, query_time_s)
        
        # Fast path: if there is an exact sample at query_time_s, return its velocity
        if idx < len(t) and np.isclose(t[idx], query_time_s):
            return v[idx].copy()
        
        # Edge case: query falls before the first sample boundary (numerical quirks); clamp to first
        if idx == 0:
            return v[0].copy()
        
        # Edge case: query falls after the last sample boundary; clamp to last
        if idx == len(t):
            return v[-1].copy()
        
        # General case: interpolate between the bracketing samples (idx-1) and idx
        t0, t1 = t[idx-1], t[idx] # neighboring times
        v0, v1 = v[idx-1], v[idx] # neighboring velocity vectors

        # Linear interpolation weight in [0, 1]: fraction of the way from t0 to t1
        alpha = (query_time_s - t0) / (t1 - t0)

        # Return the linearly interpolated velocity vector
        return (1.0 - alpha) * v0 + alpha * v1

def print_velocities_for_required_times(csv_path: Path) -> None:
    """
    Beginner-friendly helper:
    - Builds an ECEFVelocityInterpolator from your LLA CSV.
    - Evaluates the ECEF velocity vector at two required Unix times.
    - Prints the results to the console (stdout).

    Why these two times?
    The assignment requires computing velocity at:
      1532334000  and  1532335268   (seconds since Unix epoch)
    """
    # 1) Build the interpolator (loads CSV, converts to ECEF, computes velocity samples)
    interp = ECEFVelocityInterpolator.from_lla_csv(csv_path)

    # 2) The two query times (given by the problem statement)
    query_times = [1532334000.0, 1532335268.0]  # seconds since Unix epoch

    # 3) For each time, interpolate the velocity vector and print it
    print("\nECEF velocity vectors (meters/second) at required times:\n")
    for t in query_times:
        v = interp.velocity_at(t)  # returns [Vx, Vy, Vz] in m/s
        # Print in a clean, consistent format a beginner can read
        print(f"  t = {t:.0f} -> Vx = {v[0]:.6f} m/s, Vy = {v[1]:.6f} m/s, Vz = {v[2]:.6f} m/s")
    print()  # extra newline for readability


def main(argv: list[str]) -> int:
    """
    Minimal CLI entry point:
    - Expects argv[1] to be the path to the input CSV with LLA time series.
    - Loads the data, converts to ECEF positions and velocities, and writes ALL results
      to <input_stem>_ecef.csv in the same directory.
    - Produces no console output. Returns 0 on success, 2 on incorrect usage.
    """
    # Require at least one argument: the CSV path
    if len(argv) < 2:
        return 2
    
    # Input CSV path
    csv_path = Path(argv[1])
    
    # If user asks to print the two required velocities, do that and continue
    if len(argv) >= 3 and argv[2] == "--print-vel":
        print_velocities_for_required_times(csv_path)

    # Build the interpolator (assumed to: load CSV, convert to ECEF, compute velocity)
    interp = ECEFVelocityInterpolator.from_lla_csv(csv_path)

    # Save full series (time, ECEF coords, ECEF velocity) next to the input file
    out_path = csv_path.with_name(csv_path.stem + "_ecef.csv")
    
    # Build pandas DataFrame for output from the data stored in interp
    df_ecef = pd.DataFrame(
        {
            "time_s": interp.t.astype("float64"), #the time at each sample, in seconds
            "X_m": interp.r[:, 0].astype("float64"), # ECEF X coordinate in meters
            "Y_m": interp.r[:, 1].astype("float64"), # ECEF Y coordinate in meters
            "Z_m": interp.r[:, 2].astype("float64"), # ECEF Z coordinate in meters
            "Vx_mps": interp.v[:, 0].astype("float64"), # ECEF X velocity in meters/second
            "Vy_mps": interp.v[:, 1].astype("float64"), # ECEF Y velocity in meters/second
            "Vz_mps": interp.v[:, 2].astype("float64"), # ECEF Z velocity in meters/second
        }
    )

    # Write with full float64 precision to avoid cm-level round-trip errors
    df_ecef.to_csv(out_path, index=False, float_format="%.17g")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
