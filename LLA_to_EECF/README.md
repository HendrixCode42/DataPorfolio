README — LLA to ECEF Coding Challenge
===========================================

Overview
--------
This repository implements a coding challenge for converting geodetic
coordinates (LLA: latitude (in degrees), longitude (in degrees), altitude (in kilometers)) 
to Earth-Centered, Earth-Fixed coordinates (ECEF: X,Y,Z in meters), estimating ECEF velocity, 
and verifying results end-to-end.

Core files:
A. lla_to_ecef.py: 
    1. Reads a CSV file containing unnamed columns in this order:
        - Time since the Unix epoch (seconds)
        - Latitude (degrees, WGS-84)
        - Longitude (degrees, WGS-84)
        - Altitude (kilometers above the WGS-84 ellipsoid)
    2. Performs basic input validation
        Checks for missing or non-finite values and raises clear error messages if any
        problem is found, so issues can be fixed before running verification.
    3. Converts LLA to ECEF positions (in meters)
       Uses the WGS-84 ellipsoid constants (a, f, e^2) to transform each point’s
       latitude, longitude, and altitude into Cartesian coordinates X, Y, Z, measured
       in meters from the Earth’s center.
    4. Computes ECEF velocity vectors (in meters per second)
        Estimates velocity components (Vx, Vy, Vz) using finite-difference methods. It 
        takes differences between consecutive position values divided by time intervals
        to approximate how fast and in what direction the object moves in ECEF space.
        This step also checks that timestamps are strictly increasing.
    5. Supports velocity interpolation for any time value
        Includes an interpolation class that can compute an estimated velocity at
        arbitrary times between known samples. 
    6. Writes the output to a new, high-precision CSV file
        Saves the converted coordinates and velocities as
        <input file stem>_ecef.csv in the same folder as the input file.
        Every numeric column is written with full double-precision (17 significant digits)
        to avoid rounding errors that could cause small differences (on the order of
        centimeters) during verification. The output file contains the following columns:
            - time_s : The timestamp for each observation, measured as the number of seconds 
                       since the Unix epoch (in seconds)
            - X_m : The X-coordinate (in meters) of the object’s position in the Earth-Centered, 
                    Earth-Fixed (ECEF) Cartesian coordinate system.
            - Y_m : The Y-coordinate (in meters) of the object’s position in the ECEF 
                    coordinate system.
            - Z_m : The Z-coordinate (in meters) of the object’s position in the ECEF 
                    coordinate system.
            - Vx_mps : The X-component of the ECEF velocity vector (in meters per second), 
                        representing how quickly the object’s X-coordinate is changing with time.
            - Vy_mps : The Y-component of the ECEF velocity vector (in meters per second), 
                        representing how quickly the object’s Y-coordinate is changing with time.
            - Vz_mps : The Z-component of the ECEF velocity vector(in meters per second), 
                       representing how quickly the object’s Z-coordinate is changing with time.

B. verify_ecef.py
    This script independently verifies that the ECEF coordinates produced by
    lla_to_ecef.py are mathematically correct, physically reasonable,
    and internally consistent. It performs a sequence of automated checks and
    stops immediately (fail-fast) if any check fails. If all checks pass, the script 
    prints a clear confirmation in the terminal. The script does this with the following:
        1. Loads the two required input files into pandas DataFrames
            - An original LLA CSV file containing latitude (in degrees), longitude (in degrees), 
              and altitude (in kilometers).
            - The generated ECEF CSV file (<input file stem>_ecef.csv) produced by 
              lla_to_ecef.py.

        2. Runs diagnostic checks on the data
            Prints a quick summary of the ECEF coordinate magnitudes (‖r‖) and the corresponding
            approximate altitudes to confirm the dataset’s expected range.
            This helps verify that units are correct (e.g., meters vs. kilometers) and
            that the radius of the Earth and the computed altitudes are physically realistic.

        3. Calculates radius-based tolerances using the input data
            Before running formal checks, the script dynamically sets the acceptable 
            radius range (TOL["radius_min_m"] and TOL["radius_max_m"]) using the smallest 
            and the largest altitudes found in the LLA CSV. This ensures that all radius-related 
            tests are automatically scaled to a specific dataset.
        
        4. Runs a known-point unit test
            Calls known_point_unit_test() to confirm that the WGS-84 constants and the 
            LLA to ECEF formula produces correct results for a reference coordinate.
            This acts as a sanity check before testing real data.
        5. Confirms that time values match perfectly
            Calls check_times_align(lla, ecef) to ensure that the timestamps in both the 
            LLA and ECEF files are identical and strictly increasing. This prevents mismatched 
            or misaligned data rows from skewing later comparisons. 
        6. Recomputes ECEF coordinates from the LLA data
            Calls check_recompute_xyz(lla, ecef), which uses the same forward LLA to ECEF 
            transformation as the converter to calculate new ECEF positions from the 
            input LLA data, and compares these recomputed values to those stored in the 
            ECEF CSV. If the absolute difference exceeds a very small tolerance 
            (a few micrometers), the script will stop and report the discrepancy.
        7. Performs an independent inverse check (ECEF to LLA)
            - Uses the pyproj library with the official WGS-84 coordinate reference systems
              (EPSG:4978 for ECEF and EPSG:4979 for geographic coordinates) to transform
              the saved ECEF coordinates back into latitude, longitude, and altitude. These 
              LLA values are compared to the original ones to ensure accuracy. 
              - Instead of checking angular differences (in degrees), horizontal and
                vertical errors in meters are measured using the WGS-84 ellipsoid model with 
                the following tolerance parameters:
                - Horizontal tolerance: 5 cm
                - Altitude tolerance: 5 cm
        8. Checks physical sanity (radius and altitude ranges) 
            Calls check_physical_sanity(ecef) to validate that the magnitude of each 
            ECEF position vector is consistent with Earth’s known size and with the 
            altitude range found in the input file.
        9. Verifies velocity consistency
            If the ECEF CSV includes velocity components (Vx_mps, Vy_mps, Vz_mps),
            the script compares them against the finite-difference velocities recomputed
            from the position data. This ensures that the velocity and position information 
            agree physically.


C. LLA_to_ECEF_SG_test.ipynb
    This Jupyter Notebook is an exploratory test designed to find out whether applying a 
    light Savitzky–Golay (SG) smoothing filter to the original LLA time-series data 
    improves the numerical accuracy of the LLA to ECEF conversion. It helps determine if 
    small random “noise” in the input coordinates (for example, tiny jumps in altitude 
    or latitude) should be smoothed out before running the conversion script. The test 
    uses a Savitzky–Golay filter (SG filter) from the scipy.signal library to check if 
    applying light smoothing improves the accuracy or stability of the ECEF conversion results.
    - Rationale: When converting geographic coordinates to ECEF Cartesian coordinates, 
    small fluctuations in input values can create unrealistic “jitter” in the computed 
    ECEF velocities. This notebook tests whether smoothing the input signals slightly 
    using the Savitzky-Golay filter can reduce that noise without distorting the true 
    motion of the object. By sweeping through different filter parameters, you can 
    visually and numerically verify if smoothing improves accuracy or if it introduces 
    unwanted bias.

References: 
1. LLAtoECEF.pdf :
  - The lla_to_ecef.py script directly implements the equations and constants 
    described in LLAtoECEF.pdf for converting geodetic coordinates into ECEF Cartesian 

2. PROJ Project (2025). Geodetic Transformation — PROJ Documentation.
    - Informed the design of the inverse conversion step in verify_ecef.py by demonstrating 
    how ECEF (EPSG:4978) and geodetic (EPSG:4979) coordinate reference systems are transformed 
    using pyproj. https://proj.org/en/stable/usage/transformation.html
    - Served as a cross-reference for the LLA to ECEF forward transformation logic, confirming 
    that the equations and variable naming conventions in lla_to_ecef.py match common 
    Python geodesy implementations.
    - Provided the exact usage pattern (Transformer.from_crs(..., always_xy=True)) that 
    the inverse check in verify_ecef.py follows to ensure correct axis ordering and 
    reproducible results.
3. NumPy Documentation
    - Guided the efficient array-based implementation of LLA to ECEF transformations 
    rather than per-row loops, improving runtime for satellite trajectory datasets.
    https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
4. SciPy Signal Processing (Savitzky-Golay filter)
   - Informed the preprocessing step used in the accompanying test notebook to smooth 
   LLA inputs before finite-difference velocity estimation, as described in the 
   “LLA_to_ECEF_SG_test.ipynb” workflow.
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
   https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
5. GNSS Data Preprocessing with Savitzky–Golay (Georust project)
   - Illustrated real-world GNSS preprocessing pipeline using SG smoothing before 
   coordinate transformation.
   https://github.com/gnss-sdr/gnss-sdr


Environment / Dependencies
--------------------------
- Python 3.10 or 3.11
- numpy>=1.24
- pandas>=2.0
- pyproj>= 3.6
- scipy>=1.10
- jupyter>=1.0
- ipykernel>=6.0
- matplotlib>=3.7

Quick setup (recommended):
    pip install -U pip
    pip install numpy pandas pyproj
    pip install scipy matplotlib jupyter


Build / Run
-----------
A. Convert LLA to ECEF (lla_to_ecef.py)
   From the project root: 
   python lla_to_ecef.py code_problem_data.csv --print-vel

   This will generate code_problem_data_ecef.csv in the same folder

B. Verify ECEF correctness (verify_ecef.py)
    After running lla_to_ecef.py ensure you have code_problem_data_ecef.csv
    and code_problem_data.csv available in the same folder. Use the following from 
    the project root: 
    python verify_ecef.py code_problem_data.csv code_problem_data_ecef.csv

C. Exploratory test for Savitzky–Golay (SG) smoothing filter (LLA_to_ECEF_SG_test.ipynb)
   is necessary. Ensure code_problem_data.csv is in the same folder. After installing 
   dependencies, run each cell in order

