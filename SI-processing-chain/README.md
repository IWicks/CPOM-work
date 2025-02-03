### Welcome to the Python sea ice processing chain! 🧊🛰

This chain processes elevation and sea level anomaly data to calculate sea ice freeboard, thickness, and volume. Please consider the following caveats before using this code:
- The script must be run in same folder as data (input files can be in nested folders, but specify when calling).
- Elevation and SLA files must be textfiles, which are then read in and converted to Pandas DataFrames.
- Ensure all necessary output folders, e.g., the "gridded_data" folder, are created prior to running the script.
- Ensure the ocean_fraction_file reflects the hemisphere you are interested in, i.e., Northern or Southern.

This code is loop-based, and may run slowly when processing large amounts of data. Please be patient with it! 😊

If you have any suggestions or improvements, submit a comment or pull request.
