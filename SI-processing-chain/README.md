## Welcome to the Python sea ice processing chain! 🧊🛰

This chain processes elevation and sea level anomaly data to calculate sea ice freeboard, thickness, and volume.

### Please consider the following caveats before using this code:
- The script must be run in same folder as data (input files can be in nested folders, but specify when calling).
- Elevation and SLA files must be textfiles, which are then read in and converted to Pandas DataFrames.
- Ensure all necessary output folders, e.g., the "gridded_data" folder, are created prior to running the script.
- Ensure the ocean_fraction_file reflects the hemisphere you are interested in, i.e., Northern or Southern.

### Python requirement:
Ensure **at least** Python v3.10 is installed or available before use. I recommend using miniconda for installation of Python v3.10 (or another other appropriate method, if you're comfortable). If using miniconda installation, select the Python v3.10 installer for your OS from:

https://docs.conda.io/en/latest/miniconda.html

You may need to start a new shell to refresh your environment before checking that Python v3.10 is in your path. To check that python v3.10 is now available, type:

```
python -V
```
### Packages and libraries
- The NumPy and Pandas modules are required for this script.
- I also recommend creating a new environment to download the required packages and process any data, to keep things neat and tidy.

This code is loop-based, and may run slowly when processing large amounts of data. Please be patient with it! 😊

If you have any suggestions or improvements, submit a comment or pull request.
