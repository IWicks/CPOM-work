"""
Isabelle Wicks, Northumbria University (30/1/2025)

Sea ice data processing chain.

Recreated from Tilling, R. L., Ridout, A., and Shepherd, A. (2018). Estimating Arctic sea ice thickness and volume using CryoSat-2 radar
altimeter data. Advances in Space Research, 62(6), 1203–1225. https://doi.org/10.1016/j.asr.2017.10.051

Description of processing chain's purpose and steps:
    Imports sea ice data (elevation, sea level anomaly, associated metadata etc.) from files.
    Imports cell data and creates gridded masks (ocean fraction and cell area).
    Processes elevation, sea level anomaly etc. data to calculate freeboard, sea ice height, sea ice thickness.
    Filters out spurious and extreme data.
    Creates gridded sea ice thickness, sea ice concentration, and sea ice extent masks.
    Creates gridded sea ice volume mask.
    Multiplies cell masks and sea ice volume mask to calculate total sea ice volume.
    
Description of outputs:
    final_DF.to_csv saves each Dataframe with same filename as processed elevation and SLA files.
    Final gridded data (sea ice thickness, sea ice concentration, sea ice extent, no. of samples per grid cell, sea ice volume) saved
    in compressed .npz file.

Main load() function steps:
    Reads file.
    Loads data into Pandas DataFrame.
    Delimits data and assigns column names.

Main locate_samples() function steps:
    Reads in data from DataFrame.
    Identifies and matches samples where defined array matches conditions.

Main time_to_ts() function steps:
    Reads "Time" from DataFrame.
    Converts "Time" into "Timestamps" for improved sample matching.

Caveats:
    Script must be run in same folder as data (input files can be in nested folders, but specify when calling).
    Elevation and SLA files must be textfiles.
"""

# ===============================================================================================
# Import modules
# ===============================================================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import glob
from pathlib import Path

# ===============================================================================================
# Utilities to load files - credit to Ben Palmer, CPOM
# ===============================================================================================

# Function to load elevation files, after Andy Ridout, CPOM.
def load_andy_elev_file(filename): # Use relative or absolute path, dpeneding on where file is saved.
    i_andy_data = pd.DataFrame(
        np.genfromtxt(filename)[:, :-1],
        columns=[
            "Surface type",
            "Valid",
            "Packet ID",
            "Block No",
            "Time",
            "Latitude",
            "Longitude",
            "Elevation",
            "MSS",
            "Peakiness",
            "Backscatter",
            "Sea Ice Conc",
            "Sea Ice Type",
            "Confidence",
            "Sigma of fit",
            "Error in fit",
        ],
    )
    # i_andy_data["Time"] = i_andy_data["Time"].apply(andy_time_to_ts)
    i_andy_data["Valid"] = i_andy_data["Valid"].astype(bool)
    i_andy_data["Sea Ice Type"] = i_andy_data["Sea Ice Type"].astype(int)
    i_andy_data["Confidence"] = i_andy_data["Confidence"].astype(int)
    i_andy_data["Surface type"] = i_andy_data["Surface type"].astype(int)
    i_andy_data["Packet ID"] = i_andy_data["Packet ID"].astype(int)
    i_andy_data["Block No"] = i_andy_data["Block No"].astype(int)
    return i_andy_data

# Function to load freeboard files, after Andy Ridout, CPOM.
def load_andy_fb_file(filename): # Use relative or absolute path, dpeneding on where file is saved.
    i_andy_data = pd.DataFrame(
        np.genfromtxt(filename),
        columns=[
            "Packet ID",
            "Block Number",
            "Time",
            "Latitude",
            "Longitude",
            "Freeboard",
            "Interp SLA",
            "Backscatter",
            "Ice%",
            "Sea Ice Type",
            "Confidence",
            "Snow Flag",
            "Snow depth",
            "Snow density",
            "",
        ],
    )
    i_andy_data["Time"] = i_andy_data["Time"].apply(andy_time_to_ts)
    return i_andy_data


# Function to load files, after Andy Ridout, CPOM.
def load_andy_file(filename): # Use relative or absolute path, dpeneding on where file is saved.
    extension = filename.split(".")[-1]
    match extension:
        case "sla":
            # load sea level anomaly (SLA) file
            andy_data = load_andy_sla_file(filename)
        case "fb":
            # load freeboard (FB) file
            andy_data = load_andy_fb_file(filename)
        case "elev":
            # load elevation (elev) file
            andy_data = load_andy_elev_file(filename)
        case _:
            raise TypeError(f"Not supported file type {extension}")
    return andy_data

# Function to load sea level anomaly (SLA) files, after Andy Ridout, CPOM.
def load_andy_sla_file(filename): # Use relative or absolute path, dpeneding on where file is saved.
    i_andy_data = pd.DataFrame(
        np.genfromtxt(filename),
        columns=[
            "SLA Flag",
            "Packet ID",
            "Block Number",
            "Time",
            "Latitude",
            "Longitude",
            "RAW SLA",
            "Smoothed SLA",
            "Backscatter",
            "",
        ],
    )
    i_andy_data["Time"] = i_andy_data["Time"].apply(andy_time_to_ts)
    i_andy_data["SLA Flag"] = i_andy_data["SLA Flag"].astype(bool)
    return i_andy_data

# DO NOT USE, underscore at the beginning means it is for internal use only. Defined for locate_samples_2.
def _locate_sample(packet_number:int, block_number:int, time:float, packet_array:np.ndarray, block_array:np.ndarray, time_array:np.ndarray) -> int:
    """Internal function to locate the sample position in 3 arrays using the packet number, block number and time.

    Args:
        packet_number (int): The packet number
        block_number (int): The block number
        time (float): The time
        packet_array (np.ndarray): The packet array which we're searching in
        block_array (np.ndarray): The block array which we're searching in
        time_array (np.ndarray): The time array which we're searching in

    Returns:
        int: Location of the sample
    """
    
    # These return boolean arrays where the array matches the condition.
    packet_match_array = packet_array == packet_number # Check which array items match packet number.
    block_match_array = block_array == block_number # Check which array items match block number.
    time_close_array = np.abs(time - time_array) < 1e-4 # Check which array items are within 1e-4 of the time.
    
    # Check where all 3 conditions were met using binary AND operator (&).
    find_array = packet_match_array & block_match_array & time_close_array
    
    # If there were no matches, return -1.
    # Else, return the location in the array.
    # (np.any checks if any item in the array is true)
    if not np.any(find_array):
        return -1
    else:
        return np.where(find_array)[0][0]

def locate_samples_2( # Need to define each positional argument as NumPy arrays and not Pandas series when calling the function.
    time_1: np.ndarray, 
    block_1: np.ndarray, 
    packet_1: np.ndarray, 
    time_2: np.ndarray, 
    block_2: np.ndarray, 
    packet_2: np.ndarray) -> tuple:
    """Finds the indexes of 2 sets of arrays where items are matching.
    

    Args:
        time_1 (np.ndarray): Time variable for dataset 1
        block_1 (np.ndarray): Block number variable for dataset 1
        packet_1 (np.ndarray): Packet number variable for dataset 1
        time_2 (np.ndarray): Time variable for dataset 2
        block_2 (np.ndarray): Block number variable for dataset 2
        packet_2 (np.ndarray): Packet number variable for dataset 2

    Returns:
        tuple: (Index for dataset 1, Index for dataset 2) where items were matchings
    """
    # Create placeholders to put data into when we find them.
    # Using lists and not NumPy arrays, do not know how many samples are matched in advance.
    out_1 = []
    out_2 = []
    
    # Loop through every item in the arrays, attaching the location in the original array.
    for loc, (time, block_no, packet_no) in enumerate(zip(time_1, block_1, packet_1)):
        # Use _locate_sample to find the sample number
        found_index = _locate_sample(packet_no, block_no, time, packet_2, block_2, time_2)
        
        # If a matching sample is found, add the numbers to the 'out' arrays.
        if found_index != -1:
            out_1.append(loc)
            out_2.append(found_index)
    
    # Convert to NumPy arrays wheny are complete.
    return np.asarray(out_1), np.asarray(out_2)

# ===============================================================================================
# Utility to convert time - credit to Ben Palmer, CPOM
# ===============================================================================================

def andy_time_to_ts(andy_value): # Converting time to timestamp.
    milsec_frac, days = np.modf(andy_value) # np.modf returns the fractional and integral parts of an array, element-wise.
    return (
        timedelta(days=days, milliseconds=milsec_frac * 8.64e7)
        + datetime(1950, 1, 1, tzinfo=timezone.utc)
    ).timestamp()

# ===============================================================================================
# Loading in elevation and sea level anomaly (SLA) data
# ===============================================================================================

# Assumes files are in local system. Filepaths will require chaning if running on a remote server.

# Elevation data.
elev_filepaths = glob.glob("elev_files/cry_NO_20231015T*") # Change as appropriate for your filename.

# SLA data.
sla_filepaths = glob.glob("sla_files/cry_NO_20231015T*") # Change as appropriate for your filename.

# Currently set up to analyse a day at a time, can be changed to analyse multiple days/months etc.

# ===============================================================================================
# Creating cell area and ocean fraction masks
# ===============================================================================================

# Cell area mask.
cell_area_file = pd.read_fwf("cell_area_file.dat", header=None) # File containing cell areas.
cell_area_mask = np.zeros((500,720)) # Initialise 500 x 720 array, latitude by longitude.
cell_area_lat = cell_area_file.iloc[:,0] # Locating latitudes.
cell_area_lon = cell_area_file.iloc[:,1] # Locating longitudes.
cell_area_value = cell_area_file.iloc[:,2] # Locating cell area values.
cell_area_ilat = ((cell_area_lat-40)/0.1).astype(int) # Converts latitudes into suitable grid projection, in integer form.
cell_area_ilon = ((cell_area_lon+180)/0.5).astype(int) # Converts longitudes into suitable grid projection, in integer form.
cell_area_mask[cell_area_ilat, cell_area_ilon] += cell_area_value # Creates cell area gridded mask.

# Ocean fraction mask.
ocean_frac_file = pd.read_fwf("ocean_fraction_file_N.dat", header=None) # File containing mask of fraction of cells believed to be ocean.
ocean_frac_mask = np.zeros((500,720))
ocean_frac_lat = ocean_frac_file.iloc[:,0]
ocean_frac_lon = ocean_frac_file.iloc[:,1]
ocean_frac_value = ocean_frac_file.iloc[:,2] # Locating ocean fraction values.
ocean_frac_ilat = ((ocean_frac_lat-40)/0.1).astype(int)
ocean_frac_ilon = ((ocean_frac_lon+180)/0.5).astype(int)
ocean_frac_mask[ocean_frac_ilat, ocean_frac_ilon] += ocean_frac_value # Creates ocean fraction gridded mask.

# ===============================================================================================
# Initialising sea ice volume, thickness, concentration, and extent masks
# ===============================================================================================

ice_volume = np.zeros((500,720)) # Initialise 500 x 720 array, latitude by longitude.
SIT_mask = np.zeros((500,720)) # Sea ice thickness.
SIC_mask = np.zeros((500,720)) # Sea ice concentration.
SI_extent_mask = np.zeros((500,720)) # Sea ice extent.

# This array will be to count the number of samples per cell.
nin = np.zeros((500,720))

# ===============================================================================================
# Calculating freeboard, ice thickness, and filtering data - looped over each file
# ===============================================================================================

for file in range(len(elev_filepaths)): # Should have elev-SLA file pairs, so same length.

        # ===============================================================================================
        # Calculating freeboard
        # ===============================================================================================
        
        elev_file = load_andy_file(elev_filepaths[file]) # Calling the elevation file.
        elev_file["Timestamps"] = elev_file["Time"].apply(andy_time_to_ts) # Pandas function to apply a given function to a data range.
        
        sla_file = load_andy_sla_file(sla_filepaths[file]) # Calling the SLA file.

        freeboard = [] # Elevation minus mean sea surface (MSS) and SLA, into a list of freeboard heights.
        elev = elev_file.loc[:,"Elevation"] 
        MSS = elev_file.loc[:,"MSS"] 

        # Defining positional arguments for locate_samples_2 as NumPy arrays.
        elev_time = elev_file["Timestamps"].to_numpy()
        elev_block = elev_file["Block No"].to_numpy()
        elev_packet = elev_file["Packet ID"].to_numpy()
        sla_time = sla_file["Time"].to_numpy()
        sla_block = sla_file["Block Number"].to_numpy()
        sla_packet = sla_file["Packet ID"].to_numpy()

        # sla_index - Index of all the samples in the SLA DF that can be found in the elevation DF.
        # elev_index - Index of all the samples in the elevation DF that can be found in the SLA DF.
        elev_index, sla_index = locate_samples_2(elev_time, elev_block, elev_packet, sla_time, sla_block, sla_packet) 
        sla_values = np.zeros(len(elev_file)) * np.nan # Make an array that is the same size as the elevation DF, populated by NaNs.
        sla_values[elev_index] = sla_file.loc[sla_index, "Smoothed SLA"].to_numpy() # Fill in the values where they have been matched.

        for i in range(len(elev)):
            result = elev[i] - MSS[i] - sla_values[i]
            freeboard.append(result)
        # List of freeboard measurements as floating point numbers.
        # Keep as floating points for precision, sea ice on order of cm, whereas icebergs on order of m.

        elev_file["Freeboard"] = freeboard # Appends freeboard list as a column to elev_files DF.
        
        filtered_freeboard = elev_file.loc[elev_file["Freeboard"] > -0.3]
        # DF of freeboard measurements > -0.3 m with original index values.
        # Keep some small negative values to not bias freeboard measurements too high.
        # For sea ice, anything over 3 m is excluded as it is too thick to be sea ice. Icebergs on order of m, will be > 3 m thick.
        
        # ===============================================================================================
        # Calculating sea ice thickness
        # ===============================================================================================
            
        ice_thicknesses = np.zeros(len(filtered_freeboard)) * np.nan # Initialise an array that is the same size as the filtered DF.
        freeboard_2 = filtered_freeboard.loc[:,"Freeboard"] # Locating freeboard data, cannot be same name as original freeboard series.
        where_fyi = filtered_freeboard["Sea Ice Type"] == 2 # Locating first year ice data.
        where_myi = filtered_freeboard["Sea Ice Type"] == 3 # Locating multi-year ice data.
        
        ice_thicknesses[where_fyi] = ((freeboard_2[where_fyi]*1023.9) + (0.1716*268.1507))/(1023.9 - 916.7) # First year ice density = 916.7 kgm^-3.
        ice_thicknesses[where_myi] = ((freeboard_2[where_myi]*1023.9) + (0.1716*268.1507))/(1023.9 - 882) # Multi-year ice density = 882.0 kgm^-3.
        
        filtered_freeboard["Ice Thickness"] = ice_thicknesses # Appends ice thickness array as a column to filtered_freeboard DF.
        
        # ===============================================================================================
        # Filtering out false data
        # ===============================================================================================

        # Filtering out all data flagged "false", any ice thicknesses of NaN, and filter ice thicknesses to -0.3 < x < 5 to remove outliers.
        final_DF = filtered_freeboard.dropna() # Filtering out NaN values.
        final_DF = filtered_freeboard[(filtered_freeboard["Ice Thickness"] > -0.3) & # Same as freeboard, keep slightly -ve values to not bias measurements.
                                            (filtered_freeboard["Ice Thickness"] < 5) & # Removing anything anomalous above 5 m thickness.
                                            (filtered_freeboard["Valid"] == True)] # Removing "False" flags.
        
        # Some filters that would ideally be included are not present in this section, as they may filter out necessary data. These can be inclued, if required.
        
        output_filename = Path(elev_filepaths[file]).stem+".csv"
        
        final_DF.to_csv("output_files/"+output_filename) # Change as appropriate for your filename. Ensure "output_files" is created prior to saving.
        
        # ===============================================================================================
        # Creating sea ice thickness, concentration, and extent masks
        # ===============================================================================================
        
        SIT = final_DF.loc[:,"Ice Thickness"] # Locate sea ice thickness.
        SIC = final_DF.loc[:, "Sea Ice Conc"] # Locate sea ice concentration.
        SI_extent = final_DF["Sea Ice Conc"] >= 75 # Sea ice extent, only values equal to or above 75% sea ice concentration are valid.
        
        # Converting latitude and longitude to grid projection.
        lat = final_DF.loc[:,"Latitude"] # Locate latidues in final DF.
        lon = final_DF.loc[:,"Longitude"] # Locate longitudes in final DF.
        ilat = ((lat-40)/0.1).astype(int) # Converts latitudes into suitable grid projection, in integer form.
        ilon = ((lon+180)/0.5).astype(int) # Converts longitudes into suitable grid projection, in integer form.

        # Creating masks. Each iteration of loop adds to each mask (not overwriting).
        SIT_mask[ilat, ilon] += SIT # Added to gridded mask of ilats and ilons.
        SIC_mask[ilat, ilon] += SIC
        SI_extent_mask[ilat, ilon] += SI_extent
        SI_extent_mask = SI_extent_mask.astype(bool) # Converts to true/false.
        nin[ilat, ilon] += +1 # Counts number of samples per cell.

# ===============================================================================================
# Calculating sea ice volume
# ===============================================================================================

for j in range(500): # Latitudes.
    for k in range(720): # Longitudes.
        if nin[j,k] >= 1: # If the number of samples is 1 or more.
            avg_thickness = SIT_mask[j,k]/nin[j,k] # Calculate average sea ice thickness per cell.
            avg_conc = SIC_mask[j,k]/nin[j,k] # Calculate average sea ice concentration per cell.
            ice_volume[j,k] = avg_thickness * 0.001 * avg_conc * 0.01 # Did it this way as it is what Andy did.

# Applying masks to calculated sea ice volume.
final_volume = ice_volume * SI_extent_mask * cell_area_mask * ocean_frac_mask # Gridded data (array) of final cell volumes.

# Saving masks as NumPy compressed .npz files
np.savez_compressed('gridded_data/cry_NO_20231015', SIT_mask=SIT_mask, SIC_mask=SIC_mask, SI_extent_mask=SI_extent_mask,
                    nin=nin, final_volume=final_volume) # Change file name as appropriate. Ensure "gridded_data" is created prior to saving.