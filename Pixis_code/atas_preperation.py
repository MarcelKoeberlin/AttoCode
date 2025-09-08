# -*- coding: utf-8 -*-
"""
This script processes raw data from an Attosecond Transient Absorption
Spectroscopy (ATAS) experiment. It focuses on synchronising XUV spectra
with delay stage positions based on timestamps.

The main steps are:
1.  Load XUV spectra, XUV timestamps, and delay stage trigger timestamps.
2.  Identify synchronisation patterns within the XUV timestamps. These patterns
    correspond to the start of each delay scan.
3.  Segment the continuous XUV data into chunks, where each chunk represents
    one full scan of the delay stage.
4.  Within each chunk, label individual spectra as 'ON' (pump and probe beams
    are present), 'OFF' (only probe beam is present), or 'DISCARDED' based on
    the known experimental sequence.
5.  Generate and display the final ATAS trace (ΔA) by processing the labelled
    'ON' and 'OFF' spectra.
"""
import json
import os
import re
import sys
import warnings
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np

# Assuming the 'misc_functions' module is in the specified path and contains the necessary plotting functions.
sys.path.append('C:\\Users\\Moritz\\OneDrive - ETH Zurich\\Code for ThinkPad to test!')
from misc_functions import atas_on_off_clas, plot_on_off_clas_shift_sweep

# GLOBAL CONSTANTS #############################################################
SYNC_GAPS_PATTERN = [1, 4, 3, 3, 1]  # Always the same for all measurements
XUV_EV_CALIBRATION_FILE = "Spec.txt"  # Always the same for all measurements

# MEASUREMENT CONFIGURATION ####################################################
# Manually define these parameters for each measurement
XUV_DATE = "250904"                         # YYMMDD format
XUV_SEQUENCE = 1                          # X for YYMMDD_00X
DELAY_STAGE_DATE = XUV_DATE                 # YYMMDD format (same as XUV_DATE typically)
DELAY_STAGE_SEQUENCE = 4         # X for YYMMDD_00X (can be different from XUV)
SYNC_PATTERN_TOLERANCE_US = 40000           # Tolerance for detecting sync pattern gaps
SYNC_PULSES_TO_SKIP = 3                     # Number of initial pulses to discard
PULSES_TO_SKIP_BEFORE_FIRST_SHUTTER = 5     # Number of pulses to skip before shutter
SHIFT = -1                                  # Manual shift for ON/OFF labelling


class Paths:
    """Defines the main directory paths for input data."""
    BASE_DIR = r"C:\Users\Moritz\Desktop\TESTDATA"
    SERVER_DIR = r"Z:\Attoline"  # Base server directory
    # The folder where all generated plots and data will be stored.
    OUTPUT_FOLDER = r"C:\Users\Moritz\Desktop\Pixis_data\output"


def find_xuv_files(yymmdd: str, sequence_num: int) -> Tuple[str, str]:
    """
    Find XUV .npy and .json files based on the new directory structure.
    
    Args:
        yymmdd: Date in YYMMDD format (e.g., "250903")
        sequence_num: Sequence number (e.g., 1 for _001)
        
    Returns:
        Tuple of (json_path, npy_path)
    """
    year = "20" + yymmdd[:2]  # Convert YY to YYYY
    sequence_str = f"{yymmdd}_{sequence_num:03d}"
    
    # Build the directory path
    data_dir = os.path.join(Paths.BASE_DIR, year, "STRA", yymmdd, sequence_str)
    
    json_path = os.path.join(data_dir, f"{sequence_str}.json")
    npy_path = os.path.join(data_dir, f"{sequence_str}.npy")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"NPY file not found: {npy_path}")
        
    return json_path, npy_path


def find_delay_stage_file(yymmdd: str, sequence_num: int) -> str:
    """
    Find delay stage .npz file based on the new directory structure.
    
    Directory structure: server_dir/2025/DelayStage/2025/DelayStage/YYMMDD/YYMMDD_XXX/delay_YYMMDD_SXXX.npz
    
    Args:
        yymmdd: Date in YYMMDD format (e.g., "250903")
        sequence_num: Sequence number (e.g., 2 for S002)
        
    Returns:
        Full path to the delay stage .npz file
    """
    year = "20" + yymmdd[:2]  # Convert YY to YYYY
    sequence_str = f"{yymmdd}_{sequence_num:03d}"
    
    # Build the directory path: server_dir/2025/DelayStage/2025/DelayStage/250903/250903_002/
    data_dir = os.path.join(
        Paths.SERVER_DIR, 
        year, 
        "DelayStage", 
        yymmdd, 
        sequence_str
    )
    
    # Build the filename: delay_250903_S002.npz
    filename = f"delay_{yymmdd}_S{sequence_num:03d}.npz"
    npz_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Delay stage NPZ file not found: {npz_path}")
        
    return npz_path


class Settings:
    """Contains global settings and flags for the script's execution."""
    # Plot an overview of all timestamps and detected sync points. Useful for debugging timing issues.
    PLOT_TIMING_OVERVIEW = True
    # Plot the final on/off spectra and the calculated ATAS trace.
    PLOT_ON_OFF_SPECTRUM = True


class DataLabel(IntEnum):
    """Enumeration for labelling each acquired data point."""
    OFF = 0  # Pump beam blocked
    ON = 1  # Pump and probe beams present
    TRAINING = 2  # Data not used for ON/OFF, potentially for ML
    DISCARDED = 3  # Data to be ignored entirely


def get_metadata(json_path: str) -> Dict:
    """Loads metadata from a specified JSON file.

    Args:
        json_path: Path to the metadata JSON file.

    Returns:
        A dictionary containing the metadata.
    """
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata


def load_xuv_spectra_and_timestamps(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads XUV spectra and their timestamps from a memory-mapped file.

    Args:
        json_path: Path to the metadata JSON file that describes the memmap file.

    Returns:
        A tuple containing:
        - A 2D NumPy array of spectra (N_frames, N_pixels).
        - A 1D NumPy array of timestamps in microseconds.
    """
    metadata = get_metadata(json_path)
    print('the json path is at ', json_path)
    memmap_path = os.path.join(os.path.dirname(json_path), metadata["memmap_file"])

    # Define the structured data type for the memory-mapped file
    dtype = np.dtype([
        ("spectrum", np.uint16, metadata["spectra_shape"][1]),
        ("timestamp_us", np.uint64)
    ])

    # Read the data and convert from memmap to standard numpy arrays
    mmap = np.memmap(memmap_path, dtype=dtype, mode="r")
    return np.array(mmap["spectrum"]), np.array(mmap["timestamp_us"])


def load_delay_stage_data(filename: str) -> Tuple[np.ndarray, int, int, int, Optional[float]]:
    """Reads delay stage data, extracting timestamps and experimental parameters from the .npz file.

    Args:
        filename: Path to the .npz file containing delay stage timestamps and parameters.

    Returns:
        A tuple containing:
        - timestamps_us (np.ndarray): Trigger timestamps in microseconds.
        - ppas (int): Pulses per acquisition step.
        - spss (int): Spectra per shutter step (duration of ON or OFF block).
        - spds (int): Steps per delay scan (number of ON/OFF pairs).
        - move_step_fs (float | None): Move step size in femtoseconds, if found.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
   # Load data from the .npz file
    print('filename:', filename)
    
    try:
        npz_data = np.load(filename)
        available_keys = list(npz_data.keys())
        print(f"Available arrays in .npz file: {available_keys}")
        
        # Load timestamps
        timestamp_key = None
        for key in ['timestamps', 'timestamps_us', 'data', 'arr_0']:
            if key in available_keys:
                timestamp_key = key
                break
        
        if timestamp_key is None:
            raise ValueError(f"Could not find timestamp data in .npz file. Available keys: {available_keys}")
        print('the npz data is:', npz_data[timestamp_key])
        timestamps_us = np.array(npz_data[timestamp_key])
        timestamps_us = timestamps_us[timestamps_us > 0]  # Filter out zero entries
        print(f"Using key '{timestamp_key}' for timestamp data")
        
        # Load parameters from .npz file variables
        ppas = int(npz_data['ppas']) if 'ppas' in available_keys else None
        spss = int(npz_data['spss']) if 'spss' in available_keys else None
        spds = int(npz_data['spds']) if 'spds' in available_keys else None
        move_step_fs = float(npz_data['move_step_fs']) if 'move_step_fs' in available_keys else None
      
        # Check that required parameters were found
        if ppas is None or spss is None or spds is None:
            # Fallback: try to parse from filename for backward compatibility
            print("Warning: Could not find ppas, spss, spds in .npz file, trying filename parsing...")
            match = re.search(r"ppas_(\d+)_spss_(\d+)_spds_(\d+)", filename)
            if not match:
                raise ValueError("Could not find ppas, spss, and spds in .npz file or filename.")
            ppas = int(match.group(1))
            spss = int(match.group(2))
            spds = int(match.group(3))
        
        print(f"Parameters: ppas={ppas}, spss={spss}, spds={spds}, move_step_fs={move_step_fs}")
        
        return timestamps_us, ppas, spss, spds, move_step_fs
        
    except Exception as e:
        raise ValueError(f"Error loading .npz file: {e}")
    finally:
        if 'npz_data' in locals():
            npz_data.close()


def find_sync_patterns(timestamps_us: np.ndarray, expected_gaps_us: List[int], tolerance_us: int) -> np.ndarray:
    """Detects synchronisation patterns in a timestamp array.

    A pattern is defined by a specific sequence of time gaps between consecutive timestamps.

    Args:
        timestamps_us: An array of timestamps in microseconds.
        expected_gaps_us: A list of the expected time gaps that form the sync pattern.
        tolerance_us: The allowed deviation in microseconds for each gap.

    Returns:
        An array of indices, where each index points to the start of a detected sync pattern.
    """
    sync_indices = []
    num_gaps = len(expected_gaps_us)
    i = 0
    # Iterate through the timestamps to find the pattern
    while i <= len(timestamps_us) - (num_gaps + 1):
        # Get a slice of timestamps that could form one pattern
        t_slice = timestamps_us[i:i + num_gaps + 1]
        gaps = np.diff(t_slice)

        # Check if all calculated gaps match the expected gaps within the tolerance
        if all(np.abs(g - e) < tolerance_us for g, e in zip(gaps, expected_gaps_us)):
            sync_indices.append(i)
            # Skip forward to prevent re-detecting overlapping patterns.
            # A cooldown of num_gaps is a safe choice.
            i += num_gaps
        else:
            i += 1
    return np.array(sync_indices)


def create_on_off_mask(chunk_length: int, spss: int, spds: int,
                       sync_pulses_to_skip: int, pulses_to_skip: int,
                       shift: int) -> np.ndarray:
    """Creates a mask to label each data point in a chunk as ON, OFF, TRAINING, or DISCARDED.

    Args:
        chunk_length: The total number of data points in the chunk.
        spss: Spectra per shutter step (duration of an ON or OFF block).
        spds: Steps per delay scan (number of ON/OFF pairs).
        sync_pulses_to_skip: Number of initial pulses to label as DISCARDED.
        pulses_to_skip: Number of subsequent pulses to label as TRAINING.
        shift: A manual timing shift to apply to the ON/OFF blocks.

    Returns:
        A 1D NumPy array of labels (see DataLabel enum) for the chunk.
    """
    # Start by labelling everything as TRAINING.
    mask = np.full(chunk_length, DataLabel.TRAINING, dtype=np.uint8)

    # Label initial pulses for synchronisation and stabilisation.
    mask[:sync_pulses_to_skip] = DataLabel.DISCARDED
    mask[sync_pulses_to_skip:pulses_to_skip] = DataLabel.TRAINING

    # Apply the ON/OFF pattern for each delay step.
    current_index = pulses_to_skip + shift
    for _ in range(spds):
        # ON block
        mask[current_index:current_index + spss] = DataLabel.ON
        current_index += spss
        # OFF block
        mask[current_index:current_index + spss] = DataLabel.OFF
        current_index += spss
    return mask

# def prepare_atas_inputs(selected_set_name: Optional[str] = None) -> Dict[str, object]:
#     """Prepare inputs for ATAS computation without plotting.

#     This function mirrors the loading, synchronization, chunking, and labeling
#     steps from main(), but returns the assembled arrays and metadata in a dict
#     so that MATLAB (MoritzReader.m) can compute and plot ATAS.

#     Args:
#         selected_set_name: Optional name of the training set to use (e.g., "TS0028").
#                            If None, uses TrainingSets.SELECTED.

#     Returns:
#         A dictionary with keys:
#             - 'final_spectra' (np.ndarray, float32) [N x E]
#             - 'final_identifiers' (np.ndarray, int32) [N x 3]
#             - 'xuv_energy_ev' (np.ndarray, float32) [E]
#             - 'delay_stepsize_mm' (float)
#             - 'ppas' (int), 'spss' (int), 'spds' (int)
#             - 'config_name' (str)
#     """
#     # Resolve configuration
#     config: Dict[str, Any]
#     if selected_set_name is None:
#         config = cast(Dict[str, Any], TrainingSets.SELECTED)
#     else:
#         # Allow matching by attribute name (e.g., "TS0028") or by config['Name']
#         found: Optional[Dict[str, Any]] = None
#         for attr in dir(TrainingSets):
#             if attr.startswith("TS"):
#                 cfg = getattr(TrainingSets, attr)
#                 if isinstance(cfg, dict) and (attr == selected_set_name or cfg.get("Name") == selected_set_name):
#                     found = cast(Dict[str, Any], cfg)
#                     break
#         if found is None:
#             available = [a for a in dir(TrainingSets) if a.startswith("TS")]
#             raise ValueError(f"Unknown training set '{selected_set_name}'. Available: {available}")
#         config = found

#     # Locate files
#     # xuv_data_file = [f for f in os.listdir(Paths.XUV_SPECTRA) if f.startswith(config["XUV_date"]) and f.endswith(".npy")]
#     # xuv_meta_file = [f for f in os.listdir(Paths.XUV_SPECTRA) if f.startswith(config["XUV_date"]) and f.endswith(".json")]
#     # delay_file = [f for f in os.listdir(Paths.DELAY_STAGE_TIMES) if f.startswith(config["DELAY_STAGE_date"]) and f.endswith(".npz")]
#     # if not (len(xuv_data_file) == 1 and len(xuv_meta_file) == 1 and len(delay_file) == 1):
#     #     raise FileNotFoundError("Could not find exactly one file for each data type (XUV data, XUV meta, Delay).")

#     # Load data
#     # xuv_spectra, xuv_timestamps_us = load_xuv_spectra_and_timestamps(os.path.join(Paths.XUV_SPECTRA, xuv_meta_file[0]))
#     # delay_timestamps_us, ppas, spss, spds, delay_stepsize_mm = load_delay_stage_data(os.path.join(Paths.DELAY_STAGE_TIMES, delay_file[0]))
#     # xuv_energy_ev = np.loadtxt(os.path.join(Paths.XUV_SPECTRA, config["XUV_eV_calibration_file"]))
    


#     # Synchronisation: detect start of scans via sync gaps pattern
#     # sync_gaps_us: List[int] = [int(round(float(element) * (ppas * (1 / 1030) * 1e6))) for element in config["SYNC_GAPS_PATTERN"]]
#     # sync_indices = find_sync_patterns(xuv_timestamps_us, sync_gaps_us, config["SYNC_PATTERN_TOLERANCE_US"]) + 2

#     # Segment into chunks (one per delay scan)
#     chunk_indices = np.split(np.arange(len(xuv_spectra)), sync_indices)[1:]
#     chunk_lengths = [len(c) for c in chunk_indices]
#     if len(chunk_lengths) == 0:
#         raise RuntimeError("No chunks detected from XUV sync patterns; check timestamps or pattern settings.")

#     # Build ON/OFF mask using first complete chunk length
#     chunk_length = chunk_lengths[0]
#     on_off_mask = create_on_off_mask(
#         chunk_length=chunk_length,
#         spss=spss,
#         spds=spds,
#         sync_pulses_to_skip=config["SYNC_PULSES_TO_SKIP"],
#         pulses_to_skip=config["PULSES_TO_SKIP_BEFORE_FIRST_SHUTTER"],
#         shift=config["SHIFT"],
#     )

#     final_spectra: List[np.ndarray] = []
#     # identifiers: (DataLabel, chunk_index, on_off_block_index)
#     final_identifiers: List[Tuple[int, int, int]] = []

#     for chunk_idx, indices in enumerate(chunk_indices):
#         if len(indices) < chunk_length:
#             # skip incomplete last chunk
#             continue
#         block_counter = -1
#         prev_label = None
#         for i in range(chunk_length):
#             label = on_off_mask[i]
#             if label == DataLabel.DISCARDED:
#                 continue
#             if label == DataLabel.ON and prev_label != DataLabel.ON:
#                 block_counter += 1
#             prev_label = label
#             spectrum_index = indices[i]
#             final_spectra.append(xuv_spectra[spectrum_index])
#             final_identifiers.append((int(label), int(chunk_idx), int(block_counter if label != DataLabel.TRAINING else -1)))

#     # Convert to arrays (types friendly for MATLAB)
#     final_spectra_arr = np.asarray(final_spectra, dtype=np.float32)
#     final_identifiers_arr = np.asarray(final_identifiers, dtype=np.int32)
#     xuv_energy_ev_arr = np.asarray(xuv_energy_ev, dtype=np.float32)

#     # Default delay step if missing
#     if not delay_stepsize_mm:
#         delay_stepsize_mm = 0.05

#     return {
#         "final_spectra": final_spectra_arr,
#         "final_identifiers": final_identifiers_arr,
#         "xuv_energy_ev": xuv_energy_ev_arr,
#         "delay_stepsize_mm": float(delay_stepsize_mm),
#         "ppas": int(ppas),
#         "spss": int(spss),
#         "spds": int(spds),
#         "config_name": str(config["Name"]),
#     }

def main():
    """Main function to run the data processing pipeline."""
    
    # --- 1. Setup and Data Loading ---
    print("\n--- LOADING DATA ---")
    if not os.path.exists(Paths.OUTPUT_FOLDER):
        os.makedirs(Paths.OUTPUT_FOLDER)
        print(f"Created output directory: {Paths.OUTPUT_FOLDER}")

    print(f"XUV Date: {XUV_DATE}, Sequence: {XUV_SEQUENCE}")
    print(f"Delay Stage Date: {DELAY_STAGE_DATE}, Sequence: {DELAY_STAGE_SEQUENCE}")
    
    # Find XUV files using the new directory structure
    xuv_json_path, xuv_npy_path = find_xuv_files(XUV_DATE, XUV_SEQUENCE)
    print(f"XUV JSON: {xuv_json_path}")
    print(f"XUV NPY: {xuv_npy_path}")

    # Find delay stage file using the new directory structure
    delay_file_path = find_delay_stage_file(DELAY_STAGE_DATE, DELAY_STAGE_SEQUENCE)
    print(f"Delay stage file: {delay_file_path}")

    # Load data from files
    xuv_spectra, xuv_timestamps_us = load_xuv_spectra_and_timestamps(xuv_json_path)
    print(load_delay_stage_data(delay_file_path))
    delay_timestamps_us, ppas, spss, spds, move_step_fs = load_delay_stage_data(delay_file_path)
    print('delay time stamp:', delay_timestamps_us)
    # Load energy calibration file from the workspace directory (always Spec.txt)
    spec_file_path = os.path.join(os.path.dirname(__file__), XUV_EV_CALIBRATION_FILE)
    xuv_energy_ev = np.loadtxt(spec_file_path)
    print("\nLOADED DATA SHAPES:")
    print(f"\tXUV Spectra: {xuv_spectra.shape}")
    print(f"\tXUV Timestamps: {xuv_timestamps_us.shape}")
    print(f"\tDelay Timestamps: {delay_timestamps_us.shape}")
    print(f"\tSPSS: {spss}, SPDS: {spds}, Move Step: {move_step_fs} fs")
    if move_step_fs is not None:
        print(f"\tMove step in fs: {move_step_fs:.2f} fs")
    print(f"\tPPAS: {ppas}")
    # --- 2. Synchronisation and Chunking ---
    print("\n--- SYNCHRONISING DATA ---")
    # Find sync patterns in the XUV data, which mark the beginning of each delay scan.
    # An offset of +2 is added based on empirical observation of the acquisition timing.
    sync_gaps_us = [element * (ppas * (1 / 1030) * 1e6) for element in SYNC_GAPS_PATTERN]
    sync_indices = find_sync_patterns(xuv_timestamps_us, sync_gaps_us, SYNC_PATTERN_TOLERANCE_US) + 2

    expected_scans = len(delay_timestamps_us) // 2
    print(f"Expected {expected_scans} scans based on delay stage triggers.")
    print(f"Found {len(sync_indices)} sync patterns in XUV timestamps.")
    if len(sync_indices) != expected_scans:
        warnings.warn("Mismatch between number of sync patterns and expected delay scans.")

    if Settings.PLOT_TIMING_OVERVIEW:
        plt.figure(figsize=(15, 5))
        plt.vlines(xuv_timestamps_us * 1e-6, 0, 0.8, color='blue', alpha=0.3, label='XUV Timestamps')
        plt.scatter(xuv_timestamps_us[sync_indices] * 1e-6, [1.0] * len(sync_indices), color='cyan', marker='^', s=60, label='Detected XUV Sync')
        plt.vlines(delay_timestamps_us * 1e-6, 0, 1.2, color='orange', alpha=1, label='Delay Stage Triggers')
        plt.xlabel("Time [s]")
        plt.title("Timestamp Overview")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Paths.OUTPUT_FOLDER, "sync_sequences_overview.png"), dpi=300)
        plt.show()

        # plot xuv spectra summed over the energies:
    # xuv_summed = np.sum(xuv_spectra, axis=1)
    # plt.figure()
    # plt.plot(xuv_timestamps_us, xuv_summed)
    # plt.xlabel("Time [us]")
    # plt.ylabel("Summed XUV Signal")
    # plt.title("XUV Spectra Summed Over Energies")
    # plt.grid()
    # plt.show()
    # Split the continuous data stream into chunks based on the sync indices.
    # Each chunk corresponds to one full delay scan.
    chunk_indices = np.split(np.arange(len(xuv_spectra)), sync_indices)[1:]
    # All chunks should ideally have the same length for consistent processing.
    chunk_lengths = [len(c) for c in chunk_indices]
    if len(set(chunk_lengths[:-1])) > 1: # Check all but the last chunk
        warnings.warn(f"Chunk lengths are not uniform: {chunk_lengths}")

    # --- 3. Labelling and Data Restructuring ---
    print("\n--- LABELLING AND RESTRUCTURING DATA ---")
    if not Settings.PLOT_ON_OFF_SPECTRUM:
        print("Skipping plotting as per settings.")
        return

    # Use the first complete chunk's length to create the ON/OFF mask.
    chunk_length = chunk_lengths[0]
    on_off_mask = create_on_off_mask(
        chunk_length=chunk_length,
        spss=spss,
        spds=spds,
        sync_pulses_to_skip=SYNC_PULSES_TO_SKIP,
        pulses_to_skip=PULSES_TO_SKIP_BEFORE_FIRST_SHUTTER,
        shift=SHIFT
    )

    # Prepare lists to hold the restructured data for plotting.
    final_spectra = []
    # Identifier format: (DataLabel, chunk_index, on_off_block_index)
    final_identifiers = []

    # Iterate through each chunk (delay scan)
    for chunk_idx, indices in enumerate(chunk_indices):
        if len(indices) < chunk_length:
            print(f"Skipping incomplete chunk {chunk_idx} with length {len(indices)}.")
            continue

        block_counter = -1
        prev_label = None
        # Iterate through each spectrum within the chunk
        for i in range(chunk_length):
            label = on_off_mask[i]
            if label == DataLabel.DISCARDED:
                continue

            # Increment the ON/OFF block counter at the start of each new ON block.
            if label == DataLabel.ON and prev_label != DataLabel.ON:
                block_counter += 1
            prev_label = label

            spectrum_index = indices[i]
            final_spectra.append(xuv_spectra[spectrum_index])
            
            identifier = (label, chunk_idx, block_counter if label != DataLabel.TRAINING else -1)
            final_identifiers.append(identifier)

    # Convert lists to NumPy arrays for efficient processing.
    final_spectra = np.array(final_spectra, dtype=np.float32)
    final_identifiers = np.array(final_identifiers, dtype=np.int32)

    # --- 4. Plotting Results ---
    print("\n--- PLOTTING RESULTS ---")
    
    # Convert move_step_fs to delay_stepsize_mm for plotting functions
    # If move_step_fs is not available, use a default value
    if move_step_fs is not None:
        # Convert fs to mm: 1 fs = c * 1e-15 / 2 meters (round trip)
        # c = 3e8 m/s, so 1 fs = 1.5e-7 m = 0.15 mm
        delay_stepsize_mm = move_step_fs * 0.15e-3  # Convert fs to mm
        print(f"Using move step: {move_step_fs} fs = {delay_stepsize_mm:.6f} mm")
    else:
        delay_stepsize_mm = 0.05  # Default value if not found
        print(f"Using default delay step size: {delay_stepsize_mm} mm")

    # Apply a mask to focus on a relevant energy range.
    energy_mask = xuv_energy_ev < 86
    atas_on_off_clas(
        xuv_spectra=final_spectra[:, energy_mask],
        xuv_energies_eV=xuv_energy_ev[energy_mask],
        identifiers=final_identifiers,
        delay_stepsize_mm=delay_stepsize_mm,
        shift=0  # The shift is already applied in the mask, so this is for plotting only.
    )

    plot_on_off_clas_shift_sweep(
        xuv_spectra=final_spectra,
        xuv_energies_eV=xuv_energy_ev,
        identifiers=final_identifiers,
        delay_stepsize_mm=delay_stepsize_mm,
        use_background_correction=False
    )
    print("\n--- PROCESSING COMPLETE ---")

if __name__ == "__main__":
    main()