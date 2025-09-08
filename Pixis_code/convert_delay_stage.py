#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert old format delay stage .npz files to the new format
that is compatible with atas_preparation.py

Old format: timestamps stored as raw memmap data, parameters in filename
New format: timestamps + parameters (ppas, spss, spds, move_step_fs) stored as variables in .npz
"""

import numpy as np
import os
import re
from pathlib import Path

def delay_fs_to_mm(delay_fs: float) -> float:
    """
    Converts a delay time in femtoseconds (fs) to a physical distance in millimeters (mm).
    Based on the speed of light and accounts for round trip.
    """
    c_mm_per_fs = 299_792_458 / 1e15 * 1e3  # Speed of light in mm/fs
    return abs(delay_fs * c_mm_per_fs / 2)  # Division by 2 for round trip

def mm_to_delay_fs(distance_mm: float) -> float:
    """
    Converts a distance in millimeters to delay time in femtoseconds.
    """
    c_mm_per_fs = 299_792_458 / 1e15 * 1e3  # Speed of light in mm/fs
    return abs(distance_mm * 2 / c_mm_per_fs)  # Multiply by 2 for round trip

def convert_old_to_new_format(old_file_path: str, output_file_path: str = None, move_step_fs: float = None):
    """
    Convert old format delay stage .npz file to new format.
    
    Args:
        old_file_path: Path to the old format .npz file
        output_file_path: Path for the new format file (optional, auto-generated if None)
        move_step_fs: Move step in femtoseconds (optional, calculated from filename if None)
    """
    
    if not os.path.exists(old_file_path):
        raise FileNotFoundError(f"Input file not found: {old_file_path}")
    
    print(f"Converting: {old_file_path}")
    
    # Extract parameters from filename
    filename = os.path.basename(old_file_path)
    print(f"Parsing filename: {filename}")
    
    # Parse ppas, spss, spds from filename
    match = re.search(r"ppas_(\d+)_spss_(\d+)_spds_(\d+)", filename)
    if not match:
        raise ValueError(f"Could not find ppas, spss, spds in filename: {filename}")
    
    ppas = int(match.group(1))
    spss = int(match.group(2))
    spds = int(match.group(3))
    
    print(f"Extracted parameters: ppas={ppas}, spss={spss}, spds={spds}")
    
    # Try to extract step size from filename (if available)
    step_match = re.search(r"step_([\d\.]+)mm", filename)
    if step_match and move_step_fs is None:
        step_mm = float(step_match.group(1))
        move_step_fs = mm_to_delay_fs(step_mm)
        print(f"Extracted step size: {step_mm} mm = {move_step_fs:.2f} fs")
    elif move_step_fs is None:
        # Use a reasonable default if no step size found
        move_step_fs = 20.0  # 20 fs default
        print(f"No step size found in filename, using default: {move_step_fs} fs")
    else:
        print(f"Using provided move_step_fs: {move_step_fs} fs")
    
    # Load the old format data
    try:
        # First try to load as .npz file
        try:
            old_data = np.load(old_file_path)
            print(f"Loaded as .npz file. Available keys: {list(old_data.keys())}")
            
            # Try to find timestamp data
            timestamp_key = None
            for key in ['timestamps', 'timestamps_us', 'data', 'arr_0']:
                if key in old_data.keys():
                    timestamp_key = key
                    break
            
            if timestamp_key:
                timestamps_us = np.array(old_data[timestamp_key])
                print(f"Found timestamps under key '{timestamp_key}'")
            else:
                # If no recognized key, take the first array
                first_key = list(old_data.keys())[0]
                timestamps_us = np.array(old_data[first_key])
                print(f"Using first available array '{first_key}' as timestamps")
                
            old_data.close()
            
        except:
            # If .npz loading fails, try as raw memmap
            print("Failed to load as .npz, trying as raw memmap...")
            timestamps_us = np.memmap(old_file_path, dtype='int64', mode='r')
            timestamps_us = np.array(timestamps_us)
    
    except Exception as e:
        raise ValueError(f"Could not load data from {old_file_path}: {e}")
    
    # Filter out zero entries
    timestamps_us = timestamps_us[timestamps_us > 0]
    
    print(f"Loaded {len(timestamps_us)} valid timestamps")
    print(f"Timestamp range: {timestamps_us.min()} to {timestamps_us.max()} μs")
    
    # Generate output filename if not provided
    if output_file_path is None:
        old_path = Path(old_file_path)
        # Create new filename with _converted suffix
        output_file_path = str(old_path.parent / f"{old_path.stem}_converted.npz")
    
    # Save in new format
    np.savez_compressed(
        output_file_path,
        timestamps_us=timestamps_us,
        ppas=int(ppas),
        spss=int(spss),
        spds=int(spds),
        move_step_fs=float(move_step_fs)
    )
    
    print(f"Saved converted file: {output_file_path}")
    
    # Verify the conversion
    verify_conversion(output_file_path)
    
    return output_file_path

def verify_conversion(npz_file_path: str):
    """
    Verify that the converted file has the correct format.
    """
    print(f"\nVerifying converted file: {npz_file_path}")
    
    try:
        data = np.load(npz_file_path)
        
        required_keys = ['timestamps_us', 'ppas', 'spss', 'spds', 'move_step_fs']
        available_keys = list(data.keys())
        
        print(f"Available keys: {available_keys}")
        
        for key in required_keys:
            if key not in available_keys:
                print(f"❌ Missing required key: {key}")
                return False
            else:
                value = data[key]
                if key == 'timestamps_us':
                    print(f"✓ {key}: {len(value)} timestamps")
                else:
                    print(f"✓ {key}: {value}")
        
        data.close()
        print("✓ Conversion verification successful!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """
    Main function to convert the specific file mentioned in the request.
    """
    
    # The specific file mentioned in the request
    old_file = r"Z:\Personal\MoritzJ\10_measurements\04_Delay_Stage\2025_07_24-21_55_56_delayStage_timestamps_ppas_40_spss_5_spds_180.npz"
    
    # You can also specify a custom output path and move_step_fs if needed
    # output_file = r"Z:\Path\to\output\converted_file.npz"
    # custom_move_step_fs = 20.0  # femtoseconds
    
    try:
        print("=== Delay Stage .npz Converter ===")
        print("Converting old format to new atas_preparation.py compatible format\n")
        
        # Convert with default settings
        converted_file = convert_old_to_new_format(old_file)
        
        print(f"\n=== Conversion Complete ===")
        print(f"Original file: {old_file}")
        print(f"Converted file: {converted_file}")
        print("\nThe converted file can now be used with atas_preparation.py")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
