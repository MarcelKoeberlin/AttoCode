import numpy as np
import os
import time
import json
from datetime import datetime
from pynput import keyboard
from pylablib.devices import PrincetonInstruments
import gc
import h5py

# Global stop flag
stop_loop = False

# PATHS ########################################################################
class Paths:
    BASE_DIR = r"Z:\Attoline"
#Empty statement to indicate this version also works with v2

# DIRECTORY AND FILE MANAGEMENT ###############################################
def create_hdf5_filepath():
    """
    Creates the directory structure: base_dir/YYYY/STRA_new/YYMMDD/
    and returns the full path for a new HDF5 file: YYMMDD_XXX.hdf5.
    Automatically increments XXX if the file already exists.
    """
    now = datetime.now()
    year = now.strftime("%Y")
    date_str = now.strftime("%y%m%d")
    
    # Create the base directory structure
    year_dir = os.path.join(Paths.BASE_DIR, year)
    stra_dir = os.path.join(year_dir, "STRA_new")
    date_dir = os.path.join(stra_dir, date_str)
    sequence_num = 1
    # Find the next available sequence number
    while True:
        sequence_str = f"{date_str}_{sequence_num:03d}"
        sequence_dir = os.path.join(date_dir, sequence_str)
        hdf5_filename = f"{sequence_str}.hdf5"
        final_path = os.path.join(sequence_dir, hdf5_filename)

        if not os.path.exists(final_path):
            # Ensure the directory for this sequence exists
            os.makedirs(sequence_dir, exist_ok=True)
            break

        sequence_num += 1

        # Safety check to prevent infinite loop
        if sequence_num > 999:
            raise ValueError("Too many acquisitions for this date (>999)")

    return final_path

def save_dict_to_hdf5_attrs(group, metadata_dict):
    """
    Saves dictionary items as attributes to an HDF5 group.
    Complex types like dicts or lists are serialized to JSON strings.
    """
    for key, value in metadata_dict.items():
        if isinstance(value, (dict, list, tuple)):
            group.attrs[key] = json.dumps(value)
        else:
            group.attrs[key] = value

# SETTINGS #####################################################################
class Settings:
    EXP_TIME_MS = 20
    BINNING = (1, 400)
    SPECTRA_SHAPE = (1, 1340)
    NUMBER_OF_IMAGES = int(300e3)
    ACQUISITION_TIME_VIOLATION_THRESHOLD_MS = EXP_TIME_MS * 1.8


# MAIN FUNCTION ################################################################
def main():
    global stop_loop

    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


    # Create directory structure and get HDF5 file path
    hdf5_path = create_hdf5_filepath()
    print(f"Data will be saved to: {hdf5_path}")
    # Ensure parent directory exists before opening file
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    print(f"Timestamp: {timestamp}")

    # check if camera is connected:
    print("Connected devices:")
    print(PrincetonInstruments.list_cameras())  # Should list the connected camera

    cam = PrincetonInstruments.PicamCamera('2105050003')
    print("Camera connected.")

    # Save camera attributes BEFORE acquisition
    all_attrs_at_start = cam.get_all_attribute_values()

    # Backup original attributes
    original_roi = cam.get_roi()
    original_trigger_det = cam.get_attribute_value("Trigger Determination")
    original_trigger_resp = cam.get_attribute_value("Trigger Response")
    original_shutter_delay = cam.get_attribute_value("Shutter Closing Delay")
    original_delay_res = cam.get_attribute_value("Shutter Delay Resolution")
    original_shutter_mode = cam.get_attribute_value("Shutter Timing Mode")

    # Setup camera for acquisition
    cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
    print(f"[SET] Exposure Time: {cam.get_attribute_value('Exposure Time')} ms")

    cam.set_roi(hbin=Settings.BINNING[0], vbin=Settings.BINNING[1])
    print_roi(cam)

    cam.set_attribute_value("Trigger Determination", "Positive Polarity")
    print(f"[SET] Trigger Determination: {cam.get_attribute_value('Trigger Determination')}")

    cam.set_attribute_value("Trigger Response", "Readout Per Trigger")
    print(f"[SET] Trigger Response: {cam.get_attribute_value('Trigger Response')}")

    cam.set_attribute_value("Clean Until Trigger", False)
    print(f"[SET] Clean Until Trigger: {cam.get_attribute_value('Clean Until Trigger')}")

    cam.set_attribute_value("Shutter Timing Mode", "Always Open")
    print(f"[SET] Shutter Timing Mode: {cam.get_attribute_value('Shutter Timing Mode')}")

    cam.set_attribute_value("Shutter Closing Delay", 0)
    print(f"[SET] Shutter Closing Delay: {cam.get_attribute_value('Shutter Closing Delay')}")

    time.sleep(0.2)

    # Save attributes after setup
    all_attrs_measurement = cam.get_all_attribute_values()

    # Set up acquisition
    cam.setup_acquisition(mode="sequence", nframes=Settings.NUMBER_OF_IMAGES)

    # Structured dtype for HDF5 dataset
    dtype = np.dtype([
        ("spectrum", np.uint16, Settings.SPECTRA_SHAPE[1]),
        ("timestamp_us", np.uint64)
    ])

    # Metadata dictionary
    start_time_unix = time.time()
    metadata = {
        "timestamp": timestamp,
        "exp_time_ms": Settings.EXP_TIME_MS,
        "binning": list(Settings.BINNING),
        "spectra_shape": list(Settings.SPECTRA_SHAPE),
        "number_of_images_planned": Settings.NUMBER_OF_IMAGES,
        "start_time_unix": start_time_unix,
        "dtype_spectrum": "uint16",
        "dtype_timestamp_us": "uint64",
        "camera_attributes_at_start": [list(item) for item in all_attrs_at_start.items()],
        "camera_attributes_measurement": [list(item) for item in all_attrs_measurement.items()]
    }

    i = 0
    violations = []

    # Use a try...finally block to ensure resources are closed properly
    try:
        with h5py.File(hdf5_path, 'w') as f:
            # Create a resizable dataset for the spectra and timestamps
            dset = f.create_dataset(
                "data",
                shape=(0,),
                maxshape=(Settings.NUMBER_OF_IMAGES,),
                dtype=dtype,
                chunks=(100,)  # Chunking is important for performance
            )
            
            # Create a group for metadata and save it as attributes
            meta_group = f.create_group("metadata")
            save_dict_to_hdf5_attrs(meta_group, metadata)
            #f.swmr_mode = True
            #print("Writer entered SWMR mode. Acquisition starting.")
            # Start acquisition
            print("Starting acquisition... (Press 'Esc' to stop early)")
            cam.start_acquisition()
            t_prev = time.time()
            
            # Run main acquisition loop
            while i < Settings.NUMBER_OF_IMAGES:
                data = cam.read_oldest_image()

                if data is None:
                    time.sleep(Settings.EXP_TIME_MS / 1000 / 20)
                    continue

                t_now = time.time()
                timestamp_us = int(t_now * 1e6)
                max_val = data.max()

                # Resize dataset and write new data
                dset.resize((i + 1,))
                dset[i] = (data.ravel().astype(np.uint16), timestamp_us)
                
                # Periodically flush data to disk
                if i % 100 == 0:
                    f.flush()
                    print(f"Image {i} flushed.")

                dt = t_now - t_prev
                if i % 40 == 0:
                    try:
                        print(rf"Image no {i} acquired in {dt:.4f} s, max value: {max_val}")
                    except Exception as e:
                        print(f"Error occurred while logging image {i}: {e}")

                if dt > Settings.ACQUISITION_TIME_VIOLATION_THRESHOLD_MS / 1000:
                    print(f"ACQUISITION TIME VIOLATION at {i}: {dt:.4f}s --------------------------------------")
                    violations.append(i)

                t_prev = t_now

                if stop_loop:
                    print("User interrupted acquisition with 'Esc'.")
                    break
                
                i += 1
            
            # Final metadata update
            meta_group.attrs["images_acquired"] = i
            meta_group.attrs["violations"] = json.dumps(violations)
            print("Final metadata saved.")

    finally:
        # Stop acquisition
        if cam.acquisition_in_progress():
            cam.stop_acquisition()
        cam.clear_acquisition()

        # Reset original camera settings
        print("\nResetting camera to original settings:")
        cam.set_roi(*original_roi)
        print_roi(cam)

        cam.set_attribute_value("Trigger Determination", original_trigger_det)
        print(f"[RESET] Trigger Determination: {cam.get_attribute_value('Trigger Determination')}")

        cam.set_attribute_value("Trigger Response", original_trigger_resp)
        print(f"[RESET] Trigger Response: {cam.get_attribute_value('Trigger Response')}")

        cam.set_attribute_value("Shutter Closing Delay", original_shutter_delay)
        print(f"[RESET] Shutter Closing Delay: {cam.get_attribute_value('Shutter Closing Delay')}")

        cam.set_attribute_value("Shutter Delay Resolution", original_delay_res)
        print(f"[RESET] Shutter Delay Resolution: {cam.get_attribute_value('Shutter Delay Resolution')}")

        cam.set_attribute_value("Shutter Timing Mode", original_shutter_mode)
        print(f"[RESET] Shutter Timing Mode: {cam.get_attribute_value('Shutter Timing Mode')}")

        cam.close()
        print("Camera connection closed.")

        gc.collect()

        print(f"Acquisition complete. Images acquired: {i}, Number of Violations: {len(violations)}")
        print(rf"Violations at: {violations}")


# PRINT ROI HELPER ############################################################
def print_roi(cam) -> None:
    roi = cam.get_roi()
    print("ROI settings:")
    print(f"  Horizontal start:   {roi[0]}")
    print(f"  Horizontal end:     {roi[1]}")
    print(f"  Vertical start:     {roi[2]}")
    print(f"  Vertical end:       {roi[3]}")
    print(f"  Horizontal binning: {roi[4]}")
    print(f"  Vertical binning:   {roi[5]}")


# PYNPUT CALLBACK #############################################################
def on_press(key):
    global stop_loop
    try:
        if key == keyboard.Key.esc:
            print("Detected 'Esc' key, breaking the loop.")
            stop_loop = True
            return False  # Stop the listener
    except AttributeError:
        pass


# RUN SCRIPT ##################################################################
if __name__ == "__main__":
    main()