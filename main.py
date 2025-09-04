import numpy as np
import os
import time
import json
from datetime import datetime
from pynput import keyboard
from pylablib.devices import PrincetonInstruments
import gc

# Global stop flag
stop_loop = False

# PATHS ########################################################################
class Paths:
    BASE_DIR = r"C:\Users\Moritz\Desktop\TESTDATA"


# DIRECTORY AND FILE MANAGEMENT ###############################################
def create_data_directory_and_paths():
    """
    Creates the directory structure: base_dir/YYYY/STRA/YYMMDD/YYMMDD_XXX/
    Returns the directory path and base filename (without extension).
    Automatically increments XXX if directory already exists.
    """
    now = datetime.now()
    year = now.strftime("%Y")
    date_str = now.strftime("%y%m%d")
    
    # Create the base directory structure
    year_dir = os.path.join(Paths.BASE_DIR, year)
    stra_dir = os.path.join(year_dir, "STRA")
    date_dir = os.path.join(stra_dir, date_str)
    
    # Create directories if they don't exist
    os.makedirs(date_dir, exist_ok=True)
    
    # Find the next available sequence number
    sequence_num = 1
    while True:
        sequence_str = f"{date_str}_{sequence_num:03d}"
        final_dir = os.path.join(date_dir, sequence_str)
        
        if not os.path.exists(final_dir):
            os.makedirs(final_dir, exist_ok=True)
            break
        
        sequence_num += 1
        
        # Safety check to prevent infinite loop
        if sequence_num > 999:
            raise ValueError("Too many acquisitions for this date (>999)")
    
    base_filename = sequence_str
    return final_dir, base_filename

# SETTINGS #####################################################################
class Settings:
    EXP_TIME_MS = 25
    BINNING = (1, 400)
    SPECTRA_SHAPE = (1, 1340)
    NUMBER_OF_IMAGES = int(300e3)

    ACQUISITION_TIME_VIOLATION_THRESHOLD_MS = EXP_TIME_MS*1.8


# MAIN FUNCTION ################################################################
def main():
    global stop_loop

    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Create directory structure and get file paths
    data_dir, base_filename = create_data_directory_and_paths()
    print(f"Data directory: {data_dir}")
    print(f"Base filename: {base_filename}")

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

    # Structured dtype for mmap
    dtype = np.dtype([
        ("spectrum", np.uint16, Settings.SPECTRA_SHAPE[1]),
        ("timestamp_us", np.uint64)
    ])
    mmap_path = os.path.join(data_dir, f"{base_filename}.npy")
    mmap = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(Settings.NUMBER_OF_IMAGES,))
    mmap[:] = 0
    mmap.flush()

    # Save metadata
    start_time_unix = time.time()
    metadata = {
        "timestamp": timestamp,
        "exp_time_ms": Settings.EXP_TIME_MS,
        "binning": Settings.BINNING,
        "spectra_shape": Settings.SPECTRA_SHAPE,
        "number_of_images_planned": Settings.NUMBER_OF_IMAGES,
        "start_time_unix": start_time_unix,
        "memmap_file": os.path.basename(mmap_path),
        "dtype": {
            "spectrum": "uint16",
            "timestamp_us": "uint64"
        },
        "camera_attributes_at_start": [list(item) for item in all_attrs_at_start.items()],
        "camera_attributes_measurement": [list(item) for item in all_attrs_measurement.items()]
    }
    metadata_path = os.path.join(data_dir, f"{base_filename}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Start acquisition
    print("Starting acquisition... (Press 'Esc' to stop early)")
    cam.start_acquisition()

    violations = []
    t_prev = time.time()
    i = 0
    # run main acquisition loop ########################################################
    try:
        while i < Settings.NUMBER_OF_IMAGES:
            data = cam.read_oldest_image()

            if data is None:
                time.sleep(Settings.EXP_TIME_MS / 1000 / 20)
                continue

            t_now = time.time()
            timestamp_us = int(t_now * 1e6)
            # Efficient max value extraction
            max_val = data.max()

            mmap[i] = (data.ravel().astype(np.uint16), timestamp_us)

            if i % 100 == 0:
                mmap.flush()
                print(f"Image {i} flushed.")

            dt = t_now - t_prev

            # print status:
            print(rf"Image no {i} acquired in {dt:.4f} s, max value: {max_val}")

            if dt > Settings.ACQUISITION_TIME_VIOLATION_THRESHOLD_MS / 1000:
                print(f"ACQUISITION TIME VIOLATION at {i}: {dt:.4f}s --------------------------------------")
                violations.append(i)

            t_prev = t_now

            if stop_loop:
                print("User interrupted acquisition with 'Esc'.")
                break
            
            i += 1

    finally:
        # Final flush
        mmap.flush()
        del mmap
        gc.collect()

        # remove zeros from memmap
        print("Cleaning memmap...")
        load_and_clean_memmap(mmap_path, Settings.SPECTRA_SHAPE[1])

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

        # Finalize metadata
        metadata["images_acquired"] = i
        metadata["violations"] = violations
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Acquisition complete. Images acquired: {i}, Number of Violations: {len(violations)}")
        print(rf"Violations at: {violations}")


# PRINT ROI HELPER ############################################################
def print_roi(cam) -> None:
    roi = cam.get_roi()
    print("ROI settings:")
    print(f"  Horizontal start:  {roi[0]}")
    print(f"  Horizontal end:    {roi[1]}")
    print(f"  Vertical start:    {roi[2]}")
    print(f"  Vertical end:      {roi[3]}")
    print(f"  Horizontal binning:{roi[4]}")
    print(f"  Vertical binning:  {roi[5]}")


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

# CLEAR ZEROS FROM MEMMAP #######################################################
def load_and_clean_memmap(file_path: str, spectrum_length: int) -> None:
    """
    Loads the memmap, removes all-zero rows, and overwrites the original file safely via a temp file.

    :param file_path: Path to the .npy file.
    :param spectrum_length: Length of the spectrum.
    """
    import numpy as np
    import os
    import tempfile
    import gc

    print(f"Loading memmap from: {file_path}")

    dtype = np.dtype([
        ("intensities", np.uint16, spectrum_length),
        ("timestamp_us", np.uint64)
    ])

    # Step 1: Open and filter data
    mmap = np.memmap(file_path, dtype=dtype, mode="r")
    nonzero_mask = ~(
        (mmap["timestamp_us"] == 0) &
        (np.all(mmap["intensities"] == 0, axis=1))
    )
    cleaned_data = mmap[nonzero_mask].copy()  # Load into RAM
    print(f"Original rows: {len(mmap)}, Non-zero rows: {len(cleaned_data)}")

    # Step 2: Fully release original mmap (important on Windows)
    if hasattr(mmap, '_mmap'):
        mmap._mmap.close()
    del mmap
    gc.collect()

    # Step 3: Write to a temporary file
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(file_path))
    os.close(temp_fd)

    cleaned_mmap = np.memmap(temp_path, dtype=dtype, mode="w+", shape=(len(cleaned_data),))
    cleaned_mmap[:] = cleaned_data
    cleaned_mmap.flush()

    if hasattr(cleaned_mmap, '_mmap'):
        cleaned_mmap._mmap.close()
    del cleaned_mmap
    gc.collect()

    # Step 4: Atomically replace original file
    os.replace(temp_path, file_path)

    print(f"Cleaned memmap saved to: {file_path}")

# RUN SCRIPT ##################################################################
if __name__ == "__main__":
    main()
