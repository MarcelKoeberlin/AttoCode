import numpy as np
import os
import time
import json
from datetime import datetime
from pynput import keyboard
from pylablib.devices import PrincetonInstruments
import gc
import h5py
import threading
import shutil

# Global stop flag
stop_loop = False
copy_thread_stop_event = threading.Event()

# PATHS ########################################################################
class Paths:
    BASE_DIR = r"C:\Users\Moritz\Desktop\ATAS local"
    SERVER_DIR = r"Z:\Attoline" # <--- CHANGE THIS to your server path


# DIRECTORY AND FILE MANAGEMENT ###############################################
def create_data_directory_and_paths():
    """
    Creates the directory structure: base_dir/YYYY/STRA_new/YYMMDD/YYMMDD_XXX/
    and server_dir/YYYY/STRA_new/YYMMDD/YYMMDD_XXX/
    Returns the local directory path, server directory path, and base filename.
    Automatically increments XXX if directory already exists.
    """
    now = datetime.now()
    year = now.strftime("%Y")
    date_str = now.strftime("%y%m%d")
    
    # Create the base directory structure for local and server
    local_year_dir = os.path.join(Paths.BASE_DIR, year)
    local_STRA_new_dir = os.path.join(local_year_dir, "STRA_new")
    local_date_dir = os.path.join(local_STRA_new_dir, date_str)
    
    server_year_dir = os.path.join(Paths.SERVER_DIR, year)
    server_STRA_new_dir = os.path.join(server_year_dir, "STRA_new")
    server_date_dir = os.path.join(server_STRA_new_dir, date_str)

    # Create local directories if they don't exist
    os.makedirs(local_date_dir, exist_ok=True)
    
    # Find the next available sequence number
    sequence_num = 1
    while True:
        sequence_str = f"{date_str}_{sequence_num:03d}"
        final_local_dir = os.path.join(local_date_dir, sequence_str)
        
        if not os.path.exists(final_local_dir):
            os.makedirs(final_local_dir, exist_ok=True)
            break
        
        sequence_num += 1
        
        # Safety check to prevent infinite loop
        if sequence_num > 999:
            raise ValueError("Too many acquisitions for this date (>999)")

    final_server_dir = os.path.join(server_date_dir, sequence_str)
    # The copy thread will create the server directory.

    base_filename = sequence_str
    return final_local_dir, final_server_dir, base_filename

# COPY THREAD #################################################################
def copy_to_server_thread(local_path, server_path):
    """
    Continuously tries to copy a file to the server until the stop event is set.
    """
    while not copy_thread_stop_event.is_set():
        try:
            # Check if local file exists and has content before copying
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                # Check if server file needs updating
                if not os.path.exists(server_path) or os.path.getsize(server_path) != os.path.getsize(local_path):
                    print(f"Copying {local_path} to {server_path}...")
                    server_dir = os.path.dirname(server_path)
                    os.makedirs(server_dir, exist_ok=True)
                    shutil.copy2(local_path, server_path)
                    print("Copy successful.")
        except Exception as e:
            print(f"Error copying to server: {e}. Retrying in 10 seconds...")
        
        # Wait for 10 seconds or until stop event is set
        copy_thread_stop_event.wait(10)
    
    # Final copy attempt after loop ends
    try:
        print(f"Final copy attempt for {local_path} to {server_path}...")
        server_dir = os.path.dirname(server_path)
        os.makedirs(server_dir, exist_ok=True)
        shutil.copy2(local_path, server_path)
        print("Final copy successful.")
    except Exception as e:
        print(f"Final copy attempt failed: {e}")


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
    local_data_dir, server_data_dir, base_filename = create_data_directory_and_paths()
    print(f"Local data directory: {local_data_dir}")
    print(f"Server data directory: {server_data_dir}")
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

    # HDF5 setup
    h5_path = os.path.join(local_data_dir, f"{base_filename}.h5")
    h5_server_path = os.path.join(server_data_dir, f"{base_filename}.h5")
    
    h5file = h5py.File(h5_path, 'w')
    spectra_ds = h5file.create_dataset(
        "spectra", 
        (0, Settings.SPECTRA_SHAPE[1]), 
        maxshape=(None, Settings.SPECTRA_SHAPE[1]), 
        dtype='uint16', 
        chunks=(100, Settings.SPECTRA_SHAPE[1])
    )
    timestamps_ds = h5file.create_dataset(
        "timestamps_us", 
        (0,), 
        maxshape=(None,), 
        dtype='uint64', 
        chunks=(100,)
    )

    # Save metadata
    start_time_unix = time.time()
    metadata = {
        "timestamp": timestamp,
        "exp_time_ms": Settings.EXP_TIME_MS,
        "binning": Settings.BINNING,
        "spectra_shape": Settings.SPECTRA_SHAPE,
        "number_of_images_planned": Settings.NUMBER_OF_IMAGES,
        "start_time_unix": start_time_unix,
        "hdf5_file": os.path.basename(h5_path),
        "camera_attributes_at_start": [list(item) for item in all_attrs_at_start.items()],
        "camera_attributes_measurement": [list(item) for item in all_attrs_measurement.items()]
    }
    metadata_path = os.path.join(local_data_dir, f"{base_filename}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Start the copy thread
    copier_thread = threading.Thread(target=copy_to_server_thread, args=(h5_path, h5_server_path))
    copier_thread.start()

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

            # Append data to HDF5 datasets
            spectra_ds.resize(i + 1, axis=0)
            spectra_ds[i, :] = data.ravel().astype(np.uint16)
            timestamps_ds.resize(i + 1, axis=0)
            timestamps_ds[i] = timestamp_us

            if i % 100 == 0:
                h5file.flush()
                print(f"Image {i} flushed to HDF5.")

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
        # Stop acquisition
        if cam.acquisition_in_progress():
            cam.stop_acquisition()
        cam.clear_acquisition()

        # Close HDF5 file
        if 'h5file' in locals() and h5file.id:
            h5file.close()
            print("HDF5 file closed.")

        # Signal the copy thread to stop and wait for it
        print("Signaling copy thread to stop...")
        copy_thread_stop_event.set()
        copier_thread.join()
        print("Copy thread finished.")

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
        #print(rf"Violations at: {violations}")


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


# RUN SCRIPT ##################################################################
if __name__ == "__main__":
    main()
