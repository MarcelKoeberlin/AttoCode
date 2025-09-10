import numpy as np
import os
import time
import json
from datetime import datetime
from pynput import keyboard
from pylablib.devices import PrincetonInstruments
import gc
import h5py
from typing import Tuple
import threading
from collections import deque
import subprocess
import sys
import tempfile
import atexit

# Global stop flag
stop_loop = False
# Clear terminal on Windows
os.system('cls')
# PATHS ########################################################################
class Paths:
    # Server (authoritative) root. Sequence numbers determined here.
    BASE_DIR = r"Z:\Attoline"  # server path
    # Local mirror root (assumption). Adjust if needed.
    LOCAL_BASE_DIR = r"C:\Users\ULP\Desktop\LocalData"  # assumption: change to preferred local drive
# Empty statement to indicate this version also works with v2

# DIRECTORY AND FILE MANAGEMENT ###############################################
def create_hdf5_filepath(base_dir: str) -> str:
    """Create and reserve a new HDF5 filepath under base_dir.

    Directory pattern: base_dir/YYYY/STRA_new/YYMMDD/YYMMDD_XXX/YYMMDD_XXX.hdf5
    Sequence (XXX) determined by first location (server). Returns full filepath.
    """
    now = datetime.now()
    year = now.strftime("%Y")
    date_str = now.strftime("%y%m%d")

    year_dir = os.path.join(base_dir, year)
    stra_dir = os.path.join(year_dir, "STRA_new")
    date_dir = os.path.join(stra_dir, date_str)
    sequence_num = 1
    while True:
        sequence_str = f"{date_str}_{sequence_num:03d}"
        sequence_dir = os.path.join(date_dir, sequence_str)
        hdf5_filename = f"{sequence_str}.hdf5"
        final_path = os.path.join(sequence_dir, hdf5_filename)
        if not os.path.exists(final_path):
            os.makedirs(sequence_dir, exist_ok=True)
            break
        sequence_num += 1
        if sequence_num > 999:
            raise ValueError("Too many acquisitions for this date (>999)")
    return final_path


def create_dual_filepaths() -> Tuple[str, str]:
    """Create server path (authoritative sequence) then mirror path locally.

    Returns (local_path, server_path).
    """
    server_path = create_hdf5_filepath(Paths.BASE_DIR)
    # Derive relative path segment after server base dir
    rel_path = os.path.relpath(server_path, Paths.BASE_DIR)
    local_path = os.path.join(Paths.LOCAL_BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path, server_path

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
    EXP_TIME_MS = 10
    BINNING = (1, 400)
    SPECTRA_SHAPE = (1, 1340)
    NUMBER_OF_IMAGES = int(300e3)
    ACQUISITION_TIME_VIOLATION_THRESHOLD_MS = 40


def ask_exp_time_ms(default_ms: float):
    """Open a small modal Tkinter dialog to ask the user for exposure time in ms.

    Returns the numeric value (float) entered by the user or the default on cancel/error.
    This function imports tkinter locally so it won't fail in headless or minimal environments
    until it's actually called.
    """
    try:
        import tkinter as tk
    except Exception:
        print("tkinter not available; using default exposure time.")
        return default_ms

    result = {"value": default_ms}

    def on_ok():
        try:
            v = float(entry_var.get())
            result["value"] = v
        except Exception:
            # ignore parse errors and keep default
            pass
        root.destroy()

    def on_cancel():
        root.destroy()

    root = tk.Tk()
    root.title("Set Exposure Time (ms)")
    # Small fixed size dialog
    root.geometry("320x120")
    root.resizable(False, False)

    tk.Label(root, text="Exposure time (ms):").pack(pady=(12, 2))
    entry_var = tk.StringVar(value=str(default_ms))
    entry = tk.Entry(root, textvariable=entry_var)
    entry.pack(pady=4)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)
    tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side="left", padx=6)

    entry.focus_set()
    # Keep on top so the user sees it before other windows
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    try:
        root.mainloop()
    except Exception:
        # If the GUI loop fails for some reason, return default
        return default_ms

    return result["value"]


# MAIN FUNCTION ################################################################
def main():
    global stop_loop

    # Clear console immediately on startup so the terminal window starts fresh
    os.system('cls')

    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Prepare paths for IPC buffer and log used by the GUI
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    buf_file = os.path.join(workspace_dir, 'atto_recent.npy')
    log_file = os.path.join(workspace_dir, 'atto_last_lines.log')
    stop_file = os.path.join(workspace_dir, 'atto_stop.flag')
    stopped_file = os.path.join(workspace_dir, 'atto_stopped.flag')

    # Simple rotating logger capture: we will maintain last 200 printed lines in memory
    printed_lines = deque(maxlen=200)

    # Wrapper to capture prints while still printing to real stdout
    class Tee:
        def write(self, s):
            try:
                for line in s.splitlines():
                    if line.strip() == '':
                        continue
                    printed_lines.append(line)
                sys.__stdout__.write(s)
            except Exception:
                pass

        def flush(self):
            try:
                sys.__stdout__.flush()
            except Exception:
                pass

    sys.stdout = Tee()
    sys.stderr = Tee()

    # Ensure log file is cleaned and will be written periodically
    def write_log_file():
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(list(printed_lines)[-20:]))
        except Exception:
            pass

    atexit.register(write_log_file)

    # Launch GUI subprocess (non-blocking). If it fails, continue without GUI.
    # Ensure previous GUI files don't show stale data
    try:
        # truncate log file so GUI starts with empty terminal
        with open(log_file, 'w', encoding='utf-8'):
            pass
    except Exception:
        pass

    try:
        # remove stale buffer file so GUI doesn't read previous spectrum
        if os.path.exists(buf_file):
            try:
                os.remove(buf_file)
            except Exception:
                pass
    except Exception:
        pass

    # remove any leftover stop/stopped files so GUI starts fresh
    try:
        if os.path.exists(stop_file):
            try:
                os.remove(stop_file)
            except Exception:
                pass
    except Exception:
        pass
    try:
        if os.path.exists(stopped_file):
            try:
                os.remove(stopped_file)
            except Exception:
                pass
    except Exception:
        pass
    
    # Before starting the full GUI, ask the user for exposure time (ms).
    try:
        chosen = ask_exp_time_ms(Settings.EXP_TIME_MS)
        # coerce to numeric and update Settings
        try:
            Settings.EXP_TIME_MS = float(chosen)
        except Exception:
            pass
        print(f"Exposure time set to {Settings.EXP_TIME_MS} ms")
    except Exception:
        print("Exposure dialog failed or was cancelled; using default.")

    try:
        gui_proc = subprocess.Popen([sys.executable, os.path.join(workspace_dir, 'atto_gui.py')])
    except Exception as e:
        print(f"Could not launch GUI subprocess: {e}")
        gui_proc = None

    # Create dual file paths (server decides sequence number)
    local_hdf5_path, server_hdf5_path = create_dual_filepaths()
    print(f"Local file:  {local_hdf5_path}")
    print(f"Server file: {server_hdf5_path}")
    os.makedirs(os.path.dirname(server_hdf5_path), exist_ok=True)

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
        # Open both files. Server file in SWMR writer mode.
        with h5py.File(server_hdf5_path, 'w', libver='latest') as f_server, \
             h5py.File(local_hdf5_path, 'w', libver='latest') as f_local:
            # Create datasets
            dset_server = f_server.create_dataset(
                "data", shape=(0,), maxshape=(Settings.NUMBER_OF_IMAGES,), dtype=dtype, chunks=(100,)
            )
            dset_local = f_local.create_dataset(
                "data", shape=(0,), maxshape=(Settings.NUMBER_OF_IMAGES,), dtype=dtype, chunks=(100,)
            )

            # Metadata groups
            meta_server = f_server.create_group("metadata")
            meta_local = f_local.create_group("metadata")
            save_dict_to_hdf5_attrs(meta_server, metadata)
            save_dict_to_hdf5_attrs(meta_local, metadata)

            # Enter SWMR on server side ONLY
            f_server.flush()
            f_server.swmr_mode = True
            print("Server writer entered SWMR mode.")

            # Flush local immediately (schedule at frame 0)
            dset_local.flush(); f_local.flush()
            print("Local initial flush (frame 0) complete.")

            # Scheduling parameters
            FLUSH_PERIOD = 2000
            next_flush_server = 100
            next_flush_local = next_flush_server + FLUSH_PERIOD  # first in-loop local flush frame (1-based)
            
            server_flush_failures = []      # indices where server flush failed
            server_write_failures = []      # indices where server write failed
            # store tuples of (index, spectrum_array, timestamp_us) for later retry
            server_fail = []

            print("Starting acquisition... (Press 'Esc' to stop early)")
            cam.start_acquisition()
            t_prev = time.time()

            while i < Settings.NUMBER_OF_IMAGES:
                # Check for external stop request (from GUI Stop button)
                try:
                    if os.path.exists(stop_file):
                        print("Stop flag detected from GUI. Stopping acquisition...")
                        try:
                            os.remove(stop_file)
                        except Exception:
                            pass
                        stop_loop = True
                except Exception:
                    pass

                data = cam.read_newest_image()
                if data is None:
                    time.sleep(Settings.EXP_TIME_MS / 1000 / 20)
                    continue

                t_now = time.time()
                timestamp_us = int(t_now * 1e6)
                max_val = data.max()

                # Resize and write both datasets
                new_size = i + 1
                dset_local.resize((new_size,))
                row = (data.ravel().astype(np.uint16), timestamp_us)
                dset_local[i] = row
                
                try:
                    dset_server.resize((new_size,))
                    dset_server[i] = row
                except PermissionError as e:
                    print(f"[Server write WARNING] frame {i}: {e}. Skipping server write for this frame.")
                    server_write_failures.append(i)
                    try:
                        # save the row (index, spectrum, timestamp) for retrying later
                        server_fail.append((i, data.ravel().astype(np.uint16).copy(), timestamp_us))
                    except Exception:
                        pass

                # Interleaved flush logic (use 1-based frame index for comparison)
                frame_index_1b = new_size  # 1-based

                # Local flush schedule: 0,1000,2000,... (treat 0 as already flushed)
                if frame_index_1b == next_flush_local and frame_index_1b != 0:
                    try:
                        dset_local.flush(); f_local.flush()
                        print(f"[Local flush] frame {frame_index_1b}")
                    except Exception as e:
                        print(f"[Local flush ERROR] frame {frame_index_1b}: {e}")
                    finally:
                        next_flush_local += FLUSH_PERIOD * 2  # increment by 2*FLUSH_PERIOD

                # Server flush schedule: 500,1500,2500,...
                if frame_index_1b == next_flush_server:
                    try:
                        dset_server.flush(); f_server.flush()
                        print(f"[Server SWMR flush] frame {frame_index_1b}")
                        next_flush_server += FLUSH_PERIOD * 2
                    except Exception as e:
                        print(f"[Server flush ERROR] frame {frame_index_1b}: {e}. Will retry in {2*FLUSH_PERIOD} frames.")
                        server_flush_failures.append(frame_index_1b)
                        # Skip one cycle (add 1000) to retry later as requested
                        next_flush_server += FLUSH_PERIOD * 2

                    # After a server flush attempt (whether success or failure), try to replay any stored failed writes
                    if len(server_fail) > 0:
                        try:
                            # iterate over a copy since we'll modify the list
                            for entry in list(server_fail):
                                idx, spec_arr, ts = entry
                                # ensure server dataset is large enough
                                if dset_server.shape[0] <= idx:
                                    try:
                                        dset_server.resize((idx + 1,))
                                    except Exception as e:
                                        print(f"[Server retry WARNING] cannot resize for idx {idx}: {e}")
                                        continue
                                # read existing timestamp on server; if it's zero, overwrite
                                try:
                                    existing_ts = int(dset_server[idx]["timestamp_us"])
                                except Exception:
                                    existing_ts = 0
                                if existing_ts == 0:
                                    try:
                                        dset_server[idx] = (spec_arr, ts)
                                        print(f"[Server retry] wrote frame {idx} from server_fail")
                                        try:
                                            server_fail.remove(entry)
                                            if idx in server_write_failures:
                                                server_write_failures.remove(idx)
                                        except ValueError:
                                            pass
                                    except Exception as e:
                                        print(f"[Server retry ERROR] frame {idx}: {e}")
                                else:
                                    # server already has data for this index; drop the stored retry
                                    try:
                                        server_fail.remove(entry)
                                    except ValueError:
                                        pass
                        except Exception as e:
                            print(f"[Server retry ERROR] overall: {e}")

                # Lightweight periodic info (keep existing pattern logic)
                dt = t_now - t_prev
                if (i + 50) % 100 == 0:
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

                # --- Maintain recent buffer file for GUI: append latest raw spectrum ---
                try:
                    # read existing buffer if present
                    if os.path.exists(buf_file):
                        try:
                            recent = np.load(buf_file)
                            if recent.ndim == 2:
                                recent = np.vstack([recent, data.ravel().astype(np.uint16)])
                            else:
                                recent = np.atleast_2d(data.ravel().astype(np.uint16))
                        except Exception:
                            recent = np.atleast_2d(data.ravel().astype(np.uint16))
                    else:
                        recent = np.atleast_2d(data.ravel().astype(np.uint16))

                    # keep only last 100 raw frames to bound file size
                    if recent.shape[0] > 100:
                        recent = recent[-100:]

                    np.save(buf_file, recent)
                except Exception as e:
                    print(f"Could not update GUI buffer file: {e}")

                # Update rotating log file periodically
                if i % 5 == 0:
                    write_log_file()

            # Final metadata update
            meta_server.attrs["images_acquired"] = i
            meta_local.attrs["images_acquired"] = i
            meta_server.attrs["violations"] = json.dumps(violations)
            meta_local.attrs["violations"] = json.dumps(violations)
            meta_server.attrs["server_flush_failures"] = json.dumps(server_flush_failures)
            meta_server.attrs["server_write_failures"] = json.dumps(server_write_failures)

            # Ensure final flushes (even if not on schedule)
            try:
                dset_local.flush(); f_local.flush()
            except Exception:
                pass
            # Final attempt: replay any remaining server_fail entries into the server dataset
            if len(server_fail) > 0:
                try:
                    for entry in list(server_fail):
                        idx, spec_arr, ts = entry
                        if dset_server.shape[0] <= idx:
                            try:
                                dset_server.resize((idx + 1,))
                            except Exception as e:
                                print(f"[Final Server retry WARNING] cannot resize for idx {idx}: {e}")
                                continue
                        try:
                            existing_ts = int(dset_server[idx]["timestamp_us"])
                        except Exception:
                            existing_ts = 0
                        if existing_ts == 0:
                            try:
                                dset_server[idx] = (spec_arr, ts)
                                print(f"[Final Server retry] wrote frame {idx} from server_fail")
                                try:
                                    server_fail.remove(entry)
                                    if idx in server_write_failures:
                                        server_write_failures.remove(idx)
                                except ValueError:
                                    pass
                            except Exception as e:
                                print(f"[Final Server retry ERROR] frame {idx}: {e}")
                        else:
                            try:
                                server_fail.remove(entry)
                            except ValueError:
                                pass
                except Exception as e:
                    print(f"[Final Server retry ERROR] overall: {e}")
            # Additional repair: for all acquired indices, if server entry is zeroed (timestamp==0 or spectrum all zeros),
            # copy the corresponding row from the local file at the same index.
            try:
                max_check = min(i, dset_server.shape[0], dset_local.shape[0])
                for idx in range(max_check):
                    try:
                        s_ts = int(dset_server[idx]["timestamp_us"])
                        s_spec = dset_server[idx]["spectrum"]
                        # treat as zero if timestamp is 0 or spectrum all zeros
                        if s_ts == 0 or np.all(s_spec == 0):
                            try:
                                local_row = dset_local[idx]
                                l_spec = local_row["spectrum"]
                                l_ts = int(local_row["timestamp_us"])
                                dset_server[idx] = (l_spec, l_ts)
                                print(f"[Server repair] replaced zeroed frame {idx} from local file")
                            except Exception as e:
                                print(f"[Server repair ERROR] cannot copy frame {idx} from local: {e}")
                    except Exception as e:
                        print(f"[Server repair ERROR] reading server idx {idx}: {e}")
            except Exception as e:
                print(f"[Server repair ERROR] overall scan failed: {e}")
            try:
                dset_server.flush(); f_server.flush()
            except Exception:
                pass
            print("Final metadata saved (both files).")

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
        # signal GUI that acquisition and save finished
        try:
            with open(stopped_file, 'w', encoding='utf-8') as f:
                f.write('stopped')
        except Exception:
            pass


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