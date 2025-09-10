import numpy as np
import os
import time
import json
from datetime import datetime
from pynput import keyboard
from pylablib.devices import PrincetonInstruments
import gc
import h5py
from typing import Tuple, Dict, Any, List
import threading
from collections import deque
import subprocess
import sys
import atexit
import queue

# Global stop flag
stop_loop = False
# Clear terminal on Windows
os.system('cls')

# PATHS ########################################################################
class Paths:
    BASE_DIR = r"Z:\Attoline"
    LOCAL_BASE_DIR = r"C:\Users\ULP\Desktop\LocalData"

# DIRECTORY AND FILE MANAGEMENT ###############################################
def create_hdf5_filepath(base_dir: str) -> str:
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
    server_path = create_hdf5_filepath(Paths.BASE_DIR)
    rel_path = os.path.relpath(server_path, Paths.BASE_DIR)
    local_path = os.path.join(Paths.LOCAL_BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path, server_path

def save_dict_to_hdf5_attrs(group, metadata_dict):
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
    ACQUISITION_TIME_VIOLATION_THRESHOLD_MS = 70 # Adjusted threshold

def ask_exp_time_ms(default_ms: float):
    # This function remains unchanged...
    try:
        import tkinter as tk
    except Exception:
        return default_ms
    result = {"value": default_ms}
    def on_ok():
        try: result["value"] = float(entry_var.get())
        except Exception: pass
        root.destroy()
    def on_cancel(): root.destroy()
    root = tk.Tk()
    root.title("Set Exposure Time (ms)"); root.geometry("320x120"); root.resizable(False, False)
    tk.Label(root, text="Exposure time (ms):").pack(pady=(12, 2))
    entry_var = tk.StringVar(value=str(default_ms))
    entry = tk.Entry(root, textvariable=entry_var)
    entry.pack(pady=4)
    btn_frame = tk.Frame(root); btn_frame.pack(pady=8)
    tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side="left", padx=6)
    entry.focus_set()
    try: root.attributes("-topmost", True)
    except Exception: pass
    try: root.mainloop()
    except Exception: return default_ms
    return result["value"]

################################################################################################
# SINGLE DUAL-WRITER THREAD ####################################################################
################################################################################################
WRITER_DONE = object()

def dual_writer_thread(
    q: queue.Queue,
    local_path: str,
    server_path: str,
    total_frames: int,
    dtype: Any,
    initial_metadata: Dict[str, Any],
    server_write_failures: List[int],
    server_failed_rows: List[Any],
    flush_interval: int = 500,
    repair_interval: int = 1000
):
    """Consume frames and write to BOTH local and server HDF5 files.

    - Preallocates both datasets with fixed size (fast index assignment).
    - On server write exceptions, stores the frame for later retry.
    - Periodically flushes server file (SWMR visibility) and retries failed frames.
    - Final pass attempts to repair still-missing frames.
    """
    print("[Writer] Dual-writer thread started")
    try:
        with h5py.File(local_path, 'w', libver='latest') as f_local, \
             h5py.File(server_path, 'w', libver='latest') as f_server:
            dset_local = f_local.create_dataset("data", shape=(total_frames,), dtype=dtype, chunks=(100,))
            dset_server = f_server.create_dataset("data", shape=(total_frames,), dtype=dtype, chunks=(100,))
            # Progress datasets (single scalar) written BEFORE enabling SWMR so they can be safely updated later
            progress_local = f_local.create_dataset("progress", shape=(1,), dtype='u8')  # number of frames written (writer perspective)
            progress_server = f_server.create_dataset("progress", shape=(1,), dtype='u8')
            progress_local[0] = 0; progress_server[0] = 0
            meta_local = f_local.create_group("metadata")
            meta_server = f_server.create_group("metadata")
            save_dict_to_hdf5_attrs(meta_local, initial_metadata)
            save_dict_to_hdf5_attrs(meta_server, initial_metadata)
            f_server.swmr_mode = True
            # Initial flush so SWMR readers can open
            try:
                f_server.flush(); f_local.flush()
                print("[Writer] Server file entered SWMR mode (initial flush done)")
            except Exception as e:
                print(f"[Writer][INITIAL FLUSH WARN] {e}")

            processed = 0
            # Incremental repair bookkeeping
            last_repair_index = 0  # start of next repair window
            incremental_repaired = 0
            while True:
                item = q.get()
                if item is WRITER_DONE:
                    print("[Writer] Done signal received")
                    break
                idx, spec, ts = item
                if idx >= total_frames:
                    continue
                try:
                    dset_local[idx] = (spec, ts)
                except Exception as e:
                    print(f"[Writer][LOCAL ERROR] frame {idx}: {e}")
                try:
                    dset_server[idx] = (spec, ts)
                except Exception as e:
                    server_write_failures.append(idx)
                    try:
                        server_failed_rows.append((idx, spec.copy(), ts))
                    except Exception:
                        pass
                    print(f"[Writer][SERVER FAIL] frame {idx}: {e}")
                processed += 1
                if processed % flush_interval == 0:
                    try:
                        progress_val = idx + 1
                        progress_local[0] = progress_val
                        progress_server[0] = progress_val
                    except Exception:
                        pass
                    try:
                        f_local.flush()
                    except Exception:
                        pass
                    try:
                        f_server.flush()
                        print(f"[Writer] Flushed at processed={processed} (progress={progress_val})")
                    except Exception as e:
                        print(f"[Writer][FLUSH ERROR] {e}")
                    if server_failed_rows:
                        for entry in list(server_failed_rows):
                            fidx, fspec, fts = entry
                            try:
                                existing_ts = int(dset_server[fidx]['timestamp_us'])
                            except Exception:
                                existing_ts = 0
                            if existing_ts == 0:
                                try:
                                    dset_server[fidx] = (fspec, fts)
                                    server_failed_rows.remove(entry)
                                    if fidx in server_write_failures:
                                        try: server_write_failures.remove(fidx)
                                        except ValueError: pass
                                    print(f"[Writer][RETRY OK] frame {fidx}")
                                except Exception as e:
                                    print(f"[Writer][RETRY ERROR] frame {fidx}: {e}")
                            else:
                                try: server_failed_rows.remove(entry)
                                except ValueError: pass
                    # Incremental repair pass (bounded window) to fix silent missing frames early
                    if (idx + 1) - last_repair_index >= repair_interval:
                        start = last_repair_index
                        end = idx + 1  # inclusive end -> slice exclusive
                        try:
                            ts_slice = dset_server['timestamp_us'][start:end]
                            zero_rel = np.where(ts_slice == 0)[0]
                            if zero_rel.size:
                                for rel in zero_rel:
                                    j = start + rel
                                    try:
                                        lrow = dset_local[j]
                                        dset_server[j] = (lrow['spectrum'], int(lrow['timestamp_us']))
                                        incremental_repaired += 1
                                    except Exception as e:
                                        print(f"[Writer][INCR REPAIR ERROR] frame {j}: {e}")
                                try:
                                    f_server.flush()
                                except Exception:
                                    pass
                                print(f"[Writer][INCR REPAIR] window {start}:{end} repaired {len(zero_rel)} (total incremental {incremental_repaired})")
                            last_repair_index = end
                        except Exception as e:
                            print(f"[Writer][INCR REPAIR SCAN ERROR] {e}")
            try:
                f_local.flush(); f_server.flush()
            except Exception:
                pass
            try:
                progress_local[0] = min(total_frames, processed)
                progress_server[0] = min(total_frames, processed)
            except Exception:
                pass
            if server_failed_rows:
                for entry in list(server_failed_rows):
                    fidx, fspec, fts = entry
                    try:
                        existing_ts = int(dset_server[fidx]['timestamp_us'])
                    except Exception:
                        existing_ts = 0
                    if existing_ts == 0:
                        try:
                            dset_server[fidx] = (fspec, fts)
                            server_failed_rows.remove(entry)
                            if fidx in server_write_failures:
                                try: server_write_failures.remove(fidx)
                                except ValueError: pass
                            print(f"[Writer][FINAL RETRY OK] frame {fidx}")
                        except Exception as e:
                            print(f"[Writer][FINAL RETRY ERROR] frame {fidx}: {e}")
                    else:
                        try: server_failed_rows.remove(entry)
                        except ValueError: pass
            meta_server.attrs['server_write_failures_initial'] = json.dumps(server_write_failures)
            meta_server.attrs['server_fail_buffer_remaining'] = len(server_failed_rows)
            meta_server.attrs['incremental_repaired_frames'] = incremental_repaired
    except Exception as e:
        print(f"[Writer][CRITICAL] {e}")
    print("[Writer] Dual-writer thread finished")

# MAIN FUNCTION ################################################################
def main():
    global stop_loop
    # --- Initial setup (unchanged) ---
    os.system('cls'); listener = keyboard.Listener(on_press=on_press); listener.start()
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    buf_file = os.path.join(workspace_dir, 'atto_recent.npy')
    log_file = os.path.join(workspace_dir, 'atto_last_lines.log')
    stop_file = os.path.join(workspace_dir, 'atto_stop.flag')
    stopped_file = os.path.join(workspace_dir, 'atto_stopped.flag')
    for f in [buf_file, log_file, stop_file, stopped_file]:
        if os.path.exists(f):
            try: os.remove(f)
            except OSError: pass
    printed_lines = deque(maxlen=200)
    class Tee:
        def write(self, s):
            try:
                if s.strip(): printed_lines.append(s.strip())
                sys.__stdout__.write(s)
            except Exception: pass
        def flush(self):
            try: sys.__stdout__.flush()
            except Exception: pass
    sys.stdout = Tee(); sys.stderr = Tee()
    def write_log_file():
        try:
            with open(log_file, 'w', encoding='utf-8') as f: f.write('\n'.join(list(printed_lines)[-20:]))
        except Exception: pass
    atexit.register(write_log_file)
    chosen = ask_exp_time_ms(Settings.EXP_TIME_MS)
    try: Settings.EXP_TIME_MS = float(chosen)
    except (ValueError, TypeError): pass
    print(f"Exposure time set to {Settings.EXP_TIME_MS} ms")
    try: subprocess.Popen([sys.executable, os.path.join(workspace_dir, 'atto_gui.py')])
    except Exception as e: print(f"Could not launch GUI subprocess: {e}")

    # --- File & Camera Setup (unchanged) ---
    local_hdf5_path, server_hdf5_path = create_dual_filepaths()
    print(f"Local file:  {local_hdf5_path}"); print(f"Server file: {server_hdf5_path}")
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S"); print(f"Timestamp: {timestamp}")
    print("Connecting to camera..."); cam = PrincetonInstruments.PicamCamera('2105050003'); print("Camera connected.")
    original_roi = cam.get_roi(); original_trigger_det = cam.get_attribute_value("Trigger Determination")
    cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
    cam.set_roi(hbin=Settings.BINNING[0], vbin=Settings.BINNING[1])
    cam.set_attribute_value("Trigger Determination", "Positive Polarity")
    cam.set_attribute_value("Trigger Response", "Readout Per Trigger")
    cam.set_attribute_value("Shutter Timing Mode", "Always Open")
    time.sleep(0.2); cam.setup_acquisition(mode="sequence", nframes=Settings.NUMBER_OF_IMAGES)

    # --- Metadata & Data Structure (unchanged) ---
    dtype = np.dtype([("spectrum", np.uint16, Settings.SPECTRA_SHAPE[1]), ("timestamp_us", np.uint64)])
    metadata = { "timestamp": timestamp, "exp_time_ms": Settings.EXP_TIME_MS, "binning": list(Settings.BINNING), "spectra_shape": list(Settings.SPECTRA_SHAPE), "number_of_images_planned": Settings.NUMBER_OF_IMAGES, "start_time_unix": time.time(), "camera_attributes_measurement": [list(item) for item in cam.get_all_attribute_values().items()]}

    # --- SINGLE PRODUCER -> DUAL-WRITER THREAD SETUP ---
    data_queue = queue.Queue(maxsize=4000)
    server_write_failures: list[int] = []
    server_failed_rows: list[tuple[int, np.ndarray, int]] = []
    writer = threading.Thread(
        target=dual_writer_thread,
        args=(data_queue, local_hdf5_path, server_hdf5_path, Settings.NUMBER_OF_IMAGES, dtype, metadata, server_write_failures, server_failed_rows, 500),
        daemon=True
    )
    writer.start()
    
    # --- ACQUISITION LOOP (PRODUCER) ---
    i = 0; violations = []
    try:
        print("Starting acquisition... (Press 'Esc' or use GUI to stop early)")
        cam.start_acquisition(); t_prev = time.time()
        while i < Settings.NUMBER_OF_IMAGES:
            if stop_loop or os.path.exists(stop_file):
                print("Stop signal detected. Finishing acquisition loop.")
                if os.path.exists(stop_file): os.remove(stop_file)
                stop_loop = True; break
            data = cam.read_newest_image()
            if data is None: time.sleep(Settings.EXP_TIME_MS / 1000 / 20); continue
            t_now = time.time(); timestamp_us = int(t_now * 1e6)

            # Enqueue for writer (single queue)
            item_to_queue = (i, data.ravel().astype(np.uint16), timestamp_us)
            try:
                data_queue.put(item_to_queue, block=True, timeout=1)
            except queue.Full:
                print(f"!!! QUEUE FULL at frame {i}. Writer cannot keep up. Stopping.")
                stop_loop = True; break

            # Logging and GUI updates
            dt = t_now - t_prev
            if (i + 1) % 100 == 0:
                print(f"Image {i+1}/{Settings.NUMBER_OF_IMAGES} | dt: {dt*1000:.2f}ms | Q: {data_queue.qsize()}")
                write_log_file()
            if dt > Settings.ACQUISITION_TIME_VIOLATION_THRESHOLD_MS / 1000: violations.append(i)
            if (i + 1) % 5 == 0:
                # Rolling last 100 spectra for GUI
                try:
                    vec = data.ravel().astype(np.uint16)
                    if os.path.exists(buf_file):
                        try:
                            recent = np.load(buf_file)
                        except Exception:
                            recent = None
                    else:
                        recent = None
                    if recent is None:
                        stack = vec[None, :]
                    else:
                        if recent.ndim == 1:
                            recent = recent[None, :]
                        stack = np.vstack([recent, vec])
                    if stack.shape[0] > 100:
                        stack = stack[-100:]
                    np.save(buf_file, stack)
                except Exception:
                    pass
            t_prev = t_now; i += 1
    finally:
        print("Acquisition loop finished. Waiting for writer thread...")
        data_queue.put(WRITER_DONE)
        writer.join(timeout=120)
        
        # Cleanup
        if cam.acquisition_in_progress():
            cam.stop_acquisition()
            cam.clear_acquisition()
        print("\nResetting camera...")
        cam.set_roi(*original_roi)
        cam.set_attribute_value("Trigger Determination", original_trigger_det)
        cam.close()
        print("Camera connection closed.")
        
        # Final metadata + server repair pass
        try:
            # Need write intent on both files to add/modify attributes.
            # Use 'r+' (fail if missing) then fallback to 'a'.
            try:
                f_local_ctx = h5py.File(local_hdf5_path, 'r+')
            except Exception:
                f_local_ctx = h5py.File(local_hdf5_path, 'a')
            try:
                f_server_ctx = h5py.File(server_hdf5_path, 'r+')
            except Exception:
                f_server_ctx = h5py.File(server_hdf5_path, 'a')
            with f_local_ctx as f_local, f_server_ctx as f_server:
                dset_local = f_local['data']; dset_server = f_server['data']
                max_check = min(i, dset_local.shape[0], dset_server.shape[0])
                repaired = 0
                for idx in range(max_check):
                    try:
                        ts_server = int(dset_server[idx]['timestamp_us'])
                        if ts_server == 0:
                            lrow = dset_local[idx]
                            dset_server[idx] = (lrow['spectrum'], int(lrow['timestamp_us']))
                            repaired += 1
                    except Exception:
                        pass
                meta_server = f_server['metadata']; meta_local = f_local['metadata']
                meta_local.attrs['images_acquired'] = i
                meta_server.attrs['images_acquired'] = i
                meta_local.attrs['violations'] = json.dumps(violations)
                meta_server.attrs['violations'] = json.dumps(violations)
                meta_server.attrs['server_write_failures_final'] = json.dumps(server_write_failures)
                meta_server.attrs['server_repaired_frames'] = repaired
                meta_server.attrs['server_failed_remaining'] = len(server_failed_rows)
                print(f"Server repair complete. Repaired frames: {repaired}")
        except Exception as e:
            print(f"Final metadata/repair failed: {e}")
        gc.collect()
        print("-" * 50); print(f"Acquisition complete. Images acquired: {i}, Violations: {len(violations)}"); print("-" * 50)
        try:
            with open(stopped_file, 'w') as f: f.write('stopped')
        except Exception: pass

# HELPER & CALLBACK FUNCTIONS #################################################
def on_press(key):
    global stop_loop
    if key == keyboard.Key.esc:
        print("Detected 'Esc' key, breaking the loop.")
        stop_loop = True
        return False

# RUN SCRIPT ###################################################################
if __name__ == "__main__":
    main()