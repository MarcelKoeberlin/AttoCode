# spectrum_live_refactored.py

import os
import time
import datetime
import threading
import queue
import shutil
from collections import deque

import numpy as np
import h5py
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

from pylablib.devices import PrincetonInstruments

# --- Configuration container ---------------------------------------------------
class Settings:
    """Static configuration parameters for the script."""
    # --- Camera Settings ---
    EXP_TIME_MS = 1
    BINNING = (1, 400)
    SPECTRA_SHAPE = (1, 1340)
    
    # --- Data Processing ---
    # Number of raw acquisitions to average for the live display.
    ROLLING_AVG_WINDOW = 3
    # Number of seconds of averaged spectra to buffer for the std-dev calculation.
    STD_DEV_BUFFER_SECONDS = 1

    # --- Paths and Files ---
    # Server directory for final data storage.
    REMOTE_SAVE_DIR = r'Z:\Attoline'
    # Local directory for temporary, fast saves to avoid network latency.
    LOCAL_TEMP_DIR = os.path.join(os.path.expanduser("~"), "xuv_temp_local")
    # Energy calibration file expected next to this script.
    ENERGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spec.txt")

# --- Main application class ---------------------------------------------------
class SpectrumLiveApp:
    """Encapsulates the entire live spectrum acquisition and display application."""

    def __init__(self):
        """Initialize resources, data structures, and the I/O worker thread."""
        self.cam = None
        self.animation = None

        # --- Data Buffers ---
        # Stores the last N raw spectra for rolling average calculation.
        self.rolling_raw_buffer = deque(maxlen=Settings.ROLLING_AVG_WINDOW)
        # Stores (timestamp, avg_spectrum) tuples for std-dev calculation over the last 2s.
        self.spectrum_buffer_2s = deque()
        # Stores max values of the averaged spectrum for the top trace plot.
        self.max_vals_buffer = deque([0] * 1000, maxlen=1000)
        
        self.kept_spectrum_data = None
        self.roi_indices = np.array([])
        
        # --- Thread-safe queue for file saving ---
        # The main thread puts save jobs here; the worker thread processes them.
        self.save_queue = queue.Queue()
        # Start the background worker thread. daemon=True ensures it exits when the main app does.
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # Ensure local temp directory exists.
        os.makedirs(Settings.LOCAL_TEMP_DIR, exist_ok=True)
        print(f"Using local temporary directory: {Settings.LOCAL_TEMP_DIR}")

    def _initialize_camera(self):
        """Connects to and configures the Princeton Instruments camera."""
        try:
            print("Available cameras:", PrincetonInstruments.list_cameras())
            # Replace with your camera's serial number or use list_cameras() to find it.
            self.cam = PrincetonInstruments.PicamCamera('2105050003')
            
            self.cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
            self.cam.set_roi(hbin=Settings.BINNING[0], vbin=Settings.BINNING[1])
            self.cam.set_attribute_value("Trigger Determination", "Positive Polarity")
            self.cam.set_attribute_value("Trigger Response", "Readout Per Trigger")
            self.cam.set_attribute_value("Clean Until Trigger", False)
            self.cam.set_attribute_value("Shutter Timing Mode", "Always Open")
            self.cam.set_attribute_value("Shutter Closing Delay", 0)
            time.sleep(0.2)
            
            self.cam.setup_acquisition(mode="sequence", nframes=1000)
            self.cam.start_acquisition()
            print("Camera connected and acquisition started.")
            return True
        except Exception as e:
            print(f"Error: Could not connect or configure the camera. {e}")
            return False

    def _setup_plots(self):
        """Creates the matplotlib figure, axes, and plot artists."""
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[1, 4], width_ratios=[14, 1])
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        self.ax_max_trace = self.fig.add_subplot(gs[0, 0])
        self.ax_std_gauge = self.fig.add_subplot(gs[:, 1])

        # --- Plot Artists ---
        self.line, = self.ax_spec.plot(self.energy_eV, np.zeros_like(self.energy_eV), color='#0072BD', label="Live Spectrum")
        self.ref_line, = self.ax_spec.plot(self.energy_eV, np.zeros_like(self.energy_eV), color='#D95319', linestyle='--', label="Kept Spectrum")
        self.max_line, = self.ax_max_trace.plot(list(self.max_vals_buffer), color='#A2142F')
        self.std_dev_bar_patch = self.ax_std_gauge.bar(0, 0, color='#33A02C', width=1.0)[0]
        self.save_status_text = self.ax_max_trace.text(0.75, 1.4, '', ha='center', transform=self.ax_max_trace.transAxes, color='green')
        self.ref_line.set_visible(False)
        
        # --- Axes Styling ---
        self.ax_spec.set_title("Use up/down arrow keys to adjust Y-axis", loc='left')
        self.ax_spec.set_xlabel("Energy (eV)"); self.ax_spec.set_ylabel("Counts")
        self.ax_spec.grid(True)
        xmin, xmax = 20, 75
        self.ax_spec.set_xlim(xmin, xmax)
        initial_ylim = np.max(self.cam.read_newest_image().ravel().astype(np.uint16)) * 2
        self.ax_spec.set_ylim(0, max(10000, initial_ylim))
        self.ax_spec.hlines(y=65535, xmin=xmin, xmax=xmax, colors='#7E2F8E', linestyles='--', label="Saturation (16-bit)")
        self.ax_spec.legend()

        self.ax_max_trace.set_title(f"Max (Rolling Avg over {Settings.ROLLING_AVG_WINDOW} frames)")
        self.ax_max_trace.set_xticks([]); self.ax_max_trace.set_yticks([])
        self.ax_max_trace.grid(True, linestyle='--', alpha=0.7)

        self.ax_std_gauge.set_title(f"Avg. Norm. Std\n({Settings.STD_DEV_BUFFER_SECONDS}s window)")
        self.ax_std_gauge.set_ylim(0, 5)
        self.ax_std_gauge.set_xlim(-0.5, 0.5)
        self.ax_std_gauge.set_xticks([])
        self.ax_std_gauge.yaxis.tick_right()
        self.ax_std_gauge.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    def _setup_widgets(self):
        """Creates and connects the interactive buttons and text boxes."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        keep_btn_ax = self.fig.add_axes([0.01, 0.92, 0.08, 0.06])
        self.keep_btn = Button(keep_btn_ax, "Keep", color='#D95319', hovercolor='#FF7F0E')
        self.keep_btn.on_clicked(self._on_keep_clicked)

        save_btn_ax = self.fig.add_axes([0.01, 0.84, 0.08, 0.06])
        self.save_btn = Button(save_btn_ax, "Save", color='#1F77B4', hovercolor='#4DBEEE')
        self.save_btn.on_clicked(self._on_save_clicked)

        ax_roi_min = self.fig.add_axes([0.10, 0.92, 0.1, 0.06])
        self.text_box_min = TextBox(ax_roi_min, 'Min (eV)', initial=f"{self.ax_spec.get_xlim()[0]:.0f}")
        self.text_box_min.on_submit(self._on_roi_submit)

        ax_roi_max = self.fig.add_axes([0.21, 0.92, 0.1, 0.06])
        self.text_box_max = TextBox(ax_roi_max, 'Max (eV)', initial=f"{self.ax_spec.get_xlim()[1]:.0f}")
        self.text_box_max.on_submit(self._on_roi_submit)
      
        self._on_roi_submit(None) # Initialize ROI from text boxes

    # --- Event Handlers (Callbacks) ---
    def _on_key(self, event):
        self.animation.event_source.stop()
        current_ylim = self.ax_spec.get_ylim()
        if event.key == 'up':    new_max = current_ylim[1] * 1.2
        elif event.key == 'down':  new_max = current_ylim[1] / 1.2
        else:
            self.animation.event_source.start()
            return
        self.ax_spec.set_ylim(0, max(1000, new_max))
        self.fig.canvas.draw_idle()
        self.animation.event_source.start()

    def _on_keep_clicked(self, event):
        if not self.spectrum_buffer_2s:
            print("No spectra in buffer to keep.")
            return
        spectra_to_avg = np.array([item[1] for item in self.spectrum_buffer_2s])
        self.kept_spectrum_data = np.mean(spectra_to_avg, axis=0)
        self.ref_line.set_ydata(self.kept_spectrum_data)
        self.ref_line.set_visible(True)
        print(f"Kept the average of the last {len(self.spectrum_buffer_2s)} spectra.")

    def _on_save_clicked(self, event):
        if self.kept_spectrum_data is None:
            self.save_status_text.set_text("Click 'Keep' first to select a spectrum to save.")
            print("No 'kept' spectrum to save. Click 'Keep' first.")
            return
        
        now = datetime.datetime.now()
        date_str_long = now.strftime("%y%m%d")
        save_dir = os.path.join(Settings.REMOTE_SAVE_DIR, now.strftime("%Y"), "XUV_new", date_str_long)
        
        # The path logic and data are bundled and sent to the worker thread.
        # This function returns immediately, not waiting for the save to complete.
        job = (save_dir, self.energy_eV.copy(), self.kept_spectrum_data.copy(), now)
        self.save_queue.put(job)

        self.save_status_text.set_text("Save request queued...")
        print("Save request added to the queue.")

    def _on_roi_submit(self, _):
        try:
            min_val, max_val = float(self.text_box_min.text), float(self.text_box_max.text)
            if min_val >= max_val:
                print("Min ROI must be less than Max ROI.")
                return
            self.roi_indices = np.where((self.energy_eV >= min_val) & (self.energy_eV <= max_val))[0]
            print(f"Std Dev ROI set to {min_val:.1f}-{max_val:.1f} eV.")
        except ValueError:
            print("Invalid ROI input. Please enter numbers.")

    # --- I/O Worker Thread Function ---
    def _save_worker(self):
        """
        Runs in a separate thread. Waits for save jobs from the queue and processes them.
        This isolates slow file I/O from the main acquisition loop.
        """
        while True:
            # This call blocks until an item is available in the queue.
            save_dir, energy_data, spectrum_data, timestamp = self.save_queue.get()
            
            try:
                os.makedirs(save_dir, exist_ok=True)
                
                # Find the next available file index.
                file_index = 1
                while True:
                    date_str = timestamp.strftime("%y%m%d")
                    filename = f"XUV_{date_str}_{file_index:04d}.hdf5"
                    final_filepath = os.path.join(save_dir, filename)
                    if not os.path.exists(final_filepath):
                        break
                    file_index += 1
                
                # 1. Save to a local temporary file first. This is fast and reliable.
                local_filepath = os.path.join(Settings.LOCAL_TEMP_DIR, filename)
                
                # Use swmr=True to allow other processes to read the file while it's open.
                with h5py.File(local_filepath, 'w', libver='latest') as f:
                    f.swmr_mode = True # Explicitly enable SWMR mode
                    f.create_dataset('energy', data=energy_data)
                    f.create_dataset('spectrum', data=spectrum_data)
                    f.attrs['creation_time'] = timestamp.isoformat()
                    f.attrs['description'] = 'Kept spectrum from live XUV acquisition'
                
                # 2. Move the completed file from local temp to the final network destination.
                # This operation is much safer than writing directly over the network.
                shutil.move(local_filepath, final_filepath)
                
                status_msg = f"Saved: {os.path.basename(final_filepath)}"
                print(status_msg)
                
            except Exception as e:
                status_msg = f"Error during save: {e}"
                print(status_msg)
            finally:
                # Signal that the task from the queue is done.
                self.save_queue.task_done()


    # --- Animation Update Function ---
    def _update(self, frame):
        """The main, high-frequency update loop called by FuncAnimation."""
        data = self.cam.read_newest_image()
        if data is None:
            return self.line, self.max_line, self.ref_line, self.std_dev_bar_patch, self.save_status_text

        raw_spectrum = data.ravel().astype(np.uint16)
        if raw_spectrum.shape[0] != self.energy_eV.shape[0]:
            print(f"Warning: Unexpected spectrum shape: {raw_spectrum.shape}")
            return self.line, self.max_line, self.ref_line, self.std_dev_bar_patch, self.save_status_text

        # Update rolling average
        self.rolling_raw_buffer.append(raw_spectrum)
        avg_spectrum = np.mean(self.rolling_raw_buffer, axis=0)
        self.line.set_ydata(avg_spectrum)

        # Update max trace
        self.max_vals_buffer.append(np.max(avg_spectrum))
        self.max_line.set_ydata(list(self.max_vals_buffer))
        self.ax_max_trace.set_ylim(0, max(max(self.max_vals_buffer) * 1.1, 1000))
        
        # Maintain 2-second buffer for std-dev
        current_time = time.time()
        self.spectrum_buffer_2s.append((current_time, avg_spectrum.copy()))
        while self.spectrum_buffer_2s and (current_time - self.spectrum_buffer_2s[0][0] > Settings.STD_DEV_BUFFER_SECONDS):
            self.spectrum_buffer_2s.popleft()

        # Update std-dev gauge
        if len(self.spectrum_buffer_2s) >= 2:
            spectra_over_time = np.array([item[1] for item in self.spectrum_buffer_2s])
            std_dev = np.std(spectra_over_time, axis=0)
            mean_val = np.mean(spectra_over_time, axis=0)
            
            # Calculate normalized std, avoiding division by zero
            norm_std = np.divide(std_dev, mean_val, out=np.zeros_like(std_dev), where=mean_val != 0)
            
            if self.roi_indices.size > 0:
                avg_norm_std_percent = np.mean(norm_std[self.roi_indices]) * 100
                self.std_dev_bar_patch.set_height(avg_norm_std_percent)

        return self.line, self.max_line, self.ref_line, self.std_dev_bar_patch, self.save_status_text

    def run(self):
        """Main entry point to start the application."""
        os.system('cls' if os.name == 'nt' else 'clear')

        try:
            self.energy_eV = np.loadtxt(Settings.ENERGY_FILE)
            if self.energy_eV.shape[0] != Settings.SPECTRA_SHAPE[1]:
                print(f"Error: Energy axis length mismatch.")
                return
        except Exception as e:
            print(f"Error loading energy file: {e}")
            return

        if not self._initialize_camera():
            return

        self._setup_plots()
        self._setup_widgets()
        
        self.animation = FuncAnimation(self.fig, self._update, interval=1, blit=True, cache_frame_data=False)
        plt.show()

        # --- Cleanup ---
        self.close()

    def close(self):
        """Gracefully disconnects from the camera."""
        print("Plot window closed. Disconnecting camera...")
        if self.cam:
            if self.cam.acquisition_in_progress():
                self.cam.stop_acquisition()
            self.cam.clear_acquisition()
            self.cam.close()
            print("Camera disconnected.")

# --- Script Entry Point ---
if __name__ == "__main__":
    app = SpectrumLiveApp()
    app.run()