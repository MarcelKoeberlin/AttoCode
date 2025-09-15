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
    EXP_TIME_MS = 0
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
        self.cam = None
        self.animation = None

        # Placeholder for energy axis (set in run()). Initializing avoids attribute warnings.
        self.energy_eV = np.array([])

        # Data buffers
        self.rolling_raw_buffer = deque(maxlen=Settings.ROLLING_AVG_WINDOW)
        self.spectrum_buffer_2s = deque()
        self.max_vals_buffer = deque([0] * 1000, maxlen=1000)

        # Stored spectra
        self.kept_spectrum_data = None            # orange kept spectrum
        self.kept_ref_spectrum_data = None        # green reference for transmittance

        self.roi_indices = np.array([])

        # Saving infra
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        os.makedirs(Settings.LOCAL_TEMP_DIR, exist_ok=True)
        print(f"Using local temporary directory: {Settings.LOCAL_TEMP_DIR}")

    # ---------------- Camera -----------------
    def _initialize_camera(self):
        try:
            print("Available cameras:", PrincetonInstruments.list_cameras())
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

    # ---------------- Plots ------------------
    def _setup_plots(self):
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[1, 4], width_ratios=[14, 1])
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        self.ax_max_trace = self.fig.add_subplot(gs[0, 0])
        self.ax_std_gauge = self.fig.add_subplot(gs[:, 1])

        # Secondary axis for transmittance
        self.ax_spec_right = self.ax_spec.twinx()
        self.ax_spec_right.set_ylabel("Transmittance")
        self.ax_spec_right.set_ylim(0, 1)
        self.ax_spec_right.grid(False)

        # Plot artists
        zeros = np.zeros_like(self.energy_eV)
        self.line, = self.ax_spec.plot(self.energy_eV, zeros, color='#0072BD', label="Live Spectrum")
        self.ref_line, = self.ax_spec.plot(self.energy_eV, zeros, color='#D95319', linestyle='--', label="Kept Spectrum")
        self.ref_line_ref, = self.ax_spec.plot(self.energy_eV, zeros, color='#2CA02C', linestyle='--', label="Ref Spectrum")
        self.trans_line, = self.ax_spec_right.plot(self.energy_eV, zeros, color="#757575FF", label="Transmittance", linewidth=1)
        self.max_line, = self.ax_max_trace.plot(list(self.max_vals_buffer), color='#A2142F')
        self.std_dev_bar_patch = self.ax_std_gauge.bar(0, 0, color='#33A02C', width=1.0)[0]
        self.save_status_text = self.ax_max_trace.text(0.75, 1.4, '', ha='center', transform=self.ax_max_trace.transAxes, color='green')
        self.ref_line.set_visible(False)
        self.ref_line_ref.set_visible(False)
        self.trans_line.set_visible(False)

        # Axes styling
        self.ax_spec.set_title("Use up/down arrow keys to adjust Y-axis", loc='left')
        self.ax_spec.set_xlabel("Energy (eV)")
        self.ax_spec.set_ylabel("Counts")
        self.ax_spec.grid(True)
        xmin, xmax = 20, 75
        self.ax_spec.set_xlim(xmin, xmax)
        initial_ylim = np.max(self.cam.read_newest_image().ravel().astype(np.uint16)) * 2
        # Ensure saturation level is within visible range at start
        upper = max(10000, initial_ylim, 65535 * 1.05)
        self.ax_spec.set_ylim(0, upper)
        # Saturation line (store handle for legend and possible future dynamic behavior)
        self.sat_line = self.ax_spec.axhline(y=65535, color='#7E2F8E', linestyle='--', label="Saturation (16-bit)")

        # Create initial legend including static saturation line
        self.legend = self.ax_spec.legend([self.line, self.sat_line], ["Live Spectrum", "Saturation (16-bit)"])  # stored for blitting

        self.ax_max_trace.set_title(f"Max (Rolling Avg over {Settings.ROLLING_AVG_WINDOW} frames)")
        self.ax_max_trace.set_xticks([])
        self.ax_max_trace.set_yticks([])
        self.ax_max_trace.grid(True, linestyle='--', alpha=0.7)

        self.ax_std_gauge.set_title(f"Avg. Norm. Std\n({Settings.STD_DEV_BUFFER_SECONDS}s window)")
        self.ax_std_gauge.set_ylim(0, 5)
        self.ax_std_gauge.set_xlim(-0.5, 0.5)
        self.ax_std_gauge.set_xticks([])
        self.ax_std_gauge.yaxis.tick_right()
        self.ax_std_gauge.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    def _update_legend(self):
        """Update the legend to only show visible lines with their correct colors."""
        handles_l, labels_l = self.ax_spec.get_legend_handles_labels()
        handles_r, labels_r = self.ax_spec_right.get_legend_handles_labels()
        visible_handles = []
        visible_labels = []
        for h, l in zip(handles_l, labels_l):
            if h.get_visible():
                visible_handles.append(h)
                visible_labels.append(l)
        for h, l in zip(handles_r, labels_r):
            if h.get_visible():
                visible_handles.append(h)
                visible_labels.append(l)
        if not visible_handles:
            visible_handles = [self.line]
            visible_labels = ["Live Spectrum"]
        # Always include saturation line (static reference) exactly once if visible
        if hasattr(self, 'sat_line') and self.sat_line.get_visible():
            if self.sat_line not in visible_handles:
                visible_handles.append(self.sat_line)
                visible_labels.append("Saturation (16-bit)")
        # Remove old legend if exists
        if hasattr(self, 'legend') and self.legend is not None:
            try:
                self.legend.remove()
            except Exception:
                pass
        self.legend = self.ax_spec.legend(visible_handles, visible_labels)
        # Do not force full draw here; legend will be returned in blit artists.

    # ---------------- Widgets ----------------
    def _setup_widgets(self):
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        keep_btn_ax = self.fig.add_axes([0.01, 0.92, 0.08, 0.06])
        self.keep_btn = Button(keep_btn_ax, "Keep", color='#D95319', hovercolor='#FF7F0E')
        self.keep_btn.on_clicked(self._on_keep_clicked)

        keep_ref_btn_ax = self.fig.add_axes([0.01, 0.76, 0.08, 0.06])
        self.keep_ref_btn = Button(keep_ref_btn_ax, "Keep (ref)", color='#2CA02C', hovercolor='#66C266')
        self.keep_ref_btn.on_clicked(self._on_keep_ref_clicked)

        save_btn_ax = self.fig.add_axes([0.01, 0.84, 0.08, 0.06])
        self.save_btn = Button(save_btn_ax, "Save", color='#1F77B4', hovercolor='#4DBEEE')
        self.save_btn.on_clicked(self._on_save_clicked)

        ax_roi_min = self.fig.add_axes([0.10, 0.92, 0.1, 0.06])
        self.text_box_min = TextBox(ax_roi_min, 'Min (eV)', initial=f"{self.ax_spec.get_xlim()[0]:.0f}")
        self.text_box_min.on_submit(self._on_roi_submit)

        ax_roi_max = self.fig.add_axes([0.21, 0.92, 0.1, 0.06])
        self.text_box_max = TextBox(ax_roi_max, 'Max (eV)', initial=f"{self.ax_spec.get_xlim()[1]:.0f}")
        self.text_box_max.on_submit(self._on_roi_submit)

        self._on_roi_submit(None)

    # --------------- Events ------------------
    def _on_key(self, event):
        if not self.animation:
            return
        self.animation.event_source.stop()
        current_ylim = self.ax_spec.get_ylim()
        if event.key == 'up':
            new_max = current_ylim[1] * 1.2
        elif event.key == 'down':
            new_max = current_ylim[1] / 1.2
        else:
            self.animation.event_source.start()
            return
        
        # Allow any Y limit, but ensure minimum of 1000 for usability
        final_max = max(new_max, 1000)
        
        self.ax_spec.set_ylim(0, final_max)
        self.fig.canvas.draw_idle()
        self.animation.event_source.start()

    def _on_keep_clicked(self, _event):
        if not self.spectrum_buffer_2s:
            print("No spectra in buffer to keep.")
            return
        spectra_to_avg = np.array([item[1] for item in self.spectrum_buffer_2s])
        self.kept_spectrum_data = np.mean(spectra_to_avg, axis=0)
        self.ref_line.set_ydata(self.kept_spectrum_data)
        self.ref_line.set_visible(True)
        self._update_legend()
        print(f"Kept the average of the last {len(self.spectrum_buffer_2s)} spectra.")

    def _on_keep_ref_clicked(self, _event):
        if not self.spectrum_buffer_2s:
            print("No spectra in buffer to keep as reference.")
            return
        spectra_to_avg = np.array([item[1] for item in self.spectrum_buffer_2s])
        self.kept_ref_spectrum_data = np.mean(spectra_to_avg, axis=0)
        self.ref_line_ref.set_ydata(self.kept_ref_spectrum_data)
        self.ref_line_ref.set_visible(True)
        self.trans_line.set_visible(True)
        self._update_legend()
        print(f"Kept reference (avg of last {len(self.spectrum_buffer_2s)} spectra) for transmittance.")

    def _on_save_clicked(self, _event):
        if self.kept_spectrum_data is None:
            self.save_status_text.set_text("Click 'Keep' first to select a spectrum to save.")
            print("No 'kept' spectrum to save. Click 'Keep' first.")
            return
        now = datetime.datetime.now()
        date_str_long = now.strftime("%y%m%d")
        save_dir = os.path.join(Settings.REMOTE_SAVE_DIR, now.strftime("%Y"), "XUV_new", date_str_long)
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

    # --------------- Saving Thread -----------
    def _save_worker(self):
        while True:
            save_dir, energy_data, spectrum_data, timestamp = self.save_queue.get()
            try:
                os.makedirs(save_dir, exist_ok=True)
                file_index = 1
                while True:
                    date_str = timestamp.strftime("%y%m%d")
                    filename = f"XUV_{date_str}_{file_index:04d}.hdf5"
                    final_filepath = os.path.join(save_dir, filename)
                    if not os.path.exists(final_filepath):
                        break
                    file_index += 1
                local_filepath = os.path.join(Settings.LOCAL_TEMP_DIR, filename)
                with h5py.File(local_filepath, 'w', libver='latest') as f:
                    f.swmr_mode = True
                    f.create_dataset('energy', data=energy_data)
                    f.create_dataset('spectrum', data=spectrum_data)
                    f.attrs['creation_time'] = timestamp.isoformat()
                    f.attrs['description'] = 'Kept spectrum from live XUV acquisition'
                shutil.move(local_filepath, final_filepath)
                print(f"Saved: {os.path.basename(final_filepath)}")
            except Exception as e:
                print(f"Error during save: {e}")
            finally:
                self.save_queue.task_done()

    # --------------- Animation Update -------
    def _update(self, _frame):
        data = self.cam.read_newest_image()
        if data is None:
            return (self.line, self.max_line, self.ref_line, self.ref_line_ref,
                    self.trans_line, self.std_dev_bar_patch, self.save_status_text, self.legend, self.sat_line)
        raw_spectrum = data.ravel().astype(np.uint16)
        if raw_spectrum.shape[0] != self.energy_eV.shape[0]:
            print(f"Warning: Unexpected spectrum shape: {raw_spectrum.shape}")
            return (self.line, self.max_line, self.ref_line, self.ref_line_ref,
                    self.trans_line, self.std_dev_bar_patch, self.save_status_text, self.legend, self.sat_line)

        self.rolling_raw_buffer.append(raw_spectrum)
        avg_spectrum = np.mean(self.rolling_raw_buffer, axis=0)
        self.line.set_ydata(avg_spectrum)

        if self.kept_ref_spectrum_data is not None and self.kept_ref_spectrum_data.shape == avg_spectrum.shape:
            background = np.min(self.kept_ref_spectrum_data) # The camera returns something for 90 eV and up, but its just background.
            trans = np.divide(avg_spectrum - background, self.kept_ref_spectrum_data - background, out=np.zeros_like(avg_spectrum, dtype=float), where=self.kept_ref_spectrum_data != 0)
            np.clip(trans, 0, 1, out=trans)
            self.trans_line.set_ydata(trans)

        self.max_vals_buffer.append(np.max(avg_spectrum))
        self.max_line.set_ydata(list(self.max_vals_buffer))
        self.ax_max_trace.set_ylim(0, max(max(self.max_vals_buffer) * 1.1, 1000))

        current_time = time.time()
        self.spectrum_buffer_2s.append((current_time, avg_spectrum.copy()))
        while self.spectrum_buffer_2s and (current_time - self.spectrum_buffer_2s[0][0] > Settings.STD_DEV_BUFFER_SECONDS):
            self.spectrum_buffer_2s.popleft()
        if len(self.spectrum_buffer_2s) >= 2:
            spectra_over_time = np.array([item[1] for item in self.spectrum_buffer_2s])
            std_dev = np.std(spectra_over_time, axis=0)
            mean_val = np.mean(spectra_over_time, axis=0)
            norm_std = np.divide(std_dev, mean_val, out=np.zeros_like(std_dev), where=mean_val != 0)
            if self.roi_indices.size > 0:
                avg_norm_std_percent = np.mean(norm_std[self.roi_indices]) * 100
                self.std_dev_bar_patch.set_height(avg_norm_std_percent)
        return (self.line, self.max_line, self.ref_line, self.ref_line_ref,
                self.trans_line, self.std_dev_bar_patch, self.save_status_text, self.legend, self.sat_line)

    # --------------- Run / Close -------------
    def run(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        try:
            self.energy_eV = np.loadtxt(Settings.ENERGY_FILE)
            if self.energy_eV.shape[0] != Settings.SPECTRA_SHAPE[1]:
                print("Error: Energy axis length mismatch.")
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
        self.close()

    def close(self):
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