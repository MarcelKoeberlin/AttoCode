"""
spectrum_live.py

Live display and simple control for a Princeton Instruments camera.
- Shows a rolling-averaged spectrum (rolling avg over last 5 raw acquisitions).
- Shows a rolling max trace (max of the averaged spectrum over time).
- Computes a normalized std-dev gauge over a selectable ROI using the last 2 seconds
    of averaged spectra.
- "Keep" button stores the averaged spectrum as a reference line.
- "Save" button writes the kept spectrum to an HDF5 file organized by date.

"""

import os
import time
import datetime
from collections import deque

import numpy as np
import h5py
import matplotlib
# Choose a backend suitable for interactive animations.
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

# Third-party camera API (pylablib). Remain unchanged.
from pylablib.devices import PrincetonInstruments

# --- Configuration container ---------------------------------------------------
class Settings:
        """Static configuration parameters for the script."""
        # Exposure time in milliseconds.
        EXP_TIME_MS = 1

        # Binning (hbin, vbin). Vertical binning collapses many rows into one.
        BINNING = (1, 400)

        # Expected spectrum shape: (rows, cols). After binning we expect 1 x 1340.
        SPECTRA_SHAPE = (1, 1340)

        # Energy calibration file expected next to this script.
        ENERGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spec.txt")


# --- Main application ---------------------------------------------------------
def main():
        """Initialize camera, set up plotting and widgets, and run the live animation."""
        # Clear console for cleanliness (Windows oriented).
        os.system('cls')

        # Local save directory (user-customizable). The code uses a fixed path for testing.
        #base_dir = os.path.join(os.path.expanduser("~"), "XUV_new")
        base_dir = r'Z:\Attoline'  # Intentionally overriding for target environment.

        # Load energy axis (x-axis) from file and validate its length.
        try:
                energy_eV = np.loadtxt(Settings.ENERGY_FILE)
        except Exception as e:
                print(f"Error: Failed to load energy axis from '{Settings.ENERGY_FILE}'. {e}")
                return

        if energy_eV.shape[0] != Settings.SPECTRA_SHAPE[1]:
                print(f"Error: Energy axis length ({energy_eV.shape[0]}) does not match "
                            f"spectrum length ({Settings.SPECTRA_SHAPE[1]}).")
                return

        # ROI indices used for the normalized std-dev calculation. Start as full range.
        roi_indices = np.arange(energy_eV.shape[0])

        # Buffer storing tuples (timestamp, averaged_spectrum) covering the last ~2 seconds.
        spectrum_buffer_2s = deque()

        # Rolling raw buffer used to compute the displayed rolling average (last 2 raw acquisitions).
        rolling_raw_buffer = deque(maxlen=2)

        # Connect to the Princeton Instruments camera.
        try:
                print("Available cameras:", PrincetonInstruments.list_cameras())
                cam = PrincetonInstruments.PicamCamera('2105050003')  # replace with appropriate serial
                print("Camera connected successfully.")
        except Exception as e:
                print(f"Error: Could not connect to the camera. {e}")
                return

        # Configure camera acquisition parameters. Small sleep allows settings to propagate.
        cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
        cam.set_roi(hbin=Settings.BINNING[0], vbin=Settings.BINNING[1])
        cam.set_attribute_value("Trigger Determination", "Positive Polarity")
        cam.set_attribute_value("Trigger Response", "Readout Per Trigger")
        cam.set_attribute_value("Clean Until Trigger", False)
        cam.set_attribute_value("Shutter Timing Mode", "Always Open")
        cam.set_attribute_value("Shutter Closing Delay", 0)
        time.sleep(0.2)

        # Start acquisition sequence (sequence mode to continually grab frames).
        cam.setup_acquisition(mode="sequence", nframes=1000)
        cam.start_acquisition()
        print("Starting live display... (Close the plot window to stop)")

        # -------------------- Figure and axes layout --------------------------------
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 4], width_ratios=[14, 1])

        ax_spec = fig.add_subplot(gs[1, 0])      # Main spectrum plot (bottom-left)
        ax_max_trace = fig.add_subplot(gs[0, 0]) # Rolling max trace (top-left)
        ax_std_gauge = fig.add_subplot(gs[:, 1]) # Std-dev "gauge" (right column)

        # -------------------- Plot artists -----------------------------------------
        # Main live spectrum (initially zeros)
        line, = ax_spec.plot(energy_eV, np.zeros_like(energy_eV), zorder=2,
                                                 color='#0072BD', label="Live Spectrum")

        # Reference "kept" averaged spectrum (hidden until user clicks Keep)
        ref_line, = ax_spec.plot(energy_eV, np.zeros_like(energy_eV), color='#D95319',
                                                         linestyle='--', linewidth=1, label="Kept Spectrum", zorder=1)
        ref_line.set_visible(False)

        # Max-trace: keep a buffer of recent maxima (display as a line)
        max_vals_buffer = deque([0] * 1000, maxlen=1000)
        max_line, = ax_max_trace.plot(list(max_vals_buffer), color='#A2142F')

        # Standard deviation gauge implemented as a single vertical bar (bar container returns a patch)
        bar_container = ax_std_gauge.bar(0, 0, color='#33A02C', width=1.0)
        std_dev_bar_patch = bar_container[0]

        # Small status text on the max trace axes for save feedback
        save_status_text = ax_max_trace.text(0.75, 1.4, '', ha='center', va='bottom',
                                                                                fontsize=10, color='#33A02C', transform=ax_max_trace.transAxes)
        # -------------------- Axes styling -----------------------------------------
        # Spectrum axis
        ax_spec.set_title("Use up/down arrow keys to adjust Y-axis", loc='left')
        ax_spec.set_xlabel("Energy (eV)", fontsize=14)
        ax_spec.set_ylabel("Counts", fontsize=14)
        ax_spec.grid(True)
        # Set a sensible initial Y-limit from a single camera read (multiplied to give headroom)
        initial_ylim = np.max(cam.read_newest_image().ravel().astype(np.uint16)) * 2
        ax_spec.set_ylim(0, initial_ylim)
        xmin, xmax = 20, 75
        ax_spec.set_xlim(xmin, xmax)
        ax_spec.hlines(y=65535, xmin=xmin, xmax=xmax, colors='#7E2F8E',
                                     linestyles='--', linewidth=1, label="Saturation (16-bit)")
        ax_spec.legend()

        # Max trace axis — minimal decoration as requested
        ax_max_trace.set_title("Max (Rolling Avg)", fontsize=14)
        ax_max_trace.set_xticks([])
        ax_max_trace.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax_max_trace.set_ylim(0, 65535)
        ax_max_trace.set_yticks([])
        ax_max_trace.tick_params(left=False, labelleft=False)

        # Std-dev gauge axis — show percent on right y-axis
        ax_std_gauge.set_title("Avg. Norm. Std\n(2s window over 2 acquisitions)", fontsize=12)
        ax_std_gauge.set_ylim(0, 5)  # percent scale (0-5%)
        ax_std_gauge.set_xlim(-0.5, 0.5)
        ax_std_gauge.set_xticks([])
        ax_std_gauge.yaxis.tick_right()
        ax_std_gauge.yaxis.set_label_position("right")
        ax_std_gauge.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax_std_gauge.tick_params(labelsize=10)

        # -------------------- Event handlers & widgets ------------------------------
        # The animation variable is assigned after handler definitions; handlers will refer to it via nonlocal.
        animation = None  # placeholder for nonlocal reference in handlers

        def on_key(event):
                """Adjust the Y-axis scaling of the main spectrum with up/down keys."""
                nonlocal animation
                if animation is None:
                        return

                # Pause animation to prevent redraw conflicts while changing axes.
                animation.event_source.stop()
                current_ylim = ax_spec.get_ylim()

                if event.key == 'up':
                        new_max = current_ylim[1] * 1.2
                elif event.key == 'down':
                        new_max = current_ylim[1] / 1.2
                else:
                        # Resume and ignore other keys.
                        animation.event_source.start()
                        return

                # Enforce a reasonable floor to avoid tiny y-limits.
                ax_spec.set_ylim(0, max(1000, new_max))
                fig.canvas.draw_idle()
                animation.event_source.start()

        def on_keep_clicked(event):
                """Average spectra in the 2s buffer and display as the kept/reference spectrum."""
                if not spectrum_buffer_2s:
                        print("No spectra in the buffer to average.")
                        return

                spectra_to_average = np.array([item[1] for item in spectrum_buffer_2s])
                averaged_spectrum = np.mean(spectra_to_average, axis=0)

                ref_line.set_ydata(averaged_spectrum)
                ref_line.set_visible(True)
                print("Kept the average of the last 2 seconds.")

        def on_save_clicked(event):
                """Save the currently 'kept' spectrum to an HDF5 file with date-based folders."""
                if not ref_line.get_visible():
                        save_status_text.set_text("No 'kept' spectrum to save. Click 'Keep' first.")
                        fig.canvas.draw_idle()
                        print("No 'kept' spectrum to save. Click 'Keep' first.")
                        return

                kept_spectrum = ref_line.get_ydata()
                now = datetime.datetime.now()
                year_str = now.strftime("%Y")
                date_str_long = now.strftime("%y%m%d")

                save_dir = os.path.join(base_dir, year_str, "XUV_new", date_str_long)
                os.makedirs(save_dir, exist_ok=True)

                # Find next available index for the day's files.
                file_index = 1
                while True:
                        filename = f"XUV_{date_str_long}_{file_index:04d}.hdf5"
                        filepath = os.path.join(save_dir, filename)
                        if not os.path.exists(filepath):
                                break
                        file_index += 1

                try:
                        with h5py.File(filepath, 'w') as f:
                                f.create_dataset('energy', data=energy_eV)
                                f.create_dataset('spectrum', data=kept_spectrum)
                                # Add some metadata
                                f.attrs['creation_time'] = now.isoformat()
                                f.attrs['description'] = 'Kept spectrum from live XUV acquisition'
                        
                        status_msg = f"Saved to: {filepath}"
                        print(status_msg)
                        save_status_text.set_text(status_msg)
                except Exception as e:
                        status_msg = f"Error saving file: {e}"
                        print(status_msg)
                        save_status_text.set_text(status_msg)

                fig.canvas.draw_idle()

        def on_roi_submit(_):
                """Update the ROI used for the normalized std-dev calculation from text boxes."""
                nonlocal roi_indices
                try:
                        min_val = float(text_box_min.text)
                        max_val = float(text_box_max.text)
                except ValueError:
                        print("Invalid ROI input. Please enter numbers.")
                        return

                if min_val >= max_val:
                        print("Min ROI must be less than Max ROI.")
                        return

                roi_indices = np.where((energy_eV >= min_val) & (energy_eV <= max_val))[0]
                print(f"Std Dev ROI set to {min_val:.1f}-{max_val:.1f} eV.")

        def on_std_full_clicked(event):
                """Reset the ROI used for std-dev to the full energy range and update widgets."""
                nonlocal roi_indices
                roi_indices = np.arange(energy_eV.shape[0])
                text_box_min.set_val(f"{energy_eV[0]:.1f}")
                text_box_max.set_val(f"{energy_eV[-1]:.1f}")
                print("Std Dev ROI reset to full spectrum.")

        # Connect keyboard handler
        fig.canvas.mpl_connect('key_press_event', on_key)

        # Buttons and text boxes placed in figure coordinates (manual placement).
        keep_button_ax = fig.add_axes([0.01, 0.92, 0.08, 0.06])
        keep_button = Button(keep_button_ax, "Keep", color='#D95319', hovercolor='#FF7F0E')
        keep_button.on_clicked(on_keep_clicked)

        save_button_ax = fig.add_axes([0.01, 0.84, 0.08, 0.06])
        save_button = Button(save_button_ax, "Save", color='#1F77B4', hovercolor='#4DBEEE')
        save_button.on_clicked(on_save_clicked)

        ax_roi_min = fig.add_axes([0.10, 0.92, 0.1, 0.06])
        text_box_min = TextBox(ax_roi_min, 'Min (eV)', initial=f"{xmin:.0f}", textalignment="right")
        text_box_min.label.set_horizontalalignment('left')
        text_box_min.on_submit(on_roi_submit)

        ax_roi_max = fig.add_axes([0.21, 0.92, 0.1, 0.06])
        text_box_max = TextBox(ax_roi_max, 'Max (eV)', initial=f"{xmax:.0f}", textalignment="right")
        text_box_max.label.set_horizontalalignment('left')
        text_box_max.on_submit(on_roi_submit)

        std_full_ax = fig.add_axes([0.32, 0.92, 0.1, 0.06])
        std_full_button = Button(std_full_ax, "Std Dev (Full)", color='yellow', hovercolor='orange')
        std_full_button.on_clicked(on_std_full_clicked)

        # Trigger an initial ROI parse so the ROI variables match the text boxes on startup.
        on_roi_submit(None)

        # -------------------- Animation update function -----------------------------
        def update(frame):
                """Animation update called periodically by FuncAnimation.

                Steps:
                1) Acquire newest raw spectrum from camera.
                2) Append to rolling_raw_buffer and compute rolling average (displayed).
                3) Update max-trace using max of the averaged spectrum.
                4) Maintain a time-stamped buffer of averaged spectra covering ~2 s.
                5) Compute normalized std-dev across the ROI and update vertical bar height.
                """
                current_time = time.time()

                # Acquire newest frame
                data = cam.read_newest_image()
                if data is None:
                        # Nothing new to show; return artist references to keep blitting happy.
                        return line, max_line, ref_line, std_dev_bar_patch, save_status_text

                raw_spectrum = data.ravel().astype(np.uint16)

                # If the camera unexpectedly returns a different shape, warn and skip update.
                if raw_spectrum.shape[0] != energy_eV.shape[0]:
                        print(f"Warning: Unexpected spectrum shape received: {raw_spectrum.shape}")
                        return line, max_line, ref_line, std_dev_bar_patch, save_status_text

                # Rolling average over the last N raw acquisitions (rolling_raw_buffer maxlen=5).
                rolling_raw_buffer.append(raw_spectrum)
                avg_spectrum = np.mean(rolling_raw_buffer, axis=0)

                # Update displayed spectrum
                line.set_ydata(avg_spectrum)

                # Update max trace (max of the averaged spectrum)
                max_vals_buffer.append(np.max(avg_spectrum))
                max_line.set_ydata(list(max_vals_buffer))
                max_line.set_xdata(np.arange(len(max_vals_buffer)))
                ax_max_trace.set_ylim(0, max(max(max_vals_buffer) * 1.1, 1000))

                # Maintain time-stamped buffer of averaged spectra covering the last ~2 seconds.
                spectrum_buffer_2s.append((current_time, avg_spectrum.copy()))
                while spectrum_buffer_2s and (current_time - spectrum_buffer_2s[0][0] > 2):
                        spectrum_buffer_2s.popleft()

                # Compute average normalized std-dev across ROI (require at least 2 samples).
                avg_norm_std_percent = 0.0
                if len(spectrum_buffer_2s) >= 2:
                        averaged_spectra_over_time = np.array([item[1] for item in spectrum_buffer_2s])
                        std_per_energy = np.std(averaged_spectra_over_time, axis=0)
                        mean_per_energy = np.mean(averaged_spectra_over_time, axis=0)
                        normalized_std = np.divide(std_per_energy, mean_per_energy,
                                                                            out=np.zeros_like(std_per_energy),
                                                                            where=mean_per_energy != 0)
                        if roi_indices.size > 0:
                                avg_norm_std = np.mean(normalized_std[roi_indices])
                                avg_norm_std_percent = avg_norm_std * 100

                # Update gauge bar height (interpreted as percent).
                std_dev_bar_patch.set_height(avg_norm_std_percent)

                # Return artists for blitting
                return line, max_line, ref_line, std_dev_bar_patch, save_status_text

        # -------------------- Start animation and show --------------------------------
        animation = FuncAnimation(fig, update, interval=1, blit=True, cache_frame_data=False)
        plt.show()

        # -------------------- Cleanup -------------------------------------------------
        print("Plot window closed. Disconnecting camera...")
        if cam.acquisition_in_progress():
                cam.stop_acquisition()
        cam.clear_acquisition()
        cam.close()
        print("Camera disconnected.")


# Entry point
if __name__ == "__main__":
        main()