import numpy as np
import time
import matplotlib
# Suggest a backend compatible with animations. 'QtAgg' is a robust choice.
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation
from pylablib.devices import PrincetonInstruments
from collections import deque
from matplotlib.widgets import Button, TextBox
import os

# --- Configuration ---
# A class to hold all settings for easy modification.
class Settings:
    """Holds static configuration parameters for the script."""
    # Exposure time in milliseconds. Note: In trigger mode, the camera waits for a trigger,
    # so this value is less critical than the trigger rate itself.
    EXP_TIME_MS = 1  
    # Binning settings (horizontal, vertical). Here, we bin all vertical pixels into one line.
    BINNING = (1, 400)
    # Expected shape of the spectrum data (rows, columns).
    SPECTRA_SHAPE = (1, 1340)
    # Path to the energy calibration file. Assumes it's in the same directory as the script.
    ENERGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spec.txt")

# --- Main Application Logic ---
def main():
    """Initializes the camera, sets up the plot, and runs the live display."""
    
    # --- 1. Initialization and Setup ---
    
    # Load the energy axis from the calibration file.
    try:
        energy_eV = np.loadtxt(Settings.ENERGY_FILE)
    except Exception as e:
        print(f"Error: Failed to load energy axis from '{Settings.ENERGY_FILE}'. {e}")
        return

    # Validate that the energy axis matches the expected spectrum shape.
    if energy_eV.shape[0] != Settings.SPECTRA_SHAPE[1]:
        print(f"Error: Energy axis length ({energy_eV.shape[0]}) does not match spectrum length ({Settings.SPECTRA_SHAPE[1]}).")
        return

    # Initialize the Region of Interest (ROI) for standard deviation to the full spectrum.
    roi_indices = np.arange(energy_eV.shape[0])
    
    # Create a deque (a fast, double-ended queue) to buffer recent spectra for calculations.
    spectrum_buffer_2s = deque() 

    # Connect to the Princeton Instruments camera.
    try:
        print("Available cameras:", PrincetonInstruments.list_cameras())
        cam = PrincetonInstruments.PicamCamera('2105050003') # Replace with your camera's serial number if different.
        print("Camera connected successfully.")
    except Exception as e:
        print(f"Error: Could not connect to the camera. {e}")
        return

    # Configure camera acquisition settings.
    cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
    cam.set_roi(hbin=Settings.BINNING[0], vbin=Settings.BINNING[1])
    cam.set_attribute_value("Trigger Determination", "Positive Polarity")
    cam.set_attribute_value("Trigger Response", "Readout Per Trigger")
    cam.set_attribute_value("Clean Until Trigger", False)
    cam.set_attribute_value("Shutter Timing Mode", "Always Open")
    cam.set_attribute_value("Shutter Closing Delay", 0)
    time.sleep(0.2) # A brief pause to ensure settings are applied.
    
    # Start the acquisition sequence.
    cam.setup_acquisition(mode="sequence", nframes=1000)
    cam.start_acquisition()
    print("Starting live display... (Close the plot window to stop)")

    # --- 2. Plot and Widget Layout ---

    # Create the main figure and a GridSpec layout for flexible subplot arrangement.
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 4], width_ratios=[14, 1])

    # Define the subplots within the grid.
    ax_spec = fig.add_subplot(gs[1, 0])      # Main spectrum plot (bottom left)
    ax_max_trace = fig.add_subplot(gs[0, 0])  # Max value trace plot (top left)
    ax_std_gauge = fig.add_subplot(gs[:, 1])   # Standard deviation gauge (right column)
    
    # --- 3. Plot Artists and Styling ---

    # Artists for the main spectrum plot.
    line, = ax_spec.plot(energy_eV, np.zeros_like(energy_eV), zorder=2, color='#0072BD', label="Live Spectrum")
    ref_line, = ax_spec.plot(energy_eV, np.zeros_like(energy_eV), color='#D95319', linestyle='--', linewidth=1, label="Kept Spectrum", zorder=1)
    ref_line.set_visible(False) # Hide the reference line initially.

    # Artists for the max value trace plot.
    max_vals_buffer = deque([0] * 200, maxlen=200) # Buffer for the last 200 max values.
    max_line, = ax_max_trace.plot(list(max_vals_buffer), color='#A2142F')
    
    # Artist for the standard deviation gauge (a single vertical bar).
    bar_container = ax_std_gauge.bar(0, 0, color='#33A02C', width=1.0)
    std_dev_bar_patch = bar_container[0]

    # --- Styling for the Spectrum Plot (ax_spec) ---
    ax_spec.set_title("Use up/down arrow keys to adjust Y-axis", loc='left')
    ax_spec.set_xlabel("Energy (eV)")
    ax_spec.set_ylabel("Counts")
    ax_spec.grid(True)
    initial_ylim = np.max(cam.read_newest_image().ravel().astype(np.uint16)) * 2
    ax_spec.set_ylim(0, initial_ylim)
    xmin, xmax = 20, 75 # Default X-axis limits.
    ax_spec.set_xlim(xmin, xmax)
    ax_spec.hlines(y=65535, xmin=xmin, xmax=xmax, colors='#7E2F8E', linestyles='--', linewidth=1, label="Saturation (16-bit)")
    ax_spec.legend()

    # --- Styling for the Max Trace Plot (ax_max_trace) ---
    ax_max_trace.set_title("Max Count Trace", fontsize=9)
    ax_max_trace.set_ylabel("Max", fontsize=8)
    ax_max_trace.set_xticks([]) # Hide x-axis ticks.
    ax_max_trace.tick_params(axis='y', labelsize=8)
    ax_max_trace.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax_max_trace.set_ylim(0, 65535)

    # --- Styling for the Standard Deviation Gauge (ax_std_gauge) ---
    ax_std_gauge.set_title("Avg. Norm. Std\n(2s window, %)", fontsize=9)
    ax_std_gauge.set_ylim(0, 5) # Set a fixed vertical scale in percent.
    ax_std_gauge.set_xlim(-0.5, 0.5) # Center the bar.
    ax_std_gauge.set_xticks([]) # Hide x-axis ticks.
    ax_std_gauge.yaxis.tick_right() # Move ticks and labels to the right.
    ax_std_gauge.yaxis.set_label_position("right")
    ax_std_gauge.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax_std_gauge.tick_params(labelsize=8)
    
    # --- 4. Interactive Widget Handlers ---

    def on_key(event):
        """Handles key presses to adjust the Y-axis of the spectrum plot."""
        nonlocal animation # Use nonlocal to modify the animation object defined in the outer scope.
        if animation is None: return
        
        animation.event_source.stop() # Pause animation to prevent redraw conflicts.
        current_ylim = ax_spec.get_ylim()
        if event.key == 'up':
            new_max = current_ylim[1] * 1.2
        elif event.key == 'down':
            new_max = current_ylim[1] / 1.2
        else:
            animation.event_source.start() # Resume for unhandled keys.
            return
        
        ax_spec.set_ylim(0, max(1000, new_max)) # Apply new limit, with a minimum floor.
        fig.canvas.draw_idle() # Redraw the canvas.
        animation.event_source.start() # Resume animation.

    def on_keep_clicked(event):
        """Saves the current spectrum as a reference line."""
        data = cam.read_newest_image()
        if data is not None:
            spectrum = data.ravel().astype(np.uint16)
            ref_line.set_ydata(spectrum.copy())
            ref_line.set_visible(True)
            print("Reference spectrum updated.")

    def on_roi_submit(text):
        """Updates the ROI for the standard deviation calculation from text boxes."""
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
        
        # Find the indices of the energy axis that fall within the specified range.
        roi_indices = np.where((energy_eV >= min_val) & (energy_eV <= max_val))[0]
        print(f"Std Dev ROI set to {min_val:.1f}-{max_val:.1f} eV.")

    def on_std_full_clicked(event):
        """Resets the standard deviation ROI to the full spectrum."""
        nonlocal roi_indices
        roi_indices = np.arange(energy_eV.shape[0])
        # Update text boxes to reflect the full range.
        text_box_min.set_val(f"{energy_eV[0]:.1f}")
        text_box_max.set_val(f"{energy_eV[-1]:.1f}")
        print("Std Dev ROI reset to full spectrum.")

    # --- Create and place widgets on the figure ---
    fig.canvas.mpl_connect('key_press_event', on_key)

    keep_button_ax = fig.add_axes([0.01, 0.92, 0.08, 0.06])
    keep_button = Button(keep_button_ax, "Keep", color='lightgreen', hovercolor='green')
    keep_button.on_clicked(on_keep_clicked)

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

    # Manually trigger the first ROI calculation with initial values from the text boxes.
    on_roi_submit(None)

    # --- 5. Animation Core ---
    def update(frame):
        """This function is called repeatedly to update the plot data."""
        current_time = time.time()
        
        # Read the latest image from the camera.
        data = cam.read_newest_image()
        if data is None:
            return line, max_line, ref_line, std_dev_bar_patch # Return unchanged artists if no data.

        spectrum = data.ravel().astype(np.uint16)
        if spectrum.shape[0] != energy_eV.shape[0]:
            print(f"Warning: Unexpected spectrum shape received: {spectrum.shape}")
            return line, max_line, ref_line, std_dev_bar_patch

        # --- Update Spectrum Plot ---
        line.set_ydata(spectrum)
        
        # --- Update Max Trace Plot ---
        max_vals_buffer.append(np.max(spectrum))
        max_line.set_ydata(list(max_vals_buffer))
        max_line.set_xdata(np.arange(len(max_vals_buffer)))
        # Dynamically adjust the y-axis of the max trace plot.
        ax_max_trace.set_ylim(0, max(max(max_vals_buffer) * 1.1, 1000))
        
        # --- Update Standard Deviation Gauge ---
        # 1. Add the new spectrum to the 2-second buffer and remove old ones.
        spectrum_buffer_2s.append((current_time, spectrum))
        while spectrum_buffer_2s and (current_time - spectrum_buffer_2s[0][0] > 2):
            spectrum_buffer_2s.popleft()
            
        # 2. Calculate the average normalized standard deviation.
        avg_norm_std_percent = 0.0
        if len(spectrum_buffer_2s) >= 2: # Need at least two points to calculate std dev.
            # Stack spectra from the buffer into a 2D array (time x energy).
            spectra_over_time = np.array([item[1] for item in spectrum_buffer_2s])
            
            # Calculate standard deviation and mean for each energy bin (column-wise).
            std_per_energy = np.std(spectra_over_time, axis=0)
            mean_per_energy = np.mean(spectra_over_time, axis=0)
            
            # Calculate normalized std dev (std/mean), avoiding division by zero.
            normalized_std = np.divide(std_per_energy, mean_per_energy, 
                                      out=np.zeros_like(std_per_energy), 
                                      where=mean_per_energy != 0)
            
            # Average the normalized std dev over the user-defined ROI.
            if roi_indices.size > 0:
                avg_norm_std = np.mean(normalized_std[roi_indices])
                avg_norm_std_percent = avg_norm_std * 100 # Convert to percent for display.

        # 3. Update the height of the gauge bar.
        std_dev_bar_patch.set_height(avg_norm_std_percent)

        # Return a tuple of all artists that were modified for blitting.
        return line, max_line, ref_line, std_dev_bar_patch

    # Create and start the animation.
    animation = FuncAnimation(fig, update, interval=1, blit=True, cache_frame_data=False)
    plt.show()

    # --- 6. Cleanup ---
    print("Plot window closed. Disconnecting camera...")
    if cam.acquisition_in_progress():
        cam.stop_acquisition()
    cam.clear_acquisition()
    cam.close()
    print("Camera disconnected.")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()