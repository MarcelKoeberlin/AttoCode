import numpy as np
import time
import matplotlib
# We can still suggest a good backend, though FuncAnimation is more robust
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # Import GridSpec for advanced layouts
# --- CHANGE: Import ticker for formatting the new gauge's labels ---
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation
from pylablib.devices import PrincetonInstruments
from collections import deque
from matplotlib.widgets import Button
import os

# SETTINGS #####################################################################
class Settings:
    EXP_TIME_MS = 20 #This one actually does not matter in this script :D
    BINNING = (1, 400)
    SPECTRA_SHAPE = (1, 1340)
    ENERGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spec.txt")

# MAIN FUNCTION ################################################################
def main():
    # Load energy axis from file
    ani = None
    try:
        with open(Settings.ENERGY_FILE, "r") as f:
            energy_eV = np.loadtxt(f)
    except Exception as e:
        print(f"Failed to load energy axis from file: {e}")
        return

    if energy_eV.shape[0] != Settings.SPECTRA_SHAPE[1]:
        print(f"Energy axis length ({energy_eV.shape[0]}) does not match spectrum length ({Settings.SPECTRA_SHAPE[1]})")
        return

    # --- Find indices for the Standard Deviation ROI (20-75 eV) ---
    roi_indices = np.where((energy_eV >= 20) & (energy_eV <= 75))[0]
    
    # --- Data buffer for the std dev calculation ---
    # Buffer for the last 2 seconds of total counts for calculation
    counts_buffer_2s = deque() 

    print("Connected devices:")
    print(PrincetonInstruments.list_cameras())
    print('Please standby... Code from MarcelCorp.TM loading')
    cam = PrincetonInstruments.PicamCamera('2105050003')
    print("Camera connected.")

    # Setup camera
    cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
    cam.set_roi(hbin=Settings.BINNING[0], vbin=Settings.BINNING[1])
    cam.set_attribute_value("Trigger Determination", "Positive Polarity")
    cam.set_attribute_value("Trigger Response", "Readout Per Trigger")
    cam.set_attribute_value("Clean Until Trigger", False)
    cam.set_attribute_value("Shutter Timing Mode", "Always Open")
    cam.set_attribute_value("Shutter Closing Delay", 0)
    time.sleep(0.2)
    cam.setup_acquisition(mode="sequence", nframes=1000)
    cam.start_acquisition()
    print("Starting live display... (Close the plot window to stop)")

    # --- Setup matplotlib figure and layout using GridSpec ---
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 4], width_ratios=[10, 1])

    ax_spec = fig.add_subplot(gs[1, 0])      # Bottom plot, spanning both columns
    ax_std = fig.add_subplot(gs[:, 1])   # Top-left plot
    ax_max_trace = fig.add_subplot(gs[0, 0])  # Top-right plot
    ax_std.yaxis.tick_right()
    ax_std.yaxis.set_label_position("right")
    # --- Plot Artists ---
    y = np.zeros_like(energy_eV)
    line, = ax_spec.plot(energy_eV, y, zorder=2, color='#0072BD')
    ref_line, = ax_spec.plot(energy_eV, np.zeros_like(y), color='#D95319', linestyle='--', linewidth=1, label="Kept", zorder=1)
    ref_line.set_visible(False)

    max_vals_buffer = deque([0] * 200, maxlen=200)
    max_line, = ax_max_trace.plot(list(max_vals_buffer), color='#A2142F')
    
    # --- CHANGE: Artist for the new "water fill" gauge ---
    # Create a bar container, then get the single bar patch from it.
    bar_container = ax_std.bar(0, 0, color='#0077BE', width=1.0)
    std_dev_bar_patch = bar_container[0]


    # --- Plot Styling ---
    # Bottom Spectrum Plot
    ax_spec.set_title("Use up/down arrows to change limits", loc='left')
    ax_spec.set_xlabel("Energy (eV)")
    ax_spec.set_ylabel("Counts")
    ax_spec.grid(True)
    ax_spec.set_ylim(0, np.max(cam.read_newest_image().ravel().astype(np.uint16)) * 2)
    xmin = 20
    xmax = 75
    ax_spec.set_xlim(xmin, xmax)
    ax_spec.hlines(y=65535, xmin=xmin, xmax=xmax, colors='#7E2F8E', linestyles='--', linewidth=1)

    # Top-Right Max Trace Plot
    ax_max_trace.set_title("Max Trace", fontsize=9)
    ax_max_trace.set_ylabel("Max", fontsize=8)
    ax_max_trace.set_xticks([])
    ax_max_trace.tick_params(axis='y', labelsize=8)
    ax_max_trace.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax_max_trace.set_ylim(0, 65535)

    # --- CHANGE: Styling for the new "water fill" gauge ---
    ax_std.set_title("Live Std Dev (2s window)", fontsize=9)
    # Set a fixed vertical scale from 0 to 100,000
    ax_std.set_ylim(0, 100000)
    # The x-axis is not meaningful, so hide it
    ax_std.set_xticks([])
    ax_std.set_xlim(-0.5, 0.5)
    # Format y-axis labels to be more readable (e.g., 50k)
    ax_std.yaxis.set_major_formatter(mticker.EngFormatter())
    ax_std.tick_params(labelsize=8)
    
    # --- Interaction Handlers ---
    def on_key(event):
        nonlocal ani
        if ani is None: return
        ani.event_source.stop()
        current_ylim = ax_spec.get_ylim()
        if event.key == 'up': new_max = current_ylim[1] * 1.2
        elif event.key == 'down': new_max = current_ylim[1] / 1.2
        else:
            ani.event_source.start()
            return
        ax_spec.set_ylim(0, max(1000, new_max))
        fig.canvas.draw()
        fig.canvas.flush_events()
        ani.event_source.start()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def on_keep_clicked(event):
        data = cam.read_newest_image()
        if data is not None:
            spectrum = data.ravel().astype(np.uint16)
            ref_line.set_ydata(spectrum.copy())
            ref_line.set_visible(True)
            print("Spectrum kept.")

    button_ax = fig.add_axes([0.01, 0.92, 0.08, 0.06])
    keep_button = Button(button_ax, "Keep", color='lightgray', hovercolor='lightgreen')
    keep_button.on_clicked(on_keep_clicked)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # --- Animation Core ---
    def update(frame):
        current_time = time.time()
        data = cam.read_newest_image()
        if data is None:
            return line, max_line, ref_line, std_dev_bar_patch

        spectrum = data.ravel().astype(np.uint16)
        if spectrum.shape[0] != energy_eV.shape[0]:
            print(f"Unexpected spectrum shape: {spectrum.shape}")
            return line, max_line, ref_line, std_dev_bar_patch

        # Update main spectrum plot
        line.set_ydata(spectrum)
        
        # Update max trace plot
        current_max = np.max(spectrum)
        max_vals_buffer.append(current_max)
        max_line.set_ydata(list(max_vals_buffer))
        max_line.set_xdata(np.arange(len(max_vals_buffer)))
        ax_max_trace.set_ylim(0, max(max(max_vals_buffer) * 1.1, 1000))
        
        # --- CHANGE: Update Standard Deviation GAUGE ---
        # 1. Calculate total counts in the defined ROI
        total_counts_in_roi = np.sum(spectrum[roi_indices])
        
        # 2. Add to 2-second buffer and prune old data
        counts_buffer_2s.append((current_time, total_counts_in_roi))
        while counts_buffer_2s and (current_time - counts_buffer_2s[0][0] > 2):
            counts_buffer_2s.popleft()
            
        # 3. Calculate std dev if buffer is sufficient
        std_dev = 0.0
        if len(counts_buffer_2s) >= 2:
            counts_values = [item[1] for item in counts_buffer_2s]
            std_dev = np.std(counts_values)

        # 4. Update the height of the bar patch
        std_dev_bar_patch.set_height(std_dev)

        # Return a tuple of all artists that were modified
        return line, max_line, ref_line, std_dev_bar_patch

    ani = FuncAnimation(fig, update, interval=1, blit=True, cache_frame_data=False)
    plt.show()

    # --- Cleanup ---
    print("Plot window closed. Disconnecting camera...")
    print("Thank you for choosing code from MarcelCorp. TM!")
    if cam.acquisition_in_progress():
        cam.stop_acquisition()
    cam.clear_acquisition()
    cam.close()
    print("Camera disconnected.")

# RUN SCRIPT ##################################################################
if __name__ == "__main__":
    main()