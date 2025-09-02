import numpy as np
import time
import matplotlib
# We can still suggest a good backend, though FuncAnimation is more robust
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
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

    print("Connected devices:")
    print(PrincetonInstruments.list_cameras())

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

    # Setup matplotlib
    fig, ax = plt.subplots()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Plot Artists ---
    y = np.zeros_like(energy_eV)
    line, = ax.plot(energy_eV, y, zorder=2, color='#0072BD')
    ref_line, = ax.plot(energy_eV, np.zeros_like(y), color='#D95319', linestyle='--', linewidth=1, label="Kept", zorder=1)
    ref_line.set_visible(False)

    max_vals_buffer = deque([0] * 200, maxlen=200)
    ax2 = fig.add_axes([0.69, 0.7, 0.3, 0.25])
    max_line, = ax2.plot(list(max_vals_buffer), color='#A2142F')
    
    # --- Plot Styling ---
    ax.set_title("Use up/down arrows to change limits", loc='left')
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Counts")
    ax.grid(True)
    ax.set_ylim(0, np.max(cam.read_newest_image().ravel().astype(np.uint16)) * 2)
    ax.set_xlim(20, 75)

    ax2.set_title("Max Trace", fontsize=9)
    ax2.set_ylabel("Max", fontsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    #ax2.tick_params(labelsize=8)
    ax2.set_ylim(0, 65535)

    # --- Interaction Handlers ---
# In your main() function, replace the on_key function with this one:
    def on_key(event):
            nonlocal ani
            if ani is None:
                return

            # --- Stop the animation ---
            ani.event_source.stop()

            current_ylim = ax.get_ylim()
            if event.key == 'up':
                new_max = current_ylim[1] * 1.2
            elif event.key == 'down':
                new_max = current_ylim[1] / 1.2
            else:
                # If not a key we care about, resume immediately
                ani.event_source.start()
                return
            
            ax.set_ylim(0, max(1000, new_max))
            
            # --- Manually perform a full redraw NOW ---
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            print(f"Updated ylim_max: {ax.get_ylim()[1]:.0f}")

            # --- Resume the fast animation ---
            ani.event_source.start()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def on_keep_clicked(event):
        # Read the latest data from the camera to 'freeze' it
        data = cam.read_newest_image()
        if data is not None:
            spectrum = data.ravel().astype(np.uint16)
            ref_line.set_ydata(spectrum.copy())
            ref_line.set_visible(True)
            print("Spectrum kept.")

    button_ax = fig.add_axes([0.01, 0.9, 0.08, 0.06])
    keep_button = Button(button_ax, "Keep", color='lightgray', hovercolor='lightgreen')
    keep_button.on_clicked(on_keep_clicked)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # --- Animation Core ---
    def update(frame):
        data = cam.read_newest_image()
        if data is None:
            # If no new data, return the artists unchanged
            return line, max_line, ref_line

        spectrum = data.ravel().astype(np.uint16)
        if spectrum.shape[0] != energy_eV.shape[0]:
            print(f"Unexpected spectrum shape: {spectrum.shape}")
            return line, max_line, ref_line

        # Update plot data
        line.set_ydata(spectrum)
        
        # Update max trace
        current_max = np.max(spectrum)
        max_vals_buffer.append(current_max)
        max_line.set_ydata(list(max_vals_buffer))
        max_line.set_xdata(np.arange(len(max_vals_buffer)))
        ax2.set_ylim(0, max(max(max_vals_buffer) * 1.1, 1000))
        
        # Return a tuple of all artists that were modified
        return line, max_line, ref_line

    # Create the animation object. 
    # interval=1 tries to run as fast as possible. The camera read time will be the real limit.
    # blit=True enables the high-speed updates.
    ani = FuncAnimation(fig, update, interval=1, blit=True, cache_frame_data=False)

    # Show the plot and start the animation. This is a blocking call.
    plt.show()

    # --- Cleanup ---
    # This code will only run after you close the matplotlib window
    print("Plot window closed. Disconnecting camera...")
    if cam.acquisition_in_progress():
        cam.stop_acquisition()
    cam.clear_acquisition()
    cam.close()
    print("Camera disconnected.")


# RUN SCRIPT ##################################################################
if __name__ == "__main__":
    main()