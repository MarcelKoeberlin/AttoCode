# A TRIGGER HAS TO RUN FOR THIS SCRIPT TO WORK!!!!
import numpy as np
import time
import matplotlib.pyplot as plt
from pylablib.devices import PrincetonInstruments
from collections import deque
from matplotlib.widgets import Button

# SETTINGS #####################################################################
class Settings:
    EXP_TIME_MS = 1
    BINNING = (1, 400)
    SPECTRA_SHAPE = (1, 1340)
    REFRESH_TIME_S = 0.01
    ENERGY_FILE = r"C:\Users\Moritz\Desktop\Pixis_data\Spec.txt"


# MAIN FUNCTION ################################################################
def main():
    # Load energy axis from file
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

    print("Starting live display... (Press Ctrl+C to stop)")

    # Setup matplotlib
    plt.ion()
    fig, ax = plt.subplots()

    # Create "Keep" button
    button_ax = fig.add_axes([0.01, 0.9, 0.08, 0.06])  # [left, bottom, width, height]
    keep_button = Button(button_ax, "Keep", color='lightgray', hovercolor='lightgreen')

    # Initial ylim max
    ylim_max = 60000

    def on_key(event):
        nonlocal ylim_max
        if event.key == 'up':
            ylim_max *= 1.2  # Increase upper limit
        elif event.key == 'down':
            ylim_max /= 1.2  # Decrease upper limit
        ylim_max = max(1000, ylim_max)  # Avoid too low values
        ax.set_ylim(0, ylim_max)
        fig.canvas.draw()
        print(f"Updated ylim_max: {ylim_max:.0f}")

    def on_keep_clicked(event):
        if spectrum is not None:
            ref_line.set_ydata(spectrum.copy())
            ref_line.set_visible(True)
            fig.canvas.draw()
            print("Spectrum kept.")

    keep_button.on_clicked(on_keep_clicked)

    fig.canvas.mpl_connect('key_press_event', on_key)

    y = np.zeros_like(energy_eV)
    line, = ax.plot(energy_eV, y, zorder=2)

    ref_line, = ax.plot(energy_eV, np.zeros_like(y), color='green', linestyle='--', linewidth=1.5, label="Kept",
                        zorder=1)
    ref_line.set_visible(False)  # Only show when spectrum is saved

    # Buffer to store max values
    max_vals_buffer = deque([0] * 200, maxlen=200)

    # Create a small inset axes for max value trace (top-left corner)
    ax2 = fig.add_axes([0.69, 0.7, 0.3, 0.25])  # [left, bottom, width, height] in figure coords
    ax2.set_title("Max Trace", fontsize=9)
    ax2.set_ylabel("Max", fontsize=8)
    ax2.set_xticks([])
    # add a grid
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.tick_params(labelsize=8)
    max_line, = ax2.plot(list(max_vals_buffer), color='red')
    ax2.set_ylim(0, 65535)  # Adjust based on expected range of max values

    ax.set_title("Use up/down arrows to change limits", loc='left')
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Counts")
    ax.grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()

    try:
        spectrum = None
        while True:
            data = cam.read_newest_image()

            if data is None:
                time.sleep(Settings.EXP_TIME_MS / 1000 / 10)
                continue

            spectrum = data.ravel().astype(np.uint16)

            # Update max buffer and trace
            current_max = np.max(spectrum)
            max_vals_buffer.append(current_max)

            # Update max trace plot
            max_line.set_ydata(list(max_vals_buffer))
            max_line.set_xdata(np.arange(len(max_vals_buffer)))  # X-axis: just index
            ax2.set_ylim(0, max(max(max_vals_buffer) * 1.1, 1000))  # Dynamic Y-limits
            ax2.set_xlim(0, len(max_vals_buffer))

            if spectrum.shape[0] != energy_eV.shape[0]:
                print(f"Unexpected spectrum shape: {spectrum.shape}")
                continue

            # Update plot
            try:
                line.set_ydata(spectrum)
                ax.set_ylim(0, ylim_max)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(Settings.REFRESH_TIME_S)
            except Exception as e:
                print(f"Plot update error: {e}")
                continue

    except KeyboardInterrupt:
        print("User stopped with Ctrl+C.")

    finally:
        if cam.acquisition_in_progress():
            cam.stop_acquisition()
        cam.clear_acquisition()
        cam.close()
        print("Camera disconnected.")
        plt.ioff()
        plt.show()


# RUN SCRIPT ##################################################################
if __name__ == "__main__":
    main()
