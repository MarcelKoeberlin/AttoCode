import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def select_dat_file() -> str:
    """
    Opens a file dialog for the user to select a .dat memmap file.

    :return: The selected file path or an empty string if no file was selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        title="Select .dat memmap file",
        initialdir="C:\\Users\\Moritz\\Desktop\\Pixis_data\\01_XUV_Spectra",
        filetypes=[("Memmap Files", "*.dat")]
    )

    return file_path

def load_memmap_dat(filename: str, dtype=np.uint16, entry_shape=(1, 1340)):
    """
    Loads a .dat memmap file and determines its shape dynamically.

    :param filename: Path to the .dat file.
    :param dtype: Data type (default: np.uint16).
    :param entry_shape: The shape of one entry.
    :return: Loaded memmap array.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Compute number of entries dynamically
    file_size = os.path.getsize(filename)  # File size in bytes
    entry_size = np.prod(entry_shape) * np.dtype(dtype).itemsize  # Bytes per entry

    if file_size % entry_size != 0:
        raise ValueError("File size is not a multiple of entry size. Possible corruption.")

    num_entries = file_size // entry_size  # Compute the number of (1, 1340) entries

    # Load the memmap file
    memmap_array = np.memmap(filename, dtype=dtype, mode='r', shape=(num_entries, *entry_shape))

    print(f"Memmap file loaded with shape: {memmap_array.shape}")
    return memmap_array

def main():
    """
    Main function to select a .dat memmap file, load the entire dataset,
    remove rows that are all zeros, compute min/1%/max values, and display it using imshow.
    """
    file_path = select_dat_file()

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit()

    try:
        # Load memmap
        memmap_array = load_memmap_dat(file_path)

        # Reshape to (num_entries, 1340) for visualization
        data = memmap_array[:, 0, :]  # Shape: (num_entries, 1340)

        # Remove rows that are entirely zeros
        non_zero_mask = ~np.all(data == 0, axis=1)  # True for non-zero rows
        filtered_data = data[non_zero_mask]  # Only keep non-zero rows

        if filtered_data.shape[0] == 0:
            print("All rows are zero! Nothing to display.")
            sys.exit()

        # Compute min, 1% percentile, and max values
        data_min = np.min(filtered_data)
        data_1percent = np.percentile(filtered_data, 1)
        data_max = np.max(filtered_data)

        print(f"Filtered shape: {filtered_data.shape} (after removing zero rows)")
        print(f"Min value: {data_min}")
        print(f"1% percentile value: {data_1percent}")
        print(f"Max value: {data_max}")

        # Flip the data horizontally
        flipped_data = np.fliplr(filtered_data)

        # Display the flipped memmap as an image
        plt.imshow(flipped_data, cmap='viridis', aspect='auto', vmin=data_1percent, vmax=data_max)
        plt.colorbar()
        plt.title(f"Memmap Visualization {flipped_data.shape}")
        plt.xlabel("Width (Pixels)")
        plt.ylabel("Entry Index (filtered)")
        plt.show()

    except Exception as e:
        print(f"Error loading memmap file: {e}")
        sys.exit()

if __name__ == "__main__":
    main()