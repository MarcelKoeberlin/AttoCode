import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import sys

def select_npy_file() -> str:
    """
    Opens a file dialog for the user to select a .npy file.

    :return: The selected file path or an empty string if no file was selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        title="Select .npy file",
        initialdir="C:\\Users\\Moritz\\Desktop\\Pixis_data\\test_images",
        filetypes=[("NumPy Files", "*.npy")]
    )

    return file_path

def main():


    """
    Main function to select a .npy file, load the 2D matrix, print min/1%/max values,
    and display it using imshow with a viridis colormap.
    """
    file_path = select_npy_file()

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit()

    data = np.load(file_path)

    # Compute min, 1st percentile, and max values
    data_min = np.min(data)
    data_1percent = np.percentile(data, 1)
    data_max = np.max(data)

    print(f"Min value: {data_min}")
    print(f"1% percentile value: {data_1percent}")
    print(f"Max value: {data_max}")

    # Plot with vmin set to the 1% percentile and vmax to max
    plt.imshow(data, cmap='viridis', vmin=data_1percent, vmax=data_max)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
