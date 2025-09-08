import json
import numpy as np
import os
from typing import Dict
import matplotlib.pyplot as plt
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog

# Default data path
DEFAULT_PATH = r"C:\Users\Moritz\Desktop\Pixis_data\01_XUV_Spectra"


def main():
    # GUI file picker
    metadata_file = select_metadata_file()
    if not metadata_file:
        print("No file selected. Exiting.")
        return

    # Load spectra and metadata
    spectra = get_princeton_spectra(metadata_file)
    timestamps = get_timestamps(metadata_file)
    metadata = get_metadata(metadata_file)

    # Remove rows with only zeros
    mask = ~np.all(spectra == 0, axis=1)
    filtered_spectra = spectra[mask]
    filtered_timestamps = timestamps[mask]
    skipped_rows = spectra.shape[0] - filtered_spectra.shape[0]
    print(f"Skipped {skipped_rows} rows containing only zeros.")

    # Remove the first acquired spectrum (lowest timestamp) from spectra only
    index_of_earliest = np.argmin(filtered_timestamps)
    filtered_spectra = np.delete(filtered_spectra, index_of_earliest, axis=0)
    print(f"Removed spectrum at index {index_of_earliest} (earliest timestamp).")

    # Print timestamps and violations
    print(f"\nTimestamps (us) [full length]: {timestamps}")
    print("\nViolations at indices:", metadata.get("violations", []))

    # Plot spectra as image
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        filtered_spectra,
        aspect="auto",
        cmap="viridis",
        extent=(0, filtered_spectra.shape[1], filtered_spectra.shape[0], 0)
    )
    plt.colorbar(im, label="Intensity (a.u.)")
    plt.xlabel("Pixel Index")
    plt.ylabel("Frame Index")
    plt.title("XUV Spectra Matrix (excluding first acquired frame)")
    plt.tight_layout()
    plt.show()


# FUNCTIONS TO LOAD DATA AND METADATA #########################################
def get_metadata(json_path: str) -> Dict:
    """
    Load metadata from a JSON file.

    :param json_path: Path to the metadata JSON file.
    :return: Metadata dictionary.
    """
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata


def get_timestamps(json_path: str) -> np.ndarray:
    """
    Extract timestamps from the memory-mapped file.

    :param json_path: Path to the metadata JSON file.
    :return: Array of timestamps in microseconds.
    """
    metadata = get_metadata(json_path)
    memmap_path = os.path.join(os.path.dirname(json_path), metadata["memmap_file"])

    dtype = np.dtype([
        ("spectrum", np.uint16, metadata["spectra_shape"][1]),
        ("timestamp_us", np.uint64)
    ])
    mmap = np.memmap(memmap_path, dtype=dtype, mode="r")
    return mmap["timestamp_us"]


def get_princeton_spectra(json_path: str) -> np.ndarray:
    """
    Extract the Princeton spectra as a 2D NumPy array from the structured memmap.

    :param json_path: Path to the metadata JSON file.
    :return: 2D NumPy array with shape (N_frames, N_pixels).
    """
    metadata = get_metadata(json_path)
    memmap_path = os.path.join(os.path.dirname(json_path), metadata["memmap_file"])

    dtype = np.dtype([
        ("spectrum", np.uint16, metadata["spectra_shape"][1]),
        ("timestamp_us", np.uint64)
    ])
    mmap = np.memmap(memmap_path, dtype=dtype, mode="r")
    return mmap["spectrum"]


def select_metadata_file() -> str:
    """
    Cross-platform file selection:
    - Uses Zenity on Linux (if available)
    - Falls back to Tkinter on Windows or if Zenity is unavailable

    :return: Path to the selected file, or empty string if cancelled.
    """
    system = platform.system()

    if system == "Linux":
        try:
            result = subprocess.run(
                [
                    'zenity',
                    '--file-selection',
                    '--title=Select Metadata JSON File',
                    '--file-filter=*.json'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return ""
        except FileNotFoundError:
            print("Zenity not found. Falling back to Tkinter.")

    # Tkinter fallback
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Metadata JSON File",
        filetypes=[("JSON files", "*.json")]
    )
    return path


if __name__ == "__main__":
    main()
