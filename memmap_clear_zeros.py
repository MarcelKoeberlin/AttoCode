import numpy as np
import os
import sys
from tkinter import Tk, filedialog

SPECTRUM_LENGTH = 1340
DEFAULT_PATH = r"C:\Users\Moritz\Desktop\Pixis_data\01_XUV_Spectra"


def select_npy_file() -> str:
    """
    Opens a file dialog to select a `.npy` file using tkinter on Windows.
    :return: Path to the selected .npy file or exits if cancelled.
    """
    root = Tk()
    root.withdraw()  # Hide the main tkinter window

    file_path = filedialog.askopenfilename(
        title="Select a .npy memmap file",
        initialdir=DEFAULT_PATH,
        filetypes=[("NumPy memmap files", "*.npy")]
    )

    if not file_path:
        print("File selection cancelled.")
        sys.exit(1)

    if not file_path.endswith(".npy"):
        print("Selected file is not a .npy file.")
        sys.exit(1)

    return file_path


def load_and_clean_memmap(file_path: str) -> None:
    """
    Loads the memmap, removes all-zero rows, and overwrites the original file.
    :param file_path: Path to the .npy file.
    """
    print(f"Loading memmap from: {file_path}")

    dtype = np.dtype([
        ("intensities", np.uint16, SPECTRUM_LENGTH),
        ("timestamp_us", np.uint64)
    ])

    mmap = np.memmap(file_path, dtype=dtype, mode="r")

    nonzero_mask = ~(
        (mmap["timestamp_us"] == 0) &
        (np.all(mmap["intensities"] == 0, axis=1))
    )

    cleaned_data = mmap[nonzero_mask]
    print(f"Original rows: {len(mmap)}, Non-zero rows: {len(cleaned_data)}")

    del mmap  # Unmap before overwriting
    os.remove(file_path)

    cleaned_mmap = np.memmap(file_path, dtype=dtype, mode="w+", shape=(len(cleaned_data),))
    cleaned_mmap[:] = cleaned_data
    cleaned_mmap.flush()
    print(f"Cleaned memmap saved to: {file_path}")


def main():
    file_path = select_npy_file()
    load_and_clean_memmap(file_path)


if __name__ == "__main__":
    main()
