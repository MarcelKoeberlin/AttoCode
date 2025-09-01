from lib2to3.pgen2.token import NUMBER

from pylablib.devices import PrincetonInstruments
import os
from datetime import datetime
import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# PATHS: ##############################################################
class Paths:
    BACKUP_ALL_ATTRIBUTES_AT_START = r"C:\Users\Moritz\Desktop\Pixis_data\attribute_backups"
    TEST_IMAGES = r"C:\Users\Moritz\Desktop\Pixis_data\test_images"

# PROGRAM SETTINGS: ####################################################
class Settings:
    EXP_TIME_MS = 20  # Set exposure time
    # ROI = (300, 800, 100, 350)  # (x_start, x_end, y_start, y_end) - Example ROI inside 1340x400
    BINNING = (1, 400)  # (x_binning, y_binning) - Bin all rows into 1, keeping full width

    NUMBER_OF_IMAGES = 50  # Number of images to acquire

#########################################################################
# MAIN FUNCTION ########################################################
#########################################################################
def main():
    # generate timestamp:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    print(rf"Current timestamp: {timestamp}")

    # check if camera is connected:
    print("Connected devices:")
    print(PrincetonInstruments.list_cameras())  # Should list the connected camera

    # initialize the camera object:
    cam = PrincetonInstruments.PicamCamera('2105050003')

    # print the current pixel format:
    print(rf"Current pixel format: {cam.get_attribute_value('Pixel Format')}")

    # Print the temperatures:
    print(rf"Sensor Temperature Set Point: {cam.get_attribute_value('Sensor Temperature Set Point')} K")
    print(rf"Sensor Temperature Reading: {cam.get_attribute_value('Sensor Temperature Reading')} K")

    # Set the exposure time -----------------------------------------------
    cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
    # print the set exp time:
    # print(rf"Exposure time set to {cam.cav['Exposure Time']} ms")
    print(rf"Exposure time set to {cam.get_attribute_value('Exposure Time')} ms")  # alternatively

    # save the list of all attributes to a txt file at the start: --------------------------------
    path = os.path.join(Paths.BACKUP_ALL_ATTRIBUTES_AT_START, timestamp + "_all_attributes_backup.txt")
    with open(path, "w") as file:
        file.write(str(cam.get_all_attribute_values()).removeprefix("Dictionary(").removesuffix(")"))
    print("\n")

    # Print all attributes: -------------------------------------------------------
    print("List of all camera attributes:")
    print(str(cam.get_all_attribute_values()).removeprefix("Dictionary(").removesuffix(
        ")"))  # prints all camera attributes and their values
    print("\n")

    #########################################################################
    #########################################################################
    # IMAGE ACQUISITION ####################################################
    #########################################################################
    #########################################################################
    print("\n")
    print("IMAGE ACQUISITION: ##############################################")
    print("\n")

    #########################################################################
    # BACKUP ORIGINAL SETTINGS #############################################
    #########################################################################

    # get the original ROI settings:
    ROI_ORIGINAL = cam.get_roi()
    # # get the oroginal trigger mode:
    # TRIGGER_MODE_ORIGINAL = cam.get_attribute_value("Trigger Source")

    #######################################################################
    # 1. Acquire single full frame (1340 x 400)
    #######################################################################

    # print the current ROI settings:
    print_roi(cam)

    print(cam.get_settings())

    # print the roi limits:
    print(f"ROI limits: {cam.get_roi_limits()}")

    # set the ROI to the desired settings:
    cam.set_roi(hbin=1, vbin=400)
    # set the ROI to the desired settings:
    print_roi(cam)

    # print max calculated fps:
    print(f"Max calculated fps: {cam.get_attribute_value('Frame Rate Calculation')}")

    # wait 0.5 seconds:
    time.sleep(0.2)

    # set up the acquisition settings:
    # cam.setup_acquisition(mode="snap", nframes=Settings.NUMBER_OF_IMAGES + 1)
    cam.setup_acquisition(mode="sequence", nframes=Settings.NUMBER_OF_IMAGES + 1)

    # start the timer:
    start_time = time.time()

    print("\n")
    # start the acquisition: ####################################################
    cam.start_acquisition()


    # print(cam.acquisition_in_progress())


    # print the status of the frames:
    # print_frames_status(cam)


    # data_full = cam.snap()  # single image
    # data_full = cam.grab()  # single image
    # check if new image is available:

    # data_full = cam.read_newest_image()
    # save_image(data_full, f"full_frame_{timestamp}.npy")
    # cam.snap()

    # # Acquire the images in a loop:
    # for i in range(Settings.NUMBER_OF_IMAGES + 100):
    #     print(rf"Unread frames: {cam.get_frames_status()[1]}")
    #     data_full = cam.read_newest_image()
    #     save_image(data_full, rf"{timestamp}_full_image_{i}.npy")
    #     print_frames_status(cam)

    image_count = 0
    while image_count < Settings.NUMBER_OF_IMAGES:
        # get the status of the frames:
        acquired_imgs, unread_imgs, skipped_imgs, buffer_size = cam.get_frames_status()
        # wait till images are available:
        while unread_imgs == 0:
            time.sleep(0.001) # wait for the image to be available
            acquired_imgs, unread_imgs, skipped_imgs, buffer_size = cam.get_frames_status()

        print(f"Acquired: {acquired_imgs}, Unread: {unread_imgs}, Skipped: {skipped_imgs}, Buffer size: {buffer_size}")

        data_full = cam.read_oldest_image()
        save_image(data_full, rf"{timestamp}_full_image_{image_count}.npy")
        image_count += 1

        # stop the timer:
        end_time = time.time()
        print(rf"{image_count} images acquired in {end_time - start_time} seconds.")


    # stop the acquisition:
    if cam.acquisition_in_progress():
        cam.stop_acquisition()
        print("Acquisition stopped.")

    # clear the acquisition settings:
    cam.clear_acquisition()

    # set the ROI back to the original settings:
    cam.set_roi(*ROI_ORIGINAL)

    # Close the camera ####################################################
    cam.close()
    print("\nCamera connection closed.")

#########################################################################
#########################################################################
# MISC FUNCTIONS #######################################################
#########################################################################
#########################################################################
def save_image(data, filename):
    """
    Save image as a NumPy .npy file.

    :param data: Image array
    :param filename: Filename to save the image
    """
    path = os.path.join(Paths.TEST_IMAGES, filename)
    np.save(path, data)
    print(f"Saved image to {path}")


def print_roi(cam) -> None:
    """
    Print the current ROI settings of the camera in a fancyier way, with documentation:
    Return tuple (hstart, hend, vstart, vend, hbin, vbin). hstart and hend specify horizontal image extent, vstart and vend specify vertical image extent (start is inclusive, stop is exclusive, starting from 0), hbin and vbin specify binning.
    """
    roi = cam.get_roi()
    print(f"ROI settings:")
    print(f"Horizontal start: {roi[0]}")
    print(f"Horizontal end: {roi[1]}")
    print(f"Vertical start: {roi[2]}")
    print(f"Vertical end: {roi[3]}")
    print(f"Horizontal binning: {roi[4]}")
    print(f"Vertical binning: {roi[5]}")
    print("\n")

def print_array_info(arr):
    """
    Prints information about a NumPy array including:
    - The array contents
    - Shape
    - Number of dimensions
    - Number of elements
    - Data type
    - Bytes per element
    - Total bytes
    - Detailed NumPy info

    Raises:
        TypeError: If `arr` is not a numpy.ndarray.
    """

    print("Array:\n", arr, "\n")
    print("Shape:", arr.shape)
    print("Number of dimensions:", arr.ndim)
    print("Number of elements:", arr.size)
    print("Data type (dtype):", arr.dtype)
    print("Bytes per element (itemsize):", arr.itemsize)
    print("Total bytes (nbytes):", arr.nbytes, "\n")

    # # Check if `arr` is a numpy array
    # if not isinstance(arr, np.ndarray):
    #     raise TypeError("Input must be a NumPy array (np.ndarray).")
    # print("Detailed NumPy info on the array:")
    # np.info(arr)


def print_frames_status(cam) -> None:
    """
    Calls cam.get_frames_status() and prints the acquisition and buffer status
    in a fancy format.

    :param cam: Camera object that has a method get_frames_status().
    """
    console = Console()

    # Get the status from the camera
    status = cam.get_frames_status()

    # Unpacking the tuple
    acquired, unread, skipped, buffer_size = status

    # Title panel
    title = Panel.fit("[bold cyan]Acquisition & Buffer Status[/bold cyan]",
                      border_style="cyan", padding=(1, 2))

    # Table for details
    table = Table(box=None)
    table.add_column("Metric", style="bold yellow", justify="right")
    table.add_column("Value", style="bold white", justify="left")

    table.add_row("[cyan]Acquired Frames[/cyan]", f"{acquired}")
    table.add_row("[cyan]Unread Frames[/cyan]", f"{unread}")
    table.add_row("[cyan]Skipped Frames[/cyan]", f"{skipped}")
    table.add_row("[cyan]Buffer Size[/cyan]", f"{buffer_size}")

    console.print(title)
    console.print(table)


# Run main function #############################################################
if __name__ == "__main__":
    main()
