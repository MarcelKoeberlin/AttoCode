from pylablib.devices import PrincetonInstruments
import os
from datetime import datetime
import time
import numpy as np

# PATHS: ##############################################################
class Paths:
    SPECTRA_FOLDER = r"C:\Users\Moritz\Desktop\Pixis_data\01_XUV_Spectra"

# PROGRAM SETTINGS: ####################################################
class Settings:
    EXP_TIME_MS = 28  # Set exposure time, 28ms is currently the max for 20Hz
    # ROI = (300, 800, 100, 350)  # (x_start, x_end, y_start, y_end) - Example ROI inside 1340x400
    BINNING = (1, 400)  # (x_binning, y_binning) - Bin all rows into 1, keeping full width
    SPECTRA_SHAPE = (1, 1340)  # Shape of the spectra array, CHANGE TOGETHER WITH BINNING

    NUMBER_OF_IMAGES = int(3e4)  # Number of images to acquire

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

    #########################################################################
    #########################################################################
    # IMAGE ACQUISITION ####################################################
    #########################################################################
    #########################################################################
    print("\n")
    print("IMAGE ACQUISITION: ##############################################")
    print("\n")

    #######################################################################
    # Set up the camera for acquisition ###################################
    #######################################################################

    # Set the exposure time -----------------------------------------------
    cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
    # print the set exp time:
    # print(rf"Exposure time set to {cam.cav['Exposure Time']} ms")
    print(rf"Exposure time set to {cam.get_attribute_value('Exposure Time')} ms")  # alternatively

    # Set up the ROI: ####################################################
    print("Current settings:")
    print(cam.get_settings())

    # print the roi limits:
    print(f"ROI limits: {cam.get_roi_limits()}")

    # set the ROI to the desired settings:
    cam.set_roi(hbin=Settings.BINNING[0], vbin=Settings.BINNING[1])

    # Set up the trigger: ####################################################
    # set the trigger determination to 'Positive Polarity':
    cam.set_attribute_value("Trigger Determination", "Positive Polarity")

    # set the trigger response to 'Readout Per Trigger':
    cam.set_attribute_value("Trigger Response", "Readout Per Trigger")

    # print the trigger determination:
    print(f"Trigger Determination: {cam.get_attribute_value('Trigger Determination')}")
    # print the trigger response:
    print(f"Trigger Response: {cam.get_attribute_value('Trigger Response')}")

    # Set up the shutter: ################################################

    # shutter_closing_delay = cam.get_attribute("Shutter Closing Delay")

    # set the shutter timing mode:
    cam.set_attribute_value("Shutter Timing Mode", "Always Open")
    # set shutter closing delay:
    cam.set_attribute_value("Shutter Closing Delay", 0)

    # print the shutter timing mode:
    print(f"Shutter Timing Mode: {cam.get_attribute_value('Shutter Timing Mode')}")
    # print the shutter closing delay:
    print(f"Shutter Closing Delay: {cam.get_attribute_value('Shutter Closing Delay')}")

    # print the calculated fps:
    print(f"Calculated fps: {cam.get_attribute_value('Frame Rate Calculation')}")

    # set up the acquisition settings: ####################################
    # cam.setup_acquisition(mode="sequence", nframes=Settings.NUMBER_OF_IMAGES + 1)
    cam.setup_acquisition(mode="sequence", nframes=100)

    # set up the memmap:
    memmap_path = os.path.join(Paths.SPECTRA_FOLDER, timestamp + "_spectra_memmap.dat")
    shape = (Settings.NUMBER_OF_IMAGES, Settings.SPECTRA_SHAPE[0], Settings.SPECTRA_SHAPE[1])
    dtype = np.uint16
    mmap = np.memmap(memmap_path, dtype=dtype, mode="w+", shape=shape)
    # initialize the memmap to 0 and flush to disk
    mmap[:] = 0
    # flush to disk:
    mmap.flush()

    #############################################################################
    # start the acquisition: ####################################################
    #############################################################################
    # start the timer:
    start_time = time.time()

    print("\n")
    print("Starting acquisition...")
    print("\n")
    cam.start_acquisition()

    image_count = 0
    image_acquired = True
    first_image = True
    while image_count < Settings.NUMBER_OF_IMAGES:
        # get the oldest image:
        data_full = cam.read_oldest_image()

        if data_full is None:
            if image_acquired:
                print("Waiting for image to be available...")
            # Wait for the image to be available:
            time.sleep(round((Settings.EXP_TIME_MS / 1000) / 4, 3))
            image_acquired = False
            continue

        if first_image:
            start_time = time.time()
            first_image = False
            # print_frames_status(cam)

        # add the image to the memmap:
        mmap[image_count] = data_full

        # flush the memmap every 100 images:
        if image_count % 100 == 0 or image_count == Settings.NUMBER_OF_IMAGES - 1:
            mmap.flush()
            print(f"Flushed memmap at image {image_count} ----------------------------------------------")

        image_count += 1
        image_acquired = True

        # stop the timer:
        end_time = time.time()
        print(rf"{image_count} images acquired in {end_time - start_time:.3f} seconds.")
        print(rf"Frequency: {(image_count - 1)/ (end_time - start_time):.3f} Hz")

    #######################################################################
    # Some final steps ####################################################
    #######################################################################

    # stop the acquisition:
    if cam.acquisition_in_progress():
        cam.stop_acquisition()
        print("Acquisition stopped.")

    # clear the acquisition settings:
    cam.clear_acquisition()

    # Close the camera ####################################################
    cam.close()
    print("\nCamera connection closed.")


# Run main function #############################################################
if __name__ == "__main__":
    main()
