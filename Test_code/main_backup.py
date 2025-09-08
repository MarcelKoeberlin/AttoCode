from pylablib.devices import PrincetonInstruments
import os
from datetime import datetime

# PATHS: ##############################################################
class Paths:
    BACKUP_ALL_ATTRIBUTES_AT_START = r"C:\Users\Moritz\Desktop\Pixis_data\Attribute_backups"

# PROGRAM SETTINGS: ####################################################
class Settings:
    EXP_TIME_MS = 10 # Set exposure time


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
    cam = PrincetonInstruments.PicamCamera()

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
    print(str(cam.get_all_attribute_values()).removeprefix("Dictionary(").removesuffix(")"))  # prints all camera attributes and their values
    print("\n")

    #########################################################################
    # IMAGE ACQUISITION ####################################################
    #########################################################################

    # Acquire single full frame:

    # Acquire a binned image:

    # Acquire an image with a certain ROI:



    cam.close()


# Run main function #############################################################
if __name__ == "__main__":
    main()
