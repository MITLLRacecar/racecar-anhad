"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 2B - Color Image Cone Parking
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
from enum import IntEnum

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils
import pidcontroller

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 30
CENTER_THRESHHOLD = 20
AREA_THRESHHOLD = 2000
TARGET_AREA = 26000


class State(IntEnum):
    Wandering = 1
    Go_To_Cone = 2
    Stop = 3
    Move_Backwards = 4


# The HSV range for the color orange, stored as (hsv_min, hsv_max)
ORANGE = ((10, 100, 100), (20, 255, 255))

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
counter = 0
currentState = State.Wandering
PID = pidcontroller.PID(0.8, 0.1, 0.5)

########################################################################################
# Functions
########################################################################################


def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the orange contours
        contours = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)

        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(image)


def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle
    global currentState

    # Initialize variables
    speed = 0
    angle = 0
    counter = 0
    currentState = State.Wandering

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message
    print(">> Lab 2B - Color Image Cone Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global currentState
    global counter

    # Search for contours in the current color image
    update_contour()

    # TODO: Park the car 30 cm away from the closest orange cone

    # FSM
    if currentState == State.Wandering:
        speed = 1
        angle = 1 - counter / 10
        if angle < 0.4:
            angle = 0.4
        if contour_center is not None:
            if contour_area > TARGET_AREA:
                counter = 0
                currentState = State.Move_Backwards
            else:
                currentState = State.Go_To_Cone
    if currentState == State.Go_To_Cone:  # cone is in sight
        if contour_center is None:  # cone dissapeared
            counter = 0
            currentState = State.Wandering
        elif abs(contour_area - TARGET_AREA) < AREA_THRESHHOLD:  # cone 30cm away
            currentState = State.Stop
        elif contour_area > TARGET_AREA:  # cone too close
            counter = 0
            currentState = State.Move_Backwards
        else:
            error = (contour_center[1] - rc.camera.get_width() // 2) / (
                rc.camera.get_width() // 2
            )
            angle = np.clip(Angle_PID.update(error, rc.get_delta_time()), -1, 1)
            speed = 0.5
    if currentState == State.Stop:
        speed = 0
        angle = 0
        if contour_center is None:  # cone dissapeared
            currentState = State.Wandering
        elif abs(contour_area - TARGET_AREA) > AREA_THRESHHOLD:  # cone moved
            if contour_area > TARGET_AREA:
                counter = 0
                currentState = State.Move_Backwards
            else:
                currentState = State.Go_To_Cone
    if currentState == State.Move_Backwards:  # move backwards
        if contour_center is None:
            counter = 0
            currentState = State.Wandering
        else:
            if abs(contour_area - TARGET_AREA) < AREA_THRESHHOLD:
                currentState = State.Stop
            elif contour_area > TARGET_AREA:
                speed = -0.5
                angle = 0
            else:
                currentState = State.Go_To_Cone

    counter += rc.get_delta_time()
    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)


def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # Print a line of ascii text denoting the contour area and x position
    if rc.camera.get_color_image() is None:
        # If no image is found, print all X's and don't display an image
        print("X" * 10 + " (No image) " + "X" * 10)
    else:
        # If an image is found but no contour is found, print all dashes
        if contour_center is None:
            print("-" * 32 + " : area = " + str(contour_area), " State:", currentState)

        # Otherwise, print a line of dashes with a | indicating the contour x-position
        else:
            s = ["-"] * 32
            s[int(contour_center[1] / 20)] = "|"
            print(
                "".join(s) + " : area = " + str(contour_area), " State:", currentState
            )


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
