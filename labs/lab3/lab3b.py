"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3B - Depth Camera Cone Parking
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
from enum import IntEnum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
import pidcontroller

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here


class State(IntEnum):
    Wandering = 1
    Go_To_Cone = 2
    Stop = 3


# The HSV range for the color orange, stored as (hsv_min, hsv_max)
ORANGE = ((10, 100, 100), (20, 255, 255))

MIN_coneDist = 30
TARGET_DIST = 30
DIST_THRESHHOLD = 2
PID_DIST_THRESHHOLD = 100

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
counter = 0
currentState = State.Wandering
Speed_PID = pidcontroller.PID(0.2, 0.1, 0.5)
Angle_PID = pidcontroller.PID(1, 0.1, 0.5)

########################################################################################
# Functions
########################################################################################


def getCone(color_image, depth_image):
    """Return cone center, distance"""
    if color_image is None or depth_image is None:
        contour_center = None
        dist = 0
    else:
        # Find all of the orange contours
        contours = rc_utils.find_contours(color_image, ORANGE[0], ORANGE[1])

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_coneDist)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)

            mask = np.zeros(color_image.shape[0:2], np.uint8)
            mask = cv.drawContours(mask, [contour], 0, (255), -1)
            depth_image[mask == 0] = 0
            dist = np.mean(np.ma.masked_equal(depth_image, 0))

            # Draw contour onto the image
            rc_utils.draw_contour(color_image, contour)
            rc_utils.draw_circle(color_image, contour_center)
        else:
            contour_center = None
            dist = 0

        # Display the image to the screen
        # rc.display.show_color_image(color_image)
        # rc.display.show_depth_image(depth_image)

    return contour_center, dist


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    global counter
    counter = 0
    global currentState
    currentState = State.Wandering

    # Print start message
    print(">> Lab 3B - Depth Camera Cone Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()

    contour_center, coneDist = getCone(color_image, depth_image)

    global currentState
    global counter
    global speed
    global angle

    # FSM
    if currentState == State.Wandering:
        speed = 1
        angle = 1 - counter / 10
        if angle > 0.5:
            angle = 0.5
        if contour_center is not None:  # found cone
            currentState = State.Go_To_Cone
    if currentState == State.Go_To_Cone:  # cone is in sight
        if contour_center is None:  # cone dissapeared
            counter = 0
            currentState = State.Wandering
        elif abs(coneDist - TARGET_DIST) < DIST_THRESHHOLD:  # cone 30cm away
            currentState = State.Stop
        else:
            angleError = (contour_center[1] - rc.camera.get_width() // 2) / (
                rc.camera.get_width() // 2
            )
            angle = np.clip(Angle_PID.update(angleError, rc.get_delta_time()), -1, 1)
            if coneDist > PID_DIST_THRESHHOLD:
                speed = 1
            else:
                speedError = coneDist - TARGET_DIST
                speed = np.clip(
                    Speed_PID.update(speedError, rc.get_delta_time()), -1, 1
                )
    if currentState == State.Stop:
        speed = 0
        angle = 0
        if contour_center is None:  # cone dissapeared
            counter = 0
            currentState = State.Wandering
        elif abs(coneDist - TARGET_DIST) > DIST_THRESHHOLD:  # cone moved
            currentState = State.Go_To_Cone

    counter += rc.get_delta_time()
    rc.drive.set_speed_angle(speed, angle)

    # TODO: Park the car 30 cm away from the closest orange cone.
    # Use both color and depth information to handle cones of multiple sizes.
    # You may wish to copy some of your code from lab2b.py


def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # Print a line of ascii text denoting the steering angle
    s = ["-"] * 33
    s[16 + int(angle * 16)] = "|"
    print(
        "".join(s) + " : speed = " + str(speed),
        " angle = " + str(angle),
        " State:",
        currentState,
    )


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
