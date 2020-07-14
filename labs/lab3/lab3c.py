"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3C - Depth Camera Wall Parking
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
import pidcontroller

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()


class State(IntEnum):
    Wandering = 1
    Go = 2
    Stop = 3

MIN_coneDist = 30
TARGET_DIST = 30
DIST_THRESHHOLD = 2
PID_DIST_THRESHHOLD = 100

# Add any global variables here
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
counter = 0
currentState = State.Wandering
width = 0
height = 0
Speed_PID = pidcontroller.PID(0.2, 0.1, 0.5)
Angle_PID = pidcontroller.PID(1, 0.1, 0.5)

########################################################################################
# Functions
########################################################################################

def getWall(depth_image):
    global width
    global height
    """ return distance to wall, horizontal target"""
    y = depth_image[height // 3 - 10, height//3 + 10, width//3:2*width//3]


def start():
    global width
    global height
    width = rc.camera.get_width()
    height = rc.camera.get_height()
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Lab 3C - Depth Camera Wall Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # TODO: Park the car 20 cm away from the closest wall with the car directly facing
    # the wall
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
