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
from enum import IntEnum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
import pidcontroller

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()


class State(IntEnum):
    Go = 1
    Stop = 2
    moveBackwards = 3


TARGET_DIST = 20
DIST_THRESHHOLD = 1
PID_DIST_THRESHHOLD = 60
SLOPE_THRESHHOLD = 0.005

# Add any global variables here
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
counter = 0
currentState = State.Go
width = 0
height = 0
Speed_PID = pidcontroller.PID(0.01, 0.1, 0.1)
Angle_PID = pidcontroller.PID(10, 1, 0.5)

########################################################################################
# Functions
########################################################################################


def getWall(depth_image):
    global width
    global height
    """ return distance to wall, horizontal target"""
    y = np.mean(
        depth_image[height // 3 - 30 : height // 3 + 30, width // 3 : 2 * width // 3],
        axis=0,
    )
    x = np.arange(width // 3, 2 * width // 3)
    z = np.poly1d(np.polyfit(x, y, 1))

    dist = z(width // 2)
    slope = z[1]

    return dist, slope


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

    depth_image = rc.camera.get_depth_image()

    dist, slope = getWall(depth_image)

    # Sprint(dist, " ", slope)

    global currentState
    global counter
    global speed
    global angle

    # FSM
    if currentState == State.Go:  # cone is in sight
        park = True
        if abs(dist - TARGET_DIST) > DIST_THRESHHOLD:  # control speed only
            park = False
            if dist > PID_DIST_THRESHHOLD:
                speed = 0.5
            else:
                speedError = dist - TARGET_DIST
                speed = np.clip(
                    Speed_PID.update(speedError, rc.get_delta_time()), -1, 1
                )
        else:
            speed = 0
        if abs(slope) > SLOPE_THRESHHOLD:  # control angle
            park = False
            if speed == 0 or (
                dist < 60 and abs(slope) > 0.1
            ):  # good distance but angle is bad
                counter = 0
                currentState = State.moveBackwards
            else:
                angleError = -slope
                angle = np.clip(
                    Angle_PID.update(angleError, rc.get_delta_time()), -1, 1
                )
        else:
            angle = 0
        if park:
            currentState = State.Stop
    if currentState == State.Stop:
        speed = 0
        angle = 0
        if (
            abs(slope) > SLOPE_THRESHHOLD or abs(dist - TARGET_DIST) > DIST_THRESHHOLD
        ):  # cone moved
            currentState = State.Go
    if currentState == State.moveBackwards:  # move backwards for 2 seconds
        if counter < 1:
            speed = -1
            angle = 0
        elif counter < 2:
            speed = 0
            angle = 0
        else:
            currentState = State.Go

    # print(dist, " ", slope, " ", currentState)

    counter += rc.get_delta_time()
    rc.drive.set_speed_angle(speed, angle)


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
