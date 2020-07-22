"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Final Challenge - Time Trials
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

width = 0
height = 0

# Colors, stored as a pair (hsv_min, hsv_max)
BLUE = ((90, 100, 100), (125, 255, 255))
GREEN = ((70, 137, 100), (80, 255, 255))
RED = ((170, 84, 100), (11, 255, 255))
PURPLE = ((125, 90, 100), (140, 255, 255))
ORANGE = ((14, 95, 100), (10, 255, 255))

########################################################################################
# Functions
########################################################################################


def p2c(r, t):
    """polar r (float), theta (float) to cartesian x (float), y (float)"""
    if np.isnan(r) or np.isnan(t):
        return 0, 0  # return 0, 0
    else:
        x = r * np.cos(t)
        y = r * np.sin(t)
        return r * np.cos(t), r * np.sin(t)  # x, y


def c2p(x, y):
    """cartesian x (float), y (float) to polar r (float), theta (float)"""
    if np.isnan(x) or np.isnan(y):
        return 0, 0  # return 0, 0
    else:
        r = np.sqrt(x ** 2 + y ** 2)
        t = np.arctan2(y, x)
        return r, t


Vp2c = np.vectorize(p2c, [np.float32, np.float32])
Vc2p = np.vectorize(c2p, [np.float32, np.float32])


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    global width
    global height
    width = rc.camera.get_width()
    height = rc.camera.get_height()
    rc.drive.set_max_speed(1)

    # Print start message
    print(">> Final Challenge - Time Trials")


def manualcontrol():
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    rc.drive.set_speed_angle(speed, angle)


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Use the left joystick to control the angle of the front wheels
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
