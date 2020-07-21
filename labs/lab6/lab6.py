"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 6 - Sensor Fusion
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

speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
counter = 0
width = 0
height = 0
IMUVelocity = 0
previousDepthImage = None
DIFF_THRESHHOLD = 5
old_scan_xy = 0

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    global width
    global height
    global IMUVelocity
    global previousDepthImage
    global old_scan_xy
    width = rc.camera.get_width()
    height = rc.camera.get_height()
    IMUVelocity = 0
    previousDepthImage = rc.camera.get_depth_image()
    old_scan_xy = get_scan_xy()

    # Print start message
    print(">> Lab 6 - LIDAR Safety Stop")


def p2c(r, t):
    """polar r (float), theta (float) to cartesian x (float), y (float)"""
    if np.isnan(r) or np.isnan(t):
        return 0, 0
    else:
        return r * np.cos(t), r * np.sin(t)  # x, y


def c2p(x, y):
    """cartesian x (float), y (float) to polar r (float), theta (float)"""
    if np.isnan(x) or np.isnan(y):
        return 0, 0
    else:
        r = np.sqrt(x ** 2 + y ** 2)
        t = np.arctan2(y, x)
        return r, t


Vp2c = np.vectorize(p2c, [np.float32, np.float32])
Vc2p = np.vectorize(c2p, [np.float32, np.float32])


def get_scan_xy():
    scan_r = rc.lidar.get_samples()
    scan_t = np.radians(np.arange(360, step=360.0 / rc.lidar.get_num_samples()))

    return np.array(Vp2c(scan_r, scan_t))


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

    # TODO: Estimate the car's speed with at least 3 unique methods
    global IMUVelocity
    IMUVelocity += rc.physics.get_linear_acceleration()[2] * rc.get_delta_time()

    global previousDepthImage
    depth_image = rc.camera.get_depth_image()
    diff = depth_image - previousDepthImage
    diff[np.absolute(diff) < DIFF_THRESHHOLD] = 0

    DepthVelocity = np.mean(diff) * rc.get_delta_time()
    previousDepthImage = depth_image

    scan_xy = get_scan_xy
    # TODO: get velocity from lidar data
    old_scan_xy = scan_xy

    # TODO: Fuse these sources into a single velocity measurement

    # TODO: Prevent the car from traveling over 0.5 m/s

    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
