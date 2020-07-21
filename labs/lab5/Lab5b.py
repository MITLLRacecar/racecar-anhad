"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 5B - LIDAR Wall Following
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

TARGET_ANGLE = 90
TARGET_DIST = 30
ANGLE_THRESHHOLD = 5

# Add any global variables here
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
width = 0
height = 0
Angle_PID = pidcontroller.PID(10, 1, 0.5)

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
    width = rc.camera.get_width()
    height = rc.camera.get_height()

    # Print start message
    print(">> Lab 5B - LIDAR Wall Following")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    scan = rc.lidar.get_samples()

    wallDist, wallAngle = wallData(scan, 70, 110)

    speed = 0.5

    angleError = TARGET_ANGLE - wallAngle
    distError = TARGET_DIST - wallDist
    print(
        "wall Angle",
        wallAngle,
        "angle Error",
        angleError,
        "wallDist",
        wallDist,
        "distError",
        distError,
    )

    if abs(angleError) < ANGLE_THRESHHOLD:
        error = distError
    else:
        error = np.sin(angleError)

    angle = rc_utils.clamp(Angle_PID.update(error, rc.get_delta_time()), -1, 1)

    rc.drive.set_speed_angle(speed, angle)

    # TODO: Follow the wall to the right of the car without hitting anything.


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


def wallData(scan, startAngle, endAngle):
    # lidar_average_distance =rc_utils.get_lidar_average_distance(scan)
    # scan_xy = polarToCartesian(scan, startAngle, endAngle)

    scan_r = scan[startAngle * 2 : endAngle * 2]
    scan_t = np.radians(
        np.arange(startAngle, endAngle, step=360.0 / rc.lidar.get_num_samples())
    )

    scan_xy = np.array(Vp2c(scan_r, scan_t))

    x_points = scan_xy[0, :]
    y_points = scan_xy[1, :]

    scan_polynomial = np.poly1d(np.polyfit(x_points, y_points, 1))

    # data vis:

    l = Vc2p(np.arange(-200, 200), scan_polynomial(np.arange(-200, 200)))
    # l = Vc2p(x_points, y_points)

    l = np.transpose([np.degrees(l[1]), l[0]])
    # print(l)
    s = np.zeros_like(scan)
    s[startAngle * 2 : endAngle * 2] = scan[startAngle * 2 : endAngle * 2]
    rc.display.show_lidar(s, radius=200, max_range=400, highlighted_samples=l)

    # y = mx + b
    # poly1d: [b, m]
    distance = scan_polynomial.r[0]
    angle = np.degrees(
        (np.pi / 2) - np.arctan(scan_polynomial[1])
    )  # arctan(slope = m) = angle
    return (distance, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
