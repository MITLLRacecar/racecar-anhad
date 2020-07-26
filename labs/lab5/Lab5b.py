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
MAX_LIDAR_RANGE = 60

# Add any global variables here
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
width = 0
height = 0
Angle_PID = pidcontroller.PID(4, 0.5, 0.5)  # d = 0.5

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

    x = 0
    while x < 2:
        x += rc.get_delta_time()


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    scan = rc.lidar.get_samples()  # smooth(rc.lidar.get_samples())

    rightWallDist, rightWallAngle, RightL = wallData(scan, 50, 130)

    leftWallDist, leftWallAngle, LeftL = wallData(scan, 250, 310)

    s = np.zeros_like(scan)

    s[50 * 2 : 130 * 2] = scan[50 * 2 : 130 * 2]
    s[250 * 2 : 310 * 2] = scan[250 * 2 : 310 * 2]

    rc.display.show_lidar(
        s,
        radius=200,
        max_range=200,
        highlighted_samples=np.concatenate([RightL, LeftL]),
    )

    speed = 0.5

    angleError = TARGET_ANGLE - (rightWallAngle + leftWallAngle) / 2
    distError = (rightWallDist + leftWallDist) / 2

    # if abs(angleError) < ANGLE_THRESHHOLD:
    #    error = distError
    # else:
    error = distError / 10 + np.sin(np.radians(angleError)) * 2  # angleError / 30

    angle = rc_utils.clamp(Angle_PID.update(error, rc.get_delta_time()), -1, 1)

    print(
        "Distance Error",
        distError,
        "Angle Error",
        np.sin(np.radians(angleError)),
        "Angle",
        angle,
        # "Right wall Angle",
        # rightWallAngle,
        # "Right wall Dist",
        # rightWallDist,
        # "Left wall Angle",
        # rightWallAngle,
        # "Left wall Dist",
        # leftWallDist,
    )

    rc.drive.set_speed_angle(speed, angle)

    # TODO: Follow the wall to the right of the car without hitting anything.


def smooth(scan):
    temp = np.copy(scan)

    for i in range(rc.lidar.get_num_samples()):
        if temp[i] == 0 or temp[i] > MAX_LIDAR_RANGE:
            temp[i] = temp[i - 1]

    for i in range(-8, 8):
        if i != 0:
            temp += np.roll(scan, i)

    return temp / 17


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


def wallData(scan, startAngle, endAngle):
    # lidar_average_distance =rc_utils.get_lidar_average_distance(scan)
    # scan_xy = polarToCartesian(scan, startAngle, endAngle)

    scan_r = np.copy(scan[startAngle * 2 : endAngle * 2])
    scan_t = np.radians(
        np.arange(startAngle, endAngle, step=360.0 / rc.lidar.get_num_samples())
    )

    scan_r[abs(scan_r - np.mean(scan_r)) > 2 * np.std(scan_r)] = 0

    scan_xy = np.array(Vp2c(scan_r, scan_t))

    valid_points = np.all(scan_xy != 0, axis=0)
    x_points = scan_xy[0, valid_points]
    y_points = scan_xy[1, valid_points]

    scan_polynomial = np.poly1d(np.polyfit(x_points, y_points, 1))

    # data vis:

    l = Vc2p(np.arange(-100, 100), scan_polynomial(np.arange(-100, 100)))
    # l = Vc2p(x_points, y_points)

    l = np.transpose([np.degrees(l[1]), l[0]])
    # print(l)
    # s = np.zeros_like(scan)
    # s[startAngle * 2 : endAngle * 2] = scan[startAngle * 2 : endAngle * 2]
    # rc.display.show_lidar(s, radius=200, max_range=400, highlighted_samples=l)

    # y = mx + b
    # poly1d: [b, m]
    # print(scan_polynomial)
    distance = scan_polynomial(0)
    # distance = rc_utils.get_lidar_average_distance(
    #   scan, (startAngle + (endAngle - startAngle) / 2)
    # )
    angle = np.degrees(
        (np.pi / 2) - np.arctan(scan_polynomial[1])
    )  # arctan(slope = m) = angle
    return (distance, angle, l)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
