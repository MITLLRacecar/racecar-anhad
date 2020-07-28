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
from enum import IntEnum

# import scipy.interpolate as interp
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser


sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
import IMU
import transformations as ts
import icp

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

VIS_RADIUS = 300
width = rc.camera.get_width()
height = rc.camera.get_height()

MAP_SIZE_PIXELS = 500
MAP_SIZE_METERS = 100
MIN_SAMPLES = 700


# set transformation constants
ts.VIS_RADIUS = VIS_RADIUS
ts.width = width
ts.num_samples = rc.lidar.get_num_samples()

imu = IMU.IMU(rc)
lidar = Laser(rc.lidar.get_num_samples(), 7, 360, 10000, 0, -15)

oldpoints = None

# Create an RMHC SLAM object with a laser model and optional robot model
slam = RMHC_SLAM(lidar, MAP_SIZE_PIXELS, MAP_SIZE_METERS)
# Initialize an empty trajectory
trajectory = []
# Initialize empty map
mapbytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)


class Color(IntEnum):
    Red = 0
    Blue = 1
    Green = 2
    Orange = 3
    Purple = 4
    White = 5
    Yellow = 6
    # Red1 = 6
    # Red2 = 7


HSV_RANGE = [None for i in range(len(Color))]
# Colors, stored as a pair (hsv_min, hsv_max)
HSV_RANGE[Color.Blue] = ((80, 120, 100), (125, 255, 255))
HSV_RANGE[Color.Green] = ((45, 120, 150), (75, 255, 255))
HSV_RANGE[Color.Red] = ((170, 84, 100), (180, 255, 255), (0, 84, 100), (10, 255, 255))
HSV_RANGE[Color.Purple] = ((125, 90, 100), (140, 255, 255))
HSV_RANGE[Color.Orange] = ((10, 95, 150), (25, 255, 255))

BGR = [None for i in range(len(Color))]
BGR[Color.Red] = (0, 0, 255)
BGR[Color.Blue] = (255, 127, 0)
BGR[Color.Green] = (0, 255, 0)
BGR[Color.Orange] = (0, 127, 255)
BGR[Color.Purple] = (255, 0, 127)
BGR[Color.White] = (255, 255, 255)
BGR[Color.Yellow] = (0, 255, 255)

########################################################################################
# Functions
########################################################################################


def manualControl():
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    return speed, angle


def color2TopDown(hsv_image, depth_image, incolor, crop=0):
    if len(incolor) == 4:  # red has 2 ranges
        mask = cv.bitwise_or(
            cv.inRange(hsv_image, incolor[0], incolor[1]),
            cv.inRange(hsv_image, incolor[2], incolor[3]),
        )[crop:, :]
    else:
        mask = cv.inRange(hsv_image, incolor[0], incolor[1])[crop:, :]
    # rc.display.show_color_image(mask)
    points = np.argwhere(mask != 0)  # ignores black
    points[:, 0] += crop
    depths = depth_image[points[:, 0], points[:, 1]]

    return ts.polar2TopDown(ts.camera2Polar(points, depths))


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()
    rc.drive.set_max_speed(0.05)

    global width
    global height
    width = rc.camera.get_width()
    height = rc.camera.get_height()

    global oldpoints
    oldpoints = None

    # Print start message
    print(">> SLAM")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # get orientation
    imu.update()

    # construct top down view
    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()[::8, ::8]  # subsample
    depth_image[depth_image == 0] = rc.camera.get_max_range()
    # a = np.copy(depth_image)
    depth_image = cv.resize(
        depth_image, (width, height), interpolation=cv.INTER_LINEAR_EXACT
    )
    # x = np.arange(width, step=8)
    # y = np.arange(height, step=8)
    # spline = interp.RectBivariateSpline(y, x, depth_image)
    # depth_image = spline(np.arange(height), np.arange(width), grid=True)

    # depth_image[: height // 8, : width // 8] = a

    # cv.bilateralFilter(, 9, 75, 75)

    # vis_image = np.zeros((2 * VIS_RADIUS, 2 * VIS_RADIUS, 3), np.uint8, "C")
    # hsv_image = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)

    # icp_points = np.array([[], []])
    # draw_points = np.array([[], [], []])  # x, y, color

    # for c in Color:
    #    if HSV_RANGE[c] is not None:
    #        p = color2TopDown(hsv_image, depth_image, HSV_RANGE[c], height // 2)
    # print(p.shape, c)
    #        icp_points = np.hstack((icp_points, p))
    #        draw_points = np.hstack((draw_points, [p[0], p[1], np.full(p.shape[1], c)]))
    # i = ts.topDown2Vis(p)
    # if i is not None:
    #    vis_image[i] = BGR[c]

    # scan_xy = ts.lidar2Polar(rc.lidar.get_samples())  # , 30, 330)
    scan_xy = np.array(
        [rc.lidar.get_samples(), np.arange(360, step=360 / rc.lidar.get_num_samples())]
    )
    # print(scan_xy)
    if scan_xy.size > 0:
        #    scan_xy[1] -= 15  # lidar offset
        n = scan_xy.shape[1]
    #    draw_points = np.hstack(
    #        (draw_points, [scan_xy[0], scan_xy[1], np.full(n, Color.White)])
    #    )
    # icp_points = np.hstack((icp_points, scan_xy))
    # print(scan_xy.shape)
    # i = ts.topDown2Vis(scan_xy)
    # if i is not None:
    #    vis_image[i] = BGR[Color.White]

    # n = NUM_ICP_POINTS - n
    # l = icp_points.shape[1]
    # if n < 0:  # more lidar points than needed
    #    icp_points = scan_xy[:, :NUM_ICP_POINTS]
    # elif l < n:
    #    icp_points = np.hstack((icp_points, scan_xy))
    #    l = icp_points.shape[1]  # we have less points that we want
    # else:
    #    icp_points = np.hstack(
    #        (icp_points[:, np.linspace(0, l - 1, n, dtype=int)], scan_xy)
    #    )
    #    l = NUM_ICP_POINTS  # we have exactly as many points as we want

    global oldpoints
    if n > MIN_SAMPLES:
        # print(scan_xy)
        slam.update(list(scan_xy[0]), scan_angles_degrees=list(scan_xy[1]))
        oldpoints = np.copy(scan_xy)
    elif oldpoints is not None:
        slam.update(list(oldpoints[0]), scan_angles_degrees=list(oldpoints[1]))

    # Get current robot position
    x, y, theta = slam.getpos()

    # Get current map bytes as grayscale
    slam.getmap(mapbytes)

    speed, angle = manualControl()

    slam_map = np.reshape(np.array(mapbytes), (MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))
    rc.display.show_color_image(slam_map)
    # rc.display.show_color_image(vis_image)
    # rc.display.show_depth_image(depth_image)

    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
