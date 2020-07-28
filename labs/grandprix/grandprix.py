"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020
​
Final Challenge - Time Trials
"""
​
########################################################################################
# Imports
########################################################################################
​
import sys
import cv2 as cv
import numpy as np
​
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum
import pidcontroller
import transformations as ts
​
########################################################################################
# Global variables
########################################################################################
​
rc = racecar_core.create_racecar()

Angle_PID = pidcontroller.PID(4, 0.5, 0.5)  # d = 0.5
width = 0
height = 0
old_challenge = None
cur_challenge = None

# Add any global variables here
speed = 0.0
angle = 0.0
counter = 0
RED = ((160, 50, 50), (10, 255, 255))
GREEN = ((45, 120, 100), (75, 255, 255))
#RED1 = ((170, 84, 100), (180, 255, 255))
#RED2 = ((0, 84, 100), (10, 255, 255))
PURPLE = ((125, 90, 100), (140, 255, 255))
ORANGE = ((10, 95, 100), (25, 255, 255))

MIN_CONTOUR_AREA = 700
TURN_THRESHOLD = 70
​
########################################################################################
# Functions
########################################################################################

class Challenge(enum.IntEnum):
    Line = 1
    Lane = 2
    Cones = 3
    # Slalom = 3
    # Gate = 4
    Wall = 4
    ManualControl = 5

class ConeState(IntEnum):
    DRIVE = 0
    RED = 1
    BLUE = 2
cur_conestate = ConeState.DRIVE

def manualControl():
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    return speed, angle

def smooth(scan):
    temp = np.copy(scan)

    for i in range(rc.lidar.get_num_samples()):
        if temp[i] == 0 or temp[i] > MAX_LIDAR_RANGE:
            temp[i] = temp[i - 1]

    for i in range(-8, 8):
        if i != 0:
            temp += np.roll(scan, i)

    return temp / 17

def wallData(scan, startAngle, endAngle):
    # lidar_average_distance =rc_utils.get_lidar_average_distance(scan)
    scan_xy = ts.polar2TopDown(ts.lidar2Polar(scan, startAngle, endAngle))

    #scan_r[abs(scan_r - np.mean(scan_r)) > 2 * np.std(scan_r)] = 0

    #scan_xy = np.array(Vp2c(scan_r, scan_t))

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
​
def getContourDist(contour, depth_image):
    mask = np.zeros_like(depth_image, np.uint8)
    mask = cv.drawContours(mask, [contour], 0, (255), -1)
    d = np.copy(depth_image)
    d[mask == 0] = 0
    return np.mean(np.ma.masked_equal(d, 0))


def getCone(color_image, depth_image):
    """Return cone center, distance, color"""
    color = False  # False = blue, True = Red
    if color_image is None or depth_image is None:
        contour_center = None
        dist = 0
    else:
        # Find all of the orange contours
        contours_Blue = rc_utils.find_contours(color_image, BLUE[0], BLUE[1])
        contours_Red = rc_utils.find_contours(color_image, RED[0], RED[1])

        min_dist = 0
        contour = None

        for c in contours_Red:
            if cv.contourArea(c) > MIN_CONTOUR:
                dist = getContourDist(c, depth_image)
                if dist < min_dist or min_dist == 0:
                    min_dist = dist
                    contour = c
                    color = True

        for c in contours_Blue:
            if cv.contourArea(c) > MIN_CONTOUR:
                dist = getContourDist(c, depth_image)
                if dist < min_dist or min_dist == 0:
                    min_dist = dist
                    contour = c
                    color = False

        dist = min_dist
        if contour is not None:
            # Calculate contour information
            # contour_center = rc_utils.get_contour_center(contour)

            # points = np.argwhere(mask > 0)
            # print(points)
            retval, triangle = cv.minEnclosingTriangle(contour)

            i = np.argmin(triangle[:, 0, 1])

            contour_center = np.copy(triangle[i, 0])
            triangle = triangle.flatten()

            # Draw contour onto the image
            # rc_utils.draw_contour(color_image, contour)
            # rc_utils.draw_circle(color_image, contour_center)

            # draw triangle

            """cv.line(
                color_image,
                (triangle[0], triangle[1]),
                (triangle[2], triangle[3]),
                (255, 0, 255),
                2,
            )
            cv.line(
                color_image,
                (triangle[0], triangle[1]),
                (triangle[4], triangle[5]),
                (0, 255, 0),
                2,
            )
            cv.line(
                color_image,
                (triangle[2], triangle[3]),
                (triangle[4], triangle[5]),
                (0, 255, 255),
                2,
            )"""

        else:
            contour_center = None
            dist = 0

        # Display the image to the screen
        # rc.display.show_color_image(color_image)
        # rc.display.show_depth_image(depth_image)

    return contour_center, dist, color

def start():
    global cur_challenge
    global cur_conestate
    global old_challenge
    cur_challenge = Challenge.ManualControl
    cur_conestate = ConeState.DRIVE
    old_challenge = Challenge.Line
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    global width
    width = rc.camera.get_width()
    height = rc.camera.get_height()
​
    # Print start message
    print(">> Grand Prix")
​
​
def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
​       
    scan = np.clip(
        rc.lidar.get_samples() * LIDAR_OFFSET, 0, None
    )  # smooth(rc.lidar.get_samples())

    scan_xy = None

    color_image = rc.camera.get_color_image()
    #depth_image = cv.bilateralFilter(rc.camera.get_depth_image(), 9, 75, 75)
    depth_image = rc.camera.get_depth_image()[::8, ::8]  # subsample for sim
    depth_image[depth_image == 0] = rc.camera.get_max_range()
    depth_image = cv.resize(
        depth_image, (width, height), interpolation=cv.INTER_LINEAR_EXACT
    )
    vis_image = np.zeros((2 * VIS_RADIUS, 2 * VIS_RADIUS, 3), np.uint8, "C")
    hsv_image = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)

    # FSM

    speed = 0
    angle = 0
    global currentChallenge
    global oldState
    global colorPriority

    if cur_challenge == Challenge.ManualControl:
        speed, angle = manualControl()
        if rc.controller.was_pressed(rc.controller.Button.A):
            cur_challenge = old_challenge
    else:
        if rc.controller.was_pressed(rc.controller.Button.A):
            old_challenge = cur_challenge
            cur_challenge = Challenge.ManualControl
            
    if cur_challenge == Challenge.Line:
        # Determine largest contour
        red_contours = rc_utils.find_contours(color_image, RED[0], RED[1])
        blue_contours = rc_utils.find_contours(color_image, BLUE[0], BLUE[1])
        green_contours = rc_utils.find_contours(color_image, GREEN[0], GREEN[1])

        color = red_contours
        if red_contours is None and blue_contours is None:
            color = green_contours
        elif red_contours is None and green_contours is None:
            color = blue_contours

        largest = rc_utils.get_largest_contour(color)

        contour_center = rc_utils.get_contour_center(largest)

        angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width(), -1, 1)
        
    elif cur_challenge == Challenge.Lane:
        pass

    elif cur_challenge == Challenge.Cones:
        contour_center, contour_center_distance, color = getCone(color_image, depth_image) #color: False = Blue, True = Red
        #failed to recognize cone: contour_center = None, contour_center_distance = None, color = False/True

        speed = 1

        # States
        if cur_conestate == ConeState.DRIVE:
            if contour_center is not None:
                angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width(), -1, 1)
            if 0 < contour_center_distance < TURN_THRESHOLD:
                if color:
                    cur_conestate = ConeState.RED
                    counter = 0
                else:
                    cur_conestate = ConeState.BLUE
                    counter = 0
        elif cur_conestate == ConeState.RED:
            if counter < 0.45:
                angle = 1
            elif contour_center is not None and not color: #color == Blue
                cur_conestate = ConeState.DRIVE
            else:
                angle = -1
            counter += rc.get_delta_time()
        else:
            if counter < 0.45:
                angle = -1
            elif contour_center is not None and color: #color == Red
                cur_conestate = ConeState.DRIVE
            else:
                angle = 1
            counter += rc.get_delta_time()

    elif cur_challenge == Challenge.Wall:
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

    
    rc.drive.set_speed_angle(speed, angle)
​
​def findAr(color_image, min_area):
    corners, ids = rc_utils.get_ar_markers(color_image)
    p1 = (corners[0][0][0], corners[0][0][1])
    p2 = (corners[0][1][0], corners[0][1][1])
    p3 = (corners[0][2][0], corners[0][2][1])
    p4 = (corners[0][3][0], corners[0][3][1])
    area = ((p2[0] - p1[0]) * (p3[1] - p3[1]))
    if(area > min_area):
        return corners, ids
    return 0, 0


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################
​
if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()