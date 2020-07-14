"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3A - Depth Camera Safety Stop
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
import filterOnePole

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here
MIN_DISTANCE = 20
empty_image = 0
OBJECT_THRESHHOLD = 50
CUTOFF_FREQUENCY = 20
LowpassFilter = None
test = 0
########################################################################################
# Functions
########################################################################################


def distEncoder(a, maxv):
    """nonlinear float to uint8"""
    return 255.0 * (1.0 - (1.0 - a / maxv) ** 3.0)


vDistEncoder = np.vectorize(distEncoder, [np.uint8], excluded=["maxv"])


def start():
    """
    This function is run once every time the start button is pressed
    """
    global empty_image
    empty_image = cv.imread("Empty.png", cv.IMREAD_GRAYSCALE)

    # print(empty_image[:, 40])
    # a = rc.camera.get_depth_image()[::8, ::8]
    # maxv = np.max(a)
    # a = vDistEncoder(a, maxv)
    # cv.imwrite("a.png", a)
    # print(a[::8, 40])

    global LowpassFilter
    LowpassFilter = filterOnePole.Filter(
        filterOnePole.Type.LOWPASS, CUTOFF_FREQUENCY, 0, True
    )

    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(
        ">> Lab 3A - Depth Camera Safety Stop\n"
        "\n"
        "Controls:\n"
        "   Right trigger = accelerate forward\n"
        "   Right bumper = override safety stop\n"
        "   Left trigger = accelerate backward\n"
        "   Left joystick = turn front wheels\n"
        "   A button = print current speed and angle\n"
        "   B button = print the distance at the center of the depth image"
    )


def floorRemoval(depth_image):
    height, width = np.shape(depth_image)
    maxv = np.max(depth_image)
    if maxv == 0:
        maxv = 800
    scaled_image = vDistEncoder(depth_image, maxv)

    shift_x = 0
    shift_y = 0
    shift_x = 20
    mostzeros = 0
    brightness_adjusted = np.zeros_like(empty_image)

    for y in range(5, 15):
        for z in range(-5, 5):
            brightness_adjusted[41:] = np.clip(empty_image[41:] + z, 0, 255)
            object_depth = np.copy(scaled_image)
            object_depth[
                abs(
                    object_depth
                    - brightness_adjusted[y : height + y, shift_x : width + shift_x]
                )
                < OBJECT_THRESHHOLD
            ] = 0
            candidate = np.count_nonzero(object_depth[height // 2 :] == 0)
            if candidate > mostzeros:
                mostzeros = candidate
                shift_y = y
                shift_z = z

    # shift_y = 10
    # shift_z = 0

    brightness_adjusted[41:] = np.clip(empty_image[41:] + shift_z, 0, 255)
    mostzeros = -1

    for x in range(0, 40):
        object_depth = np.copy(scaled_image)
        object_depth[
            abs(object_depth - empty_image[shift_y : height + shift_y, x : width + x])
            < OBJECT_THRESHHOLD
        ] = 0
        candidate = np.count_nonzero(object_depth[height // 2 :] == 0)
        if candidate > mostzeros:
            mostzeros = candidate
            out = np.copy(object_depth)
            shift_x = x

    global test

    if rc.controller.was_pressed(rc.controller.Button.A):
        a = rc.camera.get_depth_image()[::8, ::8]
        maxv = np.max(a)
        a = vDistEncoder(a, maxv)
        cv.imwrite(str(test) + ".png", a)
        test += 1
        # for i in range(0, height):
        #    print(out[i, width // 2], end=" ")

    lowpass = LowpassFilter.input(out, rc.get_delta_time())
    lowpass[lowpass < 1] = 0
    lowpass[out == 0] = 0

    print(shift_x, " ", shift_y, " ", shift_z)

    return lowpass

    # lowpass = np.zeros_like(out)
    # lowpass = lowpass + 5
    # lowpass[out == 0] = 0
    # lowpass[out == 0] = 0

    # return lowpass

    """ for i in range(0, 15):
        object_depth = np.copy(depth_image)
        if i > 0:
            empty_image_shift[i:] = empty_image[0:-i]
        object_depth[abs(object_depth - empty_image_shift) < OBJECT_THRESHHOLD] = 0
        candidate = np.sum(object_depth)
        # cv.imshow(str(i), object_depth)
        if i == 0 or candidate < minsum:
            minsum = candidate
            out = np.copy(object_depth)
    return out """


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Calculate the distance of the object directly in front of the car
    depth_image = rc.camera.get_depth_image()[::8, ::8]
    depth_image = floorRemoval(depth_image)
    # TODO: REMOVE SUBSAMPLING ON IRL RACECAR
    center_distance = rc_utils.get_depth_image_center_distance(depth_image)

    # TODO (warmup): Prevent forward movement if the car is about to hit something.
    if center_distance < MIN_DISTANCE:
        if speed > 0 and not rc.controller.is_down(rc.controller.Button.RB):
            speed = 0
    # Allow the user to override safety stop by holding the right bumper.

    # Use the left joystick to control the angle of the front wheels
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the depth image center distance when the B button is held down
    if rc.controller.is_down(rc.controller.Button.B):
        print("Center distance:", center_distance)

    # Display the current depth image
    rc.display.show_depth_image(depth_image)

    # TODO (stretch goal): Prevent forward movement if the car is about to drive off a
    # ledge.  ONLY TEST THIS IN THE SIMULATION, DO NOT TEST THIS WITH A REAL CAR.

    # TODO (stretch goal): Tune safety stop so that the car is still able to drive up
    # and down gentle ramps.
    # Hint: You may need to check distance at multiple points.


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
