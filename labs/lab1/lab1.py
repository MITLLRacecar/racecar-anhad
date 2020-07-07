"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 1 - Driving in Shapes
"""

########################################################################################
# Imports
########################################################################################

import sys

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Put any global variables here
counter = 0
instruction = []

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    # If we use a global variable in our function, we must list it at
    # the beginning of our function like this
    global counter

    # The start function is a great place to give initial values to global variables
    counter = 0

    # Begin at a full stop
    rc.drive.stop()

    # Print start message
    # TODO (main challenge): add a line explaining what the Y button does
    print(
        ">> Lab 1 - Driving in Shapes\n"
        "\n"
        "Controls:\n"
        "   Right trigger = accelerate forward\n"
        "   Left trigger = accelerate backward\n"
        "   Left joystick = turn front wheels\n"
        "   A button = drive in a circle\n"
        "   B button = drive in a square\n"
        "   X button = drive in a figure eight\n"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global counter, instruction

    # TODO (warmup): Implement acceleration and steering
    if rc.controller.is_down(rc.controller.Button.RB):
        s = rc.controller.get_trigger(
            rc.controller.Trigger.RIGHT
        ) - rc.controller.get_trigger(rc.controller.Trigger.LEFT)
        a = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
        rc.drive.set_speed_angle(s, a)
    else:
        rc.drive.stop()

    if rc.controller.was_pressed(rc.controller.Button.A):
        print("Driving in a circle...")
        # (main challenge): Drive in a circle
        instruction += [(1, 0.5, 10)]

    # (main challenge): Drive in a square when the B button is pressed
    if rc.controller.was_pressed(rc.controller.Button.B):
        print("Driving in a square...")
        instruction += [
            (1, 0, 1.5),
            (1, 1, 1.2),
            (0, 1, 0.4),
            (1, 0, 1.5),
            (1, 1, 1.2),
            (0, 1, 0.4),
            (1, 0, 1.5),
            (1, 1, 1.2),
            (0, 1, 0.4),
            (1, 0, 1.5),
            (1, 1, 1.2),
            (0, 1, 0.4),
        ]

    # (main challenge): Drive in a figure eight when the X button is pressed
    if rc.controller.was_pressed(rc.controller.Button.X):
        print("Driving in a figure 8...")
        instruction += [(1, 1, 6), (0, 1, 0.7), (1, -1, 5.8), (0, -1, 0.5)]

    # (main challenge): Drive in a shape of your choice when the Y button is pressed
    if rc.controller.was_pressed(rc.controller.Button.Y):
        print("Driving in a spiral...")
        instruction += [
            (1, 1, 1),
            (1, 0.95, 1),
            (1, 0.9, 1),
            (1, 0.85, 1),
            (1, 0.8, 1),
            (1, 0.75, 1),
            (1, 0.7, 1),
            (1, 0.65, 1),
            (1, 0.6, 1),
            (1, 0.55, 1),
        ]

    if len(instruction) > 0:
        if counter < instruction[0][2]:
            rc.drive.set_speed_angle(instruction[0][0], instruction[0][1])
        else:
            counter -= instruction[0][2]
            del instruction[0]
            if len(instruction) == 0:
                counter = 0
                rc.drive.stop()

    # Increases counter by the number of seconds elapsed in the previous frame
    if len(instruction) > 0:
        counter += rc.get_delta_time()


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
