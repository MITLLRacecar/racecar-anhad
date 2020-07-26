# Sensor fusion for the micropython board. 25th June 2015
# Ported to MicroPython by Peter Hinch.
# Released under the MIT License (MIT)
# Copyright (c) 2017, 2018 Peter Hinch

# V0.9 Time calculations devolved to deltat.py
# V0.8 Calibrate wait argument can be a function or an integer in ms.
# V0.7 Yaw replaced with heading
# V0.65 waitfunc now optional

# Supports 6 and 9 degrees of freedom sensors. Tested with InvenSense MPU-9150 9DOF sensor.
# Source https://github.com/xioTechnologies/Open-Source-AHRS-With-x-IMU.git
# also https://github.com/kriswiner/MPU-9250.git
# Ported to Python. Integrator timing adapted for pyboard.
# See README.md for documentation.

# Portability: the only assumption is presence of time.sleep() used in mag
# calibration.
from math import sqrt, atan2, asin, degrees, radians


class IMU:
    """
    Class provides sensor fusion allowing heading, pitch and roll to be extracted. This uses the Madgwick algorithm.
    The update method must be called peiodically. The calculations take 1.6mS on the Pyboard.
    """

    declination = 0  # Optional offset for true north. A +ve value adds to heading

    def __init__(self, racecar=None):  # , timediff=None):
        # self.magbias = (0, 0, 0)            # local magnetic bias factors: set from calibration
        # self.deltat = DeltaT(timediff)      # Time between updates
        self.q = [1.0, 0.0, 0.0, 0.0]  # vector to hold quaternion
        GyroMeasError = radians(
            40
        )  # Original code indicates this leads to a 2 sec response time
        self.beta = sqrt(3.0 / 4.0) * GyroMeasError  # compute beta (see README)
        self.pitch = 0
        self.heading = 0
        self.roll = 0
        self.rc = racecar

    # def calibrate(self, getxyz, stopfunc, wait=0):
    #    magmax = list(getxyz())             # Initialise max and min lists with current values
    #    magmin = magmax[:]
    #    while not stopfunc():
    #        if wait != 0:
    #            if callable(wait):
    #                wait()
    #            else:
    #                time.sleep(wait/1000)  # Portable
    #        magxyz = tuple(getxyz())
    #        for x in range(3):
    #            magmax[x] = max(magmax[x], magxyz[x])
    #            magmin[x] = min(magmin[x], magxyz[x])
    #    self.magbias = tuple(map(lambda a, b: (a +b)/2, magmin, magmax))

    def update(self, accel=None, gyro=None, deltat=None):
        # 3-tuples (x, y, z) for accel, gyro
        if deltat is None:
            if self.rc is None:
                raise ValueError(
                    "Updating IMU without providing values and without providing racecar object"
                )
            else:
                accel = self.rc.physics.get_linear_acceleration()
                gyro = self.rc.physics.get_angular_velocity()
                deltat = self.rc.get_delta_time()

        ax, ay, az = accel  # Units G (but later normalised)
        gx, gy, gz = gyro  # (radians(x) for x in gyro) #gyro is in rad/sec
        q1, q2, q3, q4 = (
            self.q[x] for x in range(4)
        )  # short name local variable for readability
        # Auxiliary variables to avoid repeated arithmetic
        _2q1 = 2 * q1
        _2q2 = 2 * q2
        _2q3 = 2 * q3
        _2q4 = 2 * q4
        _4q1 = 4 * q1
        _4q2 = 4 * q2
        _4q3 = 4 * q3
        _8q2 = 8 * q2
        _8q3 = 8 * q3
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3
        q4q4 = q4 * q4

        # Normalise accelerometer measurement
        norm = sqrt(ax * ax + ay * ay + az * az)
        if norm == 0:
            return  # handle NaN
        norm = 1 / norm  # use reciprocal for division
        ax *= norm
        ay *= norm
        az *= norm

        # Gradient decent algorithm corrective step
        s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
        s2 = (
            _4q2 * q4q4
            - _2q4 * ax
            + 4 * q1q1 * q2
            - _2q1 * ay
            - _4q2
            + _8q2 * q2q2
            + _8q2 * q3q3
            + _4q2 * az
        )
        s3 = (
            4 * q1q1 * q3
            + _2q1 * ax
            + _4q3 * q4q4
            - _2q4 * ay
            - _4q3
            + _8q3 * q2q2
            + _8q3 * q3q3
            + _4q3 * az
        )
        s4 = 4 * q2q2 * q4 - _2q2 * ax + 4 * q3q3 * q4 - _2q3 * ay
        norm = 1 / sqrt(
            s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4
        )  # normalise step magnitude
        s1 *= norm
        s2 *= norm
        s3 *= norm
        s4 *= norm

        # Compute rate of change of quaternion
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        # Integrate to yield quaternion
        q1 += qDot1 * deltat
        q2 += qDot2 * deltat
        q3 += qDot3 * deltat
        q4 += qDot4 * deltat
        norm = 1 / sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)  # normalise quaternion
        self.q = q1 * norm, q2 * norm, q3 * norm, q4 * norm
        self.heading = 0
        self.pitch = degrees(
            -asin(2.0 * (self.q[1] * self.q[3] - self.q[0] * self.q[2]))
        )
        self.roll = degrees(
            atan2(
                2.0 * (self.q[0] * self.q[1] + self.q[2] * self.q[3]),
                self.q[0] * self.q[0]
                - self.q[1] * self.q[1]
                - self.q[2] * self.q[2]
                + self.q[3] * self.q[3],
            )
        )

