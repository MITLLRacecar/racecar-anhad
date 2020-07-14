import pidcontroller
import numpy as np

# Doesnt work, don't use


class SA_PID:
    """PID Controller """

    def __init__(self, xP, xI, xD, yP, yI, yD, racecar):
        self.Speed_PID = pidcontroller.PID(xP, xI, xD)
        self.Angle_PID = pidcontroller.PID(yP, yI, yD)
        self.racecar = racecar
        self.speed = 0
        self.angle = 0
        self.setTarget(self.racecar.camera.get_width() // 2)
        self.setTargetDist(0)

    def update(
        self, currentX, currentDist, overrideSpeed=0.0
    ):  # assumes current and target are pixels on screen
        """Calculates PID values for speed and angle
        """
        errorX = (currentX - self.targetX) / (self.racecar.camera.get_width() // 2)
        errorD = currentDist - self.targetDist

        dt = self.racecar.get_delta_time()

        if overrideSpeed == 0:
            self.speed = np.clip(self.Speed_PID.update(errorD, dt), -1, 1)
        else:
            self.speed = np.clip(overrideSpeed, -1, 1)

        if self.speed >= 0:
            self.angle = np.clip(
                self.Angle_PID.update(errorX, dt, 1 - 0.3 * self.speed), -1, 1
            )
        elif self.speed < 0:
            self.angle = np.clip(
                self.Angle_PID.update(errorX, dt, -1 - 0.3 * self.speed), -1, 1
            )

        return (self.speed, self.angle)

    def setTarget(self, x):
        self.targetX = x

    def setTargetDist(self, d):
        self.targetDist = d
