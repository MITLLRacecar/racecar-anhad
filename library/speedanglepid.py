#TODO: Unfinished

import pidcontroller

class SpeedAnglePID:
    """PID Controller """

    def __init__(self, xP=0.0, xI=0.0, xD=0.0, yP=0.0, yI=0.0, yD=0.0, racecar):
        self.Speed_PID = pidcontroller.PID(xP, xI, xD)
        self.Angle_PID = pidcontroller.PID(yP, yI, yD)
        self.racecar = racecar
        self.speed = 0
        self.angle = 0
        self.targetX = self.racecar.camera.get_width() // 2
        self.targetY = self.racecar.camera.get_height() // 2

    def update(self, currentX, currentY, dt): #assumes current and target are pixels on screen
        """Calculates PID values for speed and angle
        """
        errorX = (currentX - targetX) / (racecar.camera.get_width() // 2)
        errorY = (currentY - targetY) / (racecar.camera.get_height() // 2)

        dt = self.racecar.get_delta_time()

        self.speed = self.Speed_PID.update(errorY, dt)
        self.angle = self.Angle_PID.update(errorX, dt, Self.speed)

        return (self.speed, self.angle)

    def commit(self):
        self.racecar.drive.set_speed_angle(self.speed, self.angle)

    def setTarget(self, x, y):
        self.targetX = x
        self.targetY = y