from enum import IntEnum
import math
import numpy as np


class Type(IntEnum):
    LOWPASS = 1
    HIGHPASS = 2
    INTEGRATOR = 3
    DIFFERENTIATOR = 4


class Filter:
    """One Pole High/Lowpass filter, differentiator and integrator"""

    def __init__(self, type, cutoff, initialvalue, ndarray=False):
        self.type = type
        self.setFrequency(cutoff)
        self.ndarray = ndarray
        self.clear(initialvalue)

    def setFrequency(self, cutoff):
        self.tau = 1.0 / (math.pi * 2 * cutoff)

    def clear(self, initialvalue):
        if self.ndarray:
            self.y = np.copy(initialvalue)
            self.ylast = np.copy(initialvalue)
            self.x = np.copy(initialvalue)
        else:
            self.y = initialvalue
            self.ylast = initialvalue
            self.x = initialvalue

    def input(self, value, dt):
        if self.ndarray:
            self.ylast = np.copy(self.y)
        else:
            self.ylast = self.y  # shift the data values

        self.x = value  # this is now the most recent input value
        # tau is set by the user in microseconds, but must be converted to samples here

        # tau is set by the user in seconds, but must be converted to samples here
        tauSamps = self.tau / dt

        ampFactor = math.exp(-1.0 / tauSamps)

        self.y = (1.0 - ampFactor) * self.x + ampFactor * self.ylast
        # set the new value

        return self.output()

    def output(self):
        if self.type == Type.LOWPASS:
            return self.y
        elif self.type == Type.INTEGRATOR:
            return self.y * self.tau
        elif self.type == Type.HIGHPASS:
            return self.x - self.y
        else:  # self.type == Type.DIFFERENTIATOR:
            return (self.x - self.y) / self.tau

    def __repr__(self):
        return f"Y: {self.y} Ylast: {self.ylast} X: {self.x} tau: {self.tau}"
