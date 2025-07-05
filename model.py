import numpy as np


class TwoWheels:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length

    def model(self, xi_minus_1, ui_minus_1):
        x3 = xi_minus_1[2]
        u1 = ui_minus_1[0]
        u2 = ui_minus_1[1]
        dx1_dt = (self.radius / 2) * (u1 + u2) * np.cos(x3)
        dx2_dt = (self.radius / 2) * (u1 + u2) * np.sin(x3)
        dx3_dt = (self.radius / self.length) * (u1 - u2)
        dx_dt = np.array([dx1_dt, dx2_dt, dx3_dt])
        return dx_dt
