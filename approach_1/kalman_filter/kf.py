
import numpy as np


class KF:
    """
    Class adapted from https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/
    """
    
    DT = 1 / 30.0
    U_X = 1
    U_Y = 1
    STD_ACC = 0.01
    X_STD_MEAS = 0.001
    Y_STD_MEAS = 0.001

    def __init__(self):
        self.u = np.matrix([[KF.U_X], [KF.U_Y]])
        self.A = np.matrix([[1, 0, KF.DT, 0],
                            [0, 1, 0, KF.DT],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.B = np.matrix([[0.5 * np.power(KF.DT, 2), 0],
                            [0, 0.5 * np.power(KF.DT, 2)],
                            [KF.DT, 0],
                            [0, KF.DT]])
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.matrix([[0.25 * np.power(KF.DT, 4), 0, 0.5 * np.power(KF.DT, 3), 0],
                            [0, 0.25 * np.power(KF.DT, 4), 0, 0.5 * np.power(KF.DT, 3)],
                            [0.5 * np.power(KF.DT, 3), 0, np.power(KF.DT, 2), 0],
                            [0, 0.5 * np.power(KF.DT, 3), 0, np.power(KF.DT, 2)]]) * np.power(KF.STD_ACC, 2)
        self.R = np.matrix([[np.power(KF.X_STD_MEAS, 2), 0],
                            [0, np.power(KF.Y_STD_MEAS, 2)]])
        self.x = np.matrix(np.zeros((4, 1)))
        self.P = np.eye(4)
        self.I = np.eye(4)

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0].item(), self.x[1].item()

    def update(self, cx, cy):
        z = np.array([[cx], [cy]])
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
        self.P = (self.I - (K * self.H)) * self.P
