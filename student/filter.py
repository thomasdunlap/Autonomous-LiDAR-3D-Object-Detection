# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state 
        self.dt = params.dt 
        self.q = params.q 

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = self.dt
        return np.matrix([[1, 0, 0, dt, 0,  0 ],
                          [0, 1, 0, 0,  dt, 0 ],
                          [0, 0, 1, 0,  0,  dt],
                          [0, 0, 0, 1,  0,  0 ],
                          [0, 0, 0, 0,  1,  0 ],
                          [0, 0, 0, 0,  0,  1 ]])
        
        ############
        # END studnent code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        q = self.q 
        dt = self.dt
        q1 = dt**3 / 3 * q
        q2 = dt**2 / 2 * q
        q3 = dt * q 
        return np.matrix([[q1, 0,  0,  q2, 0,  0 ],
                          [0,  q1, 0,  0,  q2, 0 ],
                          [0,  0,  q1, 0,  0,  q2],
                          [q2, 0,  0,  q3, 0,  0 ],
                          [0,  q2, 0,  0,  q3, 0 ],
                          [0,  0,  q2, 0,  0,  q3]])
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F = self.F()
        x = F * track.x
        P = F * track.P * F.transpose() + self.Q
        track.set_x(x)
        track.set_P(P) 
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        # H -> Measurement matrix
        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)
        # S -> Covariance matrix of the residual
        S = self.S(track, meas, H)
        # K -> Kalman gain
        K = track.P * H.transpose() * S.I
        # x -> State (state update below)
        x = track.x + K * gamma 
        I = np.identity(self.dim_state)
        # P -> Covariance matrix update
        P = (I - K * H) * track.P
        track.set_x(x)
        track.set_P(P)
        
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        return meas.z - meas.sensor.get_hx(track.x)
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        return H * track.P * H.transpose() + meas.R
        
        ############
        # END student code
        ############ 