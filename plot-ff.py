#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Mininec_Gain:

    def __init__ (self, filename):
        self.filename = filename
        self.pattern  = {}
        self.read_file ()
    # end def __init__

    def read_file (self):
        guard = 'not set'
        delimiter = guard
        with open (self.filename, 'r') as f:
            for line in f:
                line = line.strip ()
                # File might end with Ctrl-Z (DOS EOF)
                if line.startswith ('\x1a'):
                    break
                if delimiter != guard:
                    d = delimiter
                    zen, azi, vp, hp, tot = (float (l) for l in line.split (d))
                    self.pattern [(zen, azi)] = tot
                if line.endswith (',D'):
                    delimiter = ','
                    continue
                if line.startswith ('ANGLE') and line.endswith ('(DB)'):
                    delimiter = None
                    continue
    # end def read_file

    def plot (self):
        thetas = set ()
        phis   = set ()
        gains  = []
        for theta, phi in sorted (self.pattern):
            gains.append (self.pattern [(theta, phi)])
            thetas.add (theta)
            phis.add   (phi)
        thetas = np.array (list (sorted (thetas))) * np.pi / 180
        phis   = np.array (list (sorted (phis)))   * np.pi / 180
        gains  = np.reshape (np.array (gains), (thetas.shape [0], -1))
        gains  = 10.0 ** (gains / 10.0)
        # Thetas are upside down (count from top)
        theta = -theta + np.pi
        P, T   = np.meshgrid (phis, thetas)
        X = np.cos (P) * np.sin (T) * gains
        Y = np.sin (P) * np.sin (T) * gains
        Z = np.cos (T) * gains
        fig = plt.figure ()
        ax  = fig.gca (projection='3d')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array \
            ( [ X.max () - X.min ()
              , Y.max () - Y.min ()
              , Z.max () - Z.min ()
              ]
            ).max() / 2.0
        mid_x = (X.max () + X.min ()) * 0.5
        mid_y = (Y.max () + Y.min ()) * 0.5
        mid_z = (Z.max () + Z.min ()) * 0.5
        ax.set_xlim (mid_x - max_range, mid_x + max_range)
        ax.set_ylim (mid_y - max_range, mid_y + max_range)
        ax.set_zlim (mid_z - max_range, mid_z + max_range)

        ax.plot_wireframe (X, Y, Z, color = 'r')
        plt.show ()
    # end def plot
# end class Mininec_Gain

if __name__ == '__main__':
    mg = Mininec_Gain (sys.argv [1])
    mg.plot ()
