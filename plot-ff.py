#!/usr/bin/python3

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser
from matplotlib import cm

class Scaler:

    def set_ticks (self, ax):
        g = self.ticks
        ax.set_rticks (self.scale (0, g))
        ax.set_yticklabels ([''] + [('%d' % i) for i in g [1:]], ha = 'center')
    # end def set_ticks

# end class Scaler

class Linear_Scaler (Scaler):
    ticks = np.array ([0, -3, -6, -10])
    title = 'Linear scale'

    def scale (self, max_gain, gains):
        return 10 ** ((gains - max_gain) / 10)
    # end def scale

# end class Linear_Scaler

scale_linear = Linear_Scaler ()

class Linear_Voltage_Scaler (Scaler):
    ticks = np.array ([0, -3, -6, -10, -20])
    title = 'Linear voltage'

    def scale (self, max_gain, gains):
        return 10 ** ((gains - max_gain) / 20)
    # end def scale

# end class Linear_Voltage_Scaler

scale_linear_voltage = Linear_Voltage_Scaler ()

class ARRL_Scaler (Scaler):
    ticks = np.array ([0, -3, -10, -20, -30])
    title = 'ARRL scale'

    def scale (self, max_gain, gains):
        return (1 / 0.89) ** ((gains - max_gain) / 2)
    # end def scale

# end class ARRL_Scaler

scale_arrl = ARRL_Scaler ()

class Linear_dB_Scaler (Scaler):
    title = 'Linear dB'

    def __init__ (self, min_db = -50):
        if min_db >= 0:
            raise ValueError ("min_db must be < 0")
        self.min_db = min_db
        self.ticks = np.arange (0, self.min_db - 10, -10)
    # end def __init__

    def scale (self, max_gain, gains):
        return \
            ( (np.maximum (self.min_db, (gains - max_gain)) - self.min_db)
            / -self.min_db
            )
    # end def scale

# end class Linear_dB_Scaler

class Mininec_Gain:

    def __init__ (self, filename):
        self.filename = filename
        self.f        = None
        # Default title from filename
        self.title    = os.path.splitext (filename) [0]
        self.pattern  = {}
        # This might override title
        self.read_file ()
        thetas = set ()
        phis   = set ()
        gains  = []
        for theta, phi in sorted (self.pattern):
            gains.append (self.pattern [(theta, phi)])
            thetas.add (theta)
            phis.add   (phi)
        self.thetas_d = np.array (list (sorted (thetas)))
        self.phis_d   = np.array (list (sorted (phis)))
        self.thetas   = self.thetas_d * np.pi / 180
        self.phis     = self.phis_d   * np.pi / 180
        self.maxg     = max (gains)
        self.gains    = np.reshape \
            (np.array (gains), (self.thetas.shape [0], -1))
        idx = np.unravel_index (self.gains.argmax (), self.gains.shape)
        self.theta_maxidx, self.phi_maxidx = idx
        self.theta_max = self.thetas_d [self.theta_maxidx]
        self.phi_max   = self.phis_d   [self.phi_maxidx]
        self.desc      = ['Title: %s' % self.title]
        if self.f is not None:
            self.desc.append ('Frequency: %.2f MHz' % self.f)
        self.desc.append ('Outer ring: %.2f dBi' % self.maxg)
        self.lbl_deg   = 0
    # end def __init__

    def read_file (self):
        guard = 'not set'
        delimiter = guard
        with open (self.filename, 'r') as f:
            for line in f:
                line = line.strip ()
                if self.f is None and line.startswith ('FREQUENCY'):
                    self.f = float (line.split (':') [1].split () [0])
                # File might end with Ctrl-Z (DOS EOF)
                if line.startswith ('\x1a'):
                    break
                if delimiter != guard:
                    if not line:
                        break
                    d = delimiter
                    fields = line.split (d)
                    zen, azi, vp, hp, tot = (float (l) for l in fields [:5])
                    self.pattern [(zen, azi)] = tot
                if line.endswith (',D'):
                    delimiter = ','
                    continue
                if line.startswith ('ANGLE') and line.endswith ('(DB)'):
                    delimiter = None
                    continue
                # Also read NEC files
                if  (   line.startswith ('DEGREES   DEGREES        DB')
                    and line.endswith ('VOLTS/M   DEGREES')
                    ):
                    delimiter = None
                    continue
    # end def read_file

    def azimuth (self, scaler):
        self.desc.insert (0, 'Azimuth Pattern')
        self.desc.append ('Scaling: %s' % scaler.title)
        self.desc.append ('Elevation: %.2f°' % (90 - self.theta_max))
        self.lbl_deg  = self.phi_max
        gains = scaler.scale (self.maxg, self.gains)
        self.polargains = gains [self.theta_maxidx]
        self.angles = self.phis
        self.polarplot (scaler)
    # end def azimuth

    def elevation (self, scaler):
        """ Elevation is a little more complicated due to theta counting
            from zenith.
            OK for both 90° and 180° plots:
            self.angles = p2 - self.thetas
            self.polargains = gains1
            OK for both 90° and ^80° plots:
            self.angles = p2 + self.thetas
            self.polargains = gains2
            But second half must (both) be flipped to avoid crossover
        """
        self.desc.insert (0, 'Elevation Pattern')
        self.desc.append ('Scaling: %s' % scaler.title)
        self.lbl_deg  = 90 - self.theta_max
        gains = scaler.scale (self.maxg, self.gains)
        gains1 = gains.T [self.phi_maxidx].T
        # Find index of the other side of the azimuth
        pmx = self.phis.shape [0] - self.phis.shape [0] % 2
        idx = (self.phi_maxidx + pmx // 2) % pmx
        assert idx != self.phi_maxidx
        eps = 1e-9
        assert abs (self.phis [idx] - self.phis [self.phi_maxidx]) - np.pi < eps
        gains2 = gains.T [idx].T
        p2 = np.pi / 2
        self.polargains = np.append (gains1, np.flip (gains2))
        self.angles = np.append (p2 - self.thetas, np.flip (p2 + self.thetas))
        self.polarplot (scaler)
    # end def elevation

    def polarplot (self, scaler):
        dpi = 80
        fig = plt.figure (dpi = dpi, figsize = (512 / dpi, 384 / dpi))
        ax  = plt.subplot (111, projection = 'polar')
        ax.set_rmax (1)
        ax.set_rlabel_position (self.lbl_deg  or 0)
        ax.set_thetagrids (range (0, 360, 15))
        plt.figtext (0.005, 0.01, '\n'.join (self.desc))
        args = dict (linestyle = 'solid', linewidth = 1.5)
        ax.plot (self.angles, self.polargains, **args)
        #ax.grid (True)
        scaler.set_ticks (ax)
        # Might add color and size labelcolor='r' labelsize = 8
        ax.tick_params (axis = 'y', rotation = 'auto')
        plt.show ()
        #fig.savefig ('zoppel.png')
    # end def polarplot

    def plot3d (self, scaler):
        gains  = scaler.scale (self.maxg, self.gains)
        P, T   = np.meshgrid (self.phis, self.thetas)
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
        ax.set_xlabel ('X')
        ax.set_ylabel ('Y')
        ax.set_zlabel ('Z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        #norm = plt.Normalize (vmin = 0.0, vmax = 1.0, clip = True)
        #colors = cm.rainbow (norm (gains))
        colors = cm.rainbow (gains)
        rc, cc = gains.shape

        #ax.plot_wireframe (X, Y, Z, color = 'r', linewidth = 0.5)
        surf = ax.plot_surface \
            ( X, Y, Z, rcount=rc, ccount=cc, facecolors=colors, shade=False
            #, cmap = cm.rainbow, norm = norm
            )
        #import pdb; pdb.set_trace()
        #surf.set_facecolor ((0, 0, 0, 0))
        #surf.set_edgecolor ((.5, .5, .5, 1))
        plt.show ()
    # end def plot3d
# end class Mininec_Gain

if __name__ == '__main__':
    cmd = ArgumentParser ()
    scaling = ['arrl', 'linear', 'linear_db', 'linear_voltage']
    cmd.add_argument \
        ( 'filename'
        , help    = 'File to parse and plot'
        )
    cmd.add_argument \
        ( '--azimuth'
        , help    = 'Do an azimuth plot'
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '--elevation'
        , help    = 'Do an elevation plot'
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '--plot3d'
        , help    = 'Do a 3D plot'
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '--scaling-method'
        , help    = 'Scaling method to use, default=%%(default)s, one of %s'
                  % (', '.join (scaling))
        , default = 'arrl'
        )
    cmd.add_argument \
        ( '--scaling-mindb'
        , help    = 'Minimum decibels linear dB scaling, default=%(default)s'
        , type    = float
        , default = -50
        )
    args = cmd.parse_args ()
    mg = Mininec_Gain (args.filename)

    scale_linear_db = Linear_dB_Scaler (args.scaling_mindb)

    scaler = globals () ['scale_' + args.scaling_method]
    if not args.azimuth and not args.elevation and not args.plot3d:
        args.plot3d = True
    if args.azimuth:
        mg.azimuth (scaler = scaler)
    if args.elevation:
        mg.elevation (scaler = scaler)
    if args.plot3d:
        mg.plot3d (scaler = scaler)
