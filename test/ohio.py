#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mininec import *
from zmatrix import matrix_ohio_r, matrix_ohio_i

class Near_Far_Comparison:

    def __init__ (self):
        w = [Wire (10,0,0,17,0,0,17.5,0.0001)]
        x = Excitation (1+0j)
        m = Mininec (299.8, w, [ideal_ground])
        m.register_source (x, 5)
        self.m = m
        m.compute ()
        height = self.height = 300
        self.phi = Angle (0, 1, 1)
        self.a = []
        self.d = []
        self.xd = [2000, 20000]
        for d in self.xd:
            r = np.arange (d, d + 1.05, .05)
            a = np.arctan (r / height) / np.pi * 180
            d = np.linalg.norm \
                (np.array ([r, np.ones (21) * height]).T, axis = 1)
            self.a.append (a)
            self.d.append (d)
    # end def __init__

    def output_currents (self):
        return self.m.currents_as_mininec ()
    # end def output_currents

    def output_near (self):
        r = []
        m = self.m
        m.compute_near_field ([self.xd [0],0,self.height], [.05,1,1], [21,1,1])
        r.append (m.near_field_e_as_mininec ())
        m.compute_near_field ([self.xd [1],0,self.height], [.05,1,1], [21,1,1])
        r.append (m.near_field_e_as_mininec ())
        return '\n'.join (r)
    # end def output_near

    def output_far (self):
        r = []
        m = self.m
        for a, d in zip (self.a, self.d):
            for aa, dd in zip (a, d):
                m.compute_far_field (Angle (aa, 1, 1), self.phi, dist = dd)
                r.append (m.far_field_absolute_as_mininec ())
        return '\n'.join (r)
    # end def output_far

    def output_mininec_params (self):
        """ For creation of (parts of) .mini file
        """
        for a, d in zip (self.a, self.d):
            for aa, dd in zip (a, d):
                print ("P\nV\nN\n%.8f\n%.8f,1,1\n0,1,1\nN" % (dd, aa))
    # end def output_mininec_params

    def nf (self, value):
        p1 = 0j
        p2 = 0.0
        for v in value:
            a = np.angle (v)
            b = np.abs (v)
            b2 = b ** 2
            p1 += b2 * np.e ** (-2j * a)
            p2 += b2.real
        return np.sqrt (p2 / 2 + abs (p1) / 2)
    # end def nf

    def plot_near_far (self):
        m   = self.m
        x   = []
        for d in self.xd:
            x.append (np.arange (d, d + 1.05, .05))
        nfs = [[], []]
        ffs = [[], []]
        for i in (0, 1):
            m.compute_near_field \
                ([self.xd [i],0,self.height], [.05,1,1], [21,1,1])
            for value in m.e_field:
                nfs [i].append (self.nf (value))
        for n, (a, d) in enumerate (zip (self.a, self.d)):
            for aa, dd in zip (a, d):
                m.compute_far_field (Angle (aa, 1, 1), self.phi, dist = dd)
                assert len (m.far_field_by_angle) == 1
                ff = next (iter (m.far_field_by_angle.values ()))
                assert ff.e_phi == 0
                ffs [n].append (np.abs (ff.e_theta))
        nfs = np.array (nfs)
        ffs = np.array (ffs)
        for i in (0, 1):
            fig = plt.figure ()
            ax  = plt.subplot (111)
            ax.plot (x [i], nfs [i])
            ax.plot (x [i], ffs [i])
            plt.show ()
            fig = plt.figure ()
            ax  = plt.subplot (111)
            ax.plot (x [i], (nfs [i] - ffs [i]) / ffs [i] * 100)
            plt.show ()
    # end def plot_near_far

    def plot_z_errors (self):
        """ Plot differences in z-matrix of Basic an Python implementation
        """
        err_r_p = []
        err_i_p = []
        err_r_b = []
        err_i_b = []
        mat = np.array (matrix_ohio_r) + 1j * np.array (matrix_ohio_i)
        mat = mat.flatten ()
        z   = self.m.Z.flatten ()
        err   = mat - z
        err_r = err.real
        err_i = err.imag
        err_r_b = err_r / mat.real * 100
        err_r_p = err_r / z.real   * 100
        err_i_b = err_i / mat.imag * 100
        err_i_p = err_i / z.imag   * 100
        x = np.arange (0, len (err_i_p), 1)
        fig = plt.figure ()
        ax  = plt.subplot (111)
        ax.plot (x, err_r_p)
        ax.plot (x, err_i_p)
        plt.show ()
        #fig = plt.figure ()
        #ax  = plt.subplot (111)
        #ax.plot (x, err_r_b)
        #ax.plot (x, err_i_b)
        #plt.show ()
    # end def plot_z_errors

# end class Near_Far_Comparison

if __name__ == '__main__':
    nfc = Near_Far_Comparison ()
    nfc.plot_near_far ()
    #nfc.plot_z_errors ()
    #print (nfc.output_currents ())
    #print (nfc.output_near ())
    #print (nfc.output_far  ())
    ##nfc.output_mininec_params ()
