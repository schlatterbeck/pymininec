# Copyright (C) 2022 Ralf Schlatterbeck. All rights reserved
# Reichergasse 131, A-3411 Weidling
# ****************************************************************************
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software. 
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ****************************************************************************

import os
import unittest
import pytest
import doctest
import mininec
import numpy as np
from mininec import *

class _Test_Base_With_File:

    def simple_setup (self, filename, mininec):
        with open (os.path.join ('test', filename), 'r') as f:
            self.expected_output = f.read ()
        self.expected_output = self.expected_output.rstrip ('\n')
        mininec.compute ()
    # end def simple_setup

    def dipole_7mhz (self, wire_dia, filename):
        w = []
        w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, wire_dia))
        s = Excitation (4, 1, 0)
        m = Mininec (7, w, [s])
        self.simple_setup (filename, m)
        zenith  = Angle (0, 10, 19)
        azimuth = Angle (0, 10, 37)
        m.compute_far_field (zenith, azimuth)
        return m
    # end def dipole_7mhz

    def vertical_dipole (self, wire_dia, filename, media = None):
        w = []
        w.append (Wire (10, 0, 0, 7.33, 0, 0, 12.7, wire_dia))
        s = Excitation (4, 1, 0)
        m = Mininec (28.074, w, [s], media = media)
        self.simple_setup (filename, m)
        zlen = 19
        if media:
            zlen = 10
        zenith  = Angle (0, 10, zlen)
        azimuth = Angle (0, 10, 37)
        m.compute_far_field (zenith, azimuth)
        return m
    # end def vertical_dipole

# end class _Test_Base_With_File

class Test_Case_Known_Structure (_Test_Base_With_File, unittest.TestCase):

    matrix_ideal_ground_from_mininec = \
        [ [ -24.25143-4.057954E-02j,    11.87853-4.004717E-02j
          , .8545649-3.865042E-02j,     .2144457-3.647149E-02j
          , 9.295902E-02-.0336251j,     4.777939E-02-3.024885E-02j
          , 2.533447E-02-2.649694E-02j, 1.235796E-02-2.253033E-02j
          , 4.29464E-03-1.850793E-02j
          ]
        , [ 11.87853-4.004717E-02j,     -24.25214-4.020253E-02j
          , 11.878-3.950967E-02j,       .8542481 -3.802047E-02j
          , .2143418-3.581698E-02j,     9.305359E-02-3.300821E-02j
          , 4.804171E-02-2.972181E-02j, 2.572245E-02-2.609828E-02j
          , 1.282296E-02-2.228332E-02j
          ]
        , [ .854565-3.865037E-02j,       11.878-.0395097j
          , -24.25245-3.957258E-02j,     11.8779-3.885516E-02j
          , .8543426-3.740353E-02j,      .2146041-3.528988E-02j
          , 9.344161E-02-3.260954E-02j,  4.850671E-02-2.947475E-02j
          , 2.621375E-02-2.601021E-02j
          ]
        , [ .2144457-3.647152E-02j,      .8542481-3.802047E-02j
          , 11.8779-3.885519E-02j,       -24.25236-3.895564E-02j
          , 11.87816-3.832813E-02j,      .8547306-.0370049j
          , .2150691-3.504282E-02j,      9.393284E-02-3.252147E-02j
          , .0489762-2.953798E-02j
          ]
        , [ 9.295902E-02-.0336251j,      .2143418-3.581696E-02j
          , .8543426-3.740353E-02j,      11.87816-3.832813E-02j
          , -24.25197-.038557j,          11.87863-3.808102E-02j
          , .8552219-3.691685E-02j,      .2155386-3.510608E-02j
          , 9.433927E-02-3.271535E-02j
          ]
        , [ 4.777937E-02-3.024887E-02j,  9.305353E-02-3.300817E-02j
          , .2146042-3.528992E-02j,      .8547306-.0370049j
          , 11.87863-3.808102E-02j,      -24.25148-3.846894E-02j
          , 11.8791-3.814429E-02j,       .8556283 -3.711072E-02j
          , .21585-3.540048E-02j
          ]
        , [ 2.533448E-02-2.649699E-02j,  4.804171E-02-2.972181E-02j
          , 9.344155E-02-3.260949E-02j,  .2150691-3.504282E-02j
          , .8552219-3.691685E-02j,      11.8791-3.814429E-02j
          , -24.25107-3.866283E-02j,     11.87941-3.843868E-02j
          , .8558246-3.746882E-02j
          ]
        , [ 1.235796E-02-2.253033E-02j,  2.572251E-02-2.609833E-02j
          , 4.850671E-02-2.947475E-02j,  9.393284E-02-3.252147E-02j
          , .2155386-3.510608E-02j,      .8556283-3.711072E-02j
          , 11.87941-3.843868E-02j,      -24.25088-3.902092E-02j
          , 11.87948-3.882118E-02j
          ]
        , [ 4.294615E-03-1.850792E-02j,  1.282297E-02-2.228327E-02j
          , 2.621375E-02-2.601026E-02j,  .0489762-2.953798E-02j
          , 9.433927E-02-3.271535E-02j,  .21585-3.540048E-02j
          , .8558246-3.746882E-02j,      11.87948-3.882118E-02j
          , -24.25092-3.939013E-02j
          ]
        ]

    def test_excitation (self):
        """ Test error cases
        """
        self.assertRaises (ValueError, Excitation, -1, cvolt = 5)
        self.assertRaises (ValueError, Excitation, 1, 1, 1, 1)
    # end def test_excitation

    def test_medium (self):
        """ Test error cases
        """
        self.assertRaises (ValueError, Medium, 0, 0, nradials = 1)
        self.assertRaises (ValueError, Medium, 0, 0, height = 1)
        self.assertRaises (ValueError, Medium, 1, 1, nradials = 1)
        self.assertRaises (ValueError, Medium, 1, 0)
        w = []
        w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        x = Excitation (4, 1, 0)
        self.assertRaises (ValueError, Mininec, 7, w, [x], media = [])
        ideal = mininec.ideal_ground
        media = [ideal, ideal]
        x = Excitation (4, 1, 0)
        self.assertRaises (ValueError, Mininec, 7, w, [x], media = media)
        rad = Medium (1, 1, nradials = 1, radius = 1, dist = 1)
        media = [rad, rad]
        x = Excitation (4, 1, 0)
        self.assertRaises (ValueError, Mininec, 7, w, [x], media = media)
        #Mininec (7, w, [x], media = media)
    # end def test_excitation

    def test_wire (self):
        """ Test error cases
        """
        self.assertRaises (ValueError, Wire, 7, 1, 1, 1, 2, 2, 2, 0)
        self.assertRaises (ValueError, Wire, 7, 1, 1, 1, 1, 1, 1, 1)
        wire = Wire (7, 0, 0, 0, 1, 1, 0, 0.01)
        ideal = mininec.ideal_ground
        self.assertRaises (ValueError, wire.compute_ground, 0, ideal)
        wire = Wire (7, 0, 0, -1, 1, 1, -1, 0.01)
        self.assertRaises (ValueError, wire.compute_ground, 0, ideal)
    # end def test_excitation

    def test_source_index (self):
        w = []
        w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        # Wrong index: Exceeds valid segments
        s = Excitation (10, 1, 0)
        self.assertRaises (ValueError, Mininec, 7, w, [s])
    # end def test_source_index

    def test_matrix_fill_ideal_ground (self):
        """ This uses assertAlmostEqual number of decimal places to
            compare significant digits (approximately)
        """
        mat   = self.matrix_ideal_ground_from_mininec
        ideal = ideal_ground
        m = self.vertical_dipole \
            (wire_dia = 0.01, filename = 'vdipole-01.pout', media = [ideal])
        for i in range (len (m.w_per)):
            for j in range (len (m.w_per)):
                f = int (np.log (abs (mat [i][j].real)) / np.log (10))
                self.assertAlmostEqual (mat [i][j].real, m.Z [i][j].real, 3-f)
                f = int (np.log (abs (mat [i][j].imag)) / np.log (10))
                self.assertAlmostEqual (mat [i][j].imag, m.Z [i][j].imag, 3-f)
    # end def test_matrix_fill_ideal_ground

    def test_dipole_wiredia_01 (self):
        m = self.dipole_7mhz (wire_dia = 0.01, filename = 'dipole-01.pout')
        self.assertEqual (self.expected_output, m.as_mininec ())
    # end def test_dipole_wiredia_01

    def test_dipole_wiredia_001 (self):
        m = self.dipole_7mhz (wire_dia = 0.001, filename = 'dipole-001.pout')
        self.assertEqual (self.expected_output, m.as_mininec ())
    # end def test_dipole_wiredia_001

    def test_vdipole_wiredia_01 (self):
        m = self.vertical_dipole (wire_dia = 0.01, filename = 'vdipole-01.pout')
        self.assertEqual (self.expected_output, m.as_mininec ())
    # end def test_dipole_wiredia_01

#    def test_vdipole_wiredia_001 (self):
#        m = self.vertical_dipole (wire_dia = 0.01, filename = 'vdipole-01.pout')
#        self.assertEqual (self.expected_output, m.as_mininec ())
#    # end def test_dipole_wiredia_001

# end class Test_Case_Known_Structure

class Test_Doctest (unittest.TestCase):

    flags = doctest.NORMALIZE_WHITESPACE

    def test_mininec (self):
        num_tests = 123
        f, t  = doctest.testmod \
            (mininec, verbose = False, optionflags = self.flags)
        fn = os.path.basename (mininec.__file__)
        format_ok  = '%(fn)s passes all of %(t)s doc-tests'
        format_nok = '%(fn)s fails %(f)s of %(t)s doc-tests'
        if f:
            msg = format_nok % locals ()
        else:
            msg = format_ok % locals ()
        exp = 'mininec.py passes all of %d doc-tests' % num_tests
        self.assertEqual (exp, msg)
    # end def test_mininec

# end class Test_Doctest
