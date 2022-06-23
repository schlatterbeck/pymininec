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
        num_tests = 118
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
