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
import filecmp
from mininec.plot_antenna import main

class Test_Plot (unittest.TestCase):
    outfile = 'zoppel-test.png'

    @pytest.fixture (autouse=True)
    def cleanup (self):
        yield
        try:
            os.unlink (self.outfile)
        except FileNotFoundError:
            pass
    # end def cleanup

    def test_cmdline_err (self):
        infile = "test/12-el-5deg.pout"
        args = ["--scaling-method=linear_db", "--scaling-mindb=7", infile]
        self.assertRaises (ValueError, main, args)
    # end def test_cmdline_err

    def test_azimuth (self):
        infile = "test/12-el-1deg.pout"
        golden = "test/12-el-azimuth.png"
        args = ["--azi", "--out=%s" % self.outfile, infile]
        main (args)
        self.assertTrue (filecmp.cmp (golden, self.outfile, shallow = False))
    # end def test_azimuth

    def test_azimuth_linear (self):
        infile = "test/12-el-5deg.pout"
        golden = "test/12-el-azimuth-linear.png"
        args = [ "--azi", "--scaling-method=linear"
               , "--out=%s" % self.outfile, infile
               ]
        main (args)
        self.assertTrue (filecmp.cmp (golden, self.outfile, shallow = False))
    # end def test_azimuth_linear

    def test_azimuth_linear_voltage (self):
        infile = "test/12-el-5deg.pout"
        golden = "test/12-el-azimuth-linear-v.png"
        args = [ "--azi", "--scaling-method=linear_voltage"
               , "--out=%s" % self.outfile, infile
               ]
        main (args)
        self.assertTrue (filecmp.cmp (golden, self.outfile, shallow = False))
    # end def test_azimuth_linear_voltage

    def test_azimuth_db (self):
        infile = "test/12-el-5deg.pout"
        golden = "test/12-el-azimuth-db.png"
        args = [ "--azi", "--scaling-method=linear_db"
               , "--out=%s" % self.outfile, infile
               ]
        main (args)
        self.assertTrue (filecmp.cmp (golden, self.outfile, shallow = False))
    # end def test_azimuth_db

    def test_elevation (self):
        infile = "test/12-el-1deg.pout"
        golden = "test/12-el-elevation.png"
        args = ["--ele", "--out=%s" % self.outfile, infile]
        main (args)
        self.assertTrue (filecmp.cmp (golden, self.outfile, shallow = False))
    # end def test_elevation

    def test_3d (self):
        infile = "test/12-el-5deg.pout"
        golden = "test/12-el-3d.png"
        args = ["--out=%s" % self.outfile, infile]
        main (args)
        self.assertTrue (filecmp.cmp (golden, self.outfile, shallow = False))
    # end def test_3d

# end class Test_Plot
