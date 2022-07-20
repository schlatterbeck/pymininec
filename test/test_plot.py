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
import hashlib
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

    def compare_cs (self, checksums):
        with open (self.outfile, 'rb') as f:
            contents = f.read ()
        cs = hashlib.sha1 (contents).hexdigest ()
        assert cs in checksums
    # end def compare_cs

    def test_cmdline_err (self):
        infile = "test/12-el-5deg.pout"
        args = ["--scaling-method=linear_db", "--scaling-mindb=7", infile]
        self.assertRaises (ValueError, main, args)
    # end def test_cmdline_err

    def test_azimuth (self):
        checksums = set \
            (( '87e13e39a92e12e79eab915c8954d5100f90b5fd'
             , '968bd23b42eaf39dd1a512e282ebcc50cc7aaaea'
             , '70a43a76cb5ac3b4c65c18e73d4b370652667e0c'
            ))
        infile = "test/12-el-1deg.pout"
        args = ["--azi", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_azimuth

    def test_azimuth_linear (self):
        checksums = set \
            (( '3793e726c2dc1ad0bf1b6cd5c138182578ca0936'
             , 'bf9c5804cbc546a54921745d1746a696bf40a0e9'
             , '6c12919897a6dddf0b7ac5c42e5682d586deb44a'
            ))
        infile = "test/12-el-5deg.pout"
        args = [ "--azi", "--scaling-method=linear"
               , "--out=%s" % self.outfile, infile
               ]
        main (args)
        self.compare_cs (checksums)
    # end def test_azimuth_linear

    def test_azimuth_linear_voltage (self):
        checksums = set \
            (( '4a9044274f104f74cab5c484625d2090230385f5'
             , 'aec07c0157118d7cf194d5f1a805be3286b80b5d'
             , 'f79c1ff393bc8ef1e95f44cb012f5de4f238e95b'
            ))
        infile = "test/12-el-5deg.pout"
        args = [ "--azi", "--scaling-method=linear_voltage"
               , "--out=%s" % self.outfile, infile
               ]
        main (args)
        self.compare_cs (checksums)
    # end def test_azimuth_linear_voltage

    def test_azimuth_db (self):
        checksums = set \
            (( '78de072c12ea607ed8dfd0932dc4d8318710bdb5'
             , 'cf1098a0b9454498b893567d34c8f6fb610ccfb8'
             , '7e51e4b9709dd695feb7ccace5c2eccae3d9dee1'
            ))
        infile = "test/12-el-5deg.pout"
        args = [ "--azi", "--scaling-method=linear_db"
               , "--out=%s" % self.outfile, infile
               ]
        main (args)
        self.compare_cs (checksums)
    # end def test_azimuth_db

    def test_elevation (self):
        checksums = set \
            (( '111bb4f7a5d99be148e67fd3a5c0976cab58e225'
             , 'ee17b416807e7e86cc609b0ec04197547766a475'
             , '03fba22dd91c16a193a4a62beed621a7a3c0001d'
            ))
        infile = "test/12-el-1deg.pout"
        args = ["--ele", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_elevation

    def test_3d (self):
        checksums = set \
            (( 'bf0053e8fafbf5b7a28fc1dd3f40a66a500fb797'
             , 'f106dccd3ccff1938d47664cba5f659b2560d27b'
             , '67d5f8a4cd30154754ececc0e8d22c6eba92a7c5'
            ))
        infile = "test/12-el-5deg.pout"
        args = ["--plot3d", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_3d

    def test_plotall (self):
        checksums = set \
            (( 'feb1e0ff03d8204ac13ab28e4d888e11507a0327'
             , 'a951418c5a0cb42f6acdd12e028e51ad91d720a8'
             , '7b18b3fcba97fee155d05c5b2df898523553fc99'
            ))
        infile = "test/inverted-v.pout"
        args = ["--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_3d

    def test_vswr (self):
        checksums = set \
            (( 'beb1826625148d40f313d631c4631509cbb06cac'
             , 'da37992d6480f05cd51f023c8d7776c15cc7a2cb'
             , 'e0abdef36874e61b525221b135f866542e5c4c14'
            ))
        infile = "test/inverted-v.pout"
        args = ["--vswr", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_3d

    def test_basic_output (self):
        checksums = set \
            (( '8a303a9907121e917e0047451110b8905ab4526c'
             , '0d490c964bc1f740d2a94e8cd777b24e30bb469a'
             , '210f8f0ddb2ed12fe9c92e52fab03358ff9e7de9'
            ))
        infile = "test/vdipole-01.bout"
        args = ["--ele", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_3d

    def test_gainfile (self):
        """ Original basic implementation can save gains to a file
        """
        checksums = set \
            (( '55c6cb4a713d0950c24529a1e0b220b1465f8047'
             , '7ff88a12c6ea96289e9a27dc8c7ba51cd5b5f43b'
             , 'baa9403ed2893533886799cf730a298a84fa59f1'
            ))
        infile = "test/DP001.GNN"
        args = ["--ele", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_3d

    def test_necfile (self):
        """ We also can parse nec2c output
        """
        checksums = set \
            (( '1e62d599284cd08170668e3ba5ce279fa36f36bb'
             , 'aec56687ccb700c60a558adf02d5cf81fcab7436'
             , '2ac05564733a3b7765e84eaa1e24d515f7146e95'
            ))
        infile = "test/12-el.nout"
        args = ["--azi", "--swr", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_3d

# end class Test_Plot
