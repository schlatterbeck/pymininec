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
            (( '510744aa65305c05b5ff5dc0463ba9aa701c4a41'
             , '6385000c17613ea4c8e887f6a684bf2da1fcceb7'
            ))
        infile = "test/12-el-1deg.pout"
        args = ["--azi", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_azimuth

    def test_azimuth_linear (self):
        checksums = set \
            (( '62cd94a5f62a982c13d8fec5db5b50fd954816f8'
             , '9025a68762d052dcdf78dc6ef6fedc08b4d4c9df'
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
            (( 'e94f8b26985b49b9abdf1c7a11ce0ce06d365cde'
             , 'eb90775ff13dd8c789023399b328c0fd7fb550dc'
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
            (( '1c1223ee1d36afee510bbc96ca4e530ea4068580'
             , '3e6865c0bcb570851284461fbb2b93c502efea22'
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
            (( 'ef0800ab1923e1a28919388b02b94cdc7d89ae95'
             , 'c6e9eb508cac2e1a50f4dd5251c2a1b20aa961b7'
            ))
        infile = "test/12-el-1deg.pout"
        args = ["--ele", "--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_elevation

    def test_3d (self):
        checksums = set \
            (( 'b9af4211ac0a2b77703e3c0f4de02686d8cacf20'
             , 'f53738e86eac0720001b66a7d7777c0276ca7ca9'
            ))
        infile = "test/12-el-5deg.pout"
        args = ["--out=%s" % self.outfile, infile]
        main (args)
        self.compare_cs (checksums)
    # end def test_3d

# end class Test_Plot
