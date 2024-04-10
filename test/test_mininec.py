# Copyright (C) 2022-24 Ralf Schlatterbeck. All rights reserved
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
import pytest
import doctest
import numpy as np
import mininec
from mininec.mininec import *
from mininec.mininec import main
from zmatrix import *
from ohio import Near_Far_Comparison

class _Test_Base_With_File:

    def simple_setup (self, filename, mininec = None):
        with open (os.path.join ('test', filename), 'r') as f:
            self.expected_output = f.read ()
        self.expected_output = self.expected_output.rstrip ('\n')
        if mininec:
            mininec.compute ()
    # end def simple_setup

    def dipole_7mhz (self, wire_dia, filename):
        w = []
        w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, wire_dia))
        s = Excitation (1, 0)
        m = Mininec (7, w)
        m.register_source (s, 4)
        self.simple_setup (filename, m)
        if filename.endswith ('near.pout'):
            fs = np.array ([-2, -2, 0])
            fi = np.ones (3)
            fn = np.ones (3) * 5
            m.compute_near_field (fs, fi, fn, 100)
        else:
            zenith  = Angle (0, 10, 19)
            azimuth = Angle (0, 10, 37)
            m.compute_far_field (zenith, azimuth)
        return m
    # end def dipole_7mhz

    def vertical_dipole (self, wire_dia, filename, media = None, load = None):
        w = []
        w.append (Wire (10, 0, 0, 7.33, 0, 0, 12.7, wire_dia))
        s = Excitation (1, 0)
        m = Mininec (28.074, w, media = media)
        m.register_source (s, 4)
        if load:
            m.register_load (load)
        if filename is None:
            m.compute ()
        else:
            self.simple_setup (filename, m)
            zlen = 19
            if media:
                zlen = 10
            zenith  = Angle (0, 10, zlen)
            azimuth = Angle (0, 10, 37)
            m.compute_far_field (zenith, azimuth)
        return m
    # end def vertical_dipole

    def folded_dipole (self, filename):
        """ Folded dipole with 1" wire distance and 16.6' total length
            according to L. B. Cebik. Under the limits: MININEC (3.13).
            In Antenna Modeling Notes, volume 1, antenneX Online
            Magazine, 2003, chapter 2, pages 23–35. We use tapering of
            segment lengths.
            Note that this produces values completely different from the
            article. Probably the folded dipole in the article has 1'
            distance (foot) not 1" (inch), probably a typo.
            The values with double precision differ considerably from
            the single precision computatations in Basic, it still looks
            good enough.
        """
        i    = 0.0254
        ft   = 0.3048
        hl   = 16.6 * ft / 2
        d    = i
        cls  = Gauge_Wire
        w = []
        w.append (cls ( 1, -hl,          0, 0, -hl +  1 * i, 0, 0, 18))
        w.append (cls ( 1, -hl +  1 * i, 0, 0, -hl +  3 * i, 0, 0, 18))
        w.append (cls ( 1, -hl +  3 * i, 0, 0, -hl +  7 * i, 0, 0, 18))
        w.append (cls ( 1, -hl +  7 * i, 0, 0, -hl + 15 * i, 0, 0, 18))
        w.append (cls ( 6, -hl + 15 * i, 0, 0,            0, 0, 0, 18))
        w.append (cls ( 1, -hl,          0, 0, -hl,          0, d, 18))
        w.append (cls ( 1, -hl,          0, d, -hl +  1 * i, 0, d, 18))
        w.append (cls ( 1, -hl +  1 * i, 0, d, -hl +  3 * i, 0, d, 18))
        w.append (cls ( 1, -hl +  3 * i, 0, d, -hl +  7 * i, 0, d, 18))
        w.append (cls ( 1, -hl +  7 * i, 0, d, -hl + 15 * i, 0, d, 18))
        w.append (cls ( 6, -hl + 15 * i, 0, d,            0, 0, d, 18))

        w.append (cls ( 1,  hl,          0, 0,  hl -  1 * i, 0, 0, 18))
        w.append (cls ( 1,  hl -  1 * i, 0, 0,  hl -  3 * i, 0, 0, 18))
        w.append (cls ( 1,  hl -  3 * i, 0, 0,  hl -  7 * i, 0, 0, 18))
        w.append (cls ( 1,  hl -  7 * i, 0, 0,  hl - 15 * i, 0, 0, 18))
        w.append (cls ( 6,  hl - 15 * i, 0, 0,            0, 0, 0, 18))
        w.append (cls ( 1,  hl,          0, d,  hl,          0, 0, 18))
        w.append (cls ( 1,  hl,          0, d,  hl -  1 * i, 0, d, 18))
        w.append (cls ( 1,  hl -  1 * i, 0, d,  hl -  3 * i, 0, d, 18))
        w.append (cls ( 1,  hl -  3 * i, 0, d,  hl -  7 * i, 0, d, 18))
        w.append (cls ( 1,  hl -  7 * i, 0, d,  hl - 15 * i, 0, d, 18))
        w.append (cls ( 6,  hl - 15 * i, 0, d,            0, 0, d, 18))
        s = Excitation (1, 0)
        m = Mininec (28.5, w, media = None)
        m.register_source (s, 30)
        self.simple_setup (filename, m)
        zenith  = Angle (0, 10, 19)
        azimuth = Angle (0, 10, 37)
        m.compute_far_field (zenith, azimuth)
        return m
    # end def folded_dipole

    def vertical_quarterwave \
        ( self, filename, media
        , load = None, inv = False, dia = 0.0254, opt = None
        ):
        """ Vertical 1/4 lambda directly connected to ground and fed at
            the junction to ground. Using L. B. Cebik. Verticals at and
            over ground. In Antenna Modeling Notes volume 1, antenneX
            Online Magazine, 2003, chapter 12, pages 161–178.
            The wire is 1" diameter aluminium, 20 segments at 7.15MHz
            For now we ignore aluminium loading (not yet implemented)
            With the inv flag we create the wire upside-down for
            increasing test coverage.
        """
        i    = 0.0254
        wl   = 397 * i
        r    = dia / 2
        w = []
        if inv:
            w.append (Wire (20, 0, 0, wl, 0, 0, 0, r))
            ex = 19
        else:
            w.append (Wire (20, 0, 0, 0, 0, 0, wl, r))
            ex = 0
        if opt is None:
            opt = {}
        s = Excitation (1, 0)
        m = Mininec (7.15, w, media = media)
        m.register_source (s, ex)
        if load is not None:
            # Load only the first (index 0) segment
            m.register_load (load, 0)
        if filename is None:
            m.compute ()
        elif filename.endswith ('near.pout'):
            self.simple_setup (filename, m)
            fs = np.array ([-1, -1, 0])
            fi = np.ones (3)
            fn = np.array ([3, 3, 2])
            m.compute_near_field (fs, fi, fn)
        else:
            self.simple_setup (filename, m)
            zenith  = Angle (0,  5, 19)
            azimuth = Angle (0, 10, 37)
            m.compute_far_field (zenith, azimuth, **opt)
        return m
    # end def vertical_quarterwave

    def inverted_l (self, filename):
        w  = []
        w.append (Wire (4, 0, 0,    0, 0,    0, .191, .004))
        w.append (Wire (6, 0, 0, .191, 0, .309, .191, .004))
        m  = Mininec (299.8, w, media = [ideal_ground])
        ex = Excitation (1, 0)
        m.register_source (ex, 0)
        if filename is None:
            m.compute ()
        else:
            self.simple_setup (filename, m)
            zenith  = Angle (0, 5, 19)
            azimuth = Angle (0, 5, 73)
            m.compute_far_field (zenith, azimuth)
        return m
    # end def inverted_l

    def t_antenna (self, filename, r = .004):
        w  = []
        w.append (Wire ( 8, 0,        0,      0, 0, 0, .07958, r))
        w.append (Wire (17, 0, -.170423, .07958, 0, 0, .07958, r))
        w.append (Wire (17, 0,  .170423, .07958, 0, 0, .07958, r))
        m  = Mininec (299.8, w, media = [ideal_ground])
        ex = Excitation (1, 0)
        m.register_source (ex, 0)
        if filename is None:
            m.compute ()
        else:
            self.simple_setup (filename, m)
            zenith  = Angle (0, 10, 10)
            azimuth = Angle (0, 90,  2)
            m.compute_far_field (zenith, azimuth)
        return m
    # end def t_antenna

    def compare_far_field_data \
        (self, m, may_fail_last_digit = False, do_currents = False):
        """ dB values below -200 contain large rounding errors.
            This makes tests fail on different architectures, notably on
            Intel vs. AMD CPUs. But also on different python versions
            (self-compiled python10 vs debian bullseye python9).
            Seems the trigonometric functions are slightly different on
            these architectures. We compare values above -200dB exactly,
            and assert that the value is below -200 for the others.
            In addition if may_fail_last_digit is True the last digit of
            a dB value may differ.
            If do_currents is True, we compare currents using
            compare_currents instead of line by line, this takes care of
            cases where currents differ slightly for different
            architectures.
        """
        ex  = self.expected_output.split ('\n')
        ac  = m.as_mininec ().split ('\n')
        idx = self.expected_output.find ('PATTERN DATA')
        l   = len (self.expected_output [:idx].split ('\n'))
        off = l + 2
        if do_currents:
            self.compare_currents (ex [:off], ac [:off])
        else:
            assert ex [:off] == ac [:off]
        for e, a in zip (ex [off:], ac [off:]):
            el = e.strip ().split ()
            al = a.strip ().split ()
            assert el [:2] == al [:2]
            for ef, af in zip (el, al):
                eff = float (ef)
                aff = float (af)
                if eff > -200:
                    if may_fail_last_digit:
                        m = min (len (ef), len (af))
                        assert ef [:m-1] == af [:m-1]
                    else:
                        assert ef == af
                else:
                    assert aff < -200
    # end def compare_far_field_data

    def compare_currents (self, ex, ac):
        """ This is used when the CURRENT line in the feed pulse is
            slightly off on different architectures. In addition the
            individual CURRENT DATA lines are also compared
            approximately. We get lists of lines already split.
        """
        off = ex.index (self.source_data) + 2
        assert ex [:off] == ac [:off]
        exc = ex [off].strip ()
        acc = ac [off].strip ()
        assert acc.startswith ('CURRENT = (')
        assert acc.endswith (' J)')
        exc = [float (x) for x in exc [11:-3].split (',')]
        acc = [float (x) for x in acc [11:-3].split (',')]
        assert np.allclose (exc, acc, atol=1e-12)
        off += 1
        state = 0
        for l_ex, l_ac in zip (ex [off:], ac [off:]):
            if state == 0 or state == 2:
                assert l_ex == l_ac
            if '(DEGREES)' in l_ex:
                state = 1
                continue
            if state == 1:
                sp_ex = l_ex.strip ().split ()
                sp_ac = l_ac.strip ().split ()
                if len (sp_ex) != 5:
                    state = 2
                    assert l_ex == l_ac
                    continue
                assert sp_ex [0] == sp_ac [0]
                num_ex = [float (x) for x in sp_ex [1:]]
                num_ac = [float (x) for x in sp_ac [1:]]
                for a, b in zip (num_ex [:-1], num_ac [:-1]):
                    assert round (abs (a - b), 11) == 0
                assert round (abs (num_ex [-1] - num_ac [-1]), 10) == 0
                if sp_ex [0] == 'E':
                    state = 2
                    continue
    # end def compare_currents

    def compare_near_field_data (self, m, opts = None):
        """ Near field data below absolute values of 1e-15 may be
            different on different architectures
        """
        if opts is None:
            opts = set (('near-field',))
        ex  = self.expected_output.rstrip ().split ('\n')
        ac  = m.as_mininec (opts).rstrip ().split ('\n')
        idx = self.expected_output.find ('FIELD POINT: X')
        l   = len (self.expected_output [:idx].split ('\n'))
        off = l - 2
        self.compare_currents (ex [:off], ac [:off])
        for e, a in zip (ex [off:], ac [off:]):
            el = e.strip ().split ()
            al = a.strip ().split ()
            if not el:
                assert e == a
                continue
            if el [0] in 'XYZ':
                fe = [float (x) for x in el [1:]]
                fa = [float (x) for x in al [1:]]
                found = False
                for n, (ffe, ffa) in enumerate (zip (fe, fa)):
                    # Don't compare angle if below threshold
                    if found and n == 3:
                        continue
                    if ffe != 0 and abs (ffe) < 1e-15:
                        assert abs (ffa) < 1e-15
                        found = True
                    else:
                        assert ffe == ffa
            else:
                assert e == a
    # end def compare_near_field_data

    def setup_generic_file (self, basename, azi = None, ele = None):
        path = os.path.join ('test', basename)
        pym  = path + '.pym'
        args = []
        with open (pym, 'r') as f:
            for line in f:
                if line.startswith ('#'):
                    continue
                args.append (line)
        args = ' '.join (args)
        args = args.split ()
        m    = main (args, return_mininec = True)
        self.simple_setup (basename + '.pout')
        elevation = ele or Angle (0, 10, 10)
        azimuth   = azi or Angle (0, 10, 37)
        m.compute ()
        m.compute_far_field (elevation, azimuth)
        return m
    # end def setup_generic_file

# end class _Test_Base_With_File

class Test_Case_Known_Structure (_Test_Base_With_File):
    # Constant delimiter lines in text output
    st = '*' * 20
    sp = ' ' *  4
    current_data = st + sp + 'CURRENT DATA' + sp + st
    far_field    = st + sp + ' FAR FIELD  ' + sp + st
    source_data  = st + sp + 'SOURCE DATA ' + sp + st

    def test_excitation (self):
        """ Test error cases
        """
        with pytest.raises (ValueError):
            Excitation (1+1j, 1)
        w = []
        w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        m = Mininec (7, w)
        x = Excitation (cvolt = 5)
        # Pulse index must be > 0
        with pytest.raises (ValueError):
            m.register_source (x, -1)
        # Invalid pulse
        with pytest.raises (ValueError):
            m.register_source (x, 55)
        # Invalid pulse for wire
        with pytest.raises (ValueError):
            m.register_source (x, 11, 0)
        # Invalid *first* pulse:
        # We create two 1-seg wires, the first will have no pulse
        w = []
        w.append (Wire (1, 0, 0, 0, 1, 0, 0, 0.01))
        w.append (Wire (1, 1, 0, 0, 2, 0, 0, 0.01))
        m = Mininec (7, w)
        with pytest.raises (ValueError):
            m.register_source (x, 0, 0)
    # end def test_excitation

    def test_load (self):
        """ Test error case
        """
        with pytest.raises (ValueError):
            Laplace_Load ([], [])
    # end def test_load

    def test_medium (self):
        """ Test error cases
        """
        with pytest.raises (ValueError):
            Medium (0, 0, nradials = 1)
        with pytest.raises (ValueError):
            Medium (0, 0, height = 1)
        with pytest.raises (ValueError):
            Medium (1, 1, nradials = 1)
        with pytest.raises (ValueError):
            Medium (1, 0)
        with pytest.raises (ValueError):
            Medium (1, 1, nradials = 1, coord = 5, dist = 7)
        w = []
        w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        with pytest.raises (ValueError):
            Mininec (7, w, media = [])
        ideal = ideal_ground
        media = [ideal, ideal]
        with pytest.raises (ValueError):
            Mininec (7, w, media = media)
        rad = Medium (1, 1, nradials = 1, radius = 1, dist = 1)
        media = [rad, rad]
        with pytest.raises (ValueError):
            Mininec (7, w, media = media)
        # Radials may not be the only medium
        with pytest.raises (ValueError):
            Mininec (7, w, media = [rad])
    # end def test_medium

    def test_wire (self):
        """ Test error cases
        """
        with pytest.raises (ValueError):
            Wire (7, 1, 1, 1, 2, 2, 2, 0)
        with pytest.raises (ValueError):
            Wire (7, 1, 1, 1, 1, 1, 1, 1)
        wire = Wire (7, 0, 0, 0, 1, 1, 0, 0.01)
        ideal = ideal_ground
        with pytest.raises (ValueError):
            wire.compute_ground (0, ideal)
        wire = Wire (7, 0, 0, -1, 1, 1, -1, 0.01)
        with pytest.raises (ValueError):
            wire.compute_ground (0, ideal)
    # end def test_excitation

    def test_source_index (self):
        w = []
        w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        # Wrong index: Exceeds valid segments
        s = Excitation (1, 0)
        m = Mininec (7, w)
        with pytest.raises (ValueError):
            m.register_source (s, 10)
    # end def test_source_index

    def test_matrix_fill_vdipole_ideal_ground (self):
        """ This uses rounding to a number of decimal places to compare
            significant digits (approximately)
        """
        mat   = matrix_ideal_ground_vdipole_from_mininec
        ideal = ideal_ground
        m = self.vertical_dipole \
            (wire_dia = 0.01, filename = None, media = [ideal])
        for i in range (len (m.pulses)):
            for j in range (len (m.pulses)):
                f = int (np.log (abs (mat [i][j].real)) / np.log (10))
                assert round (abs (mat [i][j].real - m.Z [i][j].real), 3-f) == 0
                f = int (np.log (abs (mat [i][j].imag)) / np.log (10))
                assert round (abs (mat [i][j].imag - m.Z [i][j].imag), 3-f) == 0
    # end def test_matrix_fill_vdipole_ideal_ground

    def test_matrix_fill_quarter_ideal_ground (self):
        """ This uses rounding to a number of decimal places to compare
            significant digits (approximately)
        """
        mat   = np.array (matrix_ideal_ground_quarter_from_mininec_r) \
              + 1j * np.array (matrix_ideal_ground_quarter_from_mininec_i)
        ideal = [ideal_ground]
        m = self.vertical_quarterwave (filename = None, media = ideal)
        for i in range (len (m.pulses)):
            for j in range (len (m.pulses)):
                f = int (np.log (abs (mat [i][j].real)) / np.log (10))
                assert round (abs (mat [i][j].real - m.Z [i][j].real), 3-f) == 0
                f = int (np.log (abs (mat [i][j].imag)) / np.log (10))
                assert round (abs (mat [i][j].imag - m.Z [i][j].imag), 3-f) == 0
    # end def test_matrix_fill_quarter_ideal_ground

    def test_matrix_fill_quarter_ideal_ground_load (self):
        """ This uses rounding to a number of decimal places to compare
            significant digits (approximately)
        """
        mat   = np.array (matrix_ideal_ground_quarter_l_from_mininec_r) \
              + 1j * np.array (matrix_ideal_ground_quarter_l_from_mininec_i)
        ideal = [ideal_ground]
        l = Laplace_Load (b = (1., 0), a = (0., -2.193644e-9))
        m = self.vertical_quarterwave \
            (filename = None, media = ideal, load = l, dia = 0.002)
        for i in range (len (m.pulses)):
            for j in range (1, len (m.pulses)):
                f = int (np.log (abs (mat [i][j].real)) / np.log (10))
                assert round (abs (mat [i][j].real - m.Z [i][j].real), 3-f) == 0
                f = int (np.log (abs (mat [i][j].imag)) / np.log (10))
                assert round (abs (mat [i][j].imag - m.Z [i][j].imag), 3-f) == 0
    # end def test_matrix_fill_quarter_ideal_ground_load

    def test_matrix_fill_inverted_l (self):
        """ This uses rounding to a number of decimal places to compare
            significant digits (approximately)
        """
        mat   = np.array (matrix_inverted_l_r) \
              + 1j * np.array (matrix_inverted_l_i)
        m = self.inverted_l (filename = None)
        for i in range (len (m.pulses)):
            for j in range (len (m.pulses)):
                f = int (np.log (abs (mat [i][j].real)) / np.log (10))
                assert round (abs (mat [i][j].real - m.Z [i][j].real), 3-f) == 0
                f = int (np.log (abs (mat [i][j].imag)) / np.log (10))
                assert round (abs (mat [i][j].imag - m.Z [i][j].imag), 3-f) == 0
    # end def test_matrix_fill_inverted_l

    def test_matrix_fill_quarter_radials (self):
        """ This uses rounding to a number of decimal places to compare
            significant digits (approximately)
            Note that the matrix is identical to the ideal ground case.
            This is because mininec computes the currents in the wires
            using ideal ground (which in turn make problems with wires
            too near to ground with a parallel component).
        """
        mat   = np.array (matrix_ideal_ground_quarter_from_mininec_r) \
              + 1j * np.array (matrix_ideal_ground_quarter_from_mininec_i)
        m1 = Medium (20, .0303, 0, coord = 5, nradials = 16, radius = 0.001)
        m2 = Medium (5, 0.001, -5)
        media = [m1, m2]
        m = self.vertical_quarterwave \
            (filename = 'vertical-rad.pout', media = media)
        for i in range (len (m.pulses)):
            for j in range (len (m.pulses)):
                f = int (np.log (abs (mat [i][j].real)) / np.log (10))
                assert round (abs (mat [i][j].real - m.Z [i][j].real), 3-f) == 0
                f = int (np.log (abs (mat [i][j].imag)) / np.log (10))
                assert round (abs (mat [i][j].imag - m.Z [i][j].imag), 3-f) == 0
    # end def test_matrix_fill_quarter_radials

    def test_matrix_fill_ohio_example (self):
        """ This uses rounding to a number of decimal places to compare
            significant digits (approximately)
            Note that the matrix is identical to the ideal ground case.
            This is because mininec computes the currents in the wires
            using ideal ground (which in turn make problems with wires
            too near to ground with a parallel component).
        """
        mat = np.array (matrix_ohio_r) + 1j * np.array (matrix_ohio_i)
        nfc = Near_Far_Comparison ()
        m   = nfc.m
        for i in range (len (m.pulses)):
            for j in range (len (m.pulses)):
                f = int (np.log (abs (mat [i][j].real)) / np.log (10))
                assert round (abs (mat [i][j].real - m.Z [i][j].real), 3-f) == 0
                f = int (np.log (abs (mat [i][j].imag)) / np.log (10))
                assert round (abs (mat [i][j].imag - m.Z [i][j].imag), 3-f) == 0
    # end def test_matrix_fill_ohio_example

    def test_dipole_wiredia_01 (self):
        m = self.dipole_7mhz (wire_dia = 0.01, filename = 'dipole-01.pout')
        out = m.as_mininec ()
        assert self.expected_output == out
    # end def test_dipole_wiredia_01

    def test_dipole_wiredia_001 (self):
        m = self.dipole_7mhz (wire_dia = 0.001, filename = 'dipole-001.pout')
        assert self.expected_output == m.as_mininec ()
    # end def test_dipole_wiredia_001

    def test_vdipole_wiredia_01 (self):
        m = self.vertical_dipole (wire_dia = 0.01, filename = 'vdipole-01.pout')
        assert self.expected_output == m.as_mininec ()
    # end def test_vdipole_wiredia_01

    def test_vdipole_wiredia_01_ground (self):
        ideal = [ideal_ground]
        m = self.vertical_dipole \
            (wire_dia = 0.01, filename = 'vdipole-01g0.pout', media = ideal)
        assert self.expected_output == m.as_mininec ()
    # end def test_vdipole_wiredia_01_ground

    def test_vdipole_wiredia_01_ground_loaded (self):
        ideal = [ideal_ground]
        load  = Impedance_Load (2e-6)
        m = self.vertical_dipole \
            ( wire_dia = 0.01
            , filename = 'vdipole-01g0l.pout'
            , media    = ideal
            , load     = load
            )
        # Attach to *all* segments
        assert self.expected_output == m.as_mininec ()
    # end def test_vdipole_wiredia_01_ground_loaded

    def test_vdipole_wiredia_001_ground (self):
        ideal = [ideal_ground]
        m = self.vertical_dipole \
            (wire_dia = 0.001, filename = 'vdipole-001g0.pout', media = ideal)
        assert self.expected_output == m.as_mininec ()
    # end def test_vdipole_wiredia_001_ground

    def test_vdipole_wiredia_01_avg_ground (self):
        """ This test computes different results on Intel vs. AMD
            processors. The gain at 90° theta falls below the -300 dBi
            cutoff point on AMD while it is about -297 dBi on Intel.
            This is probably due to differences in sin/cos and/or
            complex exponentials. The Basic version returns a little
            less than -117 dBi there. Note that this is an error that is
            accumulated over a sum of many contributions of each
            segment.
            We test here that the symmetric structure is equal for every
            phi angle. Then we compare the results to the expected
            result for each angle. For the special case of 90° we assert
            that the result is within the result of the Basic version
            and -999.
        """
        avg = [Medium (13, 0.005)]
        m  = self.vertical_dipole \
            (wire_dia = 0.01, filename = 'vdipole-01gavg.pout', media = avg)
        self.compare_far_field_data (m)
    # end def test_vdipole_wiredia_01_avg_ground

    def test_folded_dipole (self):
        m = self.folded_dipole ('folded-18.pout')
        self.compare_far_field_data \
            (m, may_fail_last_digit = True, do_currents = True)
    # end def test_folded_dipole

    def test_vertical_ideal_ground (self):
        ideal = [ideal_ground]
        m = self.vertical_quarterwave ('vertical-ig.pout', ideal)
        assert self.expected_output == m.as_mininec ()
    # end def test_vertical_ideal_ground

    def test_vertical_ideal_ground_upside_down (self):
        ideal = [ideal_ground]
        m = self.vertical_quarterwave ('vertical-ig-ud.pout', ideal, inv = True)
        assert self.expected_output == m.as_mininec ()
    # end def test_vertical_ideal_ground_upside_down

    def test_vertical_radials (self):
        m1 = Medium (20, .0303, 0, coord = 5, nradials = 16, radius = 0.001)
        m2 = Medium (5, 0.001, -5)
        media = [m1, m2]
        m = self.vertical_quarterwave ('vertical-rad.pout', media = media)
        assert self.expected_output == m.as_mininec ()
    # end def test_vertical_radials

    def test_inverted_l (self):
        m = self.inverted_l ('inv-l.pout')
        assert self.expected_output == m.as_mininec ()
    # end def test_inverted_l

    def test_t_ant (self):
        m = self.t_antenna ('t-ant.pout')
        self.compare_far_field_data (m)
    # end def test_t_ant

    def test_t_ant_thin (self):
        m = self.t_antenna ('t-ant-thin.pout', r = 5e-5)
        self.compare_far_field_data (m)
    # end def test_t_ant_thin

    def test_dipole_wiredia_01_near (self):
        m = self.dipole_7mhz (wire_dia = 0.01, filename = 'dipole-01-near.pout')
        opts = set (('near-field',))
        #with open ('z.out', 'w') as f:
        #    print (m.as_mininec (opts).rstrip (), file = f)
        actual_output = m.as_mininec (opts).rstrip ()
        assert self.expected_output == actual_output
    # end def test_dipole_wiredia_01_near

    def test_vertical_ideal_ground_near (self):
        ideal = [ideal_ground]
        l = Laplace_Load (b = (0., 225.998e-9), a = (1., 0.))
        m = self.vertical_quarterwave \
            ('vertical-ig-near.pout', ideal, dia = 0.002, load = l)
        self.compare_near_field_data (m)
    # end def test_vertical_ideal_ground_near

    def test_vertical_ideal_ground_far_abs (self):
        ideal = [ideal_ground]
        opt = dict (pwr = 100, dist = 1000)
        m = self.vertical_quarterwave \
            ('vertical-ig-ffabs.pout', ideal, opt = opt)
        opts = set (('far-field-absolute',))
        actual_output = m.as_mininec (opts).rstrip ()
        assert self.expected_output == actual_output
    # end def test_vertical_ideal_ground_far_abs

    def test_near_far (self):
        r = []
        nfc = Near_Far_Comparison ()
        r.append (nfc.output_currents ())
        r.append (nfc.output_near ())
        r.append (nfc.output_far  ())
        actual_output = '\n'.join (r).rstrip ()
        with open (os.path.join ('test', 'ohio.pout'), 'r') as f:
            self.expected_output = f.read ().rstrip ()
        assert self.expected_output == actual_output
    # end def test_near_far

    def test_hloop40_14 (self):
        m = self.setup_generic_file ('hloop40-14')
        self.compare_far_field_data (m)
    # end def test_hloop40_14

    def test_hloop40_7 (self):
        m = self.setup_generic_file ('hloop40-7')
        self.compare_far_field_data (m)
    # end def test_hloop40_7

    def test_vloop20 (self):
        m = self.setup_generic_file ('vloop20')
        self.compare_far_field_data (m, may_fail_last_digit = True)
    # end def test_vloop20

    def test_vloop20_t45 (self):
        m = self.setup_generic_file ('vloop20-t45')
        self.compare_far_field_data (m, may_fail_last_digit = True)
    # end def test_vloop20_t45

    def test_lzh20 (self):
        m = self.setup_generic_file ('lzh20')
        self.compare_far_field_data (m)
    # end def test_lzh20

    def test_inve802B (self):
        ele = Angle (0, 11, 9)
        m   = self.setup_generic_file ('inve802B', ele = ele)
        opt = set (['far-field'])
        out =  m.frq_independent_as_mininec ()  + '\n'
        out += m.frq_dependent_as_mininec (opt) + '\n'
        m.f = 14
        m.compute ()
        m.compute_far_field (ele, Angle (0, 10, 37))
        out += m.frq_dependent_as_mininec (opt)
        assert self.expected_output == out
    # end def test_inve802B

# end class Test_Case_Known_Structure

class Test_Doctest:

    flags = doctest.NORMALIZE_WHITESPACE

    def run_test (self, module, n):
        f, t  = doctest.testmod \
            (module, verbose = False, optionflags = self.flags)
        fn = os.path.basename (module.__file__)
        format_ok  = '%(fn)s passes all of %(t)s doc-tests'
        format_nok = '%(fn)s fails %(f)s of %(t)s doc-tests'
        if f:
            msg = format_nok % locals ()
        else:
            msg = format_ok % locals ()
        exp = '%s passes all of %d doc-tests' % (fn, n)
        assert exp == msg
    # end def run_test

    def test_mininec (self):
        num_tests = 382
        self.run_test (mininec.mininec, num_tests)
    # end def test_mininec

    def test_util (self):
        num_tests = 1
        self.run_test (mininec.util, num_tests)
    # end def test_util

# end class Test_Doctest
