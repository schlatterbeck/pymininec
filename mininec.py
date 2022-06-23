#!/usr/bin/python3
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

#!/usr/bin/python3

import sys
import copy
import numpy as np

def format_float (floats, use_e = 0):
    """ Reproduce floating-point formatting of the Basic code
    """
    r = []
    e = 1e-9 # An epsilon to avoid python rounding down on exactly .5
    for f in floats:
        if f == 0:
            fmt = '% .1f'
        else:
            fmt = '%% .%df' % (6 - int (np.log (abs (f)) / np.log (10)))
        if use_e and abs (f) < 1e-1:
            fmt = '%e'
        f += e
        s = fmt % f
        if fmt != '%e':
            if '.' in s:
                s = s [:9]
                s = s.rstrip ('0')
                s = s.rstrip ('.')
                if s.startswith (' 0.') or s.startswith ('-0.'):
                    s = s [0] + s [2:]
                s = '%-9s' % s
        else:
            s = s.upper ()
        r.append (s)
    return tuple (r)
# end def format_float

class Angle:
    """ Represents the stepping of Zenith and Azimuth angles
    """
    def __init__ (self, initial, inc, number):
        self.initial = initial
        self.inc     = inc
        self.number  = number
    # end def __init__

    def iter (self):
        for i in range (self.number):
            angle = self.initial + i * self.inc
            yield (angle, angle / 180 * np.pi)
    # end def iter
# end class Angle

class Excitation:
    """ This is the "PULSE" definition in mininec.
        The idx is the index into the segments (0-based)
        For convenience phase is in degrees (and converted internally)
        Magnitude is in volts
    """
    def __init__ (self, idx, magnitude = None, phase = None, cvolt = None):
        if idx < 0:
            raise ValueError ("Index must be >= 0")
        if  (  magnitude is not None and phase is None
            or phase is not None and magnitude is None
            or magnitude is not None and cvolt is not None
            or phase is not None and cvolt is not None
            ):
            raise ValueError \
                ("Either specify magnitude/phase or complex voltage")

        self.parent    = None
        self.idx       = idx
        if magnitude:
            self.magnitude = magnitude
            self.phase_d   = phase
            self.phase     = phase / 180. * np.pi
            self.voltage   = magnitude * np.e ** (1j * self.phase)
        else:
            self.voltage   = cvolt
            self.magnitude = np.abs (cvolt)
            self.phase     = np.angle (cvolt)
            self.phase_d   = self.phase / np.pi * 180
    # end def __init__

    @property
    def current (self):
        """ Note that this is defined only when the parent has solved
            the impedance matrix.
        """
        return self.parent.current [self.idx]
    # end def current

    @property
    def power (self):
        """ Return only the real part of the power
        """
        return (0.5 * self.voltage * np.conj (self.current)).real
    # end def power

    @property
    def impedance (self):
        return self.voltage / self.current
    # end def impedance

    def as_mininec (self):
        r = []
        r.append \
            ( 'PULSE %2d      VOLTAGE = ( %s , %s J)'
            % ( self.idx + 1
              , format_float ([self.voltage.real]) [0].strip ()
              , format_float ([self.voltage.imag]) [0].strip ()
              )
            )
        r.append \
            ( '%sCURRENT = ( %s , %s J)'
            % ( ' ' * 14
              , format_float ([self.current.real], 1) [0]
              , format_float ([self.current.imag], 1) [0]
              )
            )
        r.append \
            ( '%sIMPEDANCE = ( %s , %s J)'
            % ( ' ' * 14
              , format_float ([self.impedance.real]) [0]
              , format_float ([self.impedance.imag]) [0]
              )
            )
        r.append \
            ( '%sPOWER =  %s  WATTS'
            % ( ' ' * 14
              , format_float ([self.power], 1) [0]
              )
            )
        return '\n'.join (r)
    # end def as_mininec

    def as_mininec_short (self):
        """ Only the input data
        """
        r = ['PULSE NO., VOLTAGE MAGNITUDE, PHASE (DEGREES):']
        r.append \
            ('%2d ,%2d ,%2d' % (self.idx + 1, self.magnitude, self.phase_d))
        return ' '.join (r)
    # end def as_mininec_short

    def register (self, parent):
        assert self.parent is None
        self.parent = parent
    # end def register
# end class Excitation

class Far_Field_Pattern:
    """ Values in dBi and/or V/m for far field
        Angles are in degrees for printing
    """
    def __init__ (self, theta, phi, gain = None, e_theta = None, e_phi = None):
        self.theta    = theta
        self.phi      = phi
        self.gain     = gain
        self.e_theta  = e_theta
        self.e_phi    = e_phi
    # end def __init__
# end def Far_Field_Pattern

class Medium:
    """ This encapsulates the media (e.g. ground screen etc.)
        With diel and cond zero we asume ideal ground.
        Note that it seems only the first medium can have a
        ground screen of radials
    >>> med = Medium (3, 4)
    >>> imp = med.impedance (30)
    >>> print ('%.4f%+.4fj' % (imp.real, imp.imag))
    0.0144+0.0144j
    """
    def __init__ \
        (self, diel, cond, height = 0, nradials = 0, radius = 0, dist = 0):
        self.diel     = diel     # dielectric constant T(I)
        self.cond     = cond     # conductivity V(I)
        self.nradials = nradials # number of radials (NR)
        self.dist     = dist     # radial distance (RD)
        self.radius   = radius   # radial wire radius (RR)
        self.coord    = 1e6      # U(I)
        self.height   = height   # H(I)
        self.is_ideal = False
        if diel == 0 and cond == 0:
            self.is_ideal = True
            self.Z        = 0+0j
            self.coord    = 0
            if self.nradials:
                raise ValueError ("Ideal ground may not use radials")
            if self.height != 0:
                raise ValueError ("Ideal ground must have height 0")
        if self.nradials:
            if self.radius <= 0 or self.dist <= 0:
                raise ValueError ("Radial radius and distance must be >0")
        else:
            self.radius = 0.0
            self.dist   = 0.0
        # Code from Line 628-633 in far field calculation
        if diel:
            if not self.cond:
                raise ValueError \
                    ("Non-ideal ground must have non-zero ground parameters")
    # end def __init__

    def impedance (self, f):
        t = 2 * np.pi * f * 8.85e-6
        return 1 / np.sqrt (self.diel + -1j * self.cond / t)
    # end def impedance

# end class Medium

ideal_ground = Medium (0, 0)

class Wire:
    """ A NEC-like wire
        The original variable names are
        x1, y1, z1, x2, y2, z2 (X1, Y1, Z1, X2, Y2, Z2)
        n_segments (S1)
        wire_len (D)
        seg_len  (S)
        dirs (CA, CB, CG)
    >>> wire = Wire (1, 0, 0, 0, 0, 0, 25, 0.001)
    >>> wire
    Wire [0 0 0]-[ 0  0 25], r=0.001, seg_start=None, seg_end=None
    """
    def __init__ (self, n_segments, x1, y1, z1, x2, y2, z2, r):
        self.n_segments = n_segments
        self.p1 = np.array ([x1, y1, z1])
        self.p2 = np.array ([x2, y2, z2])
        self.r  = r
        if r <= 0:
            raise ValueError ("Radius must be >0")
        diff = self.p2 - self.p1
        if (diff == 0).all ():
            raise ValueError ("Zero length wire: %s %s" % (self.p1, self.p2))
        self.wire_len = np.linalg.norm (diff)
        self.seg_len  = self.wire_len / self.n_segments
        # Original comment: compute direction cosines
        self.dirs = diff / self.wire_len
        self.seg_start = None
        self.seg_end   = None
        self.j2 = [None, None]
    # end def __init__

    def compute_ground (self, n, media):
        self.n = n
        # If we are in free space, nothing to do here
        if media is None:
            return
        # Wire end is grounded if Z coordinate is 0
        # In the original implementation this is kept in J1
        # with: 0: not grounded -1: start grounded 1: end grounded
        self.is_ground_start = (self.p1 [-1] == 0)
        self.is_ground_end   = (self.p2 [-1] == 0)
        if self.is_ground_start and self.is_ground_end:
            raise ValueError ("Both ends of a wire may not be grounded")
        if self.p1 [-1] < 0 or self.p2 [-1] < 0:
            raise ValueError ("height cannot not be negative with ground")
    # end def compute_ground

    def __str__ (self):
        return 'Wire %s-%s, r=%s, seg_start=%s, seg_end=%s' \
            % (self.p1, self.p2, self.r, self.seg_start, self.seg_end)
    __repr__ = __str__

# end class Wire

class Mininec:
    """ A mininec implementation in Python
    >>> w = []
    >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.001))
    >>> s = Excitation (4, cvolt = 1+0j)
    >>> m = Mininec (20, w, [s])
    >>> print (m.wires_as_mininec ())
    NO. OF WIRES: 1
    <BLANKLINE>
    WIRE NO. 1
                COORDINATES                                 END         NO. OF
       X             Y             Z          RADIUS     CONNECTION     SEGMENTS
     0             0             0                           0
     21.41429      0             0             .001          0             10
    <BLANKLINE>
                      **** ANTENNA GEOMETRY ****
    <BLANKLINE>
    WIRE NO.  1  COORDINATES                                CONNECTION PULSE
    X             Y             Z             RADIUS        END1 END2  NO.
     2.141429      0             0             .001           0    1    1
     4.282857      0             0             .001           1    1    2
     6.424286      0             0             .001           1    1    3
     8.565714      0             0             .001           1    1    4
     10.70714      0             0             .001           1    1    5
     12.84857      0             0             .001           1    1    6
     14.99         0             0             .001           1    1    7
     17.13143      0             0             .001           1    1    8
     19.27286      0             0             .001           1    0    9

    """
    # INTRINSIC IMPEDANCE OF FREE SPACE DIVIDED BY 2 PI
    g0 = 29.979221
    # Q-VECTOR FOR GAUSSIAN QUADRATURE
    q = np.array \
        ([ .288675135, .5,         .430568156, .173927423
         , .169990522, .326072577, .480144928, .050614268
         , .398333239, .111190517, .262766205, .156853323
         , .091717321, .181341892
        ])
    # E-VECTOR FOR COEFFICIENTS OF ELLIPTIC INTEGRAL
    # In the code these are C0--C9
    cx = np.array \
        ([ 1.38629436112, .09666344259, .03590092383, .03742563713, .01451196212
         ,  .5,           .12498593397, .06880248576, .0332835346,  .00441787012
        ])
    c = 299.8 # speed of light

    def __init__ (self, f, geo, sources, media = None, boundary = 1):
        """ Initialize, no interactive input is done here
            f:   Frequency in MHz, (F)
            media: sequence of Medium objects, if empty use perfect ground
                   if None (the default) use free space
            sources: List of Excitation objects
            geo: A sequence of Wire objects
            # Obsolete:
            g:   +1 for free space, -1 for groundplane (G)
                 This now uses the media parameter which is
                 - None for free space
                 - empty sequence for perfect ground (we might introduce
                   a perfect ground object at some point)
                 - list of Medium object otherwise
            nm:  Number of media (NM)
                 0: perfectly conducting ground
                 See class Medium above
            boundary:  Type of boundary (1: linear, 2: circular) (TB)
                 only if nm > 1
                 See class Medium above, if we have radials it's
                 circular
            Computed:
            w:   Wavelength in m, (W)
            s0:  virtual dipole lenght for near field calculation (S0)
            m:   ?  Comment: 1 / (4 * PI * OMEGA * EPSILON)
            srm: SMALL RADIUS MODIFICATION CONDITION
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.001))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])
        >>> print (m.seg_idx)
        [[1 1]
         [1 1]
         [1 1]
         [1 1]
         [1 1]
         [1 1]
         [1 1]
         [1 1]
         [1 1]]
        """
        self.f        = f
        self.media    = media
        self.boundary = boundary
        self.sources  = sources
        self.geo      = geo
        self.wavelen  = w = 299.8 / f
        if not self.media or len (self.media) == 1:
            self.boundary = 1
        for s in self.sources:
            s.register (self)
        # virtual dipole length for near field calculation:
        self.s0      = .001 * w
        # 1 / (4 * PI * OMEGA * EPSILON)
        self.m       = 4.77783352 * w
        # set small radius modification condition:
        self.srm     = .0001 * w
        self.w       = 2 * np.pi / w
        self.w2      = self.w ** 2 / 2
        self.flg     = 0
        self.check_ground ()
        self.check_geo ()
        self.compute_connectivity ()
        # Check source indeces
        for n, s in enumerate (self.sources):
            if s.idx >= len (self.c_per):
                raise ValueError \
                    ( "Index %d of source %d exceeds segments (%d)"
                    % (s.idx, n + 1, len (self.c_per))
                    )
        self.output_date = False
    # end __init__

    def check_geo (self):
        for n, wire in enumerate (self.geo):
            wire.compute_ground (n, self.media)
    # end def check_geo

    def check_ground (self):
        if not self.media:
            if self.media is not None:
                raise ValueError ("Media must be None for free space")
            return
        for n, g in enumerate (self.media):
            if g.is_ideal and len (self.media) != 1:
                raise ValueError ("Ideal ground must be the only medium")
            if g.nradials and n != 0:
                raise ValueError ("Medium with radials must be first")
    # end def check_ground

    def compute (self):
        """ Compute the currents (solution of the impedance matrix)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])
        >>> m.compute ()
        >>> print (m.sources_as_mininec ())
        NO. OF SOURCES :  1
        PULSE NO., VOLTAGE MAGNITUDE, PHASE (DEGREES):  5 , 1 , 0
        >>> print (m.source_data_as_mininec ())
        ********************    SOURCE DATA     ********************
        PULSE  5      VOLTAGE = ( 1 , 0 J)
                      CURRENT = ( 1.006964E-02 , -5.166079E-03 J)
                      IMPEDANCE = ( 78.61622 , 40.33289 J)
                      POWER =  5.034822E-03  WATTS
        """
        self.compute_impedance_matrix ()
        #self.compute_impedance_matrix_loads ()
        self.compute_rhs ()
        self.current = np.linalg.solve (self.Z, self.rhs)
        # Used by far field calculation
        self.power = sum (s.power for s in self.sources)
    # end def compute

    def compute_connectivity (self):
        """ In the original code this is done while parsing the
            individual wires from the geometry information.

            The vector N in the original code holds the index of the
            start and end segments for each wire. We store this as
            seg_start and seg_end directly into the Wire object.
            We also make sure the indeces are 0-based not 1-based.
            There is a special case if the wire has only one segment. In
            that case the original code put a 0 into N(X,1) (the start
            segment) if it is the first wire. Otherwise N(X,2) (the end
            segment) is set to 0. Since we do 0-based indexing we store
            None in that case into the start segment. The idea is
            probably to not count segments twice.

            Index computation is 1-based (!), so we increment the
            running index i (which is 0-based) by one where it matters.

            The original code first computes the connectivity of the
            wires with other wires and with ground. Ground computation
            is done directly in the Wire object now.

            Then the segments are computed.
            In the original code this starts on line 1198 with the
            comment "compute connectivity data (pulses N1 to N)"
            The variable seg replaces the X, Y, Z arrays in the original
            code, it has consequently dimension 3.
            Looks like index n1 is the index of the next start segment.
            And n is the index of the next end segment.
        """
        n = 0
        self.c_per = c_per = {}
        self.w_per = w_per = {}
        self.seg   = seg   = {}
        for i, w in enumerate (self.geo):
            # This part starts at 1298 comment "connections"
            # We do not use the E, L, M array with the X, Y, Z
            # coordinates of the start of the wires and the second half
            # with the end of the wires, we use the Wire objects
            # instead.
            gflag = False
            i1 = i2 = 0
            w.j2 [:] = (-(i + 1), -(i + 1))
            # check for ground connection
            if self.media:
                if w.is_ground_start:
                    i1   = -(i + 1)
                if w.is_ground_end:
                    i2   = -(i + 1)
                    # FIXME: Why is the gflag only set for the second
                    #        ground case??
                    gflag = True
            for j in range (i):
                # Check start -> start
                # Original comment: check for end1 to end1
                if (w.p1 == self.geo [j].p1).all ():
                    #print ("p1 p1 %d %d" % (i, j))
                    i1 = -(j + 1)
                    w.j2 [0] = j + 1
                    if self.geo [j].j2 [0] == -(j + 1):
                        self.geo [j].j2 [0] = j + 1
                    break
                # Check start -> end
                # Original comment: check for end1 to end2
                if (w.p1 == self.geo [j].p2).all ():
                    #print ("p1 p2 %d %d" % (i, j))
                    i1 = (j + 1)
                    w.j2 [0] = j + 1
                    if self.geo [j].j2 [1] == -(j + 1):
                        self.geo [j].j2 [1] = j + 1
                    break
            if not gflag:
                for j in range (i):
                    # Check end -> end
                    # Original comment: check end2 to end2
                    if (w.p2 == self.geo [j].p2).all ():
                        #print ("p2 p2 %d %d" % (i, j))
                        i2 = -(j + 1)
                        w.j2 [1] = j + 1
                        if self.geo [j].j2 [1] == -(j + 1):
                            self.geo [j].j2 [1] = j + 1
                        break
                    # Check end -> start
                    # Original comment: check for end2 to end1
                    if (w.p2 == self.geo [j].p1).all ():
                        #print ("p2 p1 %d %d" % (i, j))
                        i2 = (j + 1)
                        w.j2 [1] = j + 1
                        if self.geo [j].j2 [0] == -(j + 1):
                            self.geo [j].j2 [0] = j + 1
                        break
            # Here we used to print the geometry, we do this separately.

            # This part starts at 1198
            # compute connectivity data (pulses n1 to n)
            # We make seg_start/seg_end 0-based and use None instead of 0
            n1 = n + 1
            self.geo [i].seg_start = n1 - 1
            if w.n_segments == 1 and i1 == 0:
                self.geo [i].seg_start = None
            n = n1 + w.n_segments
            if i1 == 0:
                n = n - 1
            if i2 == 0:
                n = n - 1
            self.geo [i].seg_end = n - 1
            if w.n_segments == 1 and i2 == 0:
                self.geo [i].seg_end = None
            # This used to be a Goto 1247 with comment
            # single segmen 0 pulse case
            if n < n1:
                i1 = n1 + 2 * i
                seg [i1] = w.p1
                i1 += 1
                seg [i1] = w.p2
                continue
            for j in range (n1, n + 1):
                c_per [(j, 1)] = i + 1
                c_per [(j, 2)] = i + 1
                w_per [j] = i + 1
            c_per [(n1, 1)] = i1
            c_per [(n, 2)]  = i2
            # Here comment says "compute coordinates of break points"
            i1 = n1 + 2 * i
            i3 = i1
            seg [i1] = w.p1
            # This used to be an inverse comparison with a goto 1230
            if c_per [(n1, 1)] != 0:
                i2 = abs (c_per [(n1, 1)])
                # We compute a vector f3 here to special-case the 3rd element
                f3 = ( np.ones (3)
                     * np.sign (c_per [(n1, 1)])
                     * self.geo [i2].seg_len
                     )
                if c_per [(n1, 1)] == -(i + 1):
                    f3 [-1] = -f3 [-1]
                seg [i1] = seg [i1] - f3 * self.geo [i2].dirs
                i3 += 1

            i6 = n + 2 * (i + 1)
            for i4 in range (i1 + 1, i6 + 1):
                j = i4 - i3
                seg [i4] = w.p1 + j * w.dirs * w.seg_len
            # This used to be an inverse comparison with a goto 1245
            if c_per [(n, 2)] != 0:
                i2 = abs (c_per [(n, 2)])
                # We compute a vector f3 here to special-case the 3rd element
                f3 = ( np.ones (3)
                     * np.sign (c_per [(n, 2)])
                     * self.geo [i2].seg_len
                     )
                i3 = i6 - 1
                if i + 1 == -c_per [(n, 2)]:
                    f3 [-1] = -f3 [-1]
                seg [i6] = seg [i3] + f3 * self.geo [i2].dirs
        self.w_per = np.array ([w_per [i] for i in sorted (w_per)])
        c_iter = iter (sorted (c_per))
        self.c_per = np.array \
            ([[c_per [k1], c_per [k2]] for k1, k2 in zip (c_iter, c_iter)])
        self.seg = np.array ([self.seg [i] for i in sorted (self.seg)])
        # This fills the 0-values in c_per with values from w_per
        # See code in lines 1282, 1283, a side-effect hidden in the
        # print routine :-(
        # Note that we keep c_per for printing (it has 0 entries)
        self.seg_idx = copy.deepcopy (self.c_per)
        # Fill 0s with values from w_per
        for idx in range (2):
            cnull = self.c_per.T [idx] == 0
            self.seg_idx.T [idx][cnull] = self.w_per [cnull]
        # make indeces 0-based
        self.w_per   -= 1
    # end def compute_connectivity

    def compute_far_field (self, zenith_angle, azimuth_angle):
        """ Compute far field
            Original code starts at 620 (and is called at 621)
            The angles are instances of Angle.
        """
        # For now ground calculation is skipped, see lines 624-633
        # We also only use calculation in dBi for now, see 641-654
        # Note that the volts/meter calculation asks about the power and
        # about the radial distance (?)
        # 685 has code to print radials, only used for volts/meter
        # Original vars:
        # AA: Azimuth angle initial
        # AC: Azimuth increment
        # NA: Azimuth number
        # ZA: Zenith angle initial
        # ZC: Zenith increment
        # ZA: Zenith number
        # X1, Y1, Z1: vec real
        # X2, Y2, Z2: vec imag
        self.far_field_angles = (zenith_angle, azimuth_angle)
        k9 = .016678 / self.power
        self.far_field_by_angle = {}
        for azi_d, azi in azimuth_angle.iter ():
            # exchange real/imag
            v12 = np.conj (np.e ** (1j * azi) * -1j)
            for zen_d, zen in zenith_angle.iter ():
                # Can we somehow reduce this with complex artithmetics?
                # What is/was the intention of that code?
                rt3  = np.e ** (1j * zen)
                rt1  = -rt3.imag * v12.imag + 1j * (rt3.real * v12.imag)
                rt2  =  rt3.imag * v12.real - 1j * (rt3.real * v12.real)
                rvec = np.array ([rt1, rt2, rt3])
                vec  = np.zeros (3, dtype = complex)
                for k in self.image_iter ():
                    kvec  = np.array ([1, 1, k])
                    kvec2 = np.array ([k, k, 1])
                    for i in range (len (self.w_per)):
                        s_x = self.seg_idx [i]
                        # Code at 716, 717
                        if k <= 0 and s_x [0] == -s_x [1]:
                            continue
                        j = 2 * self.w_per [i] + i + 1
                        # for each end of pulse compute
                        # a contribution to e-field
                        # End of this loop (goto for continue) is 812
                        for f5 in range (2):
                            l = abs (s_x [f5]) - 1
                            wire = self.geo [l]
                            f3 = np.sign (s_x [f5]) * self.w * wire.seg_len / 2
                            # Line 723, 724
                            if (s_x [0] == s_x [1] and f3 < 0):
                                continue # f5
                            # Standard case (condition Line 725, 726)
                            if  (  k == 1
                                or not self.media
                                or self.media [0].is_ideal
                                ):
                                seg = self.seg [j]
                                s2  = self.w * sum (seg * rvec.real * kvec)
                                s   = np.conj (np.e ** (1j * s2))
                                b   = f3 * s * self.current [i]
                                # Line 733
                                if s_x [0] == -s_x [1]:
                                    # grounded ends, only update last axis
                                    v = np.array ([0, 0, 1])
                                    vec += 2 * b * wire.dirs * v
                                    continue
                                vec += kvec2 * b * wire.dirs
                                continue
                            else: # real ground case (Line 747)
                                nr = 0
                                rr = 0
                                if self.media:
                                    med = self.media [0]
                                    nr = med.nradials
                                    rr = med.radius
                                # begin by finding specular distance
                                t4 = 1e5
                                if rt3.real != 0:
                                    t4 = -self.seg [j][2] * rt3.imag / rt3.real
                                b9 = t4 * v12.imag + self.seg [j][0]
                                if self.boundary != 1:
                                    # Hmm this is pythagoras?
                                    b9 *= b9
                                    b9 += (self.seg [j][1] - t4 * v1) ** 2
                                    b9 = np.sqrt (b9)
                                # search for the corresponding medium
                                j2 = len (self.media)
                                if self.media:
                                    coord = [m.coord for m in self.media]
                                    j2 = np.argmin (b9 - coord)
                                # obtain impedance at specular point
                                # FIXME: It's unclear if loop above
                                # always finds a medium, later
                                # comparison of j2 suggests not
                                idx = max (j2, len (self.media) - 1)
                                z45 = self.media [idx].impedance (self.f)
                                # Line 764, 765
                                if nr != 0 or b9 > coord [0]:
                                    prod = nr * rr
                                    r = b9 + prod
                                    z8  = self.w * r * np.log (r / prod) / nr
                                    s89 = z45 * z8 * 1j
                                    t89 = z45 + (z8 * 1j)
                                    d   = t89.real ** 2 + t89.imag ** 2
                                    z45 = s89 * np.conj (t89) / d
                                    # form SQR(1-Z^2*SIN^2)
                                    w67 = np.sqrt (z45 ** 2 * rt3.imag ** 2)
                                    # vertical reflection coefficient
                                    s89 = rt3.real - w67 * z45
                                    t89 = rt3.real + w67 * z45
                                    d   = t89.real ** 2 + t89.imag ** 2
                                    v89 = s89 * np.conj (t89) / d
                                    # horizontal reflection coefficient
                                    s89 = w67 - rt3.real * z45
                                    t89 = w67 + rt3.real * z45
                                    d   = t89.real ** 2 + t89.imag ** 2
                                    h89 = s89 * np.conj (t89) / d - v89
                                    # compute contribution to sum
                                    h   = 0
                                    if self.media and j2 < len (self.media):
                                        h = self.media [j2].height
                                    seg = self.seg [j]
                                    hv  = np.array ([0, 0, 2 * h])
                                    s2  = self.w * sum ((seg - h) * rvec)
                                    s   = np.conj (np.e ** (1j * s2))
                                    b   = f3 * s * self.current [i]
                                    w67 = v12 * v89
                                    w   = self.geo [l]
                                    d   = v12 * (w.dirs [0] + 1j * w.dirs [1])
                                    z67 = d * b * h89
                                    tm1 = np.array \
                                        ([         v12.real * z67.real
                                           + 1j * (v12.real * z67.imag)
                                         ,         v12.imag * z67.real
                                           + 1j * (v12.imag * z67.imag)
                                         , 1
                                        ])
                                    tm2 = np.array ([-1, -1, 1])
                                    vec += (w.dirs * w67 + tm1) * tm2
                h12 = sum (vec * rvec.imag) * self.g0 * -1j
                vv  = np.array ([v12.real, v12.imag])
                x34 = sum (vec [:2] * vv) * self.g0 * -1j
                rd = 0
                if self.media:
                    rd = self.media [0].dist
                # pattern in dBi
                p123 = np.ones (3) * -999
                t1 = k9 * (h12.real ** 2 + h12.imag ** 2)
                t2 = k9 * (x34.real ** 2 + x34.imag ** 2)
                t3 = t1 + t2
                # calculate values in dBi
                t123 = np.array ([t1, t2, t3])
                cond = t123 > 1e-30
                p123 [cond] = np.log (t123 [cond]) / np.log (10) * 10
                if rd != 0:
                    h12 /= rd
                    x34 /= rd
                # pattern in volts/meter
                # magnitude and phase of e(theta)
                s1 = np.abs   (h12)
                s2 = np.angle (h12)
                # magnitude and phase of e(phi)
                s3 = np.abs   (x34)
                s4 = np.angle (x34)
                self.far_field_by_angle [(zen_d, azi_d)] = \
                    Far_Field_Pattern (zen_d, azi_d, p123, [s1, s2], [s3, s4])
    # end def compute_far_field

    def compute_impedance_matrix (self):
        """ This starts at line 195 (with entry-point for gosub at 196)
            in the original basic code.
            Note that we're using seg_idx instead of c_per
        >>> s  = Excitation (4, 1, 0)
        >>> s2 = Excitation (4, 1, 30)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> m = Mininec (7, w, [s2])
        >>> m.compute_impedance_matrix ()
        >>> n = len (m.w_per)
        >>> for i in range (n):
        ...     print ("row %d" % (i+1))
        ...     for j in range (n):
        ...        print ('%s%s%sj'
        ...              % ( format_float ([m.Z [i][j].real], 1) [0].strip ()
        ...                , '++-' [int (np.sign (m.Z [i][j].imag))]
        ...                , format_float
        ...                   ([abs (m.Z [i][j].imag)], 1) [0].strip ()
        ...                )
        ...              )
        row 1
        -8.485114-9.668585E-03j
        4.240447-9.572989E-03j
        .214366-9.290246E-03j
        5.390034E-02-8.832276E-03j
        2.352013E-02-8.218219E-03j
        1.223068E-02-7.473395E-03j
        6.605724E-03-6.627950E-03j
        3.326365E-03-5.715208E-03j
        1.258103E-03-4.769989E-03j
        row 2
        4.240447-9.572989E-03j
        -8.485114-9.668585E-03j
        4.240447-9.572989E-03j
        .214366-9.290246E-03j
        5.390034E-02-8.832276E-03j
        2.352013E-02-8.218219E-03j
        1.223068E-02-7.473395E-03j
        6.605724E-03-6.627950E-03j
        3.326365E-03-5.715208E-03j
        row 3
        .214366-9.290246E-03j
        4.240447-9.572989E-03j
        -8.485114-9.668585E-03j
        4.240447-9.572989E-03j
        .214366-9.290246E-03j
        5.390034E-02-8.832276E-03j
        2.352013E-02-8.218219E-03j
        1.223068E-02-7.473395E-03j
        6.605724E-03-6.627950E-03j
        row 4
        5.390034E-02-8.832276E-03j
        .214366-9.290246E-03j
        4.240447-9.572989E-03j
        -8.485114-9.668585E-03j
        4.240447-9.572989E-03j
        .214366-9.290246E-03j
        5.390034E-02-8.832276E-03j
        2.352013E-02-8.218219E-03j
        1.223068E-02-7.473395E-03j
        row 5
        2.352013E-02-8.218219E-03j
        5.390034E-02-8.832276E-03j
        .214366-9.290246E-03j
        4.240447-9.572989E-03j
        -8.485114-9.668585E-03j
        4.240447-9.572989E-03j
        .214366-9.290246E-03j
        5.390034E-02-8.832276E-03j
        2.352013E-02-8.218219E-03j
        row 6
        1.223068E-02-7.473395E-03j
        2.352013E-02-8.218219E-03j
        5.390034E-02-8.832276E-03j
        .214366-9.290246E-03j
        4.240447-9.572989E-03j
        -8.485114-9.668585E-03j
        4.240447-9.572989E-03j
        .214366-9.290246E-03j
        5.390034E-02-8.832276E-03j
        row 7
        6.605724E-03-6.627950E-03j
        1.223068E-02-7.473395E-03j
        2.352013E-02-8.218219E-03j
        5.390034E-02-8.832276E-03j
        .214366-9.290246E-03j
        4.240447-9.572989E-03j
        -8.485114-9.668585E-03j
        4.240447-9.572989E-03j
        .214366-9.290246E-03j
        row 8
        3.326365E-03-5.715208E-03j
        6.605724E-03-6.627950E-03j
        1.223068E-02-7.473395E-03j
        2.352013E-02-8.218219E-03j
        5.390034E-02-8.832276E-03j
        .214366-9.290246E-03j
        4.240447-9.572989E-03j
        -8.485114-9.668585E-03j
        4.240447-9.572989E-03j
        row 9
        1.258103E-03-4.769989E-03j
        3.326365E-03-5.715208E-03j
        6.605724E-03-6.627950E-03j
        1.223068E-02-7.473395E-03j
        2.352013E-02-8.218219E-03j
        5.390034E-02-8.832276E-03j
        .214366-9.290246E-03j
        4.240447-9.572989E-03j
        -8.485114-9.668585E-03j
        >>> m.compute_rhs ()
        >>> for r in m.rhs:
        ...     sgn = '++-' [int (np.sign (r.imag))]
        ...     print ('%.8f%s%.8fj' % (r.real, sgn, abs (r.imag)))
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00244346-0.00423220j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        >>> m.sources [0] = s
        >>> m.compute_rhs ()
        >>> for r in m.rhs:
        ...     sgn = '++-' [int (np.sign (r.imag))]
        ...     print ('%.8f%s%.8fj' % (r.real, sgn, abs (r.imag)))
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000-0.00488692j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        0.00000000+0.00000000j
        """
        n    = len (self.w_per)
        s    = np.array ([w.seg_len for w in self.geo])
        self.Z = np.zeros ((n, n), dtype=complex)
        # Since seg_idx is 1-based, we subtract 1 to make i1v/i2v 0-based
        i1v  = np.abs (self.seg_idx.T [0]) - 1
        i2v  = np.abs (self.seg_idx.T [1]) - 1
        f4   = np.sign (self.seg_idx.T [0]) * s [i1v]
        f5   = np.sign (self.seg_idx.T [1]) * s [i2v]
        d    = np.array  ([w.dirs for w in self.geo])
        # The t567 matrix replaces vectors T5, T6, T7
        t567 = \
            ( np.tile (f4, (3, 1)).T * d [i1v]
            + np.tile (f5, (3, 1)).T * d [i2v]
            )
        # Compute the special case in line 220 in one go
        ix = self.seg_idx.T [0] == -self.seg_idx.T [1]
        t567.T [-1][ix] = (s [i1v] * (d.T [-1][i1v] + d.T [-1][i2v])) [ix]
        # Instead of I1, I2 use i1v [i], i2v [i]
        for i in range (n):
            # Instead of J1, J2 use i1v [j], i2v [j]
            for j in range (n):
                f6 = f7 = 1
                # 230
                for k in self.image_iter ():
                    if self.seg_idx [j][0] == -self.seg_idx [j][1]:
                        if k < 0:
                            continue
                        f6 = np.sign (self.seg_idx [j][0])
                        f7 = np.sign (self.seg_idx [j][1])
                    f8 = 0
                    if k >= 0:
                        di = self.geo [i1v [i]].dirs
                        dj = self.geo [i1v [j]].dirs
                        # set flag to avoid redundant calculations
                        if  (   i1v [i] == i2v [i]
                            and (  di [0] + di [1] == 0
                                or self.seg_idx [i][0] == self.seg_idx [i][1]
                                )
                            and i1v [j] == i2v [j]
                            and (  dj [0] + dj [1] == 0
                                or self.seg_idx [j][0] == self.seg_idx [j][1]
                                )
                            ):
                            if i1v [i] == i1v [j]:
                                f8 = 1
                            if i == j:
                                f8 = 2
                    # This was a conditional goto 317 in line 246
                    if k < 0 or self.Z [i][j].real == 0:
                        p1 = 2 * self.w_per [i] + i + 1
                        p2 = 2 * self.w_per [j] + j + 1
                        p3 = p2 + 0.5
                        p4 = i2v [j]
                        vp = self.vector_potential (k, p1, p2, p3, p4, i, j)
                        u = vp * np.sign (self.seg_idx [j][1])
                        # compute PSI(M,N-1/2,N)
                        p3 = p2
                        p2 -= 0.5
                        p4 = i1v [j]
                        if f8 < 2:
                            vp = self.vector_potential (k, p1, p2, p3, p4, i, j)
                        v = vp * np.sign (self.seg_idx [j][0])
                        # S(N+1/2)*PSI(M,N,N+1/2) + S(N-1/2)*PSI(M,N-1/2,N)
                        f7v  = np.array ([1, 1, f7])
                        f6v  = np.array ([1, 1, f6])
                        di1  = self.geo [i1v [j]].dirs
                        di2  = self.geo [i2v [j]].dirs
                        kvec = np.array ([1, 1, k])
                        # We do real and imaginary part in one go:
                        vec3 = f7v * u * di2 + f6v * v * di1 * kvec
                        d    = self.w2 * sum (vec3 * t567 [i])
                        # compute PSI(M+1/2,N,N+1)
                        p1 += 0.5
                        if f8 == 2:
                            p1 -= 1
                        p2 = p3
                        p3 += 1
                        p4 = i2v [j]
                        if f8 < 2 or f8 == 1:
                            if f8 == 1:
                                u56 = np.sign (self.seg_idx [j][1]) * u + vp
                            else:
                                u56 = self.scalar_potential \
                                    (k, p1, p2, p3, p4, i, j)
                            # compute PSI(M-1/2,N,N+1)
                            # Code at 291
                            p1 -= 1
                            sp = self.scalar_potential \
                                (k, p1, p2, p3, p4, i, j)
                            seglen = self.geo [i2v [j]].seg_len
                            u12 = (sp - u56) / seglen
                            # compute PSI(M+1/2,N-1,N)
                            p1  += 1
                            p3  = p2
                            p2  -= 1
                            p4  = i1v [j]
                            u34 = self.scalar_potential \
                                (k, p1, p2, p3, p4, i, j)
                            # compute PSI(M-1/2,N-1,N)
                            if f8 >= 1:
                                sp = u56
                            else:
                                p1 -= 1
                                sp = self.scalar_potential \
                                    (k, p1, p2, p3, p4, i, j)
                            # gradient of scalar potential contribution
                            seglen = self.geo [i1v [j]].seg_len
                            u12 += (u34 - sp) / seglen
                        else:
                            sp = self.scalar_potential (k, p1, p2, p3, p4, i, j)
                            seglen = self.geo [i1v [j]].seg_len
                            sg  = np.sign (self.seg_idx [j][1])
                            u12 = (2 * sp - 4 * u * sg) / seglen
                        # 314
                        # sum into impedance matrix
                        self.Z [i][j] = self.Z [i][j] + k * (d + u12)

                    # avoid redundant calculations
                    # 317
                    if j < i or f8 == 0:
                        continue
                    self.Z [j][i] = self.Z [i][j]
                    # segments on same wire same distance apart
                    # have same Z
                    p1 = j + 1
                    if p1 >= n:
                        continue
                    if self.seg_idx [p1][0] != self.seg_idx [p1][1]:
                        continue
                    d = self.geo [i2v [j]].dirs
                    if  (   self.seg_idx [p1, 1] != self.seg_idx [j][1]
                        and (  self.seg_idx [p1, 1] != -self.seg_idx [j][1]
                            or d [0] + d [1] != 0
                            )
                        ):
                        continue
                    p2 = i + 1
                    if p2 > n:
                        continue
                    self.Z [p2][p1] = self.Z [i][j]
            # Here follows a GOSUB 1599 which calculates the remaining time,
            # not implemented
        # end matrix fill time calculation
        # Here follows a GOSUB 1589 which calculates elapsed time,
        # not implemented
        # addition of loads happens in compute_impedance_matrix_loads
    # end def compute_impedance_matrix

    def compute_rhs (self):
        rhs = np.zeros (len (self.c_per), dtype=complex)
        for src in self.sources:
            f2 = -1j/self.m
            if self.seg_idx [src.idx][0] == -self.seg_idx [src.idx][1]:
                f2 = -2j/self.m
            rhs [src.idx] = f2 * src.voltage
        self.rhs = rhs
    # end def compute_rhs

    def image_iter (self):
        """ This replaces the image loop which loops over [1, -1] when a
            ground plane exists, over only [1] otherwise. At some point
            we may want to refactor this.
        """
        if self.media is None:
            return iter ([1])
        return iter ([1, -1])
    # end def image_iter

    def integral_i2_i3 (self, vec2, vecv, k, t, p4, exact_kernel = False) :
        """ Starts line 28
            Uses variables:
            vec2 (originally (X2, Y2, Z2))
            vecv (originally (V1, V2, V3))
            k, t, exact_kernel
            c0 - c9  # Parameter of elliptic integral
            w: 2 * pi * f / c (constant in program)
            srm: small radius modification condition
                 0.0001 * c / f
            a(p4): wire radius
            t3, t4: Integrals I2 and I3 (yes, they *are* named off-by-one)

            Temporary variables:
            d3, d, b, b1, w0, w1, v0, vec3 (originally (X3, Y3, Z3))

            Note when comparing results to the BASIC implementation: The
            basic implementation *adds* its results to the *existing*
            values of t3 and t4!
        # First with thin-wire approximation, doesn't make a difference
        # if we use a exact_kernel or not
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.001))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])
        >>> vv = np.array ([3.21214267693, 0, 0])
        >>> v2 = np.array ([1.07071418591, 0, 0])
        >>> t  = 0.980144947186
        >>> r = m.integral_i2_i3 (v2, vv, 1, t, 0, False)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819959 -0.1414754j
        >>> r = m.integral_i2_i3 (v2, vv, 1, t, 0, True)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819959 -0.1414754j

        # Then a thick wire without exact kernel
        # Original produces
        # 0.2819941 -0.1414753j
        >>> w [0].r = 0.01
        >>> r = m.integral_i2_i3 (v2, vv, 1, t, 0, False)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819941 -0.1414753j

        # Then a thick wire *with* exact kernel
        # Original produces
        # -2.290341 -0.1467051j
        >>> vv = np.array ([ 1.07071418591, 0, 0])
        >>> v2 = np.array ([-1.07071418591, 0, 0])
        >>> t  = 0.4900725
        >>> r = m.integral_i2_i3 (v2, vv, 1, t, 0, True)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -2.2903412 -0.1467052j

        # Original produces
        # -4.219833E-02 -4.820928E-02j
        >>> vv = np.array ([16.06072, 0, 0])
        >>> v2 = np.array ([13.91929, 0, 0])
        >>> t  = 0.7886752
        >>> r = m.integral_i2_i3 (v2, vv, 1, t, 0, False)
        >>> print ("%.8f %.8fj" % (r.real, r.imag))
        -0.04219836 -0.04820921j

        # Original produces
        # -7.783058E-02 -.1079738j
        # But *ADDED TO THE PREVIOUS RESULT*
        >>> vv = np.array ([16.06072, 0, 0])
        >>> v2 = np.array ([13.91929, 0, 0])
        >>> t  = 0.2113249
        >>> r = m.integral_i2_i3 (v2, vv, 1, t, 0, False)
        >>> print ("%.8f %.8fj" % (r.real, r.imag))
        -0.03563231 -0.05976447j
        """
        wire = self.geo [p4]
        t3 = t4 = 0.0
        if k < 0:
            vec3 = vecv + t * (vec2 - vecv)
        else:
            vec3 = vec2 + t * (vecv - vec2)
        d = d3 = np.linalg.norm (vec3)
        # MOD FOR SMALL RADIUS TO WAVELENGTH RATIO
        if wire.r > self.srm:
            # SQUARE OF WIRE RADIUS
            a2 = wire.r * wire.r
            d3 = d3 * d3
            d  = np.sqrt (d3 + a2)
            # CRITERIA FOR USING REDUCED KERNEL
            if exact_kernel:
                (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9) = self.cx
                # EXACT KERNEL CALCULATION WITH ELLIPTIC INTEGRAL
                b = d3 / (d3 + 4 * a2)
                w0 = c0 + b * (c1 + b * (c2 + b * (c3 + b * c4)))
                w1 = c5 + b * (c6 + b * (c7 + b * (c8 + b * c9)))
                v0 = (w0 - w1 * np.log (b)) * np.sqrt (1 - b)
                t3 += ( (v0 + np.log (d3 / (64 * a2)) / 2)
                      / np.pi / wire.r - 1 / d
                      )
        b1 = d * self.w
        # EXP(-J*K*R)/R
        t3 += np.cos (b1) / d
        t4 -= np.sin (b1) / d
        return t3 + t4 * 1j
    #end def integral_i2_i3

    def psi ( self, vec1, vec2, vecv, k, p2, p3, p4, i, j
            , fvs = 0, is_near = False
            ):
        """ Common code for entry points at 56, 87, and 102.
            This code starts at line 135.
            The variable fvs is used to distiguish code path at the end.
            The variable p2 is the index of the segment, seems this can
            be a float in which case the middle of two segs is used.
            The variable p4 is the index of the wire.
            vec1 is the original input vector (X1, Y1, Z1)
            vec2 replaces (X2, Y2, Z2)
            vecv replaces (V1, V2, V3)
            i6: Use reduced kernel if 0, this was I6! (single precision)
                So beware: condition "I6!=0" means variable I6! is == 0
                The exclamation mark is part of the variable name :-(

            Input:
            vec1, vec2, vecv
            k:
            p2:  segment index 1 (0-based)
            p3:  segment index 2 (0-based)
            p4:  wire index (0-based)
            fvs: scalar vs. vector potential
            is_near: This originally tested input C$ for "N" which is
                     the selection of near field compuation
            Note: p2 and p3 are used only as differences, so if both are
                  1-based produces same result as when both are 0-based.

            Output:
            vec2:
            vecv:
            t1:
            t2:

            Temp:
            i4, i5, s4, l, f2, t, d0, d3, i6
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])

        # Original produces:
        # 5.330494 -0.1568644j
        >>> vec1 = np.array ([2.141429, 0, 0])
        >>> vec2 = np.zeros (3)
        >>> vecv = np.array ([1.070714, 0, 0])
        >>> r = m.psi (vec1, vec2, vecv, 1, 1, 1.5, 0, 0, 0)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        5.3304831 -0.1568644j

        # Original produces:
        # -8.333431E-02 -0.1156091j
        >>> vec1 = np.array ([3.212143, 0, 0])
        >>> vec2 = np.array ([13.91929, 0, 0])
        >>> vecv = np.array ([16.06072, 0, 0])
        >>> x = m.psi
        >>> r = x (vec1, vec2, vecv, k=1, p2=8, p3=9, p4=0, i=0, j=8, fvs = 1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -0.0833344 -0.1156090j
        """
        wire = self.geo [p4]
        # MAGNITUDE OF S(U) - S(M)
        d0 = np.linalg.norm (vec2)
        # MAGNITUDE OF S(V) - S(M)
        d3 = np.linalg.norm (vecv)
        # MAGNITUDE OF S(V) - S(U)
        s4 = (p3 - p2) * wire.seg_len
        # ORDER OF INTEGRATION
        # LTH ORDER GAUSSIAN QUADRATURE
        tret = 0+0j
        i6 = 0
        f2 = 1
        l = 7
        t = (d0 + d3) / wire.seg_len
        # CRITERIA FOR EXACT KERNEL
        assert self.w_per [i] >= 0
        assert self.w_per [j] >= 0
        wire_i_j2 = self.geo [self.w_per [i]].j2
        wire_j_j2 = self.geo [self.w_per [j]].j2
        wires_differ = \
            (   wire_i_j2 [0] != wire_j_j2 [0]
            and wire_i_j2 [0] != wire_j_j2 [1]
            and wire_i_j2 [1] != wire_j_j2 [0]
            and wire_i_j2 [1] != wire_j_j2 [1]
            )
        if t > 1.1 or is_near or wires_differ:
            # This starts line 165
            if t > 6:
                l = 3
            if t > 10:
                l = 1
        elif not wires_differ:
            if wire.r <= self.srm:
                if fvs == 1:
                    t1 = 2 * np.log (wire.seg_len / wire.r)
                    t2 = -self.w * wire.seg_len
                else:
                    t1 = np.log (wire.seg_len / wire.r)
                    t2 = -self.w * wire.seg_len / 2
                return t1 + t2 * 1j
            # The following starts line 162
            f2 = 2 * (p3 - p2)
            i6 = (1 - np.log (s4 / f2 / 8 / wire.r)) / np.pi / wire.r
        # The following starts line 167
        i5 = l + l

        # This runs from line 168 and backjump condition is in line 178
        # This really *updated* t3 and t4 *in place* in the gosub for
        # computing the integral (!)
        # Note how the index l is incremented twice below.
        while l < i5:
            ret = self.integral_i2_i3 \
                ( vec2, vecv, k
                , (self.q [l - 1] + .5) / f2
                , p4 = p4
                , exact_kernel = bool (i6)
                )
            ret += self.integral_i2_i3 \
                ( vec2, vecv, k
                , (.5 - self.q [l - 1]) / f2
                , p4 = p4
                , exact_kernel = bool (i6)
                )
            l = l + 1
            tret += ret * self.q [l - 1]
            l = l + 1
        tret = (tret + i6) * s4
        return tret
    # end def psi

    def psi_common_vec1_vecv (self, vec1, k, p2, p3):
        """ Compute vec2 (originally (X2, Y2, Z2))
            and vecv (originally (V1, V2, V3))
            common to scalar and vector potential.
            This originally was an entry point at 113 used by scalar and
            vector potential computation.
            The variable p2 is the index of the segment, seems this can
            be a float in which case the middle of two segs is used.
            Note that this is tested by scalar_potential and
            vector_potential tests above.
            Note that p2, p3 are now 0-based.
        """
        i4 = int (p2)
        seg  = self.seg [i4]
        # S(U)-S(M) GOES IN (X2,Y2,Z2) (this is now vec2)
        kvec = np.ones (3)
        kvec [-1] = k
        if i4 == p2:
            vec2 = k * self.seg [i4] - vec1
        else:
            i5 = i4 + 1
            vec2 = k * (self.seg [i4] + self.seg [i5]) / 2 - vec1
        # S(V)-S(M) GOES IN (V1,V2,V3) (this is now vecv)
        i4 = int (p3)
        if i4 == p3:
            vecv = kvec * self.seg [i4] - vec1
        else:
            i5 = i4 + 1
            vecv = kvec * (self.seg [i4] + self.seg [i5]) / 2 - vec1
        return vec2, vecv
    # end def psi_common_vec1_vecv

    def psi_near_field_56 (self, vec0, vect, k, p1, p2, p3, p4, i, j):
        """ Compute psi used several times during computation of near field
            Original entry point in line 56
            vec0 originally is (X0, Y0, Z0)
            vect originally is (T5, T6, T7)
            Note that p1 is the only non-zero-based variable, it's not
            used as an index but as a factor.
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])

        # Original produces:
        # 0.5496336 -0.3002106j
        >>> vec0 = np.array ([0, -1, -1])
        >>> vect = np.array ([8.565715E-02, 0, 0])
        >>> method = m.psi_near_field_56
        >>> r = method (vec0, vect, k=1, p1=0.5, p2=1, p3=2, p4=0, i=0, j=0)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.5496335 -0.3002106j
        """
        kvec = np.ones (3)
        kvec [-1] = k
        vec1 = vec0 + p1 * vect / 2
        vec2 = vec1 - kvec * self.seg [p2]
        vecv = vec1 - kvec * self.seg [p3]
        return self.psi (vec1, vec2, vecv, k, p2, p3, p4, i, j, is_near = True)
    # end def psi_near_field_56

    def psi_near_field_66 (self, vec0, vec1, k, p2, p3, p4, i, j):
        """ Compute psi used during computation of near field
            Original entry point in line 66
            vec0 originally is (X0, Y0, Z0)
            vec1 originally is (X1, Y1, Z1)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])

        # Original produces:
        # 0.4792338 -0.1544592j
        >>> vec0 = np.array ([0, -1, -1])
        >>> vec1 = np.array ([3.212143, 0, 0])
        >>> method = m.psi_near_field_66
        >>> r = method (vec0, vec1, k=1, p2=0.5, p3=1, p4=0, i=0, j=0)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.4792338 -0.1544592j
        """
        kvec = np.ones (3)
        kvec [-1] = k
        i4 = int (p2)
        i5 = i4 + 1
        vec2 = vec0 - kvec * (self.seg [i4] + self.seg [i5]) / 2
        vecv = vec0 - kvec * self.seg [p3]
        return self.psi (vec1, vec2, vecv, k, p2, p3, p4, i, j, is_near = True)
    # end def psi_near_field_66

    def psi_near_field_75 (self, vec0, vec1, k, p2, p3, p4, i, j):
        """ Compute psi used during computation of near field
            Original entry point in line 75
            vec0 originally is (X0, Y0, Z0)
            vec1 originally is (X1, Y1, Z1)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])

        # Original produces:
        # 0.3218219 -.1519149j
        >>> vec0 = np.array ([0, -1, -1])
        >>> vec1 = np.array ([3.212143, 0, 0])
        >>> method = m.psi_near_field_75
        >>> r = method (vec0, vec1, k=1, p2=1, p3=1.5, p4=0, i=0, j=0)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.3218219 -0.1519149j
        """
        kvec = np.ones (3)
        kvec [-1] = k
        i4 = int (p3)
        i5 = i4 + 1
        vec2 = vec0 - kvec * self.seg [p2]
        vecv = vec0 - kvec * (self.seg [i4] + self.seg [i5]) / 2
        return self.psi (vec1, vec2, vecv, k, p2, p3, p4, i, j, is_near = True)
    # end def psi_near_field_75

    def scalar_potential (self, k, p1, p2, p3, p4, i, j):
        """ Compute scalar potential
            Original entry point in line 87.
            Original comment:
            entries required for impedance matrix calculation
            S(M) goes in (X1,Y1,Z1) for scalar potential
            mod for small radius to wave length ratio

            This *used* to use A(P4), S(P4), where P4 is the index into
            the wire datastructures, A(P4) is the wire radius and S(P4)
            is the segment length of the wire

            Inputs:
            k, p1, p2, p3, p4, i, j
            Note that p1, p2, p3, i, j, p4 are 0-based now.
            accesses self.seg, originally X(I4),Y(I4),Z(I4), X(I5),Y(I5),Z(I5)
            Outputs:
            t1, t2
            Temp:
            i4, i5
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])

        # Original produces:
        # -8.333431E-02 -0.1156091j
        >>> method = m.scalar_potential
        >>> r = method (k=1, p1=1.5, p2=8, p3=9, p4=0, i=0, j=8)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -0.0833344 -0.1156091j
        >>> w [0].r = 0.001
        >>> r = method (k=1, p1=0.5, p2=1, p3=2, p4=0, i=0, j=0)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        1.0497691 -0.3085993j
        """
        wire = self.geo [p4]
        if  (  k < 1
            or wire.r > self.srm
            or p3 == p2 + 1
            or p1 != (p2 + p3) / 2
            ):
            i4 = int (p1)
            i5 = i4 + 1
            vec1 = (self.seg [i4] + self.seg [i5]) / 2
            vec2, vecv = self.psi_common_vec1_vecv (vec1, k, p2, p3)
            return self.psi (vec1, vec2, vecv, k, p2, p3, p4, i, j, fvs = 1)
        t1 = 2 * np.log (wire.seg_len / wire.r)
        t2 = -self.w * wire.seg_len
        return t1 + t2 * 1j
    # end def scalar_potential

    def vector_potential (self, k, p1, p2, p3, p4, i, j):
        """ Compute vector potential
            Original entry point in line 102.
            Original comment:
            S(M) goes in (X1,Y1,Z1) for vector potential
            mod for small radius to wave length ratio

            This *used* to use A(P4), S(P4), where P4 is the index into
            the wire datastructures, A(P4) is the wire radius and S(P4)
            is the segment length of the wire, we still use p4 as the
            wire index.
            The variable p1 is the index of the segment.
            Inputs:
            k, p2, p3, x(p1),y(p1),z(p1)
            Outputs:
            t1, t2
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (4, 1, 0)
        >>> m = Mininec (7, w, [s])

        # Original produces:
        # 0.6747199 -.1555772j
        >>> method = m.vector_potential
        >>> r = method (k=1, p1=1, p2=1.5, p3=2, p4=0, i=0, j=1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.6747199 -0.1555773j
        """
        wire = self.geo [p4]
        if k < 1 or wire.r >= self.srm or (i != j or p3 == p2 + .5):
            vec1 = self.seg [p1]
            vec2, vecv = self.psi_common_vec1_vecv (vec1, k, p2, p3)
            return self.psi (vec1, vec2, vecv, k, p2, p3, p4, i, j, fvs = 0)
        t1 = np.log (wire.seg_len / wire.r)
        t2 = -self.w * wire.seg_len / 2
        return t1 + t2 * 1j
    # end def vector_potential

    # All the *as_mininec methods

    def as_mininec (self):
        r = []
        r.append (self.header_as_mininec ())
        r.append (self.frequency_as_mininec ())
        #r.append (self.environment_as_mininec ())
        r.append (self.wires_as_mininec ())
        r.append ('')
        r.append (self.sources_as_mininec ())
        r.append ('')
        #r.append (self.loads_as_mininec ())
        r.append (self.source_data_as_mininec ())
        r.append (self.far_field_as_mininec ())
        return '\n'.join (r)
    # end def as_mininec

    def far_field_as_mininec (self):
        """ Print far field in mininec format
            This may only be called when compute_far_field has already
            been called before.
        """
        r = []
        zenith, azimuth = self.far_field_angles
        r.append \
            ( 'ZENITH ANGLE : INITIAL,INCREMENT,NUMBER:%3d ,%3d ,%3d'
            % (zenith.initial, zenith.inc, zenith.number)
            )
        r.append \
            ( 'AZIMUTH ANGLE: INITIAL,INCREMENT,NUMBER:%3d ,%3d ,%3d'
            % (azimuth.initial, azimuth.inc, azimuth.number)
            )
        r.append ('')
        r.append ('*' * 20 + '    PATTERN DATA    ' + '*' * 20)
        r.append \
            ( 'ZENITH%sAZIMUTH%sVERTICAL%sHORIZONTAL%sTOTAL'
            % tuple (' ' * x for x in (8, 7, 6, 4))
            )
        r.append \
            ( ' ANGLE%sANGLE%sPATTERN (DB)  PATTERN (DB)  PATTERN (DB)'
            % tuple (' ' * x for x in (9, 8))
            )
        srt = lambda x: (x [1], x [0])
        for zen, azi in sorted (self.far_field_by_angle, key = srt):
            ff = self.far_field_by_angle [(zen, azi)]
            v, h, t = ff.gain
            r.append \
                ( ('%s     ' * 4 + '%s')
                % ( format_float ((zen, azi))
                  + format_float ((v, h, t))
                  )
                )
        return '\n'.join (r)
    # end def far_field_as_mininec

    def frequency_as_mininec (self):
        r = []
        r.append ('FREQUENCY (MHZ): %s' % format_float ([self.f]) [0].strip ())
        r.append \
            ('    WAVE LENGTH =  %s  METERS'
            % format_float ([self.wavelen]) [0].strip ()
            )
        r.append ('')
        return '\n'.join (r)
    # end def frequency_as_mininec

    def header_as_mininec (self):
        r = []
        r.append (' ' * 19 + '*' * 40)
        r.append (' ' * 21 + 'MINI-NUMERICAL ELECTROMAGNETICS CODE')
        r.append (' ' * 35 + 'MININEC')
        fmt = '%m-%d-%y' + ' ' * 14 + '%H:%M%:%S'
        if self.output_date:
            r.append (' ' * 23 + datetime.now ().strftime (fmt))
        r.append (' ' * 19 + '*' * 40)
        r.append ('')
        return '\n'.join (r)
    # end def header_as_mininec

    def sources_as_mininec (self):
        r = []
        r.append ('NO. OF SOURCES : %2d' % len (self.sources))
        for s in self.sources:
            r.append (s.as_mininec_short ())
        return '\n'.join (r)
    # end def sources_as_mininec

    def source_data_as_mininec (self):
        r = []
        r.append ('*' * 20 + '    SOURCE DATA     ' + '*' * 20)
        for s in self.sources:
            r.append (s.as_mininec ())
        return '\n'.join (r)
    # end def source_data_as_mininec

    def wires_as_mininec (self):
        r = []
        r.append ('NO. OF WIRES: %d' % len (self.geo))
        r.append ('')
        for i, wire in enumerate (self.geo):
            r.append ('WIRE NO. %d' % (i + 1))
            r.append \
                ( '%sCOORDINATES%sEND%sNO. OF'
                % (' ' * 12, ' ' * 33, ' ' * 9)
                )
            r.append \
                ( '   X%sY%sZ%sRADIUS%sCONNECTION%sSEGMENTS'
                % (' ' * 13, ' ' * 13, ' ' * 10, ' ' * 5, ' ' * 5)
                )
            l = []
            l.append ((' ' + '%-13s ' * 3) % format_float (wire.p1))
            conn = wire.j2 [0]
            if conn == -(wire.n + 1):
                conn = 0
            l.append ('%s%2d' % (' ' * 13, conn))
            r.append (''.join (l))
            l = []
            l.append ((' ' + '%-13s ' * 3) % format_float (wire.p2))
            l.append ('%-13s' % format_float ([wire.r]))
            conn = wire.j2 [1]
            if conn == -(wire.n + 1):
                conn = 0
            l.append ('%2d%15d' % (conn, wire.n_segments))
            r.append (''.join (l))
        r.append ('')
        r.append (' ' * 18 + '**** ANTENNA GEOMETRY ****')
        for i, wire in enumerate (self.geo):
            r.append ('')
            r.append \
                ( 'WIRE NO.%3d  COORDINATES%sCONNECTION PULSE'
                % (i + 1, ' ' * 32)
                )
            r.append \
                (('%-13s ' * 4 + 'END1 END2  NO.') % ('X', 'Y', 'Z', 'RADIUS'))
            if wire.seg_start is None and wire.seg_end is None:
                r.append \
                    (('%-13s ' * 5 + '%-4s %-4s') % tuple (['-'] * 5 + ['0']))
            for k in range (wire.seg_start, wire.seg_end + 1):
                seg = tuple (self.seg [k + 1])
                l = []
                l.append ((' ' + '%-13s ' * 3) % format_float (seg))
                l.append ('%-12s' % format_float ([wire.r]))
                l.append ('%4d %4d' % tuple (self.c_per [k]))
                l.append ('%4d' % (k + 1))
                r.append (''.join (l))
        return '\n'.join (r)
    # end def wires_as_mininec

# end class Mininec

if __name__ == '__main__':
    stderr = sys.stderr
    from argparse import ArgumentParser
    cmd = ArgumentParser ()
    cmd.add_argument \
        (  '--excitation-segment'
        , help    = "Segment number for excitation, can be specified "
                    "more than once, default is the single segment 5"
        , type    = int
        , action  = 'append'
        , default = []
        , 
        )
    cmd.add_argument \
        ( '--excitation-voltage'
        , help    = "Voltage for excitation, can be specified more than "
                    "once and can be a complex number, "
                    "default is a single source of 1V"
        , type    = complex
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '-f', '--frequency'
        , type    = float
        , default = 7.0
        , help    = 'Frequency in MHz, default=%(default)s'
        )
    cmd.add_argument \
        ( '--radial_count'
        , type    = int
        , default = 0
        , help    = 'Number of radials, default=%(default)s'
        )
    cmd.add_argument \
        ( '--radial_radius'
        , type    = float
        , help    = 'Radius of radial wires'
        )
    cmd.add_argument \
        ( '--radial_dist'
        , type    = float
        , help    = 'Distance of radials'
        )
    cmd.add_argument \
        ( '--medium'
        , help    = "Media (ground), free space if not given, "
                    "specify dielectricum, condition, height, if all are "
                    "zero, ideal ground is asumed, if radials are "
                    "specified they apply to the first ground (which "
                    "cannot be ideal with radials), several media can be "
                    "specified"
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '-w', '--wire'
        , help    = 'Wire definition 8 values delimited with ",":'
                    " Number of segments,"
                    " x,y,z coordinates of wire endpoints plus wire radius, "
                    "can be specified more than once, default is a "
                    "single wire with length 21.414285"
        , action  = 'append'
        , default = []
        )
    args = cmd.parse_args ()
    if not args.wire:
        args.wire = ['10, 0, 0, 0, 21.414285, 0, 0, 0.001']
    if not args.excitation_segment:
        args.excitation_segment = [5]
    if not args.excitation_voltage:
        args.excitation_voltage = [1]
    wires = []
    for n, wire in enumerate (args.wire):
        wparams = wire.strip ().split (',')
        if len (wparams) != 8:
            print \
                ( "Invalid number of parameters for wire %d" % (n + 1)
                , file = stderr
                )
            sys.exit (23)
        try:
            seg = int (wparams [0])
            r = [float (x) for x in wparams [1:]]
        except ValueError as err:
            print ("Invalid wire %d: %s" % (n + 1, str (err)), file = stderr)
            sys.exit (23)
        wires.append (Wire (seg, *r))
    if len (args.excitation_segment) != len (args.excitation_voltage):
        print \
            ("Number of excitation segments must match voltages", file = stderr)
        sys.exit (23)

    exc = []
    for i, v in zip (args.excitation_segment, args.excitation_voltage):
        exc.append (Excitation (i - 1, cvolt = v))
    media = []
    rad = []
    if args.radial_count:
        rad = (args.radial_count, args.radial_radius, args.radial_distance)
    for n, m in enumerate (args.medium):
        p = m.split (',')
        if len (p) != 3:
            print ("Medium needs three parameters", file = stderr)
            sys.exit (23)
        try:
            p = [float (x) for x in p]
        except ValueError:
            print ("Invalid medium %d: %s" % (n + 1, str (err)), file = stderr)
            sys.exit (23)
        r = []
        if n == 0:
            r = rad
        media.append (Medium (*p, *r))
    media = media or None

    m = Mininec (args.frequency, wires, exc, media = media)
    m.compute ()
    # We're in free space
    azimuth = Angle (0, 10, 37)
    if not media:
        zenith  = Angle (0, 10, 19)
    else:
        zenith  = Angle (0, 10, 10)
    m.compute_far_field (zenith, azimuth)
    print (m.as_mininec ())

__all__ = \
    [ 'Angle'
    , 'Excitation'
    , 'Far_Field_Pattern'
    , 'Medium'
    , 'Wire'
    , 'Mininec'
    , 'ideal_ground'
    ]
