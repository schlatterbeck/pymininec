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
from datetime import datetime

def format_float (floats, use_e = 0):
    """ Reproduce floating-point formatting of the Basic code
    Test special case with large number, should usually set 'use_e':
    >>> print ('%s' % format_float ((3e7,)))
     30000000
    """
    r = []
    for f in floats:
        if f == 0:
            fmt = '% .1f'
        else:
            prec = 6 - int (np.log (abs (f)) / np.log (10))
            if prec < 0:
                prec = 0
            fmt = '%% .%df' % prec
        if use_e and abs (f) < 1e-1:
            fmt = '% e'
            if abs (f) == 0:
                fmt = '% .0f'
        s = fmt % f
        if fmt != '% e':
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

class Connected_Wires:
    """ This is used to store a set of connected wires *and* the
        corresponding segments in one data structure.
    """

    def __init__ (self):
        self.pulses = set ()
        self.wires  = set ()
        self.list   = []
    # end def __init__

    def add (self, wire, pulse_idx, sign):
        assert wire not in self.wires
        assert pulse_idx not in self.pulses
        self.pulses.add (pulse_idx)
        self.wires.add (wire)
        self.list.append ((wire, pulse_idx, sign))
    # end def add

    def pulse_iter (self):
        """ Yield pulse indeces sorted by wire index
        """
        for wire, idx, s in sorted (self.list, key = lambda x: x [0].n):
            yield (idx, s)
    # end def pulse_iter

    def __bool__ (self):
        return bool (self.list)
    # end def __bool__

# end class Connected_Wires

class Excitation:
    """ This is the pulse source definition in mininec.
        For convenience phase is in degrees (and converted internally)
        Magnitude is in volts, constructor can either directly give a
        complex number for the voltage *or* floating point voltage
        magnitude and a phase in degrees.
    """
    def __init__ (self, cvolt, phase = None):
        if isinstance (cvolt, complex) and phase is not None:
            raise ValueError \
                ("Either specify magnitude/phase or complex voltage")

        self.parent    = None
        self.idx       = None
        if phase is not None:
            self.magnitude = cvolt
            self.phase_d   = phase
            self.phase     = phase / 180. * np.pi
            self.voltage   = cvolt * np.e ** (1j * self.phase)
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
            ( '%sCURRENT = (%s , %s J)'
            % ( ' ' * 14
              , format_float ([self.current.real], 1) [0]
              , format_float ([self.current.imag], 1) [0]
              )
            )
        r.append \
            ( '%sIMPEDANCE = (%s , %s J)'
            % ( ' ' * 14
              , format_float ([self.impedance.real]) [0]
              , format_float ([self.impedance.imag]) [0]
              )
            )
        r.append \
            ( '%sPOWER = %s  WATTS'
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

    def register (self, parent, pulse):
        """ The pulse is the index into the segments/pulses (0-based)
            We asume that the correctness of the pulse has been verified
            by the parent code.
        """
        assert self.parent is None
        self.parent = parent
        self.idx = pulse
    # end def register
# end class Excitation

class Far_Field_Pattern:
    """ Values in dBi and/or V/m for far field
        Angles are in degrees for printing
    """
    def __init__ (self, theta, phi, gain, e_theta, e_phi, pwr_ratio):
        self.theta     = theta
        self.phi       = phi
        self.gain      = gain
        self._e_theta  = e_theta
        self._e_phi    = e_phi
        self.pwr_ratio = np.sqrt (pwr_ratio)
    # end def __init__

    @property
    def e_theta (self):
        return self._e_theta * self.pwr_ratio
    # end def e_theta

    @property
    def e_phi (self):
        return self._e_phi * self.pwr_ratio
    # end def e_phi

    def db_as_mininec (self):
        v, h, t = self.gain
        return \
            ( ('%s     ' * 4 + '%s')
            % ( format_float ((self.theta, self.phi))
              + format_float ((v, h, t))
              )
            )
    # end def db_as_mininec

    def abs_gain_as_mininec (self):
        e_theta_abs   = np.abs   (self.e_theta)
        e_phi_abs     = np.abs   (self.e_phi)
        e_theta_angle = np.angle (self.e_theta) / np.pi * 180
        e_phi_angle   = np.angle (self.e_phi)   / np.pi * 180
        return \
            ( '%6.2f %9.2f %20.3E % 8.2f %19.3E % 8.2f'
            % ( self.theta,  self.phi
              , e_theta_abs, e_theta_angle
              , e_phi_abs,   e_phi_angle
              )
            )
    # end def abs_gain_as_mininec

# end def Far_Field_Pattern

class _Load:
    """ Base class of impedance loading.
        A load can be re-used with several pulses. We have an interface
        to add a pulse to a load. This is convenient if a load is placed
        on every pulse, e.g. for copper loading of all elements. Note
        that the original mininec code allowed *either* S-Parameter
        loads *or* impedance loads. We support both concurrently. If a
        segment is loaded twice, loads appear to be in series.
    """

    def __init__ (self, *args, **kw):
        self.pulses = []
        self.n      = None
    # end def __init__

    def add_pulse (self, pulse):
        self.pulses.append (pulse)
    # end def add_pulse

    def impedance (self, f):
        """ Get impedance for a certain frequency
            This probably needs reimplementation in different derived
            classes. Especially if the impedance is frequency dependent.
        """
        return self._impedance
    # end def impedance

# end class _Load

class Impedance_Load (_Load):
    """ A complex load
        The original Basic code allows to specify resistance and reactance
        See below for the second case of S-parameters.
    """
    def __init__ (self, impedance):
        self._impedance = impedance
        super ().__init__ ()
    # end def __init__

# end class Impedance_Load

class Laplace_Load (_Load):
    """ Laplace S-Parameter (S = j omega) load from mininec implementation
        We get two lists of parameters. They represent the numerator and
        denominator coefficients, respectively (sequence a is the denominator
        and sequence b is the numerator).
    >>> l = Laplace_Load (b = (1., 0.), a = (0., -2.193644e-3))
    >>> z = l.impedance (7.15)
    >>> print ("%g%+gj" % (z.real, z.imag))
    -0+10.1472j
    """
    def __init__ (self, a, b):
        m = max (len (a), len (b))
        self.a = np.zeros (m)
        self.b = np.zeros (m)
        self.a [:len (a)] = a
        self.b [:len (b)] = b
        if not len (a):
            raise ValueError ("At least one denominator parameter required")
        super ().__init__ ()
    # end def __init__

    def impedance (self, f):
        w  = 2 * np.pi * f
        u1 = u2 = d1 = d2 = 0.0
        s  = 1
        for j in range (0, len (self.a), 2):
            u1 += self.b [j] * s * w ** j
            d1 += self.a [j] * s * w ** j
            l = j + 1
            u2 += self.b [l] * s * w ** l
            d2 += self.a [l] * s * w ** l
            s = -s
        u = u1 + 1j * u2
        d = d1 + 1j * d2
        return u / d
    # end def impedance

# end class Laplace_Load

class Medium:
    """ This encapsulates the media (e.g. ground screen etc.)
        With diel and cond zero we asume ideal ground.
        Note that it seems only the first medium can have a
        ground screen of radials
        The boundary is the type of boundary between different media (if
        there is more than one). It is 1 for linear (X-coordinate it
        seems) or circular. If there is more than one medium we need the
        coordinate (distance X for linear and radius R for circular
        boundary) of the *next* medium, in the original code U(I).
    >>> med = Medium (3, 4)
    >>> imp = med.impedance (30)
    >>> print ('%.4f%+.4fj' % (imp.real, imp.imag))
    0.0144+0.0144j
    >>> med = Medium (0, 0)
    >>> med.impedance (7)
    0j
    """
    def __init__ \
        ( self
        , diel, cond, height = 0
        , nradials = 0, radius = 0, dist = 0
        , boundary = 1, coord = 1e6
        ):
        self.diel     = diel     # dielectric constant T(I)
        self.cond     = cond     # conductivity        V(I)
        self.nradials = nradials # number of radials   NR
        self.radius   = radius   # radial wire radius  RR
        self.coord    = coord    # U(I)
        self.height   = height   # H(I)
        self.boundary = boundary
        self.next     = None     # next medium
        self.prev     = None     # previous medium
        self.is_ideal = False
        if diel == 0 and cond == 0:
            self.is_ideal = True
            self.coord    = 0
            if self.nradials:
                raise ValueError ("Ideal ground may not use radials")
            if self.height != 0:
                raise ValueError ("Ideal ground must have height 0")
        if self.nradials:
            self.boundary = 2
            # Radials extend to boundary of next medium
            if self.radius <= 0:
                raise ValueError ("Radius must be >0")
        else:
            self.radius = 0.0
        # Code from Line 628-633 in far field calculation
        if diel:
            if not self.cond:
                raise ValueError \
                    ("Non-ideal ground must have non-zero ground parameters")
    # end def __init__

    def as_mininec (self):
        r = []
        if not self.is_ideal:
            p = tuple \
                (x.strip () for x in format_float ((self.diel, self.cond)))
            r.append \
                ( ' RELATIVE DIELECTRIC CONSTANT, CONDUCTIVITY:'
                  '  %s , %s'
                % p
                )
        if self.nradials:
            r.append \
                ( ' NUMBER OF RADIAL WIRES IN GROUND SCREEN: %3d'
                % self.nradials
                )
            r.append \
                ( ' RADIUS OF RADIAL WIRES:  %s'
                % format_float ([self.radius]) [0].strip ()
                )
        if self.next:
            r.append \
                ( ' X OR R COORDINATE OF NEXT MEDIA INTERFACE:  %s'
                % format_float ([self.coord]) [0].strip ()
                )
        if self.prev:
            r.append \
                ( ' HEIGHT OF MEDIA: %s'
                % format_float ([self.height]) [0].strip ()
                )
        return '\n'.join (r)
    # end def as_mininec

    def impedance (self, f):
        if self.is_ideal:
            return 0+0j
        t = 2 * np.pi * f * 8.85e-6
        return 1 / np.sqrt (self.diel + -1j * self.cond / t)
    # end def impedance

    def set_next (self, next):
        self.next = next
        if next is None:
            if self.nradials:
                raise ValueError \
                    ("Radials only on first medium of more than one")
            # Thats the default in mininec meaning infinity
            self.coord = 1e6
        else:
            if self.is_ideal:
                raise ValueError ("Ideal ground must be the only medium")
            if next.nradials:
                raise ValueError ("Medium with radials must be first")
            # All media must have same boundary, currently we only use the first
            next.boundary = self.boundary
            next.prev = self
    # end def set_next

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
    Wire [0 0 0]-[ 0  0 25], r=0.001
    >>> wire.n = 23
    >>> wire
    Wire 23 [0 0 0]-[ 0  0 25], r=0.001
    """
    def __init__ (self, n_segments, x1, y1, z1, x2, y2, z2, r):
        self.n_segments = n_segments
        # whenever we need to access both ends by index we use endpoints
        self.p1         = np.array ([x1, y1, z1])
        self.p2         = np.array ([x2, y2, z2])
        self.endpoints  = np.array ([self.p1, self.p2])
        self.r = r
        if r <= 0:
            raise ValueError ("Radius must be >0")
        diff = self.p2 - self.p1
        if (diff == 0).all ():
            raise ValueError ("Zero length wire: %s %s" % (self.p1, self.p2))
        self.wire_len = np.linalg.norm (diff)
        self.seg_len  = self.wire_len / self.n_segments
        # Original comment: compute direction cosines
        self.dirs = diff / self.wire_len
        self.end_segs = [None, None]
        self.j2 = [None, None]
        self.i1 = self.i2 = 0
        # Links to previous/next connected wire (at start/end)
        self.conn = (Connected_Wires (), Connected_Wires ())
        self.n = None
    # end def __init__

    def compute_ground (self, n, media):
        self.n = n
        # If we are in free space, nothing to do here
        if media is None:
            self.is_ground = (False, False)
            return
        # Wire end is grounded if Z coordinate is 0
        # In the original implementation this is kept in J1
        # with: 0: not grounded -1: start grounded 1: end grounded
        self.is_ground = ((self.p1 [-1] == 0), (self.p2 [-1] == 0))
        if self.is_ground [0] and self.is_ground [1]:
            raise ValueError ("Both ends of a wire may not be grounded")
        if self.p1 [-1] < 0 or self.p2 [-1] < 0:
            raise ValueError ("height cannot not be negative with ground")
    # end def compute_ground

    def compute_connections (self, media):
        """ Compute links to connected wires self.conn [0] contains
            wires that link to our first end while conn [1] contains wires
            that link to our second end.
            Also compute sets of indeces of pulses.
        """
        for n, j in enumerate (self.j2):
            if j <= 0:
                continue
            if not self.is_ground [n] and j != self.n + 1:
                other = media [j - 1]
                self.conn [n].add (other, self.end_segs [n], 1)
                if (other.p1 == self.endpoints [n]).all ():
                    s = 1 if n == 1 else -1
                    other.conn [0].add (self, self.end_segs [n], s)
                else:
                    assert (other.p2 == self.endpoints [n]).all ()
                    s = 1 if n == 0 else -1
                    other.conn [1].add (self, self.end_segs [n], s)
    # end def compute_connections

    def segment_iter (self, yield_ends = True):
        if self.end_segs [0] is not None or self.end_segs [1] is not None:
            if self.end_segs [1] is None:
                if yield_ends or self.end_segs [0] not in self.conn [0].pulses:
                    yield self.end_segs [0]
            elif self.end_segs [0] is None:
                if yield_ends or self.end_segs [1] not in self.conn [1].pulses:
                    yield self.end_segs [1]
            else:
                u = self.conn [0].pulses.union (self.conn [1].pulses)
                for k in range (self.end_segs [0], self.end_segs [1] + 1):
                    if yield_ends or k not in u:
                        yield k
    # end def segment_iter

    def __str__ (self):
        if self.n is None:
            return 'Wire %s-%s, r=%s' % (self.p1, self.p2, self.r)
        return 'Wire %d %s-%s, r=%s' % (self.n, self.p1, self.p2, self.r)
    __repr__ = __str__

# end class Wire

class Gauge_Wire (Wire):
    table = dict \
        (( ( 1,    7.348e-3)
         , ( 2,    6.543e-3)
         , ( 3,    5.827e-3)
         , ( 4,    5.189e-3)
         , ( 5,    4.621e-3)
         , ( 6,    4.115e-3)
         , ( 7,    3.665e-3)
         , ( 8,    3.264e-3)
         , ( 9,    2.906e-3)
         , (10,    2.588e-3)
         , (11,    2.304e-3)
         , (12,    2.052e-3)
         , (13,    1.829e-3)
         , (14,    1.628e-3)
         , (15,    1.45e-3)
         , (16,    1.291e-3)
         , (17,    1.15e-3)
         , (18,    1.024e-3)
         , (19,    0.9119e-3)
         , (20,    0.8128e-3)
         , (21,    0.7239e-3)
         , (22,    0.6426e-3)
         , (23,    0.574e-3)
         , (24,    0.5106e-3)
         , (25,    0.4547e-3)
         , (26,    0.4038e-3)
         , (27,    0.3606e-3)
         , (28,    0.32e-3)
         , (29,    0.287e-3)
         , (30,    0.254e-3)
         , (31,    0.2261e-3)
         , (32,    0.2032e-3)
        ))

    def __init__ (self, n_segments, x1, y1, z1, x2, y2, z2, gauge):
        r = self.table [gauge] / 2
        super ().__init__ (n_segments, x1, y1, z1, x2, y2, z2, r)
    # end def __init__

# end class Gauge_Wire

class Mininec:
    """ A mininec implementation in Python
    >>> w = []
    >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.001))
    >>> s = Excitation (cvolt = 1+0j)
    >>> m = Mininec (20, w)
    >>> m.register_source (s, 4, 0)
    >>> print (m.wires_as_mininec ())
    NO. OF WIRES: 1
    <BLANKLINE>
    WIRE NO. 1
                COORDINATES                                 END         NO. OF
       X             Y             Z          RADIUS     CONNECTION     SEGMENTS
     0             0             0                          0
     21.41428      0             0             .001         0             10
    <BLANKLINE>
                      **** ANTENNA GEOMETRY ****
    <BLANKLINE>
    WIRE NO.  1  COORDINATES                                CONNECTION PULSE
    X             Y             Z             RADIUS        END1 END2  NO.
     2.141428      0             0             .001          0    1   1
     4.282857      0             0             .001          1    1   2
     6.424285      0             0             .001          1    1   3
     8.565714      0             0             .001          1    1   4
     10.70714      0             0             .001          1    1   5
     12.84857      0             0             .001          1    1   6
     14.99         0             0             .001          1    1   7
     17.13143      0             0             .001          1    1   8
     19.27286      0             0             .001          1    0   9

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

    def __init__ (self, f, geo, media = None, print_opts = None):
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
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)
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
        self.f          = f
        self.media      = media
        self.loads      = []
        self.check_ground ()
        self.sources    = []
        self.geo        = geo
        self.print_opts = print_opts or set (('far-field',))
        if not self.media or len (self.media) == 1:
            self.boundary = 1
        self.check_geo ()
        self.compute_connectivity ()
        self.output_date = False
    # end __init__

    @property
    def f (self):
        return self._f
    # en def f

    @f.setter
    def f (self, frq):
        self._f = frq
        self.wavelen = w = 299.8 / self.f
        # 1 / (4 * PI * OMEGA * EPSILON)
        self.m       = 4.77783352 * w
        # set small radius modification condition:
        self.srm     = .0001 * w
        self.w       = 2 * np.pi / w
        self.w2      = self.w ** 2 / 2
        self.currents = None
        self.rhs      = None
        self.Z        = None
    # end def f

    def check_ground (self):
        if not self.media and self.media is not None:
            raise ValueError ("Media must be None for free space")
        if self.media:
            for n, m in enumerate (self.media):
                if n == 0:
                    if len (self.media) == 1:
                        m.set_next (None)
                    self.boundary = m.boundary
                else:
                    self.media [n - 1].set_next (m)
        else:
            self.boundary = 1
    # end def check_ground

    def check_geo (self):
        for n, wire in enumerate (self.geo):
            wire.compute_ground (n, self.media)
    # end def check_geo

    def compute (self):
        """ Compute the currents (solution of the impedance matrix)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)
        >>> m.compute ()
        >>> print (m.sources_as_mininec ())
        NO. OF SOURCES :  1
        PULSE NO., VOLTAGE MAGNITUDE, PHASE (DEGREES):  5 , 1 , 0
        >>> print (m.source_data_as_mininec ())
        ********************    SOURCE DATA     ********************
        PULSE  5      VOLTAGE = ( 1 , 0 J)
                      CURRENT = ( 1.006964E-02 , -5.166080E-03 J)
                      IMPEDANCE = ( 78.61622 ,  40.33289 J)
                      POWER =  5.034821E-03  WATTS
        """
        self.compute_impedance_matrix ()
        self.compute_impedance_matrix_loads ()
        self.compute_rhs ()
        self.current = np.linalg.solve (self.Z, self.rhs)
        # Used by far field and near field calculation
        self.power = sum (s.power for s in self.sources)
    # end def compute

    def compute_connectivity (self):
        """ In the original code this is done while parsing the
            individual wires from the geometry information.

            The vector N in the original code holds the index of the
            start and end segments for each wire. We store this as
            end_segs [] directly into the Wire object.
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
                if w.is_ground [0]:
                    i1   = -(i + 1)
                if w.is_ground [1]:
                    i2   = -(i + 1)
                    # Why is the gflag only set for the second ground case??
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
            w.i1 = i1
            w.i2 = i2
            # Here we used to print the geometry, we do this separately.

            # This part starts at 1198
            # compute connectivity data (pulses n1 to n)
            # We make end_segs 0-based and use None instead of 0
            n1 = n + 1
            self.geo [i].end_segs [0] = n1 - 1
            if w.n_segments == 1 and i1 == 0:
                self.geo [i].end_segs [0] = None
            n = n1 + w.n_segments
            if i1 == 0:
                n -= 1
            if i2 == 0:
                n -= 1
            self.geo [i].end_segs [1] = n - 1
            if w.n_segments == 1 and i2 == 0:
                self.geo [i].end_segs [1] = None
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
            c_per [(n,  2)] = i2
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
                     * self.geo [i2 - 1].seg_len
                     )
                if c_per [(n1, 1)] == -(i + 1):
                    f3 [-1] = -f3 [-1]
                seg [i1] = seg [i1] - f3 * self.geo [i2 - 1].dirs
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
                     * self.geo [i2 - 1].seg_len
                     )
                i3 = i6 - 1
                if i + 1 == -c_per [(n, 2)]:
                    f3 [-1] = -f3 [-1]
                seg [i6] = seg [i3] + f3 * self.geo [i2 - 1].dirs
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
        for wire in self.geo:
            wire.compute_connections (self.geo)
    # end def compute_connectivity

    def compute_far_field \
        (self, zenith_angle, azimuth_angle, pwr = None, dist = 0):
        """ Compute far field
            Original code starts at 620 (and is called at 621)
            The angles are instances of Angle.
            Note that the radial distance dist is only used for
            calculation of the far field in V/m
        """
        # We only use calculation in dBi for now, see 641-654
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
        self.ff_dist  = dist
        self.ff_power = pwr or self.power
        self.far_field_angles = (zenith_angle, azimuth_angle)
        k9 = .016678 / self.power
        self.far_field_by_angle = {}
        for azi_d, azi in azimuth_angle.iter ():
            # exchange real/imag
            v21 = np.e ** (-1j * azi)
            for zen_d, zen in zenith_angle.iter ():
                # Can we somehow reduce this with complex artithmetics?
                # What is/was the intention of that code?
                rt3  = np.e ** (-1j * zen)
                rt1  = -rt3.imag * v21.real + 1j * (rt3.real * v21.real)
                rt2  =  rt3.imag * v21.imag - 1j * (rt3.real * v21.imag)
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
                            if s_x [0] == -s_x [1] and f3 < 0:
                                continue
                            # Standard case (condition Line 725, 726)
                            if  (  k == 1
                                or not self.media
                                or self.media [0].is_ideal
                                ):
                                seg = self.seg [j]
                                s2  = self.w * sum (seg * rvec.real * kvec)
                                s   = np.e ** (1j * s2)
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
                                assert self.media
                                med = self.media [0]
                                nr  = med.nradials
                                rr  = med.radius
                                # begin by finding specular distance
                                t4 = 1e5
                                if rt3.real != 0:
                                    t4 = -self.seg [j][2] * rt3.imag / rt3.real
                                b9 = t4 * v21.real + self.seg [j][0]
                                if self.boundary != 1:
                                    # Hmm this is pythagoras?
                                    b9 *= b9
                                    b9 += (self.seg [j][1] - t4 * v21.imag) ** 2
                                    b9 = np.sqrt (b9)
                                # search for the corresponding medium
                                # Find minimum index where b9 > coord
                                coord = [m.coord for m in self.media]
                                j2 = np.argmin ((b9 > coord) * coord)
                                z45 = self.media [j2].impedance (self.f)
                                # Line 764, 765
                                if nr != 0 and b9 <= coord [0]:
                                    prod = nr * rr
                                    r = b9 + prod
                                    z8  = self.w * r * np.log (r / prod) / nr
                                    s89 = z45 * z8 * 1j
                                    t89 = z45 + (z8 * 1j)
                                    z45 = s89 / t89
                                # form SQR(1-Z^2*SIN^2)
                                w67 = np.sqrt (1 - z45 ** 2 * rt3.imag ** 2)
                                # vertical reflection coefficient
                                s89 = rt3.real - w67 * z45
                                t89 = rt3.real + w67 * z45
                                v89 = s89 / t89
                                # horizontal reflection coefficient
                                s89 = w67 - rt3.real * z45
                                t89 = w67 + rt3.real * z45
                                h89 = s89 / t89 - v89
                                # compute contribution to sum
                                h   = 0
                                if self.media and j2 < len (self.media):
                                    h = self.media [j2].height
                                seg = self.seg [j]
                                hv  = np.array ([0, 0, 2 * h])
                                sv  = np.array ([1, 1, -1])
                                s2  = self.w * sum (sv * (seg - hv) * rvec.real)
                                s   = np.e ** (1j * s2)
                                b   = f3 * s * self.current [i]
                                w67 = b * v89
                                w   = self.geo [l]
                                # FIXME: This is only triggered with
                                # real ground and different x/y directions
                                # and is currently untested.
                                d   = v21.imag * w.dirs [0] \
                                    + v21.real * w.dirs [1]
                                z67 = d * b * h89
                                tm1 = np.array \
                                    ([         v21.imag * z67.real
                                       + 1j * (v21.imag * z67.imag)
                                     ,         v21.real * z67.real
                                       + 1j * (v21.real * z67.imag)
                                     , 0
                                    ])
                                tm2 = np.array ([-1, -1, 1])
                                vec += (w.dirs * w67 + tm1) * tm2
                h12 = sum (vec * rvec.imag) * self.g0 * -1j
                vv  = np.array ([v21.imag, v21.real])
                x34 = sum (vec [:2] * vv) * self.g0 * -1j
                rd  = dist or 0
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
                rat = self.ff_power / self.power
                self.far_field_by_angle [(zen_d, azi_d)] = \
                    Far_Field_Pattern (zen_d, azi_d, p123, h12, x34, rat)
    # end def compute_far_field

    def compute_impedance_matrix (self):
        """ This starts at line 195 (with entry-point for gosub at 196)
            in the original basic code.
            Note that we're using seg_idx instead of c_per
        >>> s  = Excitation (1, 0)
        >>> s2 = Excitation (1, 30)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> m = Mininec (7, w)
        >>> m.register_source (s2, 4)
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
        -8.485114-9.668584E-03j
        4.240447-9.572988E-03j
        .214366-9.290245E-03j
        5.390034E-02-8.832275E-03j
        2.352013E-02-8.218218E-03j
        1.223068E-02-7.473394E-03j
        6.605723E-03-6.627949E-03j
        3.326364E-03-5.715207E-03j
        1.258102E-03-4.769988E-03j
        row 2
        4.240447-9.572988E-03j
        -8.485114-9.668584E-03j
        4.240447-9.572988E-03j
        .214366-9.290245E-03j
        5.390034E-02-8.832275E-03j
        2.352013E-02-8.218218E-03j
        1.223068E-02-7.473394E-03j
        6.605723E-03-6.627949E-03j
        3.326364E-03-5.715207E-03j
        row 3
        .214366-9.290245E-03j
        4.240447-9.572988E-03j
        -8.485114-9.668584E-03j
        4.240447-9.572988E-03j
        .214366-9.290245E-03j
        5.390034E-02-8.832275E-03j
        2.352013E-02-8.218218E-03j
        1.223068E-02-7.473394E-03j
        6.605723E-03-6.627949E-03j
        row 4
        5.390034E-02-8.832275E-03j
        .214366-9.290245E-03j
        4.240447-9.572988E-03j
        -8.485114-9.668584E-03j
        4.240447-9.572988E-03j
        .214366-9.290245E-03j
        5.390034E-02-8.832275E-03j
        2.352013E-02-8.218218E-03j
        1.223068E-02-7.473394E-03j
        row 5
        2.352013E-02-8.218218E-03j
        5.390034E-02-8.832275E-03j
        .214366-9.290245E-03j
        4.240447-9.572988E-03j
        -8.485114-9.668584E-03j
        4.240447-9.572988E-03j
        .214366-9.290245E-03j
        5.390034E-02-8.832275E-03j
        2.352013E-02-8.218218E-03j
        row 6
        1.223068E-02-7.473394E-03j
        2.352013E-02-8.218218E-03j
        5.390034E-02-8.832275E-03j
        .214366-9.290245E-03j
        4.240447-9.572988E-03j
        -8.485114-9.668584E-03j
        4.240447-9.572988E-03j
        .214366-9.290245E-03j
        5.390034E-02-8.832275E-03j
        row 7
        6.605723E-03-6.627949E-03j
        1.223068E-02-7.473394E-03j
        2.352013E-02-8.218218E-03j
        5.390034E-02-8.832275E-03j
        .214366-9.290245E-03j
        4.240447-9.572988E-03j
        -8.485114-9.668584E-03j
        4.240447-9.572988E-03j
        .214366-9.290245E-03j
        row 8
        3.326364E-03-5.715207E-03j
        6.605723E-03-6.627949E-03j
        1.223068E-02-7.473394E-03j
        2.352013E-02-8.218218E-03j
        5.390034E-02-8.832275E-03j
        .214366-9.290245E-03j
        4.240447-9.572988E-03j
        -8.485114-9.668584E-03j
        4.240447-9.572988E-03j
        row 9
        1.258102E-03-4.769988E-03j
        3.326364E-03-5.715207E-03j
        6.605723E-03-6.627949E-03j
        1.223068E-02-7.473394E-03j
        2.352013E-02-8.218218E-03j
        5.390034E-02-8.832275E-03j
        .214366-9.290245E-03j
        4.240447-9.572988E-03j
        -8.485114-9.668584E-03j
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
        >>> m.sources = []
        >>> m.register_source (s, 4)
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
                        vec3 = (f7v * u * di2 + f6v * v * di1) * kvec
                        d    = self.w2 * sum (vec3 * t567 [i])
                        # compute PSI(M+1/2,N,N+1)
                        p1 += 0.5
                        if f8 == 2:
                            p1 -= 1
                        p2 = p3
                        p3 += 1
                        p4 = i2v [j]
                        if f8 < 2:
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
                    # 323
                    if p1 >= n:
                        continue
                    # 324
                    if self.seg_idx [p1][0] != self.seg_idx [p1][1]:
                        continue
                    d = self.geo [i2v [j]].dirs
                    # 325-327
                    # The following continue statement *is* reached by
                    # the current tests but is flagged as not reached by
                    # the pytest framework if the python version is
                    # below 3.10. The statement is reached for
                    # the t-ant.pym example with p1 == 8 and j == 7.
                    if  (   self.seg_idx [p1][1] != self.seg_idx [j][1]
                        and (  self.seg_idx [p1][1] != -self.seg_idx [j][1]
                            or d [0] + d [1] != 0
                            )
                        ):
                        continue
                    p2 = i + 1
                    self.Z [p2][p1] = self.Z [i][j]
            # Here follows a GOSUB 1599 which calculates the remaining time,
            # not implemented
        # end matrix fill time calculation
        # Here follows a GOSUB 1589 which calculates elapsed time,
        # not implemented
        # addition of loads happens in compute_impedance_matrix_loads
    # end def compute_impedance_matrix

    def compute_impedance_matrix_loads (self):
        for l in self.loads:
            for j in l.pulses:
                f2 = 1 / self.m
                # Looks like K in the original code line 371 is set by
                # the preceeding loop iterating over the images. So we
                # replace this with self.media is not None
                if  (   self.c_per [j][0] == -self.c_per [j][1]
                    and self.media is not None
                    ):
                    f2 *= 2
                # Weird, the imag part goes to the real Z component and
                # vice-versa, the contribution to the real part is
                # negated, the contribution to the imag part not
                self.Z [j][j] += -f2 * l.impedance (self.f) * 1j
    # end def compute_impedance_matrix_loads

    def nf_helper (self, cp, j, k, v, j12, wj, f67, f45):
        v6  = np.array ([1, 1, f67 [0]])
        v7  = np.array ([1, 1, f67 [1]])
        dir = [w.dirs for w in wj]
        j3  = np.max (j12)
        # compute psi(0,J,J+.5)
        p2  = 2 * j3 + j + 1
        p3  = p2 + .5
        p4  = j12 [1]
        u   = self.psi_near_field_75 (v, k, p2, p3, p4) * f45 [1]
        # compute psi(0,J-.5,J)
        p3 = p2
        p2 = p2 - .5
        p4 = j12 [0]
        v   = self.psi_near_field_66 (v, k, p2, p3, p4) * f45 [0]
        # real part of vector potential contribution
        # imaginary part of vector potential contribution
        kv  = np.array ([1, 1, k])
        v35 = (v * dir [0] * v6 + u * dir [1] * v7) * kv
        return v35
    # end def nf_helper

    def compute_near_field (self, start, inc, nvec, pwr = None):
        """ Near field
            Asumes that the impedance matrix has been computed.
            Note that the input paramters are three-dimensional vectors.
            If no power level (pwr) is given, the power is computed from
            the voltages and currents given with the excitation.
            start is originally  XI, YI, ZI
            inc   is originally  XC, YC, ZC
            nvec  is originally  NX, NY, NZ
            power is originally  O2
            Basic code starts at 875
        """
        self.nf_param = np.array ([list (start), list (inc), list (nvec)])
        self.e_field = []
        self.h_field = []
        # virtual dipole length for near field calculation:
        s0 = .001 * self.wavelen
        if pwr is None:
            pwr = self.power
        self.nf_power = pwr
        f_e = np.sqrt (pwr / self.power)
        f_h = f_e / s0 / (4*np.pi)
        # Compute the grid
        r = [np.arange (s, s + n * i, i)
             for (s, n, i) in
                 (zip (*(reversed (x) for x in (start, nvec, inc))))
            ]
        self.near_field_idx = np.meshgrid (*r, indexing = 'ij')

        # Looks like T5, T6, T7 are 0 except for the diagonale at each
        # iteration of I (the dimension), we build everything in one go
        t567 = np.identity (3) * 2 * s0

        for vec in self.near_field_iter ():
            # Originally X0, Y0, Z0 but only one of them is non-0 in
            # each iteration. This considers both versions of
            # J8 (the values -1,1) in the original code
            v0m = [vec + np.identity (3) * (j8 * s0 / 2)
                   for j8 in (-1, 1)
                  ]
            v0m = np.array (v0m)
            h   = np.zeros (3, dtype = complex)
            # Loop over 3 dimensions
            for i in range (3):
                u78 = 0j
                # H-field, original Basic variable K!
                # Originally this is 6X2-dimensional in Basic, every two
                # succeeding values are real and imag part, there seem
                # to be two vectors which are indexed using j9 and j8
                # takes the value -1 and 1 in the v0m initialization
                # (originally X0, Y0, Z0)
                kf  = np.zeros ((2, 3), dtype = complex)
                # Loop over segments
                for j, cp in enumerate (self.seg_idx):
                    j12 = np.abs  (cp) - 1
                    f45 = np.sign (cp)
                    f67 = np.ones  (2)
                    u56 = 0j
                    wj  = [self.geo [g] for g in j12]
                    for k in self.image_iter ():
                        if cp [0] == -cp [1]:
                            if k < 0:
                                continue
                            # compute vector potential A
                            f67 = f45
                        v35_e = self.nf_helper \
                            (cp, j, k, vec, j12, wj, f67, f45)
                        v35_h = np.array \
                            ([self.nf_helper
                                (cp, j, k, v [i], j12, wj, f67, f45)
                              for v in v0m
                            ])
                        # At this point comment notes
                        # magnetic field calculation completed
                        # and jumps to 1042 if H field
                        # We compute both, E and H and continue
                        d   = sum (v35_e * t567 [i]) * self.w2
                        # compute psi(.5,J,J+1)
                        p2  = 2 * np.max (j12) + j + 1
                        p3  = p2 + 1
                        p4  = j12 [1]
                        u   = self.psi_near_field_56 \
                            (vec, t567 [i], k, .5, p2, p3, p4)
                        # compute psi(-.5,J,J+1)
                        tmp = self.psi_near_field_56 \
                            (vec, t567 [i], k, -.5, p2, p3, p4)
                        u   = (tmp - u) / wj [1].seg_len
                        # compute psi(.5,J-1,J)
                        p3  = p2
                        p2 -= 1
                        p4  = j12 [0]
                        u34 = self.psi_near_field_56 \
                            (vec, t567 [i], k, .5, p2, p3, p4)
                        # compute psi(-.5,J-1,J)
                        tmp = self.psi_near_field_56 \
                            (vec, t567 [i], k, -.5, p2, p3, p4)
                        # gradient of scalar potential
                        u56 += (u + (u34 - tmp) / wj [0].seg_len + d) * k
                        # Here would be a GOTO 1048 (a continue of the K loop)
                        # that jumps over the H-field calculation
                        # we do both, E and H field
                        # components of vector potential A
                        kf += v35_h * self.current [j] * k
                    # The following code only for E-field (line 1050)
                    u78 += u56 * self.current [j]

                # Here the original code has a conditional backjump to
                # 964 (just before the J loop) re-initializing the
                # X0, Y0, Z0 array (v0m in this implementation).
                # This realizes the computation of both halves of v0m.
                # We do this in one go, see above for j8, j9.

                # This originally is a ON I GOTO
                # Hmm the kf array may not be consecutive real/imag
                # parts after all?
                if i == 0:
                    h [1]  = kf [0][2] - kf [1][2]
                    h [2]  = kf [1][1] - kf [0][1]
                elif i==1:
                    h [0]  = kf [1][2] - kf [0][2]
                    h [2] += (  -kf [1][0].real + kf [0][0].real
                             + (-kf [1][0].imag + kf [0][0].imag) * 1j
                             )
                else: # i==2
                    h [0] += (  -kf [1][1].real + kf [0][1].real
                             + (-kf [1][1].imag + kf [0][1].imag) * 1j
                             )
                    h [1] += (  kf [1][0].real - kf [0][0].real
                             + (kf [1][0].imag - kf [0][0].imag) * 1j
                             )

                # imaginary part of electric field
                # real part of electric field
                # Don't know why real- and imag is exchanged here
                u78 *= -1j * self.m / s0
                if not i:
                    self.e_field.append (np.zeros (3, dtype = complex))
                self.e_field [-1][i] = u78 * f_e
            self.h_field.append (h * f_h)
    # end def compute_near_field

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
            w: 2 * pi * f / c
               (omega / c, frequency-dependent constant in program)
            srm: small radius modification condition
                 0.0001 * c / f
            a(p4): wire radius
            t3, t4: Sum of Integrals I2 and I3, t3 is real and t4 imag

            Temporary variables:
            d3, d, b, b1, w0, w1, v0, vec3 (originally (X3, Y3, Z3))

            Note when comparing results to the BASIC implementation: The
            basic implementation *adds* its results to the *existing*
            values of t3 and t4!
        # First with thin-wire approximation, doesn't make a difference
        # if we use a exact_kernel or not
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.001))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)
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
        t34  = 0j
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
                # Note: This adds only a real part, the imag part of the
                # elliptic integral is 0
                t34 += ( (v0 + np.log (d3 / (64 * a2)) / 2)
                       / np.pi / wire.r - 1 / d
                       )
        b1 = d * self.w
        # EXP(-J*K*R)/R
        t34 += np.e ** (-1j * b1) / d
        return t34
    #end def integral_i2_i3

    def near_field_iter (self):
        for a in zip (*(x.flat for x in self.near_field_idx)):
            yield np.array (list (reversed (a)))
    # end def near_field_iter

    def psi (self, vec2, vecv, k, p2, p3, p4, exact = True, fvs = 0):
        """ Common code for entry points at 56, 87, and 102.
            This code starts at line 135.
            The variable fvs is used to distiguish code path at the end.
            The variable p2 is the index of the segment, seems this can
            be a float in which case the middle of two segs is used.
            The variable p4 is the index of the wire.
            vec2 replaces (X2, Y2, Z2)
            vecv replaces (V1, V2, V3)
            i6: Use reduced kernel if 0, this was I6! (single precision)
                So beware: condition "I6!=0" means variable I6! is == 0
                The exclamation mark is part of the variable name :-(

            Input:
            vec2, vecv
            k:
            p2:  segment index 1 (0-based)
            p3:  segment index 2 (0-based)
            p4:  wire index (0-based)
            fvs: scalar vs. vector potential
            is_near: This originally tested input C$ for "N" which is
                     the selection of near field compuation, this forces
                     exact kernel, see exact flag
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
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        # Original produces:
        # 5.330494 -0.1568644j
        >>> vec2 = np.zeros (3)
        >>> vecv = np.array ([1.070714, 0, 0])
        >>> r = m.psi (vec2, vecv, 1, 1, 1.5, 0, exact = False)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        5.3304831 -0.1568644j

        # Original produces:
        # -8.333431E-02 -0.1156091j
        >>> vec2 = np.array ([13.91929, 0, 0])
        >>> vecv = np.array ([16.06072, 0, 0])
        >>> x = m.psi
        >>> r = x (vec2, vecv, k=1, p2=8, p3=9, p4=0, fvs = 1)
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
        if t > 1.1 or exact:
            # This starts line 165
            if t > 6:
                l = 3
            if t > 10:
                l = 1
        elif not exact:
            if wire.r <= self.srm:
                t12 = 2 * np.log (wire.seg_len / wire.r) \
                    - self.w * wire.seg_len * 1j
                if fvs != 1:
                    t12 /= 2
                return t12
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
            vec2 = kvec * self.seg [i4] - vec1
        else:
            i5 = i4 + 1
            vec2 = kvec * (self.seg [i4] + self.seg [i5]) / 2 - vec1
        # S(V)-S(M) GOES IN (V1,V2,V3) (this is now vecv)
        i4 = int (p3)
        if i4 == p3:
            vecv = kvec * self.seg [i4] - vec1
        else:
            i5 = i4 + 1
            vecv = kvec * (self.seg [i4] + self.seg [i5]) / 2 - vec1
        return vec2, vecv
    # end def psi_common_vec1_vecv

    def psi_near_field_56 (self, vec0, vect, k, p1, p2, p3, p4):
        """ Compute psi used several times during computation of near field
            Original entry point in line 56
            vec0 originally is (X0, Y0, Z0)
            vect originally is (T5, T6, T7)
            Note that p1 is the only non-zero-based variable, it's not
            used as an index but as a factor.
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        # Original produces:
        # 0.5496336 -0.3002106j
        >>> vec0 = np.array ([0, -1, -1])
        >>> vect = np.array ([8.565715E-02, 0, 0])
        >>> r = m.psi_near_field_56 (vec0, vect, k=1, p1=0.5, p2=1, p3=2, p4=0)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.5496335 -0.3002106j
        """
        kvec = np.ones (3)
        kvec [-1] = k
        vec1 = vec0 + p1 * vect / 2
        vec2 = vec1 - kvec * self.seg [int (p2)]
        vecv = vec1 - kvec * self.seg [int (p3)]
        return self.psi (vec2, vecv, k, p2, p3, p4, exact = True)
    # end def psi_near_field_56

    def psi_near_field_66 (self, vec0, k, p2, p3, p4):
        """ Compute psi used during computation of near field
            Original entry point in line 66
            vec0 originally is (X0, Y0, Z0)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        # Original produces:
        # 0.4792338 -0.1544592j
        >>> vec0 = np.array ([0, -1, -1])
        >>> r = m.psi_near_field_66 (vec0, k=1, p2=0.5, p3=1, p4=0)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.4792338 -0.1544592j
        """
        kvec = np.ones (3)
        kvec [-1] = k
        i4 = int (p2)
        i5 = i4 + 1
        vec2 = vec0 - kvec * (self.seg [i4] + self.seg [i5]) / 2
        vecv = vec0 - kvec * self.seg [p3]
        return self.psi (vec2, vecv, k, p2, p3, p4, exact = True)
    # end def psi_near_field_66

    def psi_near_field_75 (self, vec0, k, p2, p3, p4):
        """ Compute psi used during computation of near field
            Original entry point in line 75
            vec0 originally is (X0, Y0, Z0)
            vec1 originally is (X1, Y1, Z1)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        # Original produces:
        # 0.3218219 -.1519149j
        >>> vec0 = np.array ([0, -1, -1])
        >>> r = m.psi_near_field_75 (vec0, k=1, p2=1, p3=1.5, p4=0)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.3218219 -0.1519149j
        """
        kvec = np.ones (3)
        kvec [-1] = k
        i4 = int (p3)
        i5 = i4 + 1
        vec2 = vec0 - kvec * self.seg [p2]
        vecv = vec0 - kvec * (self.seg [i4] + self.seg [i5]) / 2
        return self.psi (vec2, vecv, k, p2, p3, p4, exact = True)
    # end def psi_near_field_75

    def register_load (self, load, pulse = None, wire_idx = None):
        """ Default if no pulse is given is to add the load to *all*
            pulses. Otherwise if no wire_idx is given the pulse is an
            absolute index, otherwise it's the index of a pulse on the
            wire given by wire_idx. Indeces are 0-based.
        """
        if pulse is None:
            for wire in self.geo:
                for p in wire.segment_iter ():
                    load.add_pulse (p)
            self.loads.append (load)
        else:
            if pulse < 0:
                raise ValueError ("Pulse index must be >= 0")
            if wire_idx is not None:
                if wire_idx >= len (self.geo):
                    raise ValueError ('Invalid wire index %d' % (wire_idx))
                w = self.geo [wire_idx]
                p = w.end_segs [0] + pulse
                if p > w.end_segs [1]:
                    raise ValueError \
                        ('Invalid pulse %d for wire %d' % (pulse, wire_idx))
            elif pulse >= len (self.c_per):
                raise ValueError ('Invalid pulse %d' % pulse)
            else:
                p = pulse
            load.add_pulse (p)
            self.loads.append (load)
    # end def register_load

    def register_source (self, source, pulse, wire_idx = None):
        """ Register a source, either with absolute pulse index or with
            a pulse index relative to a wire. Indeces are 0-based.
        """
        if pulse < 0:
            raise ValueError ("Pulse index must be > 0")
        # Check source index
        if wire_idx is not None:
            w = self.geo [wire_idx]
            if w.end_segs [0] is None:
                raise ValueError \
                    ('Invalid pulse %d for wire %d' % (pulse, wire_idx))
            p = w.end_segs [0] + pulse
            if w.end_segs [1] is None or p > w.end_segs [1]:
                raise ValueError \
                    ('Invalid pulse %d for wire %d' % (pulse, wire_idx))
            self.sources.append (source)
            source.register (self, p)
        else:
            if pulse >= len (self.c_per):
                raise ValueError ('Invalid pulse %d' % pulse)
            self.sources.append (source)
            source.register (self, pulse)
    # end def register_source

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
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

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
            or p3 != p2 + 1
            or p1 != (p2 + p3) / 2
            ):
            i4 = int (p1)
            i5 = i4 + 1
            vec1 = (self.seg [i4] + self.seg [i5]) / 2
            wd   = self.wires_differ (i, j)
            vec2, vecv = self.psi_common_vec1_vecv (vec1, k, p2, p3)
            return self.psi (vec2, vecv, k, p2, p3, p4, fvs = 1, exact = wd)
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
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        # Original produces:
        # 0.6747199 -.1555772j
        >>> method = m.vector_potential
        >>> r = method (k=1, p1=1, p2=1.5, p3=2, p4=0, i=0, j=1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.6747199 -0.1555773j
        """
        wire = self.geo [p4]
        if k < 1 or wire.r >= self.srm or i != j or p3 != p2 + .5:
            vec1 = self.seg [p1]
            vec2, vecv = self.psi_common_vec1_vecv (vec1, k, p2, p3)
            wd = self.wires_differ (i, j)
            return self.psi (vec2, vecv, k, p2, p3, p4, fvs = 0, exact = wd)
        t1 = np.log (wire.seg_len / wire.r)
        t2 = -self.w * wire.seg_len / 2
        return t1 + t2 * 1j
    # end def vector_potential

    def wires_differ (self, i, j):
        assert self.w_per [i] >= 0
        assert self.w_per [j] >= 0
        wire_i_j2 = self.geo [self.w_per [i]].j2
        wire_j_j2 = self.geo [self.w_per [j]].j2
        r = \
            (   wire_i_j2 [0] != wire_j_j2 [0]
            and wire_i_j2 [0] != wire_j_j2 [1]
            and wire_i_j2 [1] != wire_j_j2 [0]
            and wire_i_j2 [1] != wire_j_j2 [1]
            )
        return r
    # end def wires_differ

    # All the *as_mininec methods

    def _options (self, options):
        if options is None:
            options = self.print_opts
        return options
    # end def _options

    def as_mininec (self, options = None):
        options = self._options (options)
        r = []
        r.append (self.header_as_mininec ())
        r.append (self.frequency_as_mininec ())
        r.append (self.environment_as_mininec ())
        r.append ('')
        r.append (self.wires_as_mininec ())
        r.append ('')
        r.append (self.sources_as_mininec ())
        r.append (self.loads_as_mininec ())
        r.append ('')
        r.append (self.source_data_as_mininec ())
        r.append ('')
        r.append (self.currents_as_mininec ())
        r.append ('')
        r.append (self.fields_as_mininec (options))
        return '\n'.join (r)
    # end def as_mininec

    def currents_as_mininec (self):
        r = []
        r.append ('*' * 20 + '    CURRENT DATA    ' + '*' * 20)
        r.append ('')
        for wire in self.geo:
            r.append ('WIRE NO.%3d :' % (wire.n + 1))
            r.append \
                ( 'PULSE%sREAL%sIMAGINARY%sMAGNITUDE%sPHASE'
                % tuple (' ' * x for x in (9, 10, 5, 5))
                )
            r.append \
                ( ' NO.%s(AMPS)%s(AMPS)%s(AMPS)%s(DEGREES)'
                % tuple (' ' * x for x in (10, 8, 8, 8))
                )
            fmt = ''.join (['%s ' * 2] * 2)
            if not wire.is_ground [0]:
                if not wire.conn [0]:
                    r.append ((' ' * 13).join (['E '] + ['0'] * 4))
                else:
                    c = 0+0j
                    for p, s in wire.conn [0].pulse_iter ():
                        c = s * self.current [p]
                    a = np.angle (c) / np.pi * 180
                    r.append \
                        ( ('J ' + ' ' * 12 + fmt)
                        % format_float
                            ((c.real, c.imag, np.abs (c), a), use_e = True)
                        )
            for k in wire.segment_iter (yield_ends = False):
                c = self.current [k]
                a = np.angle (c) / np.pi * 180
                r.append \
                    ( ('%s     ' + fmt)
                    % format_float
                        ((k + 1, c.real, c.imag, np.abs (c), a), use_e = True)
                    )
            if not wire.is_ground [1]:
                if not wire.conn [1]:
                    r.append ((' ' * 13).join (['E '] + ['0'] * 4))
                else:
                    c = 0+0j
                    for p, s in wire.conn [1].pulse_iter ():
                        c += s * self.current [p]
                    a = np.angle (c) / np.pi * 180
                    r.append \
                        ( ('J ' + ' ' * 12 + fmt)
                        % format_float
                            ((c.real, c.imag, np.abs (c), a), use_e = True)
                        )
        return '\n'.join (r)
    # end def currents_as_mininec

    def environment_as_mininec (self):
        """ Print environment (ground setup, see Medium above)
        """
        r = []
        e = 1
        if self.media:
            e = -1
            l = len (self.media)
            m = self.media [0]
            if len (self.media) == 1 and m.is_ideal:
                l = 0
        r.append \
            ('ENVIRONMENT (+1 FOR FREE SPACE, -1 FOR GROUND PLANE): %+d' % e)
        if self.media:
            r.append \
                ( ' NUMBER OF MEDIA (0 FOR PERFECTLY CONDUCTING GROUND): %2d'
                % l
                )
            if len (self.media) > 1:
                r.append \
                    ( ' TYPE OF BOUNDARY (1-LINEAR, 2-CIRCULAR):  %d'
                    % self.boundary
                    )
            if l:
                for n, m in enumerate (self.media):
                    r.append (m.as_mininec ())
        return '\n'.join (r)
    # end def environment_as_mininec

    def far_field_absolute_as_mininec (self):
        """ Far field in V/m
            This may only be called when compute_far_field has already
            been called before.
        """
        r = []
        r.append (self.far_field_header_as_mininec (is_db = False))
        d = format_float ((self.ff_dist,), use_e = True) [0].rstrip ()
        r.append ((' ' * 14 + 'RADIAL DISTANCE = %s  METERS') % d)
        p = format_float ((self.ff_power,), use_e = True) [0].rstrip ()
        r.append ((' ' * 14 + 'POWER LEVEL = %s  WATTS') % p)
        r.append \
            ( 'ZENITH%sAZIMUTH%sE(THETA)%sE(PHI)'
            % tuple (' ' * x for x in (3, 17, 20))
            )
        r.append \
            ( ' ANGLE%sANGLE%sMAG(V/M)%sPHASE(DEG)%sMAG(V/M)%sPHASE(DEG)'
            % tuple (' ' * x for x in (4, 14, 4, 6, 4))
            )
        srt = lambda x: (x [1], x [0])
        for zen, azi in sorted (self.far_field_by_angle, key = srt):
            ff = self.far_field_by_angle [(zen, azi)]
            r.append (ff.abs_gain_as_mininec ())
        return '\n'.join (r)
    # end def far_field_absolute_as_mininec

    def far_field_as_mininec (self):
        """ Print far field in dB in mininec format
            This may only be called when compute_far_field has already
            been called before.
        """
        r = []
        r.append (self.far_field_header_as_mininec ())
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
            r.append (ff.db_as_mininec ())
        return '\n'.join (r)
    # end def far_field_as_mininec

    def far_field_header_as_mininec (self, is_db = True):
        """ Print far field in dB in mininec format
            This may only be called when compute_far_field has already
            been called before.
        """
        r = []
        zenith, azimuth = self.far_field_angles
        r.append ('*' * 20 + '     FAR FIELD      ' + '*' * 20)
        r.append ('')
        if not is_db and self.ff_power != self.power:
            p = format_float ((self.ff_power,)) [0].rstrip ()
            r.append ('NEW POWER LEVEL = %s' % p)
        ze = tuple (x.rstrip () for x in format_float
            ((zenith.initial, zenith.inc, zenith.number), use_e = True))
        r.append ('ZENITH ANGLE : INITIAL,INCREMENT,NUMBER:%s ,%s ,%s' % ze)
        az = tuple (x.rstrip () for x in format_float
            ((azimuth.initial, azimuth.inc, azimuth.number), use_e = True))
        r.append ('AZIMUTH ANGLE: INITIAL,INCREMENT,NUMBER:%s ,%s ,%s' % az)
        r.append ('')
        r.append ('*' * 20 + '    PATTERN DATA    ' + '*' * 20)
        return '\n'.join (r)
    # end def far_field_header_as_mininec

    def fields_as_mininec (self, options = None):
        options = self._options (options)
        r = []
        if 'far-field' in options:
            r.append (self.far_field_as_mininec ())
        if 'far-field-absolute' in options:
            r.append (self.far_field_absolute_as_mininec ())
        if 'near-field' in options:
            r.append (self.near_field_as_mininec ())
        return '\n'.join (r)
    # end def as_mininec

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

    def frq_independent_as_mininec (self):
        r = []
        r.append (self.header_as_mininec ())
        r.append (self.environment_as_mininec ())
        r.append ('')
        r.append (self.wires_as_mininec ())
        r.append ('')
        r.append (self.sources_as_mininec ())
        r.append (self.loads_as_mininec ())
        return '\n'.join (r)
    # end def frq_independent_as_mininec

    def frq_dependent_as_mininec (self, options = None):
        options = self._options (options)
        r = []
        r.append ('')
        r.append (self.frequency_as_mininec ())
        r.append (self.source_data_as_mininec ())
        r.append ('')
        r.append (self.currents_as_mininec ())
        r.append ('')
        r.append (self.fields_as_mininec (options))
        return '\n'.join (r)
    # end def frq_dependent_as_mininec

    def header_as_mininec (self):
        """ The original mininec header
            For reproduceability the default is without date/time.
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.001))
        >>> s = Excitation (cvolt = 1+0j)
        >>> m = Mininec (20, w)
        >>> m.register_source (s, 4)
        >>> m.output_date = True
        >>> len (m.header_as_mininec ().split ('\\n'))
        6
        """
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

    def loads_as_mininec (self):
        """ Format loads
        """
        r = []
        n = 0
        for l in self.loads:
            n += len (l.pulses)
        r.append ('NUMBER OF LOADS %d' % n)
        for l in self.loads:
            imp = l.impedance (self.f)
            for p in l.pulses:
                r.append \
                    ( 'PULSE NO.,RESISTANCE,REACTANCE: %2d , %s , %s'
                    % ((p + 1,) + format_float ([imp.real, imp.imag]))
                    )
        return '\n'.join (r)
    # end def loads_as_mininec

    def near_field_as_mininec (self):
        r = []
        r.append (self.near_field_header_as_mininec ())
        r.append (self.near_field_e_as_mininec ())
        r.append (self.near_field_header_as_mininec ())
        r.append (self.near_field_h_as_mininec ())
        return '\n'.join (r)
    # end def near_field_as_mininec

    def near_field_e_as_mininec (self):
        r = []
        for v, coord in zip (self.e_field, self.near_field_iter ()):
            p1 = 0j
            p2 = 0.0
            r.append ('*' * 20 + 'NEAR ELECTRIC FIELDS' + '*' * 20)
            r.append \
                ( ' ' * 9 + 'FIELD POINT: X = %s  Y = %s  Z = %s'
                % (format_float (coord))
                )
            r.append \
                ( '  VECTOR%sREAL%sIMAGINARY%sMAGNITUDE%sPHASE' 
                % tuple (' ' * k for k in (6, 10, 5, 5))
                )
            r.append \
                ( ' COMPONENT%sV/M%sV/M%sV/M%sDEG' 
                % tuple (' ' * k for k in (5, 11, 11, 11))
                )
            ax = 'XYZ'
            for n, value in enumerate (v):
                a   = np.angle (value)
                b   = np.abs (value)
                b2  = b ** 2
                p1 += b2 * np.e ** (-2j * a)
                p2 += b2.real
                a   = a / np.pi * 180
                line = \
                    ( ('   %s' + ' ' * 10 + '%-13s ' * 4)
                    % ((ax [n],) + format_float
                         ((value.real, value.imag, b, a), use_e = True)
                      )
                    )
                r.append (line.rstrip ())
            pk = np.sqrt (p2 / 2 + abs (p1) / 2)
            r.append \
                ( '   MAXIMUM OR PEAK FIELD = %s V/M'
                % format_float ((pk,), use_e = True)
                )
            r.append ('')
        return '\n'.join (r)
    # end def near_field_e_as_mininec

    def near_field_h_as_mininec (self):
        r = []
        for v, coord in zip (self.h_field, self.near_field_iter ()):
            p1 = 0j
            p2 = 0.0
            r.append ('*' * 20 + 'NEAR MAGNETIC FIELDS' + '*' * 20)
            r.append \
                ( ' ' * 9 + 'FIELD POINT: X = %s  Y = %s  Z = %s'
                % (format_float (coord))
                )
            r.append \
                ( '  VECTOR%sREAL%sIMAGINARY%sMAGNITUDE%sPHASE' 
                % tuple (' ' * k for k in (6, 10, 5, 5))
                )
            r.append \
                ( ' COMPONENT%sAMPS/M%sAMPS/M%sAMPS/M%sDEG' 
                % tuple (' ' * k for k in (5, 8, 8, 8))
                )
            ax = 'XYZ'
            for n, value in enumerate (v):
                a   = np.angle (value)
                b   = np.abs (value)
                b2  = b ** 2
                p1 += b2 * np.e ** (-2j * a)
                p2 += b2.real
                a   = a / np.pi * 180
                line = \
                    ( ('   %s' + ' ' * 10 + '%-13s ' * 4)
                    % ((ax [n],) + format_float
                         ((value.real, value.imag, b, a), use_e = True)
                      )
                    )
                r.append (line.rstrip ())
            pk = np.sqrt (p2 / 2 + np.linalg.norm (p1) / 2)
            r.append \
                ( '   MAXIMUM OR PEAK FIELD = %s  AMPS/M'
                % format_float ((pk,), use_e = True)
                )
            r.append ('')
        return '\n'.join (r)
    # end def near_field_h_as_mininec

    def near_field_header_as_mininec (self):
        r = []
        r.append ('*' * 20 + '    NEAR FIELDS     ' + '*' * 20)
        r.append ('')
        for coord, (nf_s, nf_i, nf_c) in zip ('XYZ', self.nf_param.T):
            ff = tuple \
                (x.rstrip () for x in format_float ((nf_s, nf_i, nf_c)))
            r.append \
                ('%s-COORDINATE (M): INITIAL,INCREMENT,NUMBER : %s , %s , %s'
                % ((coord,) + ff)
                )
        r.append ('')
        if self.nf_power != self.power:
            p = format_float ((self.nf_power,)) [0].rstrip ()
            r.append ('NEW POWER LEVEL (WATTS) = %s' % p)
            r.append ('')
        return '\n'.join (r)
    # end def near_field_header_as_mininec

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

    def seg_as_mininec (self, wire, seg, idx):
        l = []
        l.append (('%-13s ' * 3) % format_float (seg))
        l.append ('%-12s' % format_float ([wire.r]))
        l.append ('%4d %4d' % tuple (self.c_per [idx]))
        l.append ('%4d' % (idx + 1))
        return  ''.join (l)
    # end def seg_as_mininec

    def wires_as_mininec (self):
        r = []
        r.append ('NO. OF WIRES: %d' % len (self.geo))
        r.append ('')
        for wire in self.geo:
            r.append ('WIRE NO. %d' % (wire.n + 1))
            r.append \
                ( '%sCOORDINATES%sEND%sNO. OF'
                % (' ' * 12, ' ' * 33, ' ' * 9)
                )
            r.append \
                ( '   X%sY%sZ%sRADIUS%sCONNECTION%sSEGMENTS'
                % (' ' * 13, ' ' * 13, ' ' * 10, ' ' * 5, ' ' * 5)
                )
            l = []
            l.append (('%-13s ' * 3) % format_float (wire.p1))
            l.append ('%s%3d' % (' ' * 12, wire.i1))
            r.append (''.join (l))
            l = []
            l.append (('%-13s ' * 3) % format_float (wire.p2))
            l.append ('%-13s' % format_float ([wire.r]))
            l.append ('%2d%15d' % (wire.i2, wire.n_segments))
            r.append (''.join (l))
            r.append ('')
        r.append (' ' * 18 + '**** ANTENNA GEOMETRY ****')
        k = 1
        j = 0
        for wire in self.geo:
            r.append ('')
            r.append \
                ( 'WIRE NO.%3d  COORDINATES%sCONNECTION PULSE'
                % (wire.n + 1, ' ' * 32)
                )
            r.append \
                (('%-13s ' * 4 + 'END1 END2  NO.') % ('X', 'Y', 'Z', 'RADIUS'))
            if wire.end_segs [0] is None and wire.end_segs [1] is None:
                r.append \
                    ( ('%-13s ' * 3 + '    %-10s %-4s %-4s %-4s')
                    % (('-',) * 6 + ('0',))
                    )
            for k in wire.segment_iter ():
                idx = 2 * self.w_per [k] + k + 1
                seg = tuple (self.seg [idx])
                r.append (self.seg_as_mininec (wire, seg, k))
        return '\n'.join (r)
    # end def wires_as_mininec

# end class Mininec

def main (argv = sys.argv [1:], f_err = sys.stderr):
    """ The main routine called from the command-line
    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--frequency-increment=.01', '--frequency-steps=2'])
    >>> args.extend (['--medium=0,0,0', '--excitation-segment=1'])
    >>> args.extend (['--theta=0,45,3', '--phi=0,180,3'])
    >>> main (args)
                       ****************************************
                         MINI-NUMERICAL ELECTROMAGNETICS CODE
                                       MININEC
                       ****************************************
    <BLANKLINE>
    ENVIRONMENT (+1 FOR FREE SPACE, -1 FOR GROUND PLANE): -1
     NUMBER OF MEDIA (0 FOR PERFECTLY CONDUCTING GROUND):  0
    <BLANKLINE>
    NO. OF WIRES: 1
    <BLANKLINE>
    WIRE NO. 1
                COORDINATES                                 END         NO. OF
       X             Y             Z          RADIUS     CONNECTION     SEGMENTS
     0             0             0                         -1
     0             0             10.0838       .0127        0              5
    <BLANKLINE>
                      **** ANTENNA GEOMETRY ****
    <BLANKLINE>
    WIRE NO.  1  COORDINATES                                CONNECTION PULSE
    X             Y             Z             RADIUS        END1 END2  NO.
     0             0             0             .0127        -1    1   1
     0             0             2.01676       .0127         1    1   2
     0             0             4.03352       .0127         1    1   3
     0             0             6.05028       .0127         1    1   4
     0             0             8.06704       .0127         1    0   5
    <BLANKLINE>
    NO. OF SOURCES :  1
    PULSE NO., VOLTAGE MAGNITUDE, PHASE (DEGREES):  1 , 1 , 0
    NUMBER OF LOADS 0
    <BLANKLINE>
    FREQUENCY (MHZ): 7.15
        WAVE LENGTH =  41.93007  METERS
    ********************    SOURCE DATA     ********************
    PULSE  1      VOLTAGE = ( 1 , 0 J)
                  CURRENT = ( 2.857798E-02 ,  1.660853E-03 J)
                  IMPEDANCE = ( 34.87418 , -2.026766 J)
                  POWER =  1.428899E-02  WATTS
    <BLANKLINE>
    ********************    CURRENT DATA    ********************
    <BLANKLINE>
    WIRE NO.  1 :
    PULSE         REAL          IMAGINARY     MAGNITUDE     PHASE
     NO.          (AMPS)        (AMPS)        (AMPS)        (DEGREES)
     1             2.857798E-02  1.660853E-03  2.862620E-02  3.32609  
     2             2.727548E-02  6.861986E-04  2.728411E-02  1.441147 
     3             2.346944E-02  8.170779E-05  2.346959E-02  .199472  
     4             1.744657E-02 -2.362219E-04  1.744817E-02 -.775722  
     5             9.607629E-03 -2.685486E-04  9.611381E-03 -1.601092 
    E              0             0             0             0
    <BLANKLINE>
    ********************     FAR FIELD      ********************
    <BLANKLINE>
    ZENITH ANGLE : INITIAL,INCREMENT,NUMBER:  0 , 45 , 3
    AZIMUTH ANGLE: INITIAL,INCREMENT,NUMBER:  0 , 180 , 3
    <BLANKLINE>
    ********************    PATTERN DATA    ********************
    ZENITH        AZIMUTH       VERTICAL      HORIZONTAL    TOTAL
     ANGLE         ANGLE        PATTERN (DB)  PATTERN (DB)  PATTERN (DB)
     0             0            -999          -999          -999     
     45            0             1.163918     -999           1.163918
     90            0             5.119285     -999           5.119285
     0             180          -999          -999          -999     
     45            180           1.163918     -999           1.163918
     90            180           5.119285     -999           5.119285
     0             360          -999          -999          -999     
     45            360           1.163918     -999           1.163918
     90            360           5.119285     -999           5.119285
    <BLANKLINE>
    FREQUENCY (MHZ): 7.16
        WAVE LENGTH =  41.87151  METERS
    <BLANKLINE>
    ********************    SOURCE DATA     ********************
    PULSE  1      VOLTAGE = ( 1 , 0 J)
                  CURRENT = ( 2.851291E-02 ,  1.014723E-03 J)
                  IMPEDANCE = ( 35.02747 , -1.246564 J)
                  POWER =  1.425646E-02  WATTS
    <BLANKLINE>
    ********************    CURRENT DATA    ********************
    <BLANKLINE>
    WIRE NO.  1 :
    PULSE         REAL          IMAGINARY     MAGNITUDE     PHASE
     NO.          (AMPS)        (AMPS)        (AMPS)        (DEGREES)
     1             2.851291E-02  1.014723E-03  2.853096E-02  2.038193 
     2             2.721271E-02  6.814281E-05  2.721279E-02  .143473  
     3             2.341369E-02 -4.509545E-04  2.341803E-02 -1.103397 
     4             1.740291E-02 -6.326804E-04  1.741441E-02 -2.082063 
     5             9.581812E-03 -4.870737E-04  9.594184E-03 -2.91002  
    E              0             0             0             0
    <BLANKLINE>
    ********************     FAR FIELD      ********************
    <BLANKLINE>
    ZENITH ANGLE : INITIAL,INCREMENT,NUMBER: 0 , 45 , 3
    AZIMUTH ANGLE: INITIAL,INCREMENT,NUMBER: 0 , 180 , 3
    <BLANKLINE>
    ********************    PATTERN DATA    ********************
    ZENITH        AZIMUTH       VERTICAL      HORIZONTAL    TOTAL
     ANGLE         ANGLE        PATTERN (DB)  PATTERN (DB)  PATTERN (DB)
     0             0            -999          -999          -999     
     45            0             1.16192      -999           1.16192 
     90            0             5.120395     -999           5.120395
     0             180          -999          -999          -999     
     45            180           1.16192      -999           1.16192 
     90            180           5.120395     -999           5.120395
     0             360          -999          -999          -999     
     45            360           1.16192      -999           1.16192 
     90            360           5.120395     -999           5.120395

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--medium=0,0,0', '--excitation-segment=1'])
    >>> args.extend (['--theta=0,45,3', '--phi=0,180,3'])
    >>> args.extend (['--ff-power=100', '--ff-distance=1000'])
    >>> args.extend (['--option=far-field-absolute'])
    >>> main (args)
                       ****************************************
                         MINI-NUMERICAL ELECTROMAGNETICS CODE
                                       MININEC
                       ****************************************
    <BLANKLINE>
    FREQUENCY (MHZ): 7.15
        WAVE LENGTH =  41.93007  METERS
    <BLANKLINE>
    ENVIRONMENT (+1 FOR FREE SPACE, -1 FOR GROUND PLANE): -1
     NUMBER OF MEDIA (0 FOR PERFECTLY CONDUCTING GROUND):  0
    <BLANKLINE>
    NO. OF WIRES: 1
    <BLANKLINE>
    WIRE NO. 1
                COORDINATES                                 END         NO. OF
       X             Y             Z          RADIUS     CONNECTION     SEGMENTS
     0             0             0                         -1
     0             0             10.0838       .0127        0              5
    <BLANKLINE>
                      **** ANTENNA GEOMETRY ****
    <BLANKLINE>
    WIRE NO.  1  COORDINATES                                CONNECTION PULSE
    X             Y             Z             RADIUS        END1 END2  NO.
     0             0             0             .0127        -1    1   1
     0             0             2.01676       .0127         1    1   2
     0             0             4.03352       .0127         1    1   3
     0             0             6.05028       .0127         1    1   4
     0             0             8.06704       .0127         1    0   5
    <BLANKLINE>
    NO. OF SOURCES :  1
    PULSE NO., VOLTAGE MAGNITUDE, PHASE (DEGREES):  1 , 1 , 0
    NUMBER OF LOADS 0
    <BLANKLINE>
    ********************    SOURCE DATA     ********************
    PULSE  1      VOLTAGE = ( 1 , 0 J)
                  CURRENT = ( 2.857798E-02 ,  1.660853E-03 J)
                  IMPEDANCE = ( 34.87418 , -2.026766 J)
                  POWER =  1.428899E-02  WATTS
    <BLANKLINE>
    ********************    CURRENT DATA    ********************
    <BLANKLINE>
    WIRE NO.  1 :
    PULSE         REAL          IMAGINARY     MAGNITUDE     PHASE
     NO.          (AMPS)        (AMPS)        (AMPS)        (DEGREES)
     1             2.857798E-02  1.660853E-03  2.862620E-02  3.32609  
     2             2.727548E-02  6.861986E-04  2.728411E-02  1.441147 
     3             2.346944E-02  8.170779E-05  2.346959E-02  .199472  
     4             1.744657E-02 -2.362219E-04  1.744817E-02 -.775722  
     5             9.607629E-03 -2.685486E-04  9.611381E-03 -1.601092 
    E              0             0             0             0
    <BLANKLINE>
    ********************     FAR FIELD      ********************
    <BLANKLINE>
    NEW POWER LEVEL =  100
    ZENITH ANGLE : INITIAL,INCREMENT,NUMBER: 0 , 45 , 3
    AZIMUTH ANGLE: INITIAL,INCREMENT,NUMBER: 0 , 180 , 3
    <BLANKLINE>
    ********************    PATTERN DATA    ********************
                  RADIAL DISTANCE =  1000  METERS
                  POWER LEVEL =  100  WATTS
    ZENITH   AZIMUTH                 E(THETA)                    E(PHI)
     ANGLE    ANGLE              MAG(V/M)    PHASE(DEG)      MAG(V/M)    PHASE(DEG)
      0.00      0.00            0.000E+00     0.00           0.000E+00     0.00
     45.00      0.00            8.854E-02    90.84           0.000E+00     0.00
     90.00      0.00            1.396E-01    90.68           0.000E+00     0.00
      0.00    180.00            0.000E+00     0.00           0.000E+00     0.00
     45.00    180.00            8.854E-02    90.84           0.000E+00     0.00
     90.00    180.00            1.396E-01    90.68           0.000E+00     0.00
      0.00    360.00            0.000E+00     0.00           0.000E+00     0.00
     45.00    360.00            8.854E-02    90.84           0.000E+00     0.00
     90.00    360.00            1.396E-01    90.68           0.000E+00     0.00

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--medium=0,0,0', '--excitation-segment=1'])
    >>> args.extend (['--near-field=1,1,1,1,1,1,1,1,1', '--nf-power=100'])
    >>> main (args)
                       ****************************************
                         MINI-NUMERICAL ELECTROMAGNETICS CODE
                                       MININEC
                       ****************************************
    <BLANKLINE>
    FREQUENCY (MHZ): 7.15
        WAVE LENGTH =  41.93007  METERS
    <BLANKLINE>
    ENVIRONMENT (+1 FOR FREE SPACE, -1 FOR GROUND PLANE): -1
     NUMBER OF MEDIA (0 FOR PERFECTLY CONDUCTING GROUND):  0
    <BLANKLINE>
    NO. OF WIRES: 1
    <BLANKLINE>
    WIRE NO. 1
                COORDINATES                                 END         NO. OF
       X             Y             Z          RADIUS     CONNECTION     SEGMENTS
     0             0             0                         -1
     0             0             10.0838       .0127        0              5
    <BLANKLINE>
                      **** ANTENNA GEOMETRY ****
    <BLANKLINE>
    WIRE NO.  1  COORDINATES                                CONNECTION PULSE
    X             Y             Z             RADIUS        END1 END2  NO.
     0             0             0             .0127        -1    1   1
     0             0             2.01676       .0127         1    1   2
     0             0             4.03352       .0127         1    1   3
     0             0             6.05028       .0127         1    1   4
     0             0             8.06704       .0127         1    0   5
    <BLANKLINE>
    NO. OF SOURCES :  1
    PULSE NO., VOLTAGE MAGNITUDE, PHASE (DEGREES):  1 , 1 , 0
    NUMBER OF LOADS 0
    <BLANKLINE>
    ********************    SOURCE DATA     ********************
    PULSE  1      VOLTAGE = ( 1 , 0 J)
                  CURRENT = ( 2.857798E-02 ,  1.660853E-03 J)
                  IMPEDANCE = ( 34.87418 , -2.026766 J)
                  POWER =  1.428899E-02  WATTS
    <BLANKLINE>
    ********************    CURRENT DATA    ********************
    <BLANKLINE>
    WIRE NO.  1 :
    PULSE         REAL          IMAGINARY     MAGNITUDE     PHASE
     NO.          (AMPS)        (AMPS)        (AMPS)        (DEGREES)
     1             2.857798E-02  1.660853E-03  2.862620E-02  3.32609  
     2             2.727548E-02  6.861986E-04  2.728411E-02  1.441147 
     3             2.346944E-02  8.170779E-05  2.346959E-02  .199472  
     4             1.744657E-02 -2.362219E-04  1.744817E-02 -.775722  
     5             9.607629E-03 -2.685486E-04  9.611381E-03 -1.601092 
    E              0             0             0             0
    <BLANKLINE>
    ********************    NEAR FIELDS     ********************
    <BLANKLINE>
    X-COORDINATE (M): INITIAL,INCREMENT,NUMBER :  1 ,  1 ,  1
    Y-COORDINATE (M): INITIAL,INCREMENT,NUMBER :  1 ,  1 ,  1
    Z-COORDINATE (M): INITIAL,INCREMENT,NUMBER :  1 ,  1 ,  1
    <BLANKLINE>
    NEW POWER LEVEL (WATTS) =  100
    <BLANKLINE>
    ********************NEAR ELECTRIC FIELDS********************
             FIELD POINT: X =  1         Y =  1         Z =  1       
      VECTOR      REAL          IMAGINARY     MAGNITUDE     PHASE
     COMPONENT     V/M           V/M           V/M           DEG
       X           4.129238     -10.56496      11.34324     -68.65233
       Y           4.129238     -10.56496      11.34324     -68.65233
       Z          -17.05298      6.501882E-02  17.05311      179.7815
       MAXIMUM OR PEAK FIELD =  19.391   V/M
    <BLANKLINE>
    ********************    NEAR FIELDS     ********************
    <BLANKLINE>
    X-COORDINATE (M): INITIAL,INCREMENT,NUMBER :  1 ,  1 ,  1
    Y-COORDINATE (M): INITIAL,INCREMENT,NUMBER :  1 ,  1 ,  1
    Z-COORDINATE (M): INITIAL,INCREMENT,NUMBER :  1 ,  1 ,  1
    <BLANKLINE>
    NEW POWER LEVEL (WATTS) =  100
    <BLANKLINE>
    ********************NEAR MAGNETIC FIELDS********************
             FIELD POINT: X =  1         Y =  1         Z =  1       
      VECTOR      REAL          IMAGINARY     MAGNITUDE     PHASE
     COMPONENT     AMPS/M        AMPS/M        AMPS/M        DEG
       X          -.187091      -4.272377E-03  .187139      -178.6918
       Y           .187091       4.272377E-03  .187139       1.308172
       Z           0             0             0             0
       MAXIMUM OR PEAK FIELD =  .264655   AMPS/M
    <BLANKLINE>

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127,extra']
    >>> r = main (args, sys.stdout)
    Invalid number of parameters for wire 1
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,extra']
    >>> r = main (args, sys.stdout)
    Invalid wire 1: could not convert string to float: 'extra'
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--excitation-segment=1', '--excitation-segment=2'])
    >>> r = main (args, sys.stdout)
    Number of excitation segments must match voltages
    >>> r
    23

    >>> args = ['--attach-load=1']
    >>> r = main (args, sys.stdout)
    Append-load needs 2-3 parameters
    >>> r
    23
    >>> args = ['--attach-load=1,2,3,4']
    >>> r = main (args, sys.stdout)
    Append-load needs 2-3 parameters
    >>> r
    23
    >>> args = ['--attach-load=1,notanint']
    >>> r = main (args, sys.stdout)
    Attach-load: invalid literal for int() with base 10: 'notanint'
    >>> r
    23
    >>> args = ['--attach-load=1,2']
    >>> r = main (args, sys.stdout)
    Load index 1 out of range
    >>> r
    23
    >>> args = ['--load=1+1j']
    >>> r = main (args, sys.stdout)
    Error: Not all loads were used
    >>> r
    23
    >>> args = ['--load=1+1j', '--attach-load=1,all', '--attach-load=5,all']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Load index 5 out of range
    >>> r
    23
    >>> args = ['--load=1+1j', '--attach-load=1,1', '--attach-load=1,7']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error attaching load: Invalid pulse 6
    >>> r
    23
    >>> args = ['--load=1+1j', '--attach-load=1,1,7']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error attaching load: Invalid wire index 6
    >>> r
    23
    >>> args = ['--load=1+1j', '--attach-load=1,7,1']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error attaching load: Invalid pulse 6 for wire 0
    >>> r
    23
    >>> args = ['--load=1+1j', '--attach-load=1,0']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error attaching load: Pulse index must be >= 0
    >>> r
    23

    >>> args = ['--laplace-load-b=1', '--laplace-load-b=1']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error in Laplace load: At least one denominator parameter required
    >>> r
    23
    >>> args = ['--laplace-load-a=1', '--laplace-load-b=1']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error: Not all loads were used
    >>> r
    23
    >>> args = ['--laplace-load-a=1', '--laplace-load-b=1']
    >>> args = ['--laplace-load-a=1,2']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error: Not all loads were used
    >>> r
    23
    >>> args = ['--laplace-load-a=1', '--laplace-load-b=1,b']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error in Laplace load B: could not convert string to float: 'b'
    >>> r
    23
    >>> args = ['--laplace-load-a=1,a', '--laplace-load-b=1']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-segment=1'])
    >>> r = main (args, sys.stdout)
    Error in Laplace load A: could not convert string to float: 'a'
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--medium=0,0,0,0,0'])
    >>> r = main (args, sys.stdout)
    Medium needs 3-4 parameters
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--medium=0,0,oops'])
    >>> r = main (args, sys.stdout)
    Invalid medium 1: could not convert string to float: 'oops'
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--excitation-segment=1'])
    >>> args.extend (['--theta=0,45,3', '--phi=0,180'])
    >>> r = main (args, sys.stdout)
    Invalid phi angle, need three comma-separated values
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--excitation-segment=1'])
    >>> args.extend (['--medium=1,1,0', '--medium=1,1,0,5'])
    >>> args.extend (['--theta=0,45,3', '--phi=0,180,nonint'])
    >>> r = main (args, sys.stdout)
    Invalid phi angle, need float, float, int
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--excitation-segment=1', '--radial-count=8'])
    >>> args.extend (['--theta=0,45', '--phi=0,180,3'])
    >>> r = main (args, sys.stdout)
    Invalid theta angle, need three comma-separated values
    >>> r
    23

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--near-field=1,2,3'])
    >>> r = main (args, sys.stdout)
    Expecting 9 near-field parameters
    >>> r
    23
    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--near-field=1,2,3,4,5,6,7,8,zz'])
    >>> r = main (args, sys.stdout)
    Error near-field counts: invalid literal for int() with base 10: 'zz'
    >>> r
    23
    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--near-field=1,2,3,4,5,aa,7,8,9'])
    >>> r = main (args, sys.stdout)
    Error near-field inc: could not convert string to float: 'aa'
    >>> r
    23
    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--near-field=1,2,bb,4,5,6,7,8,9'])
    >>> r = main (args, sys.stdout)
    Error near-field start: could not convert string to float: 'bb'
    >>> r
    23

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--option=far-field', '--option=zoppel'])
    >>> r = main (args, sys.stdout)
    Invalid print option: zoppel
    >>> r
    23

    >>> args = ['-f', '7.15']
    >>> args.extend (['--excitation-segment=1'])
    >>> args.extend (['--theta=0,45,nonint', '--phi=0,180,3'])
    >>> r = main (args, sys.stdout)
    Invalid theta angle, need float, float, int
    >>> r
    23
    """
    from argparse import ArgumentParser
    cmd = ArgumentParser ()
    cmd.add_argument \
        ( '--attach-load'
        , help    = 'Attach load with given index to pulse, needs '
                    'load-index, pulse-index, optional wire index. If '
                    'wire index is given, pulse index is relative to '
                    'wire. To attach a load to all pulses use "all" for '
                    'the pulse index (and leave the wire index blank).'
        , action  = 'append'
        , default = []
        )
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
        ( '--frequency-increment', '--f-inc'
        , type    = float
        , help    = 'Frequency increment in MHz'
        )
    cmd.add_argument \
        ( '--frequency-steps', '--n-f'
        , type    = int
        , help    = 'Number of frequency steps'
        )
    cmd.add_argument \
        ( '--ff-distance'
        , help    = "Distance used for far-field computation"
        , type    = float
        )
    cmd.add_argument \
        ( '--ff-power'
        , help    = "Power used for far-field computation"
        , type    = float
        )
    cmd.add_argument \
        ( '--laplace-load-a'
        , help    = 'Laplace load, A (denominator) parameters (comma-separated)'
                    ' if multiple load-types are given, Laplace loads'
                    ' are numbered last.'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--laplace-load-b'
        , help    = 'Laplace load, B (numerator) parameters (comma-separated)'
                    ' if multiple load-types are given, Laplace loads'
                    ' are numbered last.'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '-l', '--load'
        , type    = complex
        , help    = 'Complex load'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--medium'
        , help    = "Media (ground), free space if not given, "
                    "specify dielectricum, condition, height, if all are "
                    "zero, ideal ground is asumed, if radials are "
                    "specified they apply to the first ground (which "
                    "cannot be ideal with radials), several media can be "
                    "specified. If more than one medium is given, the "
                    "circular or linear coordinate of medium must be given."
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--near-field'
        , help    = "Near-field definition, give three comma-separated "
                    "start values, then three comma-separated "
                    "increments, then three comma-separated counts for X, "
                    "Y and Z direction, respectively."
        )
    cmd.add_argument \
        ( '--nf-power'
        , help    = "Power used for near-field computation"
        , type    = float
        )
    allowed_options = ['far-field', 'near-field', 'far-field-absolute']
    cmd.add_argument \
        ( '--option'
        , help    = "Computation/printing options, option can be repeated, "
                    "use one or several of %s. If none are given, far field "
                    " is printed if no near-field option is present, "
                    " otherwise near field is printed."
                  % ', '.join (allowed_options)
        , action  = "append"
        , default = []
        )
    cmd.add_argument \
        ( '--phi'
        , help    = "Phi angle: start, increment, count"
        , default = '0,10,37'
        )
    cmd.add_argument \
        ( '--radial-count'
        , type    = int
        , default = 0
        , help    = 'Number of radials, default=%(default)s'
        )
    cmd.add_argument \
        ( '--radial-radius'
        , type    = float
        , help    = 'Radius of radial wires'
        )
    cmd.add_argument \
        ( '--radial-distance'
        , type    = float
        , help    = 'Distance of radials'
        , default = 0
        )
    cmd.add_argument \
        ( '--theta'
        , help    = "Theta angle: start, increment, count"
        , default = '0,10,10'
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
    args = cmd.parse_args (argv)
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
                , file = f_err
                )
            return 23
        try:
            seg = int (wparams [0])
            r = [float (x) for x in wparams [1:]]
        except ValueError as err:
            print ("Invalid wire %d: %s" % (n + 1, str (err)), file = f_err)
            return 23
        wires.append (Wire (seg, *r))
    if len (args.excitation_segment) != len (args.excitation_voltage):
        print \
            ("Number of excitation segments must match voltages", file = f_err)
        return 23

    media = []
    rad = {}
    if args.radial_count:
        rad = dict \
            ( nradials = args.radial_count
            , radius   = args.radial_radius
            , dist     = args.radial_distance
            )
    for n, m in enumerate (args.medium):
        p = m.split (',')
        if not 3 <= len (p) <= 4:
            print ("Medium needs 3-4 parameters", file = f_err)
            return 23
        try:
            p = [float (x) for x in p]
        except ValueError as err:
            print ("Invalid medium %d: %s" % (n + 1, str (err)), file = f_err)
            return 23
        d = {}
        if n == 0:
            d = dict (rad)
        if len (p) > 3:
            d ['coord'] = p [3]
        p = p [:3]
        media.append (Medium (*p, **d))
    media = media or None
    m = Mininec (args.frequency, wires, media = media)
    for i, v in zip (args.excitation_segment, args.excitation_voltage):
        s = Excitation (cvolt = v)
        m.register_source (s, i - 1)
    loads = []
    for l in args.load:
        loads.append (Impedance_Load (l))
    laplace = []
    for a in args.laplace_load_a:
        try:
            af = [float (x) for x in a.split (',')]
        except ValueError as err:
            print ("Error in Laplace load A: %s" % err, file = f_err)
            return 23
        laplace.append ([af])
    for n, b in enumerate (args.laplace_load_b):
        try:
            bf = [float (x) for x in b.split (',')]
        except ValueError as err:
            print ("Error in Laplace load B: %s" % err, file = f_err)
            return 23
        if n < len (laplace):
            laplace [n].append (bf)
        else:
            laplace.append ([[], bf])
    for x in laplace:
        if len (x) == 1:
            x.append ([])
    for a, b in laplace:
        try:
            l = Laplace_Load (a = a, b = b)
        except ValueError as err:
            print ("Error in Laplace load: %s" % err, file = f_err)
            return 23
        loads.append (l)

    used_loads = set ()
    for x in args.attach_load:
        att = x.split (',')
        if not 2 <= len (att) <= 3:
            print ("Append-load needs 2-3 parameters", file = f_err)
            return 23
        if len (att) == 2 and att [-1] == 'all':
            att = att [:-1]
        try:
            att = [int (a) - 1 for a in att]
        except ValueError as err:
            print ("Attach-load: %s" % err, file = f_err)
            return 23
        if att [0] >= len (loads) or att [0] < 0:
            print ("Load index %d out of range" % (att [0] + 1), file = f_err)
            return 23
        used_loads.add (att [0])
        try:
            m.register_load (loads [att [0]], *att [1:])
        except ValueError as err:
            print ("Error attaching load: %s" % err, file = f_err)
            return 23
    if len (used_loads) != len (loads):
        print ("Error: Not all loads were used", file = f_err)
        return 23
    p = args.phi.split (',')
    if len (p) != 3:
        print \
            ( "Invalid phi angle, need three comma-separated values"
            , file = f_err
            )
        return 23
    try:
        azimuth = Angle (float (p [0]), float (p [1]), int (p [2]))
    except Exception:
        print ("Invalid phi angle, need float, float, int", file = f_err)
        return 23
    p = args.theta.split (',')
    if len (p) != 3:
        print \
            ( "Invalid theta angle, need three comma-separated values"
            , file = f_err
            )
        return 23
    try:
        zenith = Angle (float (p [0]), float (p [1]), int (p [2]))
    except Exception:
        print ("Invalid theta angle, need float, float, int", file = f_err)
        return 23
    nf_count = None
    if args.near_field:
        nf = args.near_field.split (',')
        if len (nf) != 9:
            print ("Expecting 9 near-field parameters", file = f_err)
            return 23
        try:
            nf_count = [int (x) for x in nf [6:]]
        except ValueError as err:
            print ("Error near-field counts: %s" % err, file = f_err)
            return 23
        try:
            nf_start = [float (x) for x in nf [:3]]
        except ValueError as err:
            print ("Error near-field start: %s" % err, file = f_err)
            return 23
        try:
            nf_inc = [float (x) for x in nf [3:6]]
        except ValueError as err:
            print ("Error near-field inc: %s" % err, file = f_err)
            return 23
    options = set ()
    far_field = False
    for opt in args.option:
        if opt not in allowed_options:
            print ("Invalid print option: %s" % opt)
            return 23
        options.add (opt)
        if opt.startswith ('far'):
            far_field = True
    if not options:
        if nf_count:
            options.add ('near-field')
        else:
            options.add ('far-field')
            far_field = True
    if not args.frequency_steps or not args.frequency_increment:
        args.frequency_steps = 1
        args.frequency_increment = 0.0
    for k in range (args.frequency_steps):
        m.f = args.frequency + k * args.frequency_increment
        m.compute ()
        if args.frequency_steps != 1 and k == 0:
            print (m.frq_independent_as_mininec ())
        if 'near-field' in options:
            d = {}
            if args.nf_power:
                d ['pwr'] = args.nf_power
            m.compute_near_field (nf_start, nf_inc, nf_count, **d)
        if far_field:
            d = {}
            if args.ff_power:
                d ['pwr'] = args.ff_power
            if args.ff_distance:
                d ['dist'] = args.ff_distance
            m.compute_far_field (zenith, azimuth, **d)
        if args.frequency_steps == 1:
            print (m.as_mininec (options))
        else:
            print (m.frq_dependent_as_mininec (options))
# end def main

__all__ = \
    [ 'Angle'
    , 'Excitation'
    , 'Far_Field_Pattern'
    , 'Gauge_Wire'
    , 'Impedance_Load'
    , 'Medium'
    , 'Mininec'
    , 'Laplace_Load'
    , 'Wire'
    , 'ideal_ground'
    ]
