#!/usr/bin/python3
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

#!/usr/bin/python3

import sys
import copy
import time
import numpy as np
from datetime import datetime
from scipy.special import ellipk
from scipy.integrate import fixed_quad
from mininec.util  import format_float
from mininec.pulse import Pulse_Container, Pulse

legendre_cache = {}
try:
    from scipy.integrate._quadrature import _cached_roots_legendre
    legendre_cache [2] = [x / 2 for x in _cached_roots_legendre (2)]
    legendre_cache [4] = [x / 2 for x in _cached_roots_legendre (4)]
    legendre_cache [8] = [x / 2 for x in _cached_roots_legendre (8)]
except ImportError:
    pass

class Angle:
    """ Represents the stepping of Zenith and Azimuth angles
    """
    def __init__ (self, initial, inc, number):
        self.initial = initial
        self.inc     = inc
        self.number  = number
    # end def __init__

    def angle_deg (self):
        idx = np.array (range (self.number))
        a   = self.initial + idx * self.inc
        return a
    # end def angle_deg

    def angle_rad (self):
        return self.angle_deg () / 180 * np.pi
    # end def angle_rad

# end class Angle

class Connected_Wires:
    """ This is used to store a set of connected wires *and* the
        corresponding segments in one data structure.
    """

    def __init__ (self):
        self.wires         = set ()
        self.list          = []
        self.cached_pulses = None
        self.sgn_by_wire   = {}
    # end def __init__

    @property
    def pulses (self):
        if self.cached_pulses is None:
            self.cached_pulses = set (x [0] for x in self.pulse_iter ())
        return self.cached_pulses
    # end def pulses

    def add (self, wire, other_wire, end_idx, sign, sign2):
        assert wire not in self.wires
        self.wires.add (wire)
        self.list.append ((wire, other_wire, end_idx, sign))
        self.sgn_by_wire [other_wire] = sign2
    # end def add

    def idx (self, wire):
        """ This computes I1 or I2 from the Basic code, respectively
        """
        if self.list:
            # Forward linked segments are printed as 0
            if wire.n < self.list [0][0].n:
                return 0
            return (1 + self.list [0][0].n) * self.sgn_by_wire [wire]
        return 0
    # end def idx

    def is_connected (self, other):
        return other in self.wires
    # end def is_connected

    def pulse_iter (self):
        """ Yield pulse indeces sorted by wire index
        """
        for wire, ow, idx, s in self._iter ():
            yield (ow.end_segs [idx], s)
    # end def pulse_iter

    def _iter (self):
        for wire, ow, idx, s in sorted (self.list, key = lambda x: x [0].n):
            yield (wire, ow, idx, s)
    # end def _iter

    def __bool__ (self):
        return bool (self.list)
    # end def __bool__

    def __str__ (self):
        r = []
        for wire, ow, idx, s in self._iter ():
            r.append ('w: %d idx: %d s:%d' % (wire.n, ow.end_segs [idx], s))
        return '\n'.join (r)
    __repr__ = __str__

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
    def __init__ (self, azi, zen, gain, e_theta, e_phi, pwr_ratio):
        self.azi       = azi
        self.zen       = zen
        self.gain      = gain
        self.pwr_ratio = np.sqrt (pwr_ratio)
        self.e_theta   = e_theta * self.pwr_ratio
        self.e_phi     = e_phi   * self.pwr_ratio
    # end def __init__

    def db_as_mininec (self):
        r = []
        v, h, t = self.gain.T
        for theta, phi, v, h, t in zip \
            (self.zen.flat, self.azi.flat, v.flat, h.flat, t.flat):
            r.append \
                ( ('%s     ' * 4 + '%s')
                % ( format_float ((theta, phi))
                  + format_float ((v, h, t))
                  )
                )
        return '\n'.join (r)
    # end def db_as_mininec

    def abs_gain_as_mininec (self):
        e_t_abs = np.abs   (self.e_theta)
        e_p_abs = np.abs   (self.e_phi)
        e_t_ang = np.angle (self.e_theta) / np.pi * 180
        e_p_ang = np.angle (self.e_phi)   / np.pi * 180
        r = []
        for th, ph, et_abs, ep_abs, et_ang, ep_ang in zip \
            ( self.zen.flat, self.azi.flat
            , e_t_abs.flat, e_p_abs.flat, e_t_ang.flat, e_p_ang.flat
            ):
            r.append \
                ( '%6.2f %9.2f %20.3E % 8.2f %19.3E % 8.2f'
                % (th,  ph, et_abs, et_ang, ep_abs, ep_ang)
                )
        return '\n'.join (r)
    # end def abs_gain_as_mininec

# end class Far_Field_Pattern

class _Load:
    """ Base class of impedance loading.
        A load can be re-used with several pulses. We have an interface
        to add a pulse to a load. This is convenient if a load is placed
        on every pulse, e.g. for copper loading of all elements. Note
        that the original mininec code allowed *either* S-Parameter
        loads *or* impedance loads. We support both concurrently. If a
        pulse is loaded twice, loads appear to be in series.
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

class Series_RLC_Load (_Load):
    """ A load with R, L, C in series, an unspecified value is
        considered to be a 0-Ohm resistor.
        This was not in the original mininec code.
        But it could be modelled with a Laplace load.
        Frequency is in MHz (when calling impedance), otherwise we use
        metric units Ohm, Henry, Farad.
    >>> l = Series_RLC_Load (R = 0.1, L = 60e-6)
    >>> z = l.impedance (7)
    >>> print ("%g%+gj" % (z.real, z.imag))
    0.1+2638.94j
    >>> l = Series_RLC_Load (R = 1000, C = 60e-12)
    >>> z = l.impedance (7)
    >>> print ("%g%+gj" % (z.real, z.imag))
    1000-378.94j
    """
    def __init__ (self, R = None, L = None, C = None):
        super ().__init__ ()
        self.r = R
        self.l = L
        self.c = C
    # end def __init__

    def impedance (self, f):
        """ Impedance for given frequency
        """
        w = 2 * np.pi * f * 1e6
        x = 0 + 0j
        if self.r is not None:
            x += self.r
        if self.l is not None:
            x += w * self.l * 1j
        if self.c is not None:
            x -= 1 / (w * self.c) * 1j
        return x
    # end def impedance

# end class Series_RLC_Load

class Laplace_Load (_Load):
    """ Laplace s-Parameter (s = j omega) load from mininec implementation
        We get two lists of parameters. They represent the numerator and
        denominator coefficients, respectively (sequence a is the denominator
        and sequence b is the numerator). This uses s*L for inductance,
        1/(s*C) for capacitance, and R for resistors. These are combined
        with the usual rules for parallel and serial connection.
        Contrary to the Basic implementation, inductances are in H and
        capacitances are in F (not in µH/µF as in the Basic implementation),
        this is more convenient if there are higher-order s-Parameters.
        The frequency is still given in MHz, for compatibility with the
        interface where all frequencies are in MHz.
    >>> l = Laplace_Load (b = (1., 0.), a = (0., -2.193644e-9))
    >>> z = l.impedance (7.15)
    >>> print ("%g%+gj" % (z.real, z.imag))
    -0+10.1472j
    >>> l = Laplace_Load (b = (0., 225.998e-9), a = (1,))
    >>> z = l.impedance (7.15)
    >>> print ("%g%+gj" % (z.real, z.imag))
    0+10.1529j
    >>> l = Laplace_Load (b = (0.1, 60e-6), a = (1,))
    >>> z = l.impedance (7)
    >>> print ("%g%+gj" % (z.real, z.imag))
    0.1+2638.94j
    >>> l = Laplace_Load (b = (1, 1000 * 60e-12), a = (0, 60e-12))
    >>> z = l.impedance (7)
    >>> print ("%g%+gj" % (z.real, z.imag))
    1000-378.94j
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
        """ We multiply by s^^k for k in 0..n-1 for all n parameters
            Where s = 1j * omega = 2j * pi * f
            Note that the frequency is given in MHz.
        """
        w = 2 * np.pi * f * 1e6
        u = d = 0j
        m = 1.0
        for j in range (len (self.a)):
            u += self.b [j] * m
            d += self.a [j] * m
            m *= (1j * w)
        return u / d
    # end def impedance

# end class Laplace_Load

class Trap_Load (Laplace_Load):
    """ A trap consisting of L+R in series parallel to a C.
           +---L----R---+
        ---+            +---
           +-----C------+
        This is a convenience method as it can already be specified with
        the Laplace_Load type.
        Note that you should not specify R=0, otherwise you'll get a
        division by zero at the resonance frequency.
    >>> c = 10e-12
    >>> l = 10e-6
    >>> r = 1
    >>> t = Trap_Load (r, l, c)
    >>> z = t.impedance (15.91548635144039)
    >>> print ("%g%+gj" % (z.real, z.imag))
    1e+06+2.87559e-08j
    >>> t = Trap_Load (r, l, c)
    >>> z = t.impedance (15.915)
    >>> print ("%g%+gj" % (z.real, z.imag))
    996218+60882.7j
    >>> t = Trap_Load (r, l, c)
    >>> z = t.impedance (15.916)
    >>> print ("%g%+gj" % (z.real, z.imag))
    995915-64286.3j
    >>> t = Trap_Load (1e-10, l, c)
    >>> z = t.impedance (15.915494309189534)
    >>> print ("%g%+gj" % (z.real, z.imag))
    1e+16-1000j
    """

    def __init__ (self, R, L, C):
        super ().__init__ (a = (1, R*C, L*C), b = (R, L))
    # end def __init__

# end class Trap_Load

class Medium:
    """ This encapsulates the media (e.g. ground screen etc.)
        With permittivity and conductivity zero we asume ideal ground.
        Note that it seems only the first medium can have a
        ground screen of radials.
        The boundary is the type of boundary between different media (if
        there is more than one). It is either 'linear' (X-coordinate it
        seems) or 'circular'. If there is more than one medium we need
        the coordinate (distance X for linear and radius R for circular
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
        , permittivity, conductivity, height = 0
        , nradials = 0, radius = 0, dist = 0
        , boundary = 'linear', coord = 1e6
        ):
        self.permittivity = permittivity  # (dielectric constant) T(I)
        self.conductivity = conductivity  # V(I)
        self.nradials = nradials # number of radials   NR
        self.radius   = radius   # radial wire radius  RR
        self.coord    = coord    # U(I)
        self.height   = height   # H(I)
        self.boundary = boundary
        self.next     = None     # next medium
        self.prev     = None     # previous medium
        self.is_ideal = False
        if permittivity == 0 and conductivity == 0:
            self.is_ideal = True
            self.coord    = 0
            if self.nradials:
                raise ValueError ("Ideal ground may not use radials")
            if self.height != 0:
                raise ValueError ("Ideal ground must have height 0")
        if self.nradials:
            self.boundary = 'circular'
            # Radials extend to boundary of next medium
            if self.radius <= 0:
                raise ValueError ("Radius must be >0")
        else:
            self.radius = 0.0
        # Code from Line 628-633 in far field calculation
        if permittivity:
            if not self.conductivity:
                raise ValueError \
                    ("Non-ideal ground must have non-zero ground parameters")
    # end def __init__

    def as_mininec (self):
        r = []
        if not self.is_ideal:
            p = tuple \
                ( x.strip ()
                  for x in format_float ((self.permittivity, self.conductivity))
                )
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
        return 1 / np.sqrt (self.permittivity + -1j * self.conductivity / t)
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
        dirvec (CA, CB, CG)
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
        self.pulses     = []
        self.r = r
        if r <= 0:
            raise ValueError ("Radius must be >0")
        diff = self.p2 - self.p1
        if (diff == 0).all ():
            raise ValueError ("Zero length wire: %s %s" % (self.p1, self.p2))
        self.wire_len = np.linalg.norm (diff)
        self.seg_len  = self.wire_len / self.n_segments
        # Original comment: compute direction cosines
        # Unit vector in wire direction
        self.dirvec   = diff / self.wire_len
        self.end_segs = [None, None]
        # Links to previous/next connected wire (at start/end)
        # conn [0] contains wires that link to our first end
        # while conn [1] contains wires that link to our second end.
        self.conn = (Connected_Wires (), Connected_Wires ())
        self.n = None
        # This is Integral I1 in the paper(s), in the implementation
        # used in Basic this boils down to only wire-dependent
        # Parameters. So we can avoid re-computing this for each
        # combination of segments.
        self.i6 = (1 + np.log (16 * r / self.seg_len)) / np.pi / r
    # end def __init__

    @property
    def idx_1 (self):
        return self.idx (0)
    # end def idx_1

    @property
    def idx_2 (self):
        return self.idx (1)
    # end def idx_2

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

    def compute_connections (self, parent):
        """ Compute links to connected wires
            Also compute sets of indeces of pulses.
            Note that we're using a dictionary for matching endpoints,
            this reduces two nested loops over the wires which is O(N**2)
            to a single loop O(N).

            We do not use a second loop but instead put each wire end
            into a dictionary linking to the wire. That way the
            algorithm is O(N) not O(N^2). The dictionaries for the wire
            ends is end_dict, we store a tuple of end index and wire in
            this dictionary.

            In the future we may want to introduce fuzzy matching of
            endpoints (similar to how NEC does it). This means that we
            would probably first try to match via dictionary and only if
            that doesn't return a match we might use a binary search via
            one of the coordinates (and then determine the euclidian
            distance).

            Also compute pulses.
        """

        # This rolls the end-matching computation of 4
        # explicitly-programmed cases in the Basic program lines
        # 1325-1356 into a few statements.
        # The idea is to use a dictionary of end-point coordinates.
        for n1, current_end in enumerate (self.endpoints):
            if self.is_ground [n1]:
                continue
            ep_tuple = tuple (current_end)
            if ep_tuple in parent.end_dict:
                n2, other = parent.end_dict [ep_tuple]
                s = -1 if (n2 == n1) else 1
                other.conn [n2].add (self,  self, n1, s, s)
                self.conn  [n1].add (other, self, n1, 1, s)
            else:
                parent.end_dict [ep_tuple] = (n1, self)
        self.end_segs [0] = parent.pulses.pulse_idx
        if self.n_segments == 1 and self.idx_1 == 0:
            self.end_segs [0] = None
        npulse = self.n_segments - (not self.idx_1) - (not self.idx_2)
        self.end_segs [1] = parent.pulses.pulse_idx + npulse
        if self.n_segments == 1 and self.idx_2 == 0:
            self.end_segs [1] = None
        # inversion of Z component
        invz = np.array ([1, 1, -1])
        # First segment start
        seg  = np.copy (self.p1)
        inc  = self.dirvec * self.seg_len
        pu   = parent.pulses
        # Connection to other wire(s) at end 1
        if self.idx_1 != 0 and abs (self.idx_1) - 1 != self.n:
            assert not self.is_ground [0]
            other = parent.geo [abs (self.idx_1) - 1]
            sgn   = [np.sign (self.idx_1), 1]
            oinc  = other.dirvec * other.seg_len * sgn [0]
            prev  = self.p1 - oinc
            p = Pulse (pu, self.p1, prev, seg + inc, other, self, sgn = sgn)
            if self.n_segments == 1 and self.idx_2 == 0:
                p.c_per [1] = 0
            self.pulses.append (p)
        elif self.is_ground [0]:
            s = seg + inc
            p = Pulse (pu, self.p1, s * invz, s, self, self, gnd = 0)
            self.pulses.append (p)
        s0 = seg
        s1 = seg + self.dirvec * self.seg_len
        for i in range (self.n_segments - 1):
            s2 = seg + (i + 2) * self.dirvec * self.seg_len
            p  = Pulse (pu, s1, s0, s2, self, self)
            s0 = s1
            s1 = s2
            self.pulses.append (p)
            if i == 0 and self.idx_1 == 0:
                p.c_per [0] = 0
            if i == self.n_segments - 2 and self.idx_2 == 0:
                p.c_per [1] = 0
        # Second endpoint is slightly off in original Basic computation
        # because it is computed from the first endpoint
        p2  = seg + (self.n_segments) * self.dirvec * self.seg_len
        seg = seg + (self.n_segments - 1) * self.dirvec * self.seg_len
        # Connection to other wire(s) at end 2
        if self.idx_2 != 0 and abs (self.idx_2) - 1 != self.n:
            assert not self.is_ground [1]
            other = parent.geo [abs (self.idx_2) - 1]
            sgn   = [1, np.sign (self.idx_2)]
            oinc  = other.dirvec * other.seg_len * sgn [1]
            next  = p2 + oinc
            p = Pulse (pu, p2, seg, next, self, other, sgn = sgn)
            if self.n_segments == 1 and self.idx_1 == 0:
                p.c_per [0] = 0
            self.pulses.append (p)
        elif self.is_ground [1]:
            next = p2 - self.dirvec * self.seg_len * invz
            p = Pulse (pu, self.p2, seg, next, self, self, gnd = 1)
            self.pulses.append (p)
    # end def compute_connections

    def connections (self):
        return self.conn [0].wires.union (self.conn [1].wires)
    # end def connections

    def idx (self, end_idx):
        """ The indeces I1, I2 from the Basic code, this is -self.n when
            the end is grounded, the index of a connected wire (negative
            if the direction of the wire is reversed) if connected and 0
            otherwise. Used mainly when printing wires in mininec format.
        """
        if self.is_ground [end_idx]:
            return -(self.n + 1)
        return self.conn [end_idx].idx (self)
    # end def idx

    def is_connected (self, other):
        if other is self:
            return True
        for c in self.conn:
            if c.is_connected (other):
                return True
        return bool (self.connections ().intersection (other.connections ()))
    # end def is_connected

    def conn_as_str (self):
        """ Mostly for debugging connections
        """
        r = []
        r.append ('W: %d' % self.n)
        for n, c in enumerate (self.conn):
            r.append ('conn %s:' % 'lr' [n])
            r.append (str (c))
        return '\n'.join (r)
    # end def conn_as_str

    def pulse_idx_iter (self, yield_ends = True):
        for p in self.pulse_iter (yield_ends):
            yield p.idx
    # end def pulse_idx_iter

    def pulse_iter (self, yield_ends = True):
        for p in self.pulses:
            if p.wires [0] != p.wires [1] and not yield_ends:
                continue
            yield p
    # end def pulse_iter

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

def measure_time (method):
    """ Decorator for time measurement
        Needs member variable do_timing in calling class
    """
    def timer (self, *args, **kw):
        if self.do_timing:
            start_time = time.time ()
            retval = method (self, *args, **kw)
            end_time = time.time ()
            print \
                ( 'Time %7.3f for %s'
                % (end_time - start_time, method.__name__)
                , file = sys.stderr
                )
            return retval
        return method (self, *args, **kw)
    # end def timer
    return timer
# end measure_time

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
    c  = 299.8 # speed of light

    def __init__ (self, f, geo, media = None, print_opts = None, t = False):
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
                 forced to circular
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
        >>> w = []
        >>> w.append (Wire (10, 0,          0, 0, 21.414285, 0, 0, 0.001))
        >>> w.append (Wire (10, 21.4142850, 0, 0, 33.      , 0, 0, 0.001))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> print (m.geo_as_str ())
        W: 0
        conn l:
        <BLANKLINE>
        conn r:
        w: 1 idx: 9 s:1
        W: 1
        conn l:
        w: 0 idx: 9 s:1
        conn r:
        <BLANKLINE>
        """
        self.do_timing  = t
        self.f          = f
        self.media      = media
        self.loads      = []
        self.loadidx    = set ()
        self.check_ground ()
        self.sources    = []
        self.geo        = geo
        # Dictionary of ends to compute matches
        self.end_dict   = {}
        # Pulses
        self.pulses     = Pulse_Container ()
        self.print_opts = print_opts or set (('far-field',))
        if not self.media or len (self.media) == 1:
            self.boundary = 'linear'
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
        # The wave number 2 * pi / lambda
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
            self.boundary = 'linear'
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
        self.compute_currents ()
        # Used by far field and near field calculation
        self.power = sum (s.power for s in self.sources)
    # end def compute

    def compute_connectivity (self):
        """ In the original code this is done while parsing the
            individual wires from the geometry information.
        """
        for w in self.geo:
            w.compute_connections (self)
    # end def compute_connectivity

    @measure_time
    def compute_currents (self):
        """ This just solves the matrix equation, this function is
            introduced for timing measurements.
        """
        self.current = np.linalg.solve (self.Z, self.rhs)
    # end def compute_currents

    @measure_time
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
        # X1, Y1, Z1: vec real
        # X2, Y2, Z2: vec imag
        self.ff_dist  = dist
        self.ff_power = pwr or self.power
        self.far_field_angles = (zenith_angle, azimuth_angle)
        if self.media:
            nr, rr = self.media [0].nradials, self.media [0].radius
            media_coord     = np.array ([m.coord for m in self.media])
            media_height    = np.array ([m.height for m in self.media])
            media_impedance = np.array \
                ([m.impedance (self.f) for m in self.media])
        rd   = dist or 0
        pv   = self.pulses
        f3   = pv.sign * self.w * pv.seg_len / 2
        k9   = .016678 / self.power
        # cos, -sin for azi and zen angles
        acs  = np.e ** (-1j * azimuth_angle.angle_rad ())
        zcs  = np.e ** (-1j * zenith_angle.angle_rad ())
        zcs_m, acs_m = np.meshgrid (zcs, acs)
        rvec = np.array (
            [ -zcs_m.imag * acs_m.real + 1j * (zcs_m.real * acs_m.real)
            ,  zcs_m.imag * acs_m.imag - 1j * (zcs_m.real * acs_m.imag)
            ,  zcs_m
            ]).T
        gain = np.zeros (rvec.shape, dtype = complex)
        shp  = list (rvec.shape)
        shp.insert (-1, len (pv))
        rvrp = np.reshape (np.repeat (rvec, len (pv), axis = 1), shp)
        zen_d, azi_d = np.meshgrid \
            (zenith_angle.angle_deg (), azimuth_angle.angle_deg ())
        for k in self.image_iter ():
            kvec  = np.array ([1, 1, k])
            kvec2 = np.array ([k, k, 1])
            kv2   = np.tile (kvec2, (len (pv), 2, 1))
            kv2 [pv.ground] = np.array ([0, 0, 0])
            # Vectorized computation of standard case
            if k == 1 or not self.media or self.media [0].is_ideal:
                kv2g = np.copy (kv2)
                kv2g [pv.inv_ground] = np.array ([0, 0, 2])
                if k < 0:
                    kv2g [pv.inv_ground] = np.array ([0, 0, 0])
                s2   = self.w * np.sum \
                    (pv.point * kvec * rvrp.real, axis = 3)
                s    = np.e ** (1j * s2)
                sshp = list (s.shape)
                sshp.insert (-1, 2)
                ss   = np.reshape (np.repeat (s, 2, axis = 1), sshp)
                b    = f3.T * ss * self.current
                bshp = list (b.shape)
                bshp.insert (2, 3)
                bs   = np.zeros (bshp, dtype = complex)
                bs [:, :, 0, :, :] = b
                bs [:, :, 1, :, :] = b
                bs [:, :, 2, :, :] = b
                gain += np.sum \
                    (kv2g.T * pv.dirvec.T * bs, axis = (3, 4))
            else:
                kv2 [pv.inv_ground] = np.array ([0, 0, 0])
                assert self.media
                rt3 = rvrp [:, :, :, 2]
                # begin by finding specular distance
                cond = rt3.real != 0
                t4 = np.zeros (rt3.shape)
                t4 [rt3.real == 0] = 1e5
                t4 [cond] = \
                    (-pv.point.T [2] * rt3.imag) [cond] / rt3.real [cond]
                b9 = (t4.T * acs_m.real).T + pv.point.T [0]
                if self.boundary != 'linear':
                    # Pythagoras in case of circular boundary
                    b9 *= b9
                    b9 += ((-t4.T * acs_m.imag).T + pv.point.T [1]) ** 2
                    b9 = np.sqrt (b9)
                # search for the corresponding medium
                # Find minimum index where b9 > coord
                # Note: the coord of a medium is the perimeter
                # in the linear case in x-direction, otherwise
                # circular, b9 > media_coord will be True as
                # long as b9 is greater and is False when it
                # exceeds the perimenter, the argmin finds that
                # first False value.
                shp = list (b9.shape)
                shp.insert (0, len (media_coord))
                tc = np.reshape \
                    (np.tile (media_coord, (np.prod (b9.shape),1)).T, shp)
                j2 = np.argmin (b9 > tc, axis = 0)
                z45 = media_impedance [j2]
                if nr != 0:
                    prod = nr * rr
                    r = b9 + prod
                    z8  = self.w * r * np.log (r / prod) / nr
                    s89 = z45 * z8 * 1j
                    t89 = z45 + (z8 * 1j)
                    z45 [j2 == 0] = (s89 / t89) [j2 == 0]
                # form SQR(1-Z^2*SIN^2)
                w671 = w67 = np.sqrt (1 - z45 ** 2 * rt3.imag ** 2)
                # vertical reflection coefficient
                s89 = rt3.real - w67 * z45
                t89 = rt3.real + w67 * z45
                v89 = s89 / t89
                # horizontal reflection coefficient
                s89 = w67 - rt3.real * z45
                t89 = w67 + rt3.real * z45
                h89 = s89 / t89 - v89
                # Reshape to include two ends of segments
                vsp = list (v89.shape)
                vsp.insert (2, 2)
                h89o = h89
                h89 = np.zeros (vsp, dtype = complex)
                h89 [:, :, 0, :] = h89o
                h89 [:, :, 1, :] = h89o
                vsp.insert (2, 3)
                v89o = v89
                v89 = np.zeros (vsp, dtype = complex)
                v89 [:, :, 0, 0, :] = v89o
                v89 [:, :, 0, 1, :] = v89o
                v89 [:, :, 1, 0, :] = v89o
                v89 [:, :, 1, 1, :] = v89o
                v89 [:, :, 2, 0, :] = v89o
                v89 [:, :, 2, 1, :] = v89o
                # compute contribution to sum
                shp = j2.shape + (3,)
                h   = np.zeros (shp)
                h [:, :, :, 2] = media_height [j2] * 2
                sh  = pv.point - h
                s2 = self.w * np.sum \
                    (sh * rvrp.real * kvec, axis = 3)
                s   = np.e ** (1j * s2)
                sshp = list (s.shape)
                sshp.insert (-1, 2)
                ss   = np.reshape (np.repeat (s, 2, axis = 1), tuple (sshp))
                b    = f3.T * ss * self.current
                bshp = list (b.shape)
                bshp.insert (2, 3)
                bs   = np.zeros (bshp, dtype = complex)
                bs [:, :, 0, :, :] = b
                bs [:, :, 1, :, :] = b
                bs [:, :, 2, :, :] = b
                w67  = bs * v89
                acs_ms = np.repeat (acs_m.T, len (pv) * 2, axis = 1)
                acs_ms = np.reshape (acs_ms, acs_m.T.shape + (2, len (pv)))
                d   = acs_ms.imag * pv.dirvec.T [0] \
                    + acs_ms.real * pv.dirvec.T [1]
                z67 = d * b * h89
                tm1 = np.zeros (w67.shape, dtype = complex)
                tm1 [:, :, 0, :, :] = \
                    (acs_ms.imag * z67.real + 1j * acs_ms.imag * z67.imag)
                tm1 [:, :, 1, :, :] = \
                    (acs_ms.real * z67.real + 1j * acs_ms.real * z67.imag)
                dim = end = 0
#                with open ('/tmp/bla.log', 'a') as f:
#                    for a in range (azi_d.shape [0]):
#                        for z in range (zen_d.shape [-1]):
#                            #print ('--------------------', file = f)
#                            for dim in range (3):
#                                for end in range (2):
#                                    for p in self.pulses:
#                                        v = w67 [z, a, dim, end, p.idx]
#                                        print ('dim: %d end: %d p: %d w67: %.6e %.6ej'
#                                              % (dim, end, p.idx, v.real, v.imag), file = f)
#                                        #import pdb; pdb.set_trace ()
#                                        v = h89 [z, a, end, p.idx]
#                                        #v = h89o [z, a, p.idx]
#                                        #v = v89 [z, a, dim, end, p.idx]
#                                        print ('dim: %d end: %d p: %d b: %.6e %.6ej'
#                                              % (dim, end, p.idx, v.real, v.imag), file = f)
#                                        v = zcs [z]
#                                        print ('dim: %d end: %d p: %d zcs: %.6e %.6ej'
#                                              % (dim, end, p.idx, v.real, v.imag), file = f)
                gain += np.sum \
                    ((pv.dirvec.T * w67 + tm1) * kv2.T, axis = (3, 4))
        for i_a, v21 in enumerate (acs):
            for i_z, rt3 in enumerate (zcs):
                # Can we somehow reduce this with complex artithmetics?
                # What is/was the intention of that code?
                #rt1  = -rt3.imag * v21.real + 1j * (rt3.real * v21.real)
                #rt2  =  rt3.imag * v21.imag - 1j * (rt3.real * v21.imag)
                #rvec = np.array ([rt1, rt2, rt3])
                for k in self.image_iter ():
                    kvec  = np.array ([1, 1, k])
                    kvec2 = np.array ([k, k, 1])
                    kv2   = np.tile (kvec2, (len (pv), 2, 1))
                    kv2 [pv.ground] = np.array ([0, 0, 0])
                    # Vectorized computation of standard case
                    if k == 1 or not self.media or self.media [0].is_ideal:
                        kv2g = np.copy (kv2)
                        kv2g [pv.inv_ground] = np.array ([0, 0, 2])
                        if k < 0:
                            kv2g [pv.inv_ground] = np.array ([0, 0, 0])
                        s2 = self.w * np.sum \
                            (pv.point * rvec [i_z, i_a].real * kvec, axis = 1)
                        s  = np.e ** (1j * s2)
                        b  = f3.T * s * self.current
#                        zoppel = (kv2g.T * b * pv.dirvec.T).T
#                        with open ('/tmp/bla.log', 'a') as f:
#                            for p in self.pulses:
#                                for bla in range (2):
#                                    value = zoppel [p.idx][bla]
#                                    for v in value:
#                                        print ('%.6e %.6ej ' % (v.real, v.imag), file = f, end = '')
#                                    print (file = f)
##                        gain [i_z][i_a] += np.sum \
##                            ((kv2g.T * b * pv.dirvec.T).T, axis = (0, 1))
#                        with open ('/tmp/bla.log', 'a') as f:
#                            print ('VEC:', file = f, end = ' ')
#                            for v in gain [i_z][i_a]:
#                                print ('%.6e %.6ej ' % (v.real, v.imag), file = f, end = '')
#                            print (file = f)
                        continue
                    else:
                    #elif False:
                        kv2 [pv.inv_ground] = np.array ([0, 0, 0])
                        assert self.media
                        # begin by finding specular distance
                        t4 = 1e5
                        if rt3.real != 0:
                            t4 = -pv.point.T [2] * rt3.imag / rt3.real
                        b9 = t4 * v21.real + pv.point.T [0]
                        if self.boundary != 'linear':
                            # Pythagoras in case of circular boundary
                            b9 *= b9
                            b9 += (pv.point.T [1] - t4 * v21.imag) ** 2
                            b9 = np.sqrt (b9)
                        # search for the corresponding medium
                        # Find minimum index where b9 > coord
                        # Note: the coord of a medium is the perimeter
                        # in the linear case in x-direction, otherwise
                        # circular, b9 > media_coord will be True as
                        # long as b9 is greater and is False when it
                        # exceeds the perimenter, the argmin finds that
                        # first False value.
                        tc = np.tile (media_coord, (len (b9), 1))
                        j2 = np.argmin ((b9 > tc.T).T, axis = 1)
                        z45 = media_impedance [j2]
                        if nr != 0:
                            prod = nr * rr
                            r = b9 + prod
                            z8  = self.w * r * np.log (r / prod) / nr
                            s89 = z45 * z8 * 1j
                            t89 = z45 + (z8 * 1j)
                            z45 [j2 == 0] = (s89 / t89) [j2 == 0]
                        # form SQR(1-Z^2*SIN^2)
                        w671 = w67 = np.sqrt (1 - z45 ** 2 * rt3.imag ** 2)
                        # vertical reflection coefficient
                        s89 = rt3.real - w67 * z45
                        t89 = rt3.real + w67 * z45
                        v89 = s89 / t89
                        # horizontal reflection coefficient
                        s89 = w67 - rt3.real * z45
                        t89 = w67 + rt3.real * z45
                        h89 = s89 / t89 - v89
                        # compute contribution to sum
                        z   = np.zeros (len (self.pulses))
                        h   = media_height [j2]
                        sh  = pv.point - np.array ([z, z, 2 * h]).T
                        s2 = self.w * np.sum \
                            (sh * rvec [i_z, i_a].real * kvec, axis = 1)
                        s   = np.e ** (1j * s2)
                        b   = f3.T * s * self.current
                        w67 = b * v89
                        d   = v21.imag * pv.dirvec.T [0] \
                            + v21.real * pv.dirvec.T [1]
                        z67 = d * b * h89
                        tm1 = np.array \
                            ([         v21.imag * z67.real
                               + 1j * (v21.imag * z67.imag)
                             ,         v21.real * z67.real
                               + 1j * (v21.real * z67.imag)
                             , np.zeros ((2, len (self.pulses)))
                            ])
                        dim = end = 0
#                        with open ('/tmp/bla2.log', 'a') as f:
#                            #print ('--------------------', file = f)
#                            for dim in range (3):
#                                for end in range (2):
#                                    for p in self.pulses:
#                                        v = w67 [end, p.idx]
#                                        print ('dim: %d end: %d p: %d w67: %.6e %.6ej'
#                                              % (dim, end, p.idx, v.real, v.imag), file = f)
#                                        #import pdb; pdb.set_trace ()
#                                        v = h89 [p.idx]
#                                        print ('dim: %d end: %d p: %d b: %.6e %.6ej'
#                                              % (dim, end, p.idx, v.real, v.imag), file = f)
#                                        v = zcs [i_z]
#                                        print ('dim: %d end: %d p: %d zcs: %.6e %.6ej'
#                                              % (dim, end, p.idx, v.real, v.imag), file = f)
##                        gain [i_z][i_a] += np.sum \
##                            ((pv.dirvec.T * w67 + tm1) * kv2.T, axis = (1, 2))
                        zoppel = ((pv.dirvec.T * w67 + tm1) * kv2.T).T
#                        with open ('/tmp/bla.log', 'a') as f:
#                            for p in self.pulses:
#                                for bla in range (2):
#                                    print (' s2: %.7e %.7ej' % (s2 [p.idx].real, s2 [p.idx].imag), file = f)
#                                    print ('v89: %.7e %.7ej' % (v89 [p.idx].real, v89 [p.idx].imag), file = f)
#                                    print ('w67: %.7e %.7ej' % (w67 [bla][p.idx].real, w67 [bla][p.idx].imag), file = f)
#                                    print ('z67: %.7e %.7ej' % (z67 [bla][p.idx].real, z67 [bla][p.idx].imag), file = f)
#                                    value = zoppel [p.idx][bla]
#                                    for v in value:
#                                        print ('%.6e %.6ej ' % (v.real, v.imag), file = f, end = '')
#                                    print (file = f)
                        #with open ('/tmp/bla.log', 'a') as f:
                        #    print ('VEC:', file = f, end = ' ')
                        #    for v in gain [i_z][i_a]:
                        #        print ('%.6e %.6ej ' % (v.real, v.imag), file = f, end = '')
                        #    print (file = f)
                        continue
                    for p in self.pulses:
                        # Code at 716, 717
                        # For mirror image do nothing if one end is grounded
                        if k <= 0 and p.ground.any ():
                            continue
                        # for each end of pulse compute
                        # a contribution to e-field
                        # End of this loop (goto for continue) is 812
                        for f5 in range (2):
                            wire = p.wires [f5]
                            f3 = p.sign [f5] * self.w * wire.seg_len / 2
                            # Line 723, 724
                            # No contribution by grounded end
                            if p.gnd_sgn [f5] < 0:
                                continue
                            # Standard case (condition Line 725, 726)
                            if  (  k == 1
                                or not self.media
                                or self.media [0].is_ideal
                                ):
                                s2  = self.w * sum (p.point * rvec [i_z, i_a].real * kvec)
                                s   = np.e ** (1j * s2)
                                b   = f3 * s * self.current [p.idx]
                                # Line 733
                                if p.ground.any ():
                                    # grounded ends, only update last axis
                                    v = np.array ([0, 0, 1])
                                    gain [i_z][i_a] += 2 * b * wire.dirvec * v
                                    continue
#                                with open ('/tmp/bla.log', 'a') as f:
#                                    for v in tuple (kvec2 * b * wire.dirvec):
#                                        print ('%.6e %.6ej ' % (v.real, v.imag), file = f, end = '')
#                                    print (file = f)
                                gain [i_z][i_a] += kvec2 * b * wire.dirvec
                                continue
                            else: # real ground case (Line 747)
                                assert self.media
                                # begin by finding specular distance
                                t4 = 1e5
                                if rt3.real != 0:
                                    t4 = -p.point [2] * rt3.imag / rt3.real
                                b9 = t4 * v21.real + p.point [0]
                                if self.boundary != 'linear':
                                    # Hmm this is pythagoras?
                                    b9 *= b9
                                    b9 += (p.point [1] - t4 * v21.imag) ** 2
                                    b9 = np.sqrt (b9)
                                # search for the corresponding medium
                                # Find minimum index where b9 > media_coord
                                j2 = np.argmin ((b9 > media_coord) * media_coord)
                                z45 = self.media [j2].impedance (self.f)
                                # Line 764, 765
                                if nr != 0 and b9 <= media_coord [0]:
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
                                assert self.media and j2 < len (self.media)
                                h = self.media [j2].height
                                sh  = p.point - np.array ([0, 0, 2 * h])
                                s2  = self.w * sum (kvec * sh * rvec [i_z, i_a].real)
                                s   = np.e ** (1j * s2)
                                b   = f3 * s * self.current [p.idx]
                                w67 = b * v89
                                d   = v21.imag * wire.dirvec [0] \
                                    + v21.real * wire.dirvec [1]
                                z67 = d * b * h89
                                with open ('/tmp/bla.log', 'a') as f:
                                    print (' s2: %.7e %.7ej' % (s2.real, s2.imag), file = f)
                                    print ('v89: %.7e %.7ej' % (v89.real, v89.imag), file = f)
                                    print ('w67: %.7e %.7ej' % (w67.real, w67.imag), file = f)
                                    print ('z67: %.7e %.7ej' % (z67.real, z67.imag), file = f)
                                tm1 = np.array \
                                    ([         v21.imag * z67.real
                                       + 1j * (v21.imag * z67.imag)
                                     ,         v21.real * z67.real
                                       + 1j * (v21.real * z67.imag)
                                     , 0
                                    ])
                                gain [i_z][i_a] += (wire.dirvec * w67 + tm1) * kvec2
                                with open ('/tmp/bla.log', 'a') as f:
                                    for v in (wire.dirvec * w67 + tm1) * kvec2:
                                        print ('%.6e %.6ej ' % (v.real, v.imag), file = f, end = '')
                                    print (file = f)
#                with open ('/tmp/bla.log', 'a') as f:
#                    print ('VEC:', file = f, end = ' ')
#                    for v in gain [i_z][i_a]:
#                        print ('%.6e %.6ej ' % (v.real, v.imag), file = f, end = '')
#                    print (file = f)
        h12 = np.sum (gain * rvec.imag, axis = 2) * self.g0 * -1j
        vv  = np.array ([acs_m.imag, acs_m.real]).T
        x34 = np.sum (gain [:, :, :2] * vv, axis = 2) * self.g0 * -1j
        # pattern in dBi
        p123 = np.ones ((len (zcs), len (acs), 3), dtype = float) * -999
        t1 = k9 * (h12.real ** 2 + h12.imag ** 2)
        t2 = k9 * (x34.real ** 2 + x34.imag ** 2)
        t3 = t1 + t2
        # calculate values in dBi
        t123 = np.array ([t1.T, t2.T, t3.T]).T
        cond = t123 > 1e-30
        p123 [cond] = np.log (t123 [cond]) / np.log (10) * 10
        if rd != 0:
            h12 /= rd
            x34 /= rd
        rat = self.ff_power / self.power
        self.far_field = \
            Far_Field_Pattern (azi_d, zen_d, p123, h12.T, x34.T, rat)
    # end def compute_far_field

    @measure_time
    def compute_impedance_matrix (self):
        """ This starts at line 195 (with entry-point for gosub at 196)
            in the original basic code.
        >>> s  = Excitation (1, 0)
        >>> s2 = Excitation (1, 30)
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> m = Mininec (7, w)
        >>> m.register_source (s2, 4)
        >>> m.compute_impedance_matrix ()
        >>> n = len (m.pulses)
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
        n    = len (self.pulses)
        self.Z = np.zeros ((n, n), dtype=complex)
        for p_i in self.pulses:
            i = p_i.idx
            for p_j in self.pulses:
                j = p_j.idx
                # 230
                for k in self.image_iter ():
                    if p_j.ground.any () and k < 0:
                        continue
                    f8 = 0
                    if k >= 0:
                        # set flag to avoid redundant calculations
                        if  (   p_i.wires [0] == p_i.wires [1]
                            and p_j.wires [0] == p_j.wires [1]
                            ):
                            if p_i.wires [0] == p_j.wires [0]:
                                f8 = 1
                            if p_i == p_j:
                                f8 = 2
                    # This was a conditional goto 317 in line 246
                    if k < 0 or self.Z [i][j].real == 0:
                        vp = self.vector_potential (k, p_i, p_j, 0.5)
                        u = vp * p_j.sign [1]
                        # compute PSI(M,N-1/2,N)
                        if f8 < 2:
                            vp = self.vector_potential (k, p_i, p_j, -0.5)
                        v = vp * p_j.sign [0]
                        # S(N+1/2)*PSI(M,N,N+1/2) + S(N-1/2)*PSI(M,N-1/2,N)
                        # We use p_j.sgn, the sign resulting from only
                        # the inverted wire, not from grounding
                        f6v  = np.array ([1, 1, p_j.gnd_sgn [0]])
                        f7v  = np.array ([1, 1, p_j.gnd_sgn [1]])
                        di1  = p_j.wires [0].dirvec
                        di2  = p_j.wires [1].dirvec
                        kvec = np.array ([1, 1, k])
                        # We do real and imaginary part in one go:
                        vec3 = (f7v * u * di2 + f6v * v * di1) * kvec
                        zzz = sum \
                            ( p_i.dir_sgn [idx]
                            * p_i.wires [idx].seg_len
                            * p_i.wires [idx].dirvec
                            for idx in (0, 1)
                            )
                        d    = self.w2 * sum (vec3 * zzz)
                        # compute PSI(M+1/2,N,N+1)
                        if f8 < 2:
                            if f8 == 1:
                                u56 = p_j.sign [1] * u + vp
                            else:
                                u56 = self.scalar_potential \
                                    (k, p_i, p_j, 0.5, 1)
                            # compute PSI(M-1/2,N,N+1)
                            # Code at 291
                            sp = self.scalar_potential (k, p_i, p_j, -.5, 1)
                            seglen = p_j.wires [1].seg_len
                            u12 = (sp - u56) / seglen
                            # compute PSI(M+1/2,N-1,N)
                            u34 = self.scalar_potential (k, p_i, p_j, .5, -1)
                            # compute PSI(M-1/2,N-1,N)
                            if f8 >= 1:
                                sp = u56
                            else:
                                sp = self.scalar_potential \
                                    (k, p_i, p_j, -.5, -1)
                            # gradient of scalar potential contribution
                            seglen = p_j.wires [0].seg_len
                            u12 += (u34 - sp) / seglen
                        else:
                            sp = self.scalar_potential (k, p_i, p_j, -.5, 1)
                            seglen = p_j.wires [1].seg_len
                            sg  = p_j.sign [1]
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
                    if j + 1 >= len (self.pulses):
                        continue
                    p1 = j + 1
                    p_j_next = self.pulses [j + 1]
                    if  (  p_j_next.wires [0] != p_j_next.wires [1]
                        or p_j_next.ground.any ()
                        ):
                        continue
                    d = p_j.wires [1].dirvec
                    # 325-327
                    # The original condition was reached for the t-ant.pym
                    # example with j == 7.
                    # Note: the reversed original tested for d [0] + d [1] != 0
                    # This was probably meant to check that both are nonzero.
                    # (Resulting in a test for a grounded vertical wire)
                    # Since an empty matrix element is recomputed anyway
                    # adding a large test (with questionable semantics)
                    # here doesn't make much sense.
                    # So we kept only the condition with equal wires.
                    if p_j_next.wires [1] == p_j.wires [1]:
                        self.Z [i + 1][j + 1] = self.Z [i][j]
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
                if  (   self.pulses [j].ground.any ()
                    and self.media is not None
                    ):
                    f2 *= 2
                # Weird, the imag part goes to the real Z component and
                # vice-versa, the contribution to the real part is
                # negated, the contribution to the imag part not
                self.Z [j][j] += -f2 * l.impedance (self.f) * 1j
    # end def compute_impedance_matrix_loads

    def geo_as_str (self):
        r = []
        for g in self.geo:
            r.append (g.conn_as_str ())
        return '\n'.join (r)
    # end def geo_as_str

    def nf_helper (self, k, v1, pulse):
        """ Compute potentials in near field calculation
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        >>> nf66 = 0.4792338 -0.1544592j
        >>> nf75 = 0.3218219 -0.1519149j
        >>> ex   = (nf66 + nf75) * w [0].dirvec
        >>> vec0 = np.array ([0, -1, -1])
        >>> r    = m.nf_helper (1, vec0, m.pulses [0])
        >>> assert r [1] == 0 and r [2] == 0
        >>> print ("%.7f %.7fj" % (ex [0].real, ex [0].imag))
        0.8010557 -0.3063741j
        >>> print ("%.7f %.7fj" % (r [0].real, r [0].imag))
        0.8010557 -0.3063742j
        >>> assert np.linalg.norm (ex - r) < 1e-7
        """
        kvec = np.array ([1, 1, k])
        v6   = np.array ([1, 1, pulse.gnd_sgn [0]])
        v7   = np.array ([1, 1, pulse.gnd_sgn [1]])
        dir  = [w.dirvec for w in pulse.wires]
        # compute psi(0,J,J+.5)
        v2, vv = pulse.dvecs (0.5)
        v2     = v1 - kvec * v2
        vv     = v1 - kvec * vv
        u = self.psi (v2, vv, k, 0.5, pulse.wires [1], exact = False)

        # compute psi(0,J-.5,J)
        v2, vv = pulse.dvecs (-0.5)
        v2 = v1 - kvec * v2
        vv = v1 - kvec * vv
        v  = self.psi (v2, vv, k, 0.5, pulse.wires [0], exact = False)
        v *= pulse.sign [0]

        # real part of vector potential contribution
        # imaginary part of vector potential contribution
        return (v * dir [0] * v6 + u * dir [1] * v7) * kvec
    # end def nf_helper

    @measure_time
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
                # Loop over pulses
                for p in self.pulses:
                    j   = p.idx
                    u56 = 0j
                    for k in self.image_iter ():
                        if p.ground.any () and k < 0:
                            continue
                        # compute vector potential A
                        v35_e = self.nf_helper (k, vec, p)
                        v35_h = np.array \
                            ([self.nf_helper (k, v [i], p) for v in v0m])
                        # At this point comment notes
                        # magnetic field calculation completed
                        # and jumps to 1042 if H field
                        # We compute both, E and H and continue
                        d   = sum (v35_e * t567 [i]) * self.w2
                        # compute psi(.5,J,J+1)
                        u    = self.psi_near_field_56 \
                            (vec, t567 [i], k, .5, p, 1)
                        # compute psi(-.5,J,J+1)
                        tmp = self.psi_near_field_56 \
                            (vec, t567 [i], k, -.5, p, 1)
                        u   = (tmp - u) / p.wires [1].seg_len
                        # compute psi(.5,J-1,J)
                        u34  = self.psi_near_field_56 \
                            (vec, t567 [i], k, .5, p, -1)
                        # compute psi(-.5,J-1,J)
                        tmp = self.psi_near_field_56 \
                            (vec, t567 [i], k, -.5, p, -1)
                        # gradient of scalar potential
                        u56 += (u + (u34 - tmp) / p.wires [0].seg_len + d) * k
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

    @measure_time
    def compute_rhs (self):
        rhs = np.zeros (len (self.pulses), dtype=complex)
        for src in self.sources:
            f2 = -1j/self.m
            if self.pulses [src.idx].ground.any ():
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

    def integral_i2_i3 (self, t, vec2, vecv, k, wire, exact_kernel = False):
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
        >>> wire = m.geo [0]
        >>> m.register_source (s, 4)
        >>> vv = np.array ([3.21214267693, 0, 0])
        >>> v2 = np.array ([1.07071418591, 0, 0])
        >>> t  = np.array ([0.980144947186])
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire, False)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819959 -0.1414754j
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire, True)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819959 -0.1414754j

        # Then a thick wire without exact kernel
        # Original produces
        # 0.2819941 -0.1414753j
        >>> w [0].r = 0.01
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire, False)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819941 -0.1414753j

        # Then a thick wire *with* exact kernel
        # Original produces
        # -2.290341 -0.1467051j
        >>> vv = np.array ([ 1.07071418591, 0, 0])
        >>> v2 = np.array ([-1.07071418591, 0, 0])
        >>> t  = np.array ([0.4900725])
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire, True)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -2.2903413 -0.1467052j

        # Original produces
        # -4.219833E-02 -4.820928E-02j
        >>> vv = np.array ([16.06072, 0, 0])
        >>> v2 = np.array ([13.91929, 0, 0])
        >>> t  = np.array ([0.7886752])
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire, False)
        >>> print ("%.8f %.8fj" % (r.real, r.imag))
        -0.04219836 -0.04820921j

        # Original produces
        # -7.783058E-02 -.1079738j
        # But *ADDED TO THE PREVIOUS RESULT*
        >>> vv = np.array ([16.06072, 0, 0])
        >>> v2 = np.array ([13.91929, 0, 0])
        >>> t  = np.array ([0.2113249])
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire, False)
        >>> print ("%.8f %.8fj" % (r.real, r.imag))
        -0.03563231 -0.05976447j
        """
        t34  = np.zeros (t.shape, dtype = complex)
        if k < 0:
            vec3 = vecv + (vec2 - vecv) * t[:,np.newaxis]
        else:
            vec3 = vec2 + (vecv - vec2) * t[:,np.newaxis]
        d = d3 = np.linalg.norm (vec3, axis = 1)
        # MOD FOR SMALL RADIUS TO WAVELENGTH RATIO
        if wire.r > self.srm:
            # SQUARE OF WIRE RADIUS
            a2 = wire.r * wire.r
            d3 = d3 * d3
            d  = np.sqrt (d3 + a2)
            # criteria for using reduced kernel
            if exact_kernel:
                # exact kernel calculation with elliptic integral
                b = d3 / (d3 + 4 * a2)
                v0 = ellipk (1 - b) * np.sqrt (1 - b)
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

    def fast_quad (self, a, b, args, n):
        """ This uses the idea of the original pre-scaled gauss
            parameters. Computation of the integrals in psi is a little
            faster that way. Note that fixed_quad is a pure python
            implementation which does the necessary integral bounds
            scaling. Since we're always integrating from 0 to 1/f we can
            speed things up here and avoid some multiplications.
            Note that this needs access to the internals of
            scipy.integral and we fall back to fixed_quad if this
            interface changes. Also this is a special case that works
            only for the lower bound being 0.
            The test uses a non-cached order of integration to test the
            else part of the if statement.
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)
        >>> vec2 = np.zeros (3)
        >>> vecv = np.array ([1.070714, 0, 0])
        >>> args = vec2, vecv, 1, w [0], True
        >>> r = m.fast_quad (0, 1, args, 5)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        55.5802636 -0.1465045j
        """
        if n in legendre_cache and a == 0:
            x, w = legendre_cache [n]
            y = (x + .5) * b
            return np.sum (w * self.integral_i2_i3 (y, *args), axis = -1)
        else:
            r, p = fixed_quad \
                (self.integral_i2_i3, a, b, args = args, n = n)
            return r / b
    # end def fast_quad

    def psi (self, vec2, vecv, k, scale, wire, exact = False, fvs = 0):
        """ Common code for entry points at 56, 87, and 102.
            This code starts at line 135.
            The variable fvs is used to distiguish code path at the end.
            Both p2 and p3 used to be floating-point segment indeces.
            We now directly pass the difference, it is always positive
            and can be 1 or 0.5.
            The variable p4 was the index of the wire, we now pass wire
            directly.
            vec2 replaces (X2, Y2, Z2)
            vecv replaces (V1, V2, V3)
            i6: Use reduced kernel if 0, this was I6! (single precision)
                So beware: condition "I6!=0" means variable I6! is == 0
                The exclamation mark is part of the variable name :-(

            Input:
            vec2, vecv
            k:
            scale (used to be p3 - p2)
            wire: the wire
            fvs: scalar vs. vector potential
            is_near: This originally tested input C$ for "N" which is
                     the selection of near field compuation, this forces
                     non-exact kernel, see exact flag
            Output:
            vec2:
            vecv:
            t1:
            t2:

        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        # Original produces:
        # 5.330494 -0.1568644j
        >>> vec2 = np.zeros (3)
        >>> vecv = np.array ([1.070714, 0, 0])
        >>> r = m.psi (vec2, vecv, 1, 0.5, m.geo [0], exact = True)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        5.3304830 -0.1568644j

        # Original produces:
        # -8.333431E-02 -0.1156091j
        >>> vec2 = np.array ([13.91929, 0, 0])
        >>> vecv = np.array ([16.06072, 0, 0])
        >>> x = m.psi
        >>> r = x (vec2, vecv, k=1, scale=1, wire=m.geo [0], fvs = 1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -0.0833344 -0.1156090j
        """
        # magnitude of S(U) - S(M)
        d0 = np.linalg.norm (vec2)
        # magnitude of S(V) - S(M)
        d3 = np.linalg.norm (vecv)
        # magnitude of S(V) - S(U)
        s4 = scale * wire.seg_len
        # order of integration
        # gauss_n order gaussian quadrature
        i6 = 0
        f2 = 1
        gauss_n = 8
        t = (d0 + d3) / wire.seg_len
        # CRITERIA FOR EXACT KERNEL
        if t > 1.1 or not exact:
            # This starts line 165
            if t > 6:
                gauss_n = 4
            if t > 10:
                gauss_n = 2
        elif exact:
            if wire.r <= self.srm:
                t12 = 2 * np.log (wire.seg_len / wire.r) \
                    - self.w * wire.seg_len * 1j
                if fvs != 1:
                    t12 /= 2
                return t12
            # The following starts line 162
            f2 = 2 * scale
            # Hmm, dividing by f2 here removes scale = (p3-p2) from logarithm
            # So i6 depends only on wire/seg parameters?!
            # Moved to wire. Original formula was:
            # i6 = (1 - np.log (s4 / f2 / 8 / wire.r)) / np.pi / wire.r
            # But s4 / f2 reduces to wire.seg_len / 2
            i6 = wire.i6
        # Gauss quadrature was explicitly implemented here, we use
        # fixed_quad from a scipy lib for now.
        args = vec2, vecv, k, wire, bool (i6)
        # Integrate, need to multiply by f2 below.
        r = self.fast_quad (0, 1/f2, args, gauss_n)
        return (r + i6) * s4
    # end def psi

    def psi_near_field_56 (self, vec0, vect, k, ds0, pulse2, ds2):
        """ Compute psi used several times during computation of near field
            Original entry point in line 56
            vec0 originally is (X0, Y0, Z0)
            vect originally is (T5, T6, T7)
            Note that ds0 is the only non-zero-based variable, it's not
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
        >>> r = m.psi_near_field_56 (vec0, vect, 1, 0.5, m.pulses [0], 1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.5496335 -0.3002106j
        """
        wire = pulse2.wires [ds2 > 0]
        kvec = np.array ([1, 1, k])
        vec1 = vec0 + ds0 * vect / 2
        v2, vv = pulse2.dvecs (ds2)
        v2 = vec1 - kvec * v2
        vv = vec1 - kvec * vv
        return self.psi (v2, vv, k, abs (ds2), wire, exact = False)
    # end def psi_near_field_56

    def register_load (self, load, pulse = None, wire_idx = None):
        """ Default if no pulse is given is to add the load to *all*
            pulses. Otherwise if no wire_idx is given the pulse is an
            absolute index, otherwise it's the index of a pulse on the
            wire given by wire_idx. Indeces are 0-based.
        """
        if pulse is None:
            for wire in self.geo:
                for p in wire.pulse_idx_iter ():
                    load.add_pulse (p)
            # Avoid adding same load several times
            if load.n not in self.loadidx:
                self.loads.append (load)
                self.loadidx.add (load.n)
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
            elif pulse >= len (self.pulses):
                raise ValueError ('Invalid pulse %d' % pulse)
            else:
                p = pulse
            load.add_pulse (p)
            # Avoid adding same load several times
            if load.n not in self.loadidx:
                self.loads.append (load)
                self.loadidx.add (load.n)
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
            if pulse >= len (self.pulses):
                raise ValueError ('Invalid pulse %d' % pulse)
            self.sources.append (source)
            source.register (self, pulse)
    # end def register_source

    def scalar_potential (self, k, pulse1, pulse2, ds1, ds2):
        """ Compute scalar potential
            Inputs:
            ds1 in the displacement on pulse1, we compute the endseg
            (the endpoint in positive or negative direction).
            ds2 is the displacement on pulse2, either negative or
            positive, returns the pulse endpoint (in the middle of the
            prev or next segment)

            Original entry point in line 87.
            Original comment:
            entries required for impedance matrix calculation
            S(M) goes in (X1,Y1,Z1) for scalar potential
            mod for small radius to wave length ratio
            This *used* to use A(P4), S(P4), where P4 is the index into
            the wire datastructures, A(P4) is the wire radius and S(P4)
            is the segment length of the wire

        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        # Original produces:
        # -8.333431E-02 -0.1156091j
        >>> method = m.scalar_potential
        >>> p0 = m.pulses [0]
        >>> p8 = m.pulses [8]
        >>> r = method (1, p0, p8, 0.5, -1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -0.0833344 -0.1156091j
        >>> w [0].r = 0.001
        >>> r = method (1, p0, p0, -0.5, 1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        1.0497691 -0.3085993j
        """
        wire = pulse2.wires [ds2 > 0]
        vec1 = pulse1.endseg (ds1)
        kvec = np.array ([1, 1, k])
        if  (  k < 1
            or wire.r > self.srm
            or pulse1.wire.n != pulse2.wire.n
            or pulse1.idx + ds1 != pulse2.idx + ds2 / 2
            ):
            v1 = pulse1.endseg (ds1)
            wd = self.wires_unconnected (pulse1, pulse2)
            v2, vv = pulse2.dvecs (ds2)
            v2 = kvec * v2 - vec1
            vv = kvec * vv - vec1
            return self.psi \
                (v2, vv, k, abs (ds2), wire, fvs = 1, exact = not wd)
        t1 = 2 * np.log (wire.seg_len / wire.r)
        t2 = -self.w * wire.seg_len
        return t1 + t2 * 1j
    # end def scalar_potential

    def vector_potential (self, k, pulse1, pulse2, ds):
        """ Compute vector potential
            Original entry point in line 102.
            Original comment:
            S(M) goes in (X1,Y1,Z1) for vector potential
            mod for small radius to wave length ratio

            This *used* to use A(P4), S(P4), where P4 is the index into
            the wire datastructures, A(P4) is the wire radius and S(P4)
            is the segment length of the wire, we now directly use the
            wire.
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)

        # Original produces:
        # 0.6747199 -.1555772j
        >>> method = m.vector_potential
        >>> p0 = m.pulses [0]
        >>> p1 = m.pulses [1]
        >>> r = method (1, p0, p1, -0.5)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.6747199 -0.1555773j
        """
        wire = pulse2.wires [ds > 0]
        kvec = np.array ([1, 1, k])
        if k < 1 or wire.r >= self.srm or pulse1 != pulse2:
            v1 = pulse1.point
            v2, vv = pulse2.dvecs (ds)
            v2 = kvec * v2 - v1
            vv = kvec * vv - v1
            wd = self.wires_unconnected (pulse1, pulse2)
            return self.psi (v2, vv, k, abs (ds), wire, fvs = 0, exact = not wd)
        t1 = np.log (wire.seg_len / wire.r)
        t2 = -self.w * wire.seg_len / 2
        return t1 + t2 * 1j
    # end def vector_potential

    def wires_unconnected (self, pulse1, pulse2):
        """ Check if the wires to which the given segment indeces belong
            are *not* connected.
        """
        w1 = pulse1.wire
        w2 = pulse2.wire
        # Well this should really be symmetric, so one should be enough :-)
        return not w1.is_connected (w2) and not w2.is_connected (w1)
    # end def wires_unconnected

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
                        assert p is not None
                        c = s * self.current [p]
                    a = np.angle (c) / np.pi * 180
                    r.append \
                        ( ('J ' + ' ' * 12 + fmt)
                        % format_float
                            ((c.real, c.imag, np.abs (c), a), use_e = True)
                        )
            for k in wire.pulse_idx_iter (yield_ends = False):
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
                        assert p is not None
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
                b = 1
                if self.boundary != 'linear':
                    b = 2
                r.append (' TYPE OF BOUNDARY (1-LINEAR, 2-CIRCULAR):  %d' % b)
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
        r.append (self.far_field.abs_gain_as_mininec ())
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
        r.append (self.far_field.db_as_mininec ())
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
            l.append ('%s%3d' % (' ' * 12, wire.idx_1))
            r.append (''.join (l))
            l = []
            l.append (('%-13s ' * 3) % format_float (wire.p2))
            l.append ('%-13s' % format_float ([wire.r]))
            l.append ('%2d%15d' % (wire.idx_2, wire.n_segments))
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
            for p in wire.pulse_iter ():
                r.append (p.as_mininec ())
        return '\n'.join (r)
    # end def wires_as_mininec

# end class Mininec

def parse_floatlist (s, l = 3, fill = None):
    """ Parse comma-separated list of floats with length l.
        Missing values are filled with the value of fill.
    """
    return [float (x) if x else fill for x in s.split (',')]
# end def parse_floatlist

def main (argv = sys.argv [1:], f_err = sys.stderr, return_mininec = False):
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
     2             2.727548E-02  6.861985E-04  2.728411E-02  1.441147 
     3             2.346944E-02  8.170773E-05  2.346959E-02  .199472  
     4             1.744657E-02 -2.362219E-04  1.744817E-02 -.775722  
     5             9.607629E-03 -2.685486E-04  9.611381E-03 -1.601092 
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
     2             2.721271E-02  6.814274E-05  2.721279E-02  .143473  
     3             2.341369E-02 -4.509546E-04  2.341803E-02 -1.103397 
     4             1.740291E-02 -6.326805E-04  1.741441E-02 -2.082063 
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
     2             2.727548E-02  6.861985E-04  2.728411E-02  1.441147 
     3             2.346944E-02  8.170773E-05  2.346959E-02  .199472  
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
     2             2.727548E-02  6.861985E-04  2.728411E-02  1.441147 
     3             2.346944E-02  8.170773E-05  2.346959E-02  .199472  
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
       Z          -17.05298      6.501885E-02  17.05311      179.7815
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

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--boundary', 'unknown'])
    >>> r = main (args, sys.stdout)
    Invalid boundary: unknown, must be one of "linear", "circular"
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

    >>> args = ['--rlc-load=a,b,c']
    >>> r = main (args, sys.stdout)
    Error in series RLC load: could not convert string to float: 'a'
    >>> r
    23

    >>> args = ['--trap-load=a,b,c']
    >>> r = main (args, sys.stdout)
    Error in trap load: could not convert string to float: 'a'
    >>> r
    23
    """
    boundaries = ('linear', 'circular')
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
        ( '--boundary'
        , help    = 'Boundary between different media, one of %s'
                  % ','.join ('"%s"' % b for b in boundaries)
        , default = 'linear'
        )
    cmd.add_argument \
        (  '--excitation-segment'
        , help    = "Segment number for excitation, can be specified "
                    "more than once, default is the single segment 5"
        , type    = int
        , action  = 'append'
        , default = []
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
        , help    = 'Complex load, specify complex impedance, e.g.  50+3j,'
                    ' complex loads are numbered first'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--rlc-load'
        , help    = 'RLC (series) load, specify R,L,C in Ohm, Henry, Farad,'
                    ' RLC loads are numbered second'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '-T', '--timing'
        , help    = 'Measure the time for certain parts of the algorithm'
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '--trap-load'
        , help    = 'Trap load, R+L in series parallel to C, '
                    'specify R,L,C in Ohm, Henry, Farad,'
                    ' Trap loads are numbered third'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--medium'
        , help    = "Media (ground), free space if not given, "
                    "specify permittivity (dielectric constant), "
                    "conductivity, height, if all are "
                    "zero, ideal ground is asumed, if radials are "
                    "specified they apply to the first ground (which "
                    "cannot be ideal with radials), several media can be "
                    "specified. If more than one medium is given, the "
                    "circular or linear coordinate of medium must be "
                    "given as the fourth parameter, this is the distance "
                    "to the next medium."
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
    if args.boundary not in boundaries:
        print \
            ( 'Invalid boundary: %s, must be one of %s'
            % (args.boundary, ', '.join ('"%s"' % b for b in boundaries))
            , file = f_err
            )
        return 23
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
        d.update (boundary = args.boundary)
        if len (p) > 3:
            d.update (coord = p [3])
        p = p [:3]
        media.append (Medium (*p, **d))
    media = media or None
    m = Mininec (args.frequency, wires, media = media, t = args.timing)
    for i, v in zip (args.excitation_segment, args.excitation_voltage):
        s = Excitation (cvolt = v)
        m.register_source (s, i - 1)
    loads = []
    for l in args.load:
        loads.append (Impedance_Load (l))
    for l in args.rlc_load:
        try:
            loads.append (Series_RLC_Load (*parse_floatlist (l)))
        except ValueError as err:
            print ("Error in series RLC load: %s" % err, file = f_err)
            return 23
    for l in args.trap_load:
        try:
            loads.append (Trap_Load (*parse_floatlist (l, fill=0)))
        except ValueError as err:
            print ("Error in trap load: %s" % err, file = f_err)
            return 23
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
    if return_mininec:
        return m
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

if __name__ == '__main__':
    main ()

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
