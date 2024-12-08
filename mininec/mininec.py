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
from datetime        import datetime
from itertools       import pairwise
from scipy.special   import ellipk, jv
from scipy.integrate import fixed_quad
from mininec.util    import format_float
from mininec.pulse   import Pulse_Container, Pulse
from mininec.segment import Segment
from mininec.taper   import taper1, taper2, Taper_Error
from numpy.polynomial.legendre import leggauss

legendre_cache = {}
for k in (2, 4, 8):
    legendre_cache [k] = [x / 2 for x in leggauss (k)]

# Constants
mu_0      = 1.25663706127e-6
epsilon_0 = 8.8541878188e-12
c         = 1 / np.sqrt (mu_0 * epsilon_0)

class Rotation_Matrix:

    def __init__ (self, rotation):
        rot_x = rot_y = rot_z = np.eye (3)
        if rotation [0]:
            a = rotation [0] / 180 * np.pi
            rot_x = np.array \
                ( [ [1, 0,           0         ]
                  , [0, np.cos (a), -np.sin (a)]
                  , [0, np.sin (a),  np.cos (a)]
                  ]
                )
        if rotation [1]:
            a = rotation [1] / 180 * np.pi
            rot_y = np.array \
                ( [ [ np.cos (a), 0, np.sin (a)]
                  , [ 0,          1, 0         ]
                  , [-np.sin (a), 0, np.cos (a)]
                  ]
                )
        if rotation [2]:
            a = rotation [2] / 180 * np.pi
            rot_z = np.array \
                ( [ [np.cos (a), -np.sin (a), 0]
                  , [np.sin (a),  np.cos (a), 0]
                  , [0,           0,          1]
                  ]
                )
        self.m = rot_z @ rot_y @ rot_x
    # end def __init__

    def apply (self, vec):
        return self.m @ vec
    # end def apply

# end def Rotation_Matrix

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

class Connected_Geobj:
    """ This is used to store a set of connected geo objects *and* the
        corresponding segments in one data structure.
    """

    def __init__ (self):
        self.geo           = set ()
        self.list          = []
        self.sgn_by_geobj  = {}
    # end def __init__

    def add (self, geobj, other_geobj, end_idx, sign, sign2):
        assert geobj not in self.geo
        self.geo.add (geobj)
        self.list.append ((geobj, other_geobj, end_idx, sign))
        self.sgn_by_geobj [other_geobj] = sign2
    # end def add

    def tag (self, geobj):
        """ This computes I1 or I2 from the Basic code, respectively
        """
        if self.list:
            # Forward linked segments are printed as 0
            if geobj.n < self.list [0][0].n:
                return 0
            return (self.list [0][0].tag) * self.sgn_by_geobj [geobj]
        return 0
    # end def tag

    def is_connected (self, other):
        return other in self.geo
    # end def is_connected

    def pulse_iter (self):
        """ Yield pulse indeces sorted by geo index
        """
        for geobj, ow, idx, s in self._iter ():
            yield (ow.end_segs [idx], s)
    # end def pulse_iter

    def _iter (self):
        for geobj, ow, idx, s in sorted (self.list, key = lambda x: x [0].n):
            yield (geobj, ow, idx, s)
    # end def _iter

    def __bool__ (self):
        return bool (self.list)
    # end def __bool__

    def __str__ (self):
        r = ['Connected_Ojb:']
        for geobj, ow, idx, s in self._iter ():
            r.append (' geo: %d idx: %s s:%d' % (geobj.n, ow.end_segs [idx], s))
        return '\n'.join (r)
    __repr__ = __str__

# end class Connected_Geobj

class Excitation:
    """ This is the pulse source definition in mininec.
        For convenience phase is in degrees (and converted internally)
        Magnitude is in volts, constructor can either directly give a
        complex number for the voltage *or* floating point voltage
        magnitude and a phase in degrees.
    """
    def __init__ (self, cvolt, phase = None, geo_tag = None, geo_idx = None):
        if isinstance (cvolt, complex) and phase is not None:
            raise ValueError \
                ("Either specify magnitude/phase or complex voltage")

        self.parent     = None
        self.idx        = None
        self.is_default = False
        self.geo_tag    = geo_tag
        self.geo_idx    = geo_idx
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

    def as_basic_input (self):
        """ Output in the input format of original Basic implementation.
        """
        r = []
        # PULSE NO., VOLTAGE MAGNITUDE, PHASE (DEGREES):
        r.append ('%d, %g, %g' % (self.idx + 1, self.magnitude, self.phase))
        return '\n'.join (r)
    # end def as_basic_input

    def as_cmdline (self):
        r = []
        if self.voltage != 1+0j:
            r.append \
                ( '--excitation-voltage=%g%+gj'
                % (self.voltage.real, self.voltage.imag)
                )
        if not self.is_default:
            if self.geo_tag is not None and self.geo_idx is not None:
                r.append \
                    ( '--excitation-pulse=%d,%d'
                    % (self.geo_idx + 1, self.geo_tag)
                    )
            else:
                r.append ('--excitation-pulse=%d' % (self.idx + 1))
        return '\n'.join (r)
    # end def as_cmdline

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
              , format_float ([self.impedance.real], use_e = True) [0]
              , format_float ([self.impedance.imag], use_e = True) [0]
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

    def as_cmdline_load_attach (self, parent, by_geo = False):
        """ This needs to output the actual load, done by the subclass.
            But we can output the load attachments here.
            This used to be 'as_cmdline' but since the class hierarchy
            is more involved this is now a separate method.
        """
        r = []
        # Optimize: Check if we can issue an attach statement to *all*
        # of a geo object or to *all* pulses
        pulsecount_by_geo = {}
        geo = set ()
        geo_all = set ()
        for pulse in self.pulses:
            if pulse.geobj.n not in pulsecount_by_geo:
                pulsecount_by_geo [pulse.geobj.n] = 0
            pulsecount_by_geo [pulse.geobj.n] += 1
            geo.add (pulse.geobj)
        for w in geo:
            if pulsecount_by_geo [w.n] == len (w.pulses):
                geo_all.add (w)
        if len (geo_all) == len (parent.geo):
            r.append ('--attach-load=%d,all' % (self.n + 1))
        elif geo_all:
            for w in geo_all:
                r.append ('--attach-load=%d,all,%d' % (self.n + 1, w.tag))
        for pulse in self.pulses:
            if pulse.geobj in geo_all:
                continue
            if by_geo:
                r.append \
                    ( '--attach-load=%d,%d,%d'
                    % (self.n + 1, pulse.n + 1, pulse.geobj.n + 1)
                    )
            else:
                r.append ('--attach-load=%d,%d' % (self.n + 1, pulse.idx + 1))
        return '\n'.join (r)
    # end def as_cmdline_load_attach

    def as_mininec (self, parent):
        r = []
        for pulse in self.pulses:
            imp = self.impedance (parent.f, pulse)
            r.append \
                ( 'PULSE NO.,RESISTANCE,REACTANCE: %2d , %s , %s'
                % ( (pulse.idx + 1,)
                  + format_float ([imp.real, imp.imag])
                  )
                )
        return '\n'.join (r)
    # end def as_mininec

    def impedance (self, f, pulse = None):
        """ Get impedance for a certain frequency
            This probably needs reimplementation in different derived
            classes. Especially if the impedance is frequency dependent.
            Note that some subclasses need the pulse parameter.
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

    def as_basic_input (self, args, is_s = False):
        r = []
        if is_s: # pragma: no cover
            raise NotImplementedError \
                ('Output of Impedance load as S-parameters not yet implemented')
        for pulse in self.pulses:
            # PULSE NO.,RESISTANCE,REACTANCE:
            z = self._impedance
            r.append ('%d, %g, %g' % (pulse.idx + 1, z.real, z.imag))
        return '\n'.join (r)
    # end def as_basic_input

    def as_cmdline (self, parent, by_geo = False):
        r = []
        ld = '--load=%g' % self._impedance.real
        if self._impedance.imag:
            ld += '+%gj' % self._impedance.imag
        r.append (ld)
        r.append (self.as_cmdline_load_attach (parent, by_geo))
        return '\n'.join (r)
    # end def as_cmdline

# end class Impedance_Load

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
        self.degree = max (len (self.a), len (self.b)) - 1
        if not len (a):
            raise ValueError ("At least one denominator parameter required")
        super ().__init__ ()
    # end def __init__

    def as_basic_input (self, args, is_s):
        r = []
        assert is_s
        for pulse in self.pulses:
            # PULSE NO., ORDER OF S-PARAMETER FUNCTION:
            r.append ('%d, %d' % (pulse.idx + 1, self.degree))
            for d in range (self.degree + 1):
                # Factor, L, C are in µH, µF up to version 9
                f = 10 ** (6 * d)
                if args.mininec_version != '9':
                    f = 1
                # NUMERATOR, DENOMINATOR COEFFICIENTS OF S^[d]:
                r.append ('%g, %g' % (self.b [d] * f, self.a [d] * f))
        return '\n'.join (r)
    # end def as_basic_input

    def as_cmdline (self, parent, by_geo = False):
        r = []
        r.append ('--laplace-load-b=%s' % ','.join ('%.8g' % b for b in self.b))
        r.append ('--laplace-load-a=%s' % ','.join ('%.8g' % a for a in self.a))
        r.append (self.as_cmdline_load_attach (parent, by_geo))
        return '\n'.join (r)
    # end def as_cmdline

    def as_mininec (self, parent):
        r = []
        for pulse in self.pulses:
            imp = self.impedance (parent.f, pulse)
            r.append \
                ( 'PULSE NO., ORDER OF S-PARAMETER FUNCTION:  %d , %d'
                % (pulse.idx + 1, self.degree)
                )
            for d in range (self.degree + 1):
                # Factor, L, C are in µH, µF
                f = 10 ** (6 * d)
                s = 'NUMERATOR, DENOMINATOR COEFFICIENTS OF S^%d : %g , %g'
                r.append (s % (d, self.b [d] * f, self.a [d] * f))
        return '\n'.join (r)
    # end def as_mininec

    def impedance (self, f, pulse = None):
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

class Series_RLC_Load (Laplace_Load):
    """ A load with R, L, C in series, an unspecified value is
        considered to be a 0-Ohm resistor.
        This was not in the original mininec code.
        But it can be modelled with a Laplace load.
        Frequency is in MHz (when calling impedance), otherwise we use
        metric units Ohm, Henry, Farad.
        This is convenient when converting models from NEC.
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
        self.r = R
        self.l = L
        self.c = C
        r = self.r or 0
        l = self.l or 0
        if C:
            a = np.array ([0.0, self.c])
            b = np.array ([1.0, r * self.c, l * self.c])
        else:
            a = np.array ([1.0])
            b = np.array ([r, l])
        super ().__init__ (a, b)
    # end def __init__

    def as_cmdline (self, parent, by_geo = False):
        r  = []
        ld = (self.r, self.l, self.c)
        s  = ','.join ('%g' if x else '' for x in ld)
        r.append ('--rlc-load=' + s % tuple (x for x in ld if x))
        r.append (self.as_cmdline_load_attach (parent, by_geo))
        return '\n'.join (r)
    # end def as_cmdline

# end class Series_RLC_Load

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
        self.r = R
        self.l = L
        self.c = C
        super ().__init__ (a = (1, R*C, L*C), b = (R, L))
    # end def __init__

    def as_cmdline (self, parent, by_geo = False):
        r = []
        r.append ('--trap-load=%g,%g,%g' % (self.r, self.l, self.c))
        r.append (self.as_cmdline_load_attach (parent, by_geo))
        return '\n'.join (r)
    # end def as_cmdline

# end class Trap_Load

class Distributed_Load (_Load):

    def add_pulse (self, pulse):
        # Allow adding only to our own geobj
        n = self.__class__.__name__.replace ('_', ' ')
        if self.geobj not in pulse.geo: # pragma: no cover
            # Not reachable via command line
            raise ValueError \
                ('%s can only be attached to pulses of its geo objects'
                % n
                )
        super ().add_pulse (pulse)
    # end def add_pulse

    def as_basic_input (self, args, is_s = False):
        r = []
        if is_s: # pragma: no cover
            raise NotImplementedError ('Output as S-parameters not implemented')
        for pulse in self.pulses:
            # PULSE NO.,RESISTANCE,REACTANCE:
            z = self.impedance (self.geobj.parent.parent.f, pulse)
            r.append ('%d, %g, %g' % (pulse.idx + 1, z.real, z.imag))
        return '\n'.join (r)
    # end def as_basic_input

# end class Distributed_Load

class Skin_Effect_Load (Distributed_Load):
    """ Ohmic loss due to skin effect
    """

    def __init__ (self, geobj, conductivity):
        super ().__init__ ()
        self.geobj        = geobj
        self.conductivity = conductivity
        if geobj.skin_load is not None and geobj.skin_load is not self:
            raise ValueError \
                ("Only one skin-effect load per geo object")
        geobj.skin_load = self
    # end def __init__

    def as_cmdline (self, parent, by_geo = False):
        r = []
        tag = self.geobj.tag
        r.append ('--skin-effect-conductivity=%g,%d' % (self.conductivity, tag))
        return '\n'.join (r)
    # end def as_cmdline

    def impedance (self, f, pulse):
        """ Get resistance for given frequency
        >>> e1   = np.zeros (3)
        >>> e2   = np.ones (3)
        >>> wire = Wire (1, 0, 0, 0, 0, 0, 25, 1)
        >>> wire.n = 0
        >>> pc   = Pulse_Container ()
        >>> s    = Segment (e1, e2, wire, 0)
        >>> p    = Pulse (pc, np.ones (1), np.zeros (1), np.ones (1), s, s)
        >>> for r in (1e-6, 1e-4, 1e-3):
        ...     wire = Wire (1, 0, 0, 0, 0, 0, 25, r)
        ...     p.geo = [wire]
        ...     ld = Skin_Effect_Load (wire, 2.5e6)
        ...     x  = ld.impedance (1e3, p)
        ...     print ('%.8g %+.8gj' % (x.real, x.imag))
        63662.106 +157.07947j
        33.273927 +31.556264j
        3.1622777 +3.1622777j
        """
        fhz = f * 1e6
        omg = 2 * np.pi * fhz
        x   = 0
        for i, w in enumerate (pulse.geo):
            ld    = w.skin_load
            if ld is None:
                continue
            # Cache zint in geobj
            if w.zint is None:
                k     = np.sqrt (-1j * omg * mu_0 * ld.conductivity)
                kr    = k * w.r_orig
                b     = 1j
                if abs (kr) < 110.0:
                    b = jv (0, kr) / jv (1, kr)
                zint  = k / (2 * np.pi * w.r_orig * ld.conductivity) * b
                w.zint = zint
            dv = pulse.dvecs (i - 0.5)
            l  = np.linalg.norm (dv [0] - dv [1])
            x += l * w.zint
        return x
    # end def impedance

# end class Skin_Effect_Load

class Insulation_Load (Distributed_Load):
    """ Impedance due to insulation of geobj
    """

    def __init__ (self, geobj, radius, epsilon_r):
        super ().__init__ ()
        self.geobj     = geobj
        self.epsilon_r = epsilon_r
        self.epsilon   = epsilon_r * epsilon_0
        self.radius    = radius
        if  (   geobj.coat_load is not None
            and geobj.coat_load is not self
            ):
            raise ValueError ("Only one insulation-load per geo object")
        if geobj.r >= radius:
            raise ValueError ("Insulation radius must be > geo object radius")
        geobj.coat_load = self
        self.f = None
    # end def __init__

    def as_cmdline (self, parent, by_geo = False):
        r = []
        tag = self.geobj.tag
        r.append \
            ('--insulation-load=%g,%g,%d' % (self.radius, self.epsilon_r, tag))
        return '\n'.join (r)
    # end def as_cmdline

    def impedance (self, f, pulse):
        """ Get resistance for given frequency
        """
        fhz = f * 1e6
        omg = 2 * np.pi * fhz
        x   = 0j
        for seg in pulse.segs:
            # Not both segments may be from same geobj
            geobj = seg.geobj
            ld    = geobj.coat_load
            if not ld:
                continue
            if geobj.zins is None:
                geobj.zins = \
                    ( mu_0 * (ld.epsilon_r - 1) / ld.epsilon_r
                    * np.log (ld.radius / self.geobj.r_orig)
                    / (2 * np.pi)
                    )
            x += geobj.zins * omg * 1j * (seg.seg_len / 2)
        return x
    # end def impedance

# end class Insulation_Load

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
        , nradials = 0, radius = 0
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

    def as_basic_input (self):
        """ Output in the input format of original Basic implementation.
        """
        r = []
        # Question is only asked on the first media and only if there is
        # more than one.
        if not self.prev and self.next:
            # TYPE OF BOUNDARY (1-LINEAR, 2-CIRCULAR):
            if self.boundary == 'circular':
                r.append ('2')
            else:
                r.append ('1')
        # RELATIVE DIELECTRIC CONSTANT, CONDUCTIVITY:
        r.append ('%g, %g' % (self.permittivity, self.conductivity))
        if self.prev:
            # HEIGHT OF MEDIA:
            r.append ('%g' % self.height)
        elif self.next and self.boundary == 'circular':
            # NUMBER OF RADIAL WIRES IN GROUND SCREEN:
            r.append (str (self.nradials))
            if self.nradials:
                # RADIUS OF RADIAL WIRES:
                r.append ('%g' % self.radius)
        if self.next:
            # X OR R COORDINATE OF NEXT MEDIA INTERFACE:
            r.append ('%g' % self.coord)
        return '\n'.join (r)
    # end def as_basic_input

    def as_cmdline (self):
        r = []
        v = '--medium=%g,%g,%g' % \
            (self.permittivity, self.conductivity, self.height)
        if self.next:
            v += ',%g' % self.coord
        r.append (v)
        if not self.prev:
            if self.next:
                r.append ('--boundary=%s' % self.boundary)
            if self.nradials:
                r.append ('--radial-count=%d' % self.nradials)
                r.append ('--radial-radius=%g' % self.radius)
        return '\n'.join (r)
    # end def as_cmdline

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

class Geo_Container:

    def __init__ (self, parent = None, geo = None):
        self.parent     = parent
        self.geo        = []
        self.by_tag     = {}
        self.min_seglen = None
        if geo is not None:
            for g in geo:
                self.append (g)
    # end def __init__

    def __getitem__ (self, idx):
        return self.geo [idx]
    # end def __getitem__

    def __iter__ (self):
        for geobj in self.geo:
            yield geobj
    # end def __iter__

    def __len__ (self):
        return len (self.geo)
    # end def __len__

    def append (self, geobj):
        self.geo.append (geobj)
        assert getattr (geobj, 'parent', None) is None
        geobj.parent = self
    # end def append

    def compute_ground (self):
        for n, geobj in enumerate (self.geo):
            geobj.compute_ground (n, self.parent.media)
    # end def compute_ground

    def compute_segments (self):
        for w in self:
            w.compute_segments ()
        self.min_seglen = min (w.min_seglen for w in self.geo)
        self.parent.min_seglen = self.min_seglen
    # end def compute_segments

    def compute_tags (self):
        tags_seen = set ()
        for geobj in self.geo:
            if geobj.tag is not None:
                if geobj.tag <= 0:
                    raise ValueError ('Tag "%s" not allowed' % geobj.tag)
                if geobj.tag in tags_seen:
                    raise ValueError \
                        ('Duplicate tag "%s" in geo object' % geobj.tag)
                tags_seen.add (geobj.tag)
        max_tag = 0
        if tags_seen:
            max_tag = max (tags_seen)
        for n, geobj in enumerate (self.geo):
            if geobj.tag is None:
                max_tag += 1
                geobj.tag = max_tag
            self.by_tag [geobj.tag] = geobj
        # sort by tag
        self.geo.sort (key = lambda geobj: geobj.tag)
    # end def compute_tags

    def rotate (self, rotation, tag = None):
        """ Rotate geometry object given by tag. If tag is None, the
            whole structure defined so far is rotated.
            The parameter rotation is a 3-element vector with angles in
            *degrees*.
            As in NEC order of transformations is
            - rotation about X-axis
            - rotation about Y-axis
            - rotation about Z-axis
            Unlike in NEC there is currently no provision to *copy*
            geometry objects to a new location.
        """
        rmatrix = Rotation_Matrix (rotation)
        if tag is None:
            for g in self:
                g.rotate (rmatrix)
        else:
            self.by_tag [tag].rotate (rmatrix)
    # end def rotate

    def scale (self, factor, tag = None):
        """ Scale geometry object given by tag. If tag is None (the
            typical use-case) the whole structure is scaled.
            Note that *all* parameters are scaled *including* the
            radius (but not the radius of an insulation load).
            Note that scaling happens *after* translation and rotation,
            so that parameters of translation are in the same
            coordinates as the geo objects.
        """
        if tag is None:
            for g in self:
                g.scale (factor)
        else:
            self.by_tag [tag].scale (factor)
    # end def scale

    def translate (self, translation, tag = None):
        """ Translate geometry object given by tag. If tag is None, the
            whole structure defined so far is translated.
            The parameter translation is a 3-element vector.
            Unlike in NEC there is currently no provision to *copy*
            geometry objects to a new location.
            Note that scaling happens *after* translation and rotation,
            so that parameters of translation are in the same
            coordinates as the geo objects.
        """
        if tag is None:
            for g in self:
                g.translate (translation)
        else:
            self.by_tag [tag].translate (translation)
    # end def translate

# end class Geo_Container

class Geobj:

    def __init__ (self, r, tag = None):
        if r <= 0:
            raise ValueError ("Radius must be >0")
        self._r        = r
        self.tag       = tag
        self.zint      = None
        self.zins      = None
        self.skin_load = None
        self.coat_load = None
        self.pulses    = []
        self.had_tag   = tag is not None

        self.end_segs  = [None, None]
        # Links to previous/next connected wire (at start/end)
        # conn [0] contains wires that link to our first end
        # while conn [1] contains wires that link to our second end.
        self.conn = (Connected_Geobj (), Connected_Geobj ())
        self.n    = None # index into parent.geo
    # end def __init__

    @property
    def idx_1 (self):
        return self.idx (0)
    # end def idx_1

    @property
    def idx_2 (self):
        return self.idx (1)
    # end def idx_2

    @property
    def r (self):
        """ Use equivalent radius if we have an insulation load
        """
        if not self.coat_load:
            return self._r
        a     = self._r
        b     = self.coat_load.radius
        eps_r = self.coat_load.epsilon_r
        return b * (a / b) ** (1 / eps_r)
    # end def r

    @property
    def r_orig (self):
        """ Original unmodified radius
        """
        return self._r
    # end def r_orig

    def as_basic_input (self):
        """ Output in the input format of original Basic implementation.
            If we emulate several wires we iterate over all segments
            here.
        """
        r = []
        if self.n_emulated_wires == 1:
            # NO. OF SEGMENTS:
            r.append (str (self.n_segments))
            # END ONE COORDINATES (X,Y,Z):
            r.append ('%.15g, %.15g, %.15g' % tuple (self.p1))
            # END TWO COORDINATES (X,Y,Z):
            r.append ('%.15g, %.15g, %.15g' % tuple (self.p2))
            # RADIUS:
            r.append ('%.8g' % self.r)
            # CHANGE WIRE NO.  x  (Y/N):
            r.append ('N')
        else:
            # NO. OF SEGMENTS:
            r.append ('1')
            # END ONE COORDINATES (X,Y,Z):
            r.append ('%.15g, %.15g, %.15g' % tuple (self.p1))
            # END TWO COORDINATES (X,Y,Z):
            r.append ('%.15g, %.15g, %.15g' % tuple (self.segments [0].p2))
            # RADIUS:
            r.append ('%.8g' % self.r)
            # CHANGE WIRE NO.  x  (Y/N):
            r.append ('N')
            for s in self.segments [1:]:
                # NO. OF SEGMENTS:
                r.append ('1')
                # END ONE COORDINATES (X,Y,Z):
                r.append ('%.15g, %.15g, %.15g' % tuple (s.p1))
                # END TWO COORDINATES (X,Y,Z):
                r.append ('%.15g, %.15g, %.15g' % tuple (s.p2))
                # RADIUS:
                r.append ('%.8g' % self.r)
                # CHANGE WIRE NO.  x  (Y/N):
                r.append ('N')
        return '\n'.join (r)
    # end def as_basic_input

    def _add_conn (self, parent, ep_tuple, n1):
        n2, other = parent.end_dict [ep_tuple]
        s = -1 if (n2 == n1) else 1
        other.conn [n2].add (self,  self, n1, s, s)
        self.conn  [n1].add (other, self, n1, 1, s)
    # end def _add_conn

    def compute_connections (self, parent):
        """ Compute links to connected geo objects
            Also compute sets of indeces of pulses.
            Note that we're using a dictionary for matching endpoints,
            so in the case of a match we find an existing matching
            endpoint quickly. When not matching we still need to search
            all existing geo objects.

            We adopt NEC's algorithm for matching wire ends: If the
            separation of two endpoints is less than 1e-3 of the
            shortest segment, the endpoints match. (See Part III, User's
            guide p.4).

            Also compute pulses.
        """
        # This rolls the end-matching computation of 4
        # explicitly-programmed cases in the Basic program lines
        # 1325-1356 into a few statements.
        # The idea is to use a dictionary of end-point coordinates.
        # But for fuzzy matching we still need to iterate over existing
        # segments if we do not find a match in the dictionary.
        for n1, current_end in enumerate (self.endpoints):
            if self.is_ground [n1]:
                continue
            ep_tuple = tuple (current_end)
            if ep_tuple in parent.end_dict:
                self._add_conn (parent, ep_tuple, n1)
            else:
                minlen = parent.min_seglen * 1e-3
                for tpl in parent.end_dict:
                    end_coord = np.array (list (tpl))
                    if np.linalg.norm (current_end - end_coord) <= minlen:
                        # Copy dict entry so that future ends can be found
                        parent.end_dict [ep_tuple] = parent.end_dict [tpl]
                        self._add_conn (parent, ep_tuple, n1)
                        break
                else:
                    parent.end_dict [ep_tuple] = (n1, self)
        self.end_segs [0] = parent.pulses.pulse_idx
        if self.n_segments == 1 and self.idx_1 == 0:
            self.end_segs [0] = None
        npulse = self.n_segments - (not self.idx_1) - (not self.idx_2)
        # If structure is connected to itself, deduct one from pulse count
        if self.conn [0].list and self.conn [0].list [0][0] is self:
            assert self.conn [1].list [0][0] is self
            npulse -= 1
        self.end_segs [1] = parent.pulses.pulse_idx + npulse
        if self.n_segments == 1 and self.idx_2 == 0:
            self.end_segs [1] = None
        # inversion of Z component
        invz = np.array ([1, 1, -1])

        pc = 0
        pu = parent.pulses
        # Connection to other geo object(s) at end 1
        seg0 = self.segments [0]
        if self.idx_1 != 0 and abs (self.idx_1) - 1 != self.n:
            assert not self.is_ground [0]
            other = parent.geo [abs (self.idx_1) - 1]
            sgn   = [np.sign (self.idx_1), 1]
            if sgn [0] < 0:
                oseg = other.segments [0]
            else:
                oseg = other.segments [-1]
            oinc = oseg.dirvec * oseg.seg_len * sgn [0]
            prev = self.p1 - oinc
            p = Pulse (pu, self.p1, prev, seg0.p2, oseg, seg0, sgn = sgn)
            if self.n_segments == 1 and self.idx_2 == 0:
                p.c_per [1] = 0
            p.n = pc
            pc += 1
            self.pulses.append (p)
        elif self.is_ground [0]:
            s = seg0.p2
            p = Pulse (pu, self.p1, s * invz, s, seg0, seg0, gnd = 0)
            p.n = pc
            pc += 1
            self.pulses.append (p)
        for i, seg in enumerate (self.segments [:-1]):
            nseg = self.segments [i + 1]
            p = Pulse (pu, seg.p2, seg.p1, nseg.p2, seg, nseg)
            p.n = pc
            pc += 1
            self.pulses.append (p)
            if i == 0 and self.idx_1 == 0:
                p.c_per [0] = 0
            if i == self.n_segments - 2 and self.idx_2 == 0:
                p.c_per [1] = 0
        # Connection to other geo object(s) at end 2
        lseg = self.segments [-1]
        p1   = lseg.p1
        p2   = lseg.p2
        if self.is_ground [1]:
            end2 = p2 - lseg.dirvec * lseg.seg_len * invz
            p = Pulse (pu, self.p2, p1, end2, lseg, lseg, gnd = 1)
            p.n = pc
            pc += 1
            self.pulses.append (p)
        elif self.idx_2 != 0:
            assert not self.is_ground [1]
            other = parent.geo [abs (self.idx_2) - 1]
            sgn   = [1, np.sign (self.idx_2)]
            if sgn [1] < 0:
                oseg = other.segments [-1]
            else:
                oseg = other.segments [0]
            oinc = oseg.dirvec * oseg.seg_len * sgn [1]
            p3   = p2 + oinc
            p = Pulse (pu, p2, p1, p3, lseg, oseg, sgn = sgn)
            if self.n_segments == 1 and self.idx_1 == 0:
                p.c_per [0] = 0
            p.n = pc
            pc += 1
            self.pulses.append (p)
    # end def compute_connections

    def compute_ground (self, n, media):
        self.n = n
        # If we are in free space, nothing to do here
        if media is None:
            self.is_ground = (False, False)
            return
        # Wire end is grounded if Z coordinate is 0
        # In the original implementation this is kept in J1
        # with: 0: not grounded -1: start grounded 1: end grounded
        eps = self.parent.min_seglen * 1e-3
        self.is_ground = (abs (self.p1 [-1]) < eps, abs (self.p2 [-1]) < eps)
        if self.p1 [-1] < -eps or self.p2 [-1] < -eps:
            tg = ''
            if self.tag is not None:
                tg = ' %d' % self.tag
            raise ValueError \
                ( "Geo object%s: height cannot not be negative with ground"
                % tg
                )
    # end def compute_ground

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

    def connections (self):
        return self.conn [0].geo.union (self.conn [1].geo)
    # end def connections

    def idx (self, end_idx):
        """ The indeces I1, I2 from the Basic code, this was -self.n when
            the end is grounded, the index of a connected wire (negative
            if the direction of the wire is reversed) if connected and 0
            otherwise. Used mainly when printing wires in mininec format.
            We now use the tag of the wire.
        """
        if self.is_ground [end_idx]:
            return -(self.tag)
        return self.conn [end_idx].tag (self)
    # end def idx

    def is_connected (self, other):
        if other is self:
            return True
        for c in self.conn:
            if c.is_connected (other):
                return True
        return bool (self.connections ().intersection (other.connections ()))
    # end def is_connected

    def pulse_idx_iter (self, yield_ends = True):
        for p in self.pulse_iter (yield_ends):
            yield p.idx
    # end def pulse_idx_iter

    def pulse_iter (self, yield_ends = True):
        for p in self.pulses:
            if p.geo [0] != p.geo [1] and not yield_ends:
                continue
            yield p
    # end def pulse_iter

# end class Geobj

class Arc (Geobj):
    """ A NEC-like wire arc (GA card in NEC)
        The given segments form a polygon *inscribed* within the arc.
        The arcs center is located at the origin and the axis is the
        Y-axis. If an arc of a different position or orientation is
        desired the object can be moved with one or several of the geo
        transformations.
        The first radius parameter is the radius of the arc. The r
        parameter is the wire radius. The ang1 and ang2 parameters give
        the start and end arcs measured from the X-axis in a left hand
        direction about the Y-axis in degree.
        We require at least 3 wire segments, otherwise arcs degenerate
        to a wire.
    >>> arc = Arc (6, 1, 0, 90, 0.002)
    >>> arc
    Arc radius=1, [0, 90], r=0.002
    >>> arc.n = 42
    >>> arc
    Arc 42 radius=1, [0, 90], r=0.002
    """

    name = 'ARC'

    def __init__ (self, n_segments, radius, ang1, ang2, r, tag = None):
        super ().__init__ (r, tag)
        if n_segments < 3:
            raise ValueError ('Arc needs at least three segments')
        if radius <= 0:
            raise ValueError ('Arc radius must be > 0')
        if ang1 == ang2:
            raise ValueError ('Arc angles must be different')
        if ang2 - ang1 > 360:
            raise ValueError ('Arcs must not exceed a full circle')
        self.n_segments = n_segments
        self.radius     = radius
        self.ang1       = ang1
        self.ang2       = ang2
        # We *always* need to emulate segments as wires in Basic
        self.n_emulated_wires = self.n_segments
        # Compute segments. On geo transformation these must be adapted.
        segends = []
        a1 = ang1 / 180 * np.pi
        a2 = ang2 / 180 * np.pi
        for i in range (n_segments):
            a = a1 + (a2 - a1) / n_segments * i
            segends.append ([radius * np.cos (a), 0.0, radius * np.sin (a)])
        segends.append ([radius * np.cos (a2), 0.0, radius * np.sin (a2)])
        self.segends = np.array (segends)
    # end def __init__

    @property
    def endpoints (self):
        return np.array ([self.segends [0], self.segends [-1]])
    # end def endpoints

    @property
    def p1 (self):
        return self.segends [0]
    # end def p1

    @property
    def p2 (self):
        return self.segends [-1]
    # end def p2

    def as_cmdline (self):
        r = []
        tpl = (self.n_segments, self.radius, self.ang1, self.ang2, self.r_orig)
        if self.had_tag:
            tpl = (self.tag,) + tpl
            r.append ('-w %d,%d,%.11g,%.11g,%.11g,%.11g' % tpl)
        else:
            r.append ('-w %d,%.11g,%.11g,%.11g,%.11g' % tpl)
        return '\n'.join (r)
    # end def as_cmdline

    def compute_ground (self, n, media):
        """ It *is* allowed that both ends are grounded but no
            intermediate segments may below ground
        """
        super ().compute_ground (n, media)
        if media is None:
            return
        gnd_prev  = False
        eps = self.parent.min_seglen * 1e-3
        for s in self.segends:
            if s [-1] < -eps:
                raise ValueError ('Arc may not be partially below ground')
            if abs (s [-1]) < eps:
                if gnd_prev:
                    raise ValueError \
                        ('Arc: No two adjacent segments may be grounded')
                gnd_prev  = True
                s [-1] = 0.0
            else:
                gnd_prev  = False
    # end def compute_ground

    def compute_segments (self):
        """ Loop over our segment ends and create segments.
        """
        self.segments = []
        for e1, e2 in pairwise (self.segends):
            self.segments.append (Segment (e1, e2, self, len (self.segments)))
        self.min_seglen = self.segments [0].seg_len
    # end def compute_segments

    def rotate (self, rmatrix):
        assert not getattr (self, 'segments', None)
        self.segends = rmatrix.apply (self.segends.T).T
    # end def rotate

    def scale (self, factor):
        """ Scale everything by factor including radius
            Similar to the NEC GS card "Scale Structure Dimensions"
        """
        assert not getattr (self, 'segments', None)
        self.segends = self.segends * factor
        self.radius  = self.radius  * factor
        self._r      = self._r      * factor
    # end def scale

    def translate (self, translation):
        assert not getattr (self, 'segments', None)
        self.segends = self.segends + translation
    # end def translate

    def __str__ (self):
        s = 'radius=%.11g, [%.11g, %.11g], r=%.11g' \
          % (self.radius, self.ang1, self.ang2, self.r_orig)
        if self.n is None:
            return 'Arc ' + s
        return 'Arc %d ' % self.n + s
    __repr__ = __str__

# end class Arc

class Wire (Geobj):
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

    name = 'WIRE'

    def __init__ (self, n_segments, x1, y1, z1, x2, y2, z2, r, tag = None):
        super ().__init__ (r, tag)
        self.n_segments = n_segments
        # whenever we need to access both ends by index we use endpoints
        self.p1        = np.array ([x1, y1, z1])
        self.p2        = np.array ([x2, y2, z2])
        self._segtype  = 0
        self.taper_min = None
        self.taper_max = None
        self.compute_endpoints ()
    # end def __init__

    @property
    def n_emulated_wires (self):
        """ If we have non-equal segmentation we return the number of
            segments (emulated in the Basic implementation as
            single-segment wires). Otherwise we return 1.
        """
        if self.segtype == 0:
            return 1
        return self.n_segments
    # end def n_emulated_wires

    @property
    def segtype (self):
        return self._segtype
    # end def segtype

    @segtype.setter
    def segtype (self, stype):
        """ Segmentation type:
            - 0 for equal segment lengths
            - 1 for tapered segmentation from end 1
            - 2 for tapered segmentation from end 2
            - 3 for tapered segmentation from *both* ends
        """
        assert 0 <= stype <= 3
        self._segtype = stype
    # end def segtype

    def as_cmdline (self):
        r = []
        tpl = (self.n_segments,) + tuple (self.endpoints.flat) + (self.r_orig,)
        if self.had_tag:
            tpl = (self.tag,) + tpl
            r.append \
                ('-w %d,%d,%.11g,%.11g,%.11g,%.11g,%.11g,%.11g,%.11g' % tpl)
        else:
            r.append \
                ('-w %d,%.11g,%.11g,%.11g,%.11g,%.11g,%.11g,%.11g' % tpl)
        if self.segtype:
            tpr = '--taper-wire=%d,%d' % (self.n + 1, self.segtype)
            if self.taper_min or self.taper_max:
                if self.taper_max is not None:
                    mn   = self.taper_min or 0
                    tpr += ',%.11g,%.11g' % (mn, self.taper_max)
                else:
                    tpr += ',%.11g' % self.taper_min
            r.append (tpr)
        return '\n'.join (r)
    # end def as_cmdline

    def compute_ground (self, n, media):
        super ().compute_ground (n, media)
        eps = self.parent.min_seglen * 1e-3
        if abs (self.p1 [-1]) < eps:
            self.p1 [-1] = 0.0
        if abs (self.p2 [-1]) < eps:
            self.p2 [-1] = 0.0
        self.compute_endpoints ()
        if self.is_ground [0] and self.is_ground [1]:
            raise ValueError ("Both ends of a wire may not be grounded")
    # end def compute_ground

    def compute_endpoints (self):
        self.endpoints = np.array ([self.p1, self.p2])
        self.diff = self.p2 - self.p1
        if (self.diff == 0).all ():
            raise ValueError ("Zero length wire: %s %s" % (self.p1, self.p2))
        self.wire_len = np.linalg.norm (self.diff)
    # end def compute_endpoints

    def compute_equal_segments (self):
        """ Compute segments with equal segment length
            Second endpoint is slightly off in original Basic computation
            because it is computed from the first endpoint.
        """
        # Original comment: compute direction cosines
        # Unit vector in wire direction
        dirvec  = self.diff / self.wire_len
        seg_len = self.wire_len / self.n_segments
        # First segment start
        seg     = np.copy (self.p1)
        s0      = seg
        for i in range (self.n_segments):
            s1 = seg + (i + 1) * dirvec * seg_len
            self.segments.append (Segment (s0, s1, self, len (self.segments)))
            s0 = s1
            # Avoid rounding errors, these can have influence on the
            # currents computed, even if slightly off!
            self.segments [-1].seg_len = seg_len
            self.segments [-1].dirvec  = dirvec
        self.min_seglen = seg_len
    # end def compute_equal_segments

    def compute_segments (self):
        """ Silently ignore tapering if segment size is too short.
            If tapering fails we use equal-sized segments.
            In the future we may want to issue a warning in that case,
            because even for equally spaced segments the segment size is
            shorter than the user specified.
        """
        self.segments = []
        if self.segtype == 0:
            self.compute_equal_segments ()
        elif self.segtype == 1 or self.segtype == 2:
            try:
                self.compute_taper1_segments ()
            except Taper_Error: # pragma: no cover
                # We may want to issue a warning here, maybe when there
                # is a warning framework
                assert len (self.segments) == 0
                self.segtype = 0
                self.compute_equal_segments ()
        else:
            try:
                self.compute_taper2_segments ()
            except Taper_Error: # pragma: no cover
                # We may want to issue a warning here, maybe when there
                # is a warning framework
                assert len (self.segments) == 0
                self.segtype = 0
                self.compute_equal_segments ()
    # end def compute_segments

    def compute_taper1_segments (self):
        assert 0 <= self.segtype - 1 <= 1
        d = dict (end = self.segtype - 1)
        if self.taper_max is not None:
            d.update (max_t = self.taper_max)
        if self.taper_min is not None:
            d.update (min_t = self.taper_min)
        self.min_seglen = None
        for p1, p2 in taper1 (self.p1, self.p2, self.n_segments, self.r, **d):
            self.segments.append (Segment (p1, p2, self, len (self.segments)))
            if self.min_seglen is None:
                self.min_seglen = self.segments [-1].seg_len
    # end def compute_taper1_segments

    def compute_taper2_segments (self):
        d = {}
        if self.taper_max is not None:
            d.update (max_t = self.taper_max)
        if self.taper_min is not None:
            d.update (min_t = self.taper_min)
        self.min_seglen = None
        for p1, p2 in taper2 (self.p1, self.p2, self.n_segments, self.r, **d):
            self.segments.append (Segment (p1, p2, self, len (self.segments)))
            if self.min_seglen is None:
                self.min_seglen = self.segments [-1].seg_len
    # end def compute_taper2_segments

    def rotate (self, rmatrix):
        assert not getattr (self, 'segments', None)
        self.p1 = rmatrix.apply (self.p1)
        self.p2 = rmatrix.apply (self.p2)
        self.compute_endpoints ()
    # end def rotate

    def scale (self, factor):
        """ Scale everything by factor including radius
            Similar to the NEC GS card "Scale Structure Dimensions"
        """
        assert not getattr (self, 'segments', None)
        self.p1 = self.p1 * factor
        self.p2 = self.p2 * factor
        self._r = self._r * factor
        self.compute_endpoints ()
    # end def scale

    def translate (self, translation):
        assert not getattr (self, 'segments', None)
        self.p1 = self.p1 + translation
        self.p2 = self.p2 + translation
        self.compute_endpoints ()
    # end def translate

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
    >>> m.register_source (s, 4, 1)
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
        Connected_Ojb:
        conn r:
        Connected_Ojb:
         geo: 1 idx: 9 s:1
        W: 1
        conn l:
        Connected_Ojb:
         geo: 0 idx: 9 s:1
        conn r:
        Connected_Ojb:
        """
        self.do_timing    = t
        self.f            = f
        self.media        = media
        self.loads        = []
        self.check_ground ()
        self.sources      = []
        if isinstance (geo, Geo_Container):
            self.geo = geo
            assert geo.parent is None
            geo.parent = self
        else:
            self.geo = Geo_Container (self, geo)
            self.geo.compute_tags ()
        self.output_date  = False
        # Dictionary of ends to compute matches
        self.end_dict     = {}
        # Pulses
        self.pulses       = Pulse_Container ()
        self.print_opts   = print_opts or set (('far-field',))
        if not self.media or len (self.media) == 1:
            self.boundary = 'linear'
        self.geo.compute_segments ()
        self.geo.compute_ground ()
        self.compute_connectivity ()
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
        # \lambda = c / f
        # c = 1 / sqrt (mu_0 * epsilon_0)
        # \omega = 2 * pi * c/\lambda
        # 1 / (4*pi*\omega*\epsilon_0) = \lambda/(4pi*2pi*c*\epsilon_0)
        # = \lambda * sqrt (mu_0*epsilon_0)/epsilon_0 / (8*pi**2)
        # = \lambda * sqrt (mu_0/epsilon_0) / (8*pi**2)
        # \aprox 4.771345158604122
        self.m       = 4.77783352 * w
        # set small radius modification condition:
        self.srm     = .0001 * w
        # The wave number 2 * pi / lambda
        self.w        = 2 * np.pi / w
        self.w2       = self.w ** 2 / 2
        self.currents = None
        self.rhs      = None
        self.Z        = None
    # end def f

    def as_basic_input \
        ( self
        , args
        , filename = 'MININEC.OUT'
        , azi = None, zen = None
        , near = None, pwr_nf = None
        , pwr_ff = None, ff_dist = None, ff_abs = False
        , gainfile = None
        ):
        """ Output in the input format of original Basic implementation.
            Used for cross-evaluation.
            Note: The output is returned as a string. The filename in
            the arguments is the filename the Basic code will write its
            output to.
            The original Basic implemetation issues prompts and expects
            inputs, to make some sense of the generated input we
            document the prompt from the Basic implementation as
            comments.
        """
        r = []
        # OUTPUT TO CONSOLE, PRINTER, OR DISK (C/P/D):
        r.append ('D')
        # FILENAME (NAME.OUT):
        r.append (filename)
        # FREQUENCY (MHZ):
        r.append ('%.12g' % self.f)
        # ENVIRONMENT (+1 FOR FREE SPACE, -1 FOR GROUND PLANE):
        if self.media:
            r.append ('-1')
            # NUMBER OF MEDIA (0 FOR PERFECTLY CONDUCTING GROUND):
            if len (self.media) == 1 and self.media [0].is_ideal:
                r.append ('0')
            else:
                r.append (str (len (self.media)))
                for m in self.media:
                    r.append (m.as_basic_input ())
        else:
            r.append ('+1')
        # NO. OF WIRES:
        # The original mininec code only supports straight wires with
        # equal segmentation. We ask each geometry object about the
        # number of emulated wires.
        nw = sum (w.n_emulated_wires for w in self.geo)
        r.append (str (nw))
        for w in self.geo:
            r.append (w.as_basic_input ())
        # CHANGE GEOMETRY (Y/N):
        r.append ('N')
        # NO. OF SOURCES:
        r.append (str (len (self.sources)))
        for s in self.sources:
            r.append (s.as_basic_input ())
        # NUMBER OF LOADS:
        lsum = 0
        is_s = False
        ldtypes = (Impedance_Load, Skin_Effect_Load, Insulation_Load)
        for l in self.loads:
            lsum += len (l.pulses)
            if not isinstance (l, ldtypes):
                is_s = True
        r.append (str (lsum))
        if lsum:
            # S-PARAMETER (S=jw) IMPEDANCE LOAD (Y/N):
            r.append ('Y' if is_s else 'N')
        for l in self.loads:
            r.append (l.as_basic_input (args, is_s))
        # C - COMPUTE/DISPLAY CURRENTS
        r.append ('C')
        # SAVE CURRENTS TO A FILE (Y/N):
        r.append ('N')
        if azi and zen:
            # P - COMPUTE FAR-FIELD PATTERNS
            r.append ('P')
            # CALCULATE PATTERN IN DBI OR VOLTS/METER (D/V):
            r.append ('V' if ff_abs else 'D')
            if ff_abs:
                # CHANGE POWER LEVEL (Y/N):
                if pwr_ff is not None:
                    r.append ('Y')
                    # NEW POWER LEVEL (WATTS):
                    r.append ('%g' % pwr_ff)
                    # CHANGE POWER LEVEL (Y/N):
                    r.append ('N')
                else:
                    r.append ('N')
                # RADIAL DISTANCE (METERS):
                assert ff_dist is not None
                r.append ('%g' % ff_dist)
            # ZENITH ANGLE : INITIAL,INCREMENT,NUMBER:
            r.append ('%g, %g, %g' % (zen.initial, zen.inc, zen.number))
            # AZIMUTH ANGLE: INITIAL,INCREMENT,NUMBER:
            r.append ('%g, %g, %g' % (azi.initial, azi.inc, azi.number))
            # FILE PATTERN (Y/N):
            if gainfile:
                r.append ('Y')
                r.append (gainfile)
            else:
                r.append ('N')
        if near:
            near = np.reshape (np.array (list (near)), (3, 3)).T
            for fieldtype in 'EH':
                # N - COMPUTE NEAR-FIELDS
                r.append ('N')
                # ELECTRIC OR MAGNETIC NEAR FIELDS (E/H):
                r.append (fieldtype)
                for ini, inc, n in near:
                    # X-COORDINATE (M): INITIAL,INCREMENT,NUMBER:
                    # Y-COORDINATE (M): INITIAL,INCREMENT,NUMBER:
                    # Z-COORDINATE (M): INITIAL,INCREMENT,NUMBER:
                    r.append ('%g, %g, %d' % (ini, inc, n))
                # CHANGE POWER LEVEL (Y/N):
                if pwr_nf is not None:
                    r.append ('Y')
                    # NEW POWER LEVEL (WATTS):
                    r.append ('%g' % pwr_nf)
                    # CHANGE POWER LEVEL (Y/N):
                    r.append ('N')
                else:
                    r.append ('N')
                # SAVE TO A FILE (Y/N):
                r.append ('N')
        # Q - QUIT
        r.append ('Q')
        return '\n'.join (r)
    # end def as_basic_input

    def as_cmdline \
        ( self
        , azi = None, zen = None
        , near = None, pwr_nf = None
        , pwr_ff = None, ff_dist = None
        , load_by_geo = False
        , opt = ()
        ):
        r = []
        r.append ('-f %.8g' % self.f)
        for w in self.geo:
            r.append (w.as_cmdline ())
        for s in self.sources:
            cm = s.as_cmdline ()
            if cm:
                r.append (cm)
        for g in (self.media or ()):
            r.append (g.as_cmdline ())
        for l in self.loads:
            r.append (l.as_cmdline (self, by_geo = load_by_geo))
        if zen is not None:
            r.append ('--theta=%g,%g,%d' % (zen.initial, zen.inc, zen.number))
        if azi is not None:
            r.append ('--phi=%g,%g,%d' % (azi.initial, azi.inc, azi.number))
        if near is not None:
            r.append ('--near-field=%s' % ','.join ('%g' % x for x in near))
        if pwr_ff is not None:
            r.append ('--ff-power=%g' % pwr_ff)
        if pwr_nf is not None:
            r.append ('--nf-power=%g' % pwr_nf)
        if ff_dist is not None:
            r.append ('--ff-distance=%g' % ff_dist)
        for p in opt:
            r.append ('--option=%s' % p)
        r.append ('')
        return '\n'.join (r)
    # end def as_cmdline

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
        # about the radial distance, the latter is the distance of the
        # far field measurement point in radial direction.
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
        # cos is the real, -sin the imag part
        acs  = np.e ** (-1j * azimuth_angle.angle_rad ())
        zcs  = np.e ** (-1j * zenith_angle.angle_rad ())
        zcs_m, acs_m = np.meshgrid (zcs, acs)
        # spherical coordinates??
        rvec = np.array (
            [ -zcs_m.imag * acs_m.real + 1j * (zcs_m.real * acs_m.real)
            ,  zcs_m.imag * acs_m.imag - 1j * (zcs_m.real * acs_m.imag)
            ,  zcs_m
            ]).T
        zen_d, azi_d = np.meshgrid \
            (zenith_angle.angle_deg (), azimuth_angle.angle_deg ())
        gain = np.zeros (rvec.shape, dtype = complex)
        for k in self.image_iter ():
            kvec  = np.array ([1, 1, k])
            kvec2 = np.array ([k, k, 1])
            kv2   = np.tile (kvec2, (len (pv), 2, 1))
            kv2 [pv.ground] = np.array ([0, 0, 0])
            if k == 1 or not self.media or self.media [0].is_ideal:
                kv2g = np.copy (kv2)
                kv2g [pv.inv_ground] = np.array ([0, 0, 2])
                if k < 0:
                    kv2g [pv.inv_ground] = np.array ([0, 0, 0])
            else:
                kv2 [pv.inv_ground] = np.array ([0, 0, 0])
                assert self.media
            for a_i, azi in enumerate (acs):
                shp  = list (rvec.shape)
                shp [1] = len (pv)
                rvrp = np.reshape \
                    (np.repeat (rvec [:, a_i, :], len (pv), axis = 0), shp)
                # Vectorized computation of standard case
                if k == 1 or not self.media or self.media [0].is_ideal:
                    s2   = self.w * np.sum \
                        (pv.point * kvec * rvrp.real, axis = 2)
                    s    = np.exp (1j * s2)
                    sshp = list (s.shape)
                    sshp.insert (-1, 2)
                    ss   = np.reshape (np.repeat (s, 2, axis = 0), sshp)
                    b    = f3.T * ss * self.current
                    bshp = list (b.shape)
                    bshp.insert (1, 3)
                    bs   = np.zeros (bshp, dtype = complex)
                    bs [:, 0, :, :] = b
                    bs [:, 1, :, :] = b
                    bs [:, 2, :, :] = b
                    gain [:, a_i, :] += np.sum \
                        (kv2g.T * pv.dirvec.T * bs, axis = (2, 3))
                else:
                    rt3 = rvrp [:, :, 2]
                    # begin by finding specular distance
                    cond = rt3.real != 0
                    t4 = np.zeros (rt3.shape)
                    t4 [rt3.real == 0] = 1e5
                    t4 [cond] = \
                        (-pv.point.T [2] * rt3.imag) [cond] / rt3.real [cond]
                    b9 = (t4.T * acs [a_i].real).T + pv.point.T [0]
                    if self.boundary != 'linear':
                        # Pythagoras in case of circular boundary
                        b9 *= b9
                        b9 += ((-t4.T * acs [a_i].imag).T + pv.point.T [1]) ** 2
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
                    vsp.insert (1, 2)
                    h89o = h89
                    h89 = np.zeros (vsp, dtype = complex)
                    h89 [:, 0, :] = h89o
                    h89 [:, 1, :] = h89o
                    vsp.insert (1, 3)
                    v89o = v89
                    v89 = np.zeros (vsp, dtype = complex)
                    v89 [:, 0, 0, :] = v89o
                    v89 [:, 0, 1, :] = v89o
                    v89 [:, 1, 0, :] = v89o
                    v89 [:, 1, 1, :] = v89o
                    v89 [:, 2, 0, :] = v89o
                    v89 [:, 2, 1, :] = v89o
                    # compute contribution to sum
                    shp = j2.shape + (3,)
                    h   = np.zeros (shp)
                    h [:, :, 2] = media_height [j2] * 2
                    sh  = pv.point - h
                    s2 = self.w * np.sum \
                        (sh * rvrp.real * kvec, axis = 2)
                    s   = np.e ** (1j * s2)
                    sshp = list (s.shape)
                    sshp.insert (-1, 2)
                    ss   = np.reshape (np.repeat (s, 2, axis = 0), sshp)
                    b    = f3.T * ss * self.current
                    bshp = list (b.shape)
                    bshp.insert (1, 3)
                    bs   = np.zeros (bshp, dtype = complex)
                    bs [:, 0, :, :] = b
                    bs [:, 1, :, :] = b
                    bs [:, 2, :, :] = b
                    w67  = bs * v89
                    d   = acs [a_i].imag * pv.dirvec.T [0] \
                        + acs [a_i].real * pv.dirvec.T [1]
                    z67 = d * b * h89
                    tm1 = np.zeros (w67.shape, dtype = complex)
                    tm1 [:, 0, :, :] = \
                        ( acs [a_i].imag * z67.real
                        + acs [a_i].imag * z67.imag * 1j
                        )
                    tm1 [:, 1, :, :] = \
                        ( acs [a_i].real * z67.real
                        + acs [a_i].real * z67.imag * 1j
                        )
                    gain [:, a_i, :] += np.sum \
                        ((pv.dirvec.T * w67 + tm1) * kv2.T, axis = (2, 3))
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
        n      = len (self.pulses)
        self.Z = np.zeros ((n, n), dtype=complex)

        # Independent of k
        same_geobj   = np.logical_and (*self.pulses.matrix_same_geobj)
        same_len     = np.logical_and (*self.pulses.matrix_same_len)
        same_dir     = np.logical_and (*self.pulses.matrix_same_dir)
        same_seg     = np.logical_and (same_len, same_dir)
        same         = np.logical_and (same_geobj, same_seg)
        idx0, idx1   = self.pulses.matrix_geo_idx_0
        gs           = self.pulses.matrix_gnd_sgn [1]
        sg           = self.pulses.matrix_sign [1]
        sl           = self.pulses.matrix_seg_len [1]
        f6v          = np.ones ((n, n, 3), dtype = int)
        f6v [..., 2] = gs [..., 0]
        f7v          = np.ones ((n, n, 3), dtype = int)
        f7v [..., 2] = gs [..., 1]
        dv           = self.pulses.matrix_dirvec [1]
        di1          = dv [..., 0, :]
        di2          = dv [..., 1, :]
        zzz          = np.sum \
            ( self.pulses.matrix_dir_sgn [0][..., np.newaxis]
            * self.pulses.matrix_seg_len [0][..., np.newaxis]
            * self.pulses.matrix_dirvec  [0]
            , axis = -2
            )
        diag         = np.diag_indices (n)
        zero         = np.zeros (self.Z.shape, dtype = int)
        opt          = np.zeros (self.Z.shape, dtype = int)
        opt [np.logical_and (same, (idx0 == idx1))] = 1
        opt [diag] *= 2

        # Optimization on (upper) diagonals: The Z of two pulses is the
        # same if they are the same distance as two other pulses and
        # all are on the same geo object. This amounts to checking
        # pulses on diagonals (not just the main diagonal) for being on
        # the same geo object.
        # Note that this creates probably more work than simply
        # computing the values but to get *exactly* the same numbers
        # like the original implementation we're doing this here.
        # The optimization is only applied for the above-ground case
        # (k == 1).
        cpy_src = []
        cpy_dst = []
        diag_i, diag_j = diag
        for k in range (n):
            if k == 0:
                d = np.eye (n, dtype = bool)
            else:
                d = np.zeros ((n, n), dtype = bool)
                d [diag_i [:-k], diag_j [k:]] = True
            valid = np.logical_and (d, opt > 0)
            # We can use idx0 or idx1 because we established both
            # pulses have same geo object at that point
            for geobj in np.unique (idx0 [valid]):
                v = np.logical_and (valid, idx0 == geobj)
                a, b = np.where (v)
                if len (a) < 2:
                    continue
                cpy_src.append ((a [0], b [0]))
                v [cpy_src [-1]] = False
                cpy_dst.append (v)
        # Make sure cpy_dst are not computed:
        excp = np.logical_not (np.logical_or.reduce (cpy_dst))

        # We only compute upper triangle and copy to lower
        # But only if f8 (see below) is nonzero
        # This is the case for the mirror image (k == -1)
        triu = np.zeros ((n, n), dtype = bool)
        triu [np.triu_indices (n, 1)] = True
        ngnd   = np.logical_not (self.pulses.matrix_ground [1])
        ngnd   = np.logical_and (ngnd [..., 0], ngnd [..., 1])
        for k in self.image_iter ():
            ng         = ngnd if k < 0 else True
            kvec       = np.array ([1, 1, k])
            f8         = opt if k >= 0 else zero
            # These are elements not computed but copied from other triu
            # Note that this happens only in the above-ground part (k > 0)
            copy       = np.logical_and (triu, f8 > 0)
            # And these must be computed
            compu      = np.logical_and (np.logical_not (copy.T), ng)
            if k > 0:
                compu  = np.logical_and (compu, excp)
            v          = np.zeros ((n, n), dtype = complex)
            u          = np.zeros ((n, n), dtype = complex)
            vp         = np.zeros ((n, n), dtype = complex)
            vp [compu] = self.vector_potential (k, compu, 0.5)
            u  [compu] = vp [compu] * sg [..., 1][compu]
            # compute PSI(M,N-1/2,N)
            # Here we had 'if f8 < 2'
            cond       = f8 < 2
            c          = np.logical_and (cond, compu)
            vp [c]     = self.vector_potential (k, c, -0.5)
            v [compu]  = vp [compu] * sg [..., 0] [compu]
            # S(N+1/2)*PSI(M,N,N+1/2) + S(N-1/2)*PSI(M,N-1/2,N)
            vec3       = \
                ( f7v * u [..., np.newaxis] * di2
                + f6v * v [..., np.newaxis] * di1
                ) * kvec
            d          = self.w2 * np.sum (vec3 * zzz, axis = -1)
            # compute PSI(M+1/2,N,N+1)
            c1         = np.logical_and (f8 == 1, compu)
            u56        = np.zeros (u.shape, dtype = complex)
            u56 [c1]   = \
                self.pulses.matrix_sign [1][..., 1][c1] * u [c1] + vp [c1]
            c0         = np.logical_and (f8 == 0, compu)
            u56 [c0]   = self.scalar_potential (k, c0, 0.5, 1)
            # compute PSI(M-1/2,N,N+1)
            sp         = np.zeros (u.shape, dtype = complex)
            sp [c]     = self.scalar_potential (k, c, -.5, 1)
            u12        = np.zeros (u.shape, dtype = complex)
            u12 [c]    = (sp [c] - u56 [c]) / sl [..., 1][c]
            # compute PSI(M+1/2,N-1,N)
            u34        = np.zeros (u.shape, dtype = complex)
            u34 [c]    = self.scalar_potential (k, c, .5, -1)
            sp [c1]    = u56 [c1]
            sp [c0]    = self.scalar_potential (k, c0, -.5, -1)
            u12 [c]   += (u34 [c] - sp [c]) / sl [..., 0][c]
            # Here we had else (from 'if f8 < 2' above)
            cond       = np.logical_not (cond)
            c          = np.logical_and (cond, compu)
            sp [c]     = self.scalar_potential (k, c, -.5, 1)
            u12 [c]    = (2 * sp [c] - 4 * u [c] * sg [..., 1][c]) \
                       / sl [..., 1][c]
            self.Z    += k * (d + u12)
            # Copy diagonal optimizations
            if k == 1:
                for src, dst in zip (cpy_src, cpy_dst):
                    self.Z [dst] = self.Z [src]
            # Copy to lower triangle
            self.Z.T [copy] = self.Z [copy]
    # end def compute_impedance_matrix

    def compute_impedance_matrix_loads (self):
        for l in self.loads:
            for pulse in l.pulses:
                j  = pulse.idx
                f2 = 1 / self.m
                # Looks like K in the original code line 371 is set by
                # the preceeding loop iterating over the images. So we
                # replace this with self.media is not None
                if pulse.ground.any () and self.media is not None:
                    f2 *= 2
                # Weird, the imag part goes to the real Z component and
                # vice-versa, the contribution to the real part is
                # negated, the contribution to the imag part not
                self.Z [j][j] += -f2 * l.impedance (self.f, pulse) * 1j
    # end def compute_impedance_matrix_loads

    def geo_as_str (self):
        r = []
        for g in self.geo:
            r.append (g.conn_as_str ())
        return '\n'.join (r)
    # end def geo_as_str

    def nf_helper (self, k, v1, pidx):
        """ Compute potentials in near field calculation
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)
        >>> w [0].compute_segments ()

        >>> nf66 = 0.4792338 -0.1544592j
        >>> nf75 = 0.3218219 -0.1519149j
        >>> ex   = (nf66 + nf75) * w [0].segments [0].dirvec
        >>> vec0 = np.array ([0, -1, -1])
        >>> r    = m.nf_helper (1, vec0, 0)
        >>> assert r [1] == 0 and r [2] == 0
        >>> print ("%.7f %.7fj" % (ex [0].real, ex [0].imag))
        0.8010557 -0.3063741j
        >>> print ("%.7f %.7fj" % (r [0].real, r [0].imag))
        0.8010557 -0.3063742j
        >>> assert np.linalg.norm (ex - r) < 1e-7

        # Vectorized
        >>> vec0 = np.array ([vec0, vec0])
        >>> r    = m.nf_helper (1, vec0, np.array ([0, 0]))
        >>> assert (r [0] == r [1]).all ()
        >>> assert r [0][1] == 0 and r [0][2] == 0
        >>> print ("%.7f %.7fj" % (r [0][0].real, r [0][0].imag))
        0.8010557 -0.3063742j
        """
        kvec        = np.array ([1, 1, k])
        gs          = self.pulses.gnd_sgn
        v6          = np.ones (v1.shape, dtype = int)
        v6 [..., 2] = gs.T [0][pidx]
        v7          = np.ones (v1.shape, dtype = int)
        v7 [..., 2] = gs.T [1][pidx]
        dir         = self.pulses.dirvec
        d1          = dir [:, 0, :]
        d2          = dir [:, 1, :]
        # compute psi(0,J,J+.5)
        dv          = self.pulses.dvecs (0.5)
        v2          = dv [:, 0, :]
        vv          = dv [:, 1, :]
        v2          = v1 - kvec * v2 [pidx]
        vv          = v1 - kvec * vv [pidx]
        u           = self.psi (v2, vv, k, 0.5, pidx, exact = False) \
                      [..., np.newaxis]

        # compute psi(0,J-.5,J)
        dv          = self.pulses.dvecs (-0.5)
        v2          = dv [:, 0, :]
        vv          = dv [:, 1, :]
        v2          = v1 - kvec * v2 [pidx]
        vv          = v1 - kvec * vv [pidx]
        v           = \
            ( self.psi (v2, vv, k, 0.5, pidx, exact = False)
            * self.pulses.sign [..., 0][pidx]
            ) [..., np.newaxis]

        # real part of vector potential contribution
        # imaginary part of vector potential contribution
        return (v * d1 [pidx] * v6 + u * d1 [pidx] * v7) * kvec
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
        self.nf_param = np.array ([list (start), list (inc), list (nvec)]).T
        self.e_field = []
        self.h_field = []
        # virtual dipole length for near field calculation:
        s0 = .001 * self.wavelen
        if pwr is None:
            pwr = self.power
        self.nf_power = pwr
        f_e = np.sqrt (pwr / self.power)
        f_h = f_e / s0 / (4*np.pi)
        # Compute grid, coordinate lists can be different lengths on the axes
        r = [np.arange (s, s + n * i, i)
             for (s, i, n) in reversed (self.nf_param)
            ]
        self.near_field_coord = np.flip \
            ( np.array
                ([x.flatten () for x in np.meshgrid (*r, indexing = 'ij')])
            , axis = 0
            )

        # Looks like T5, T6, T7 are 0 except for the diagonale at each
        # iteration of I (the dimension), we build everything in one go
        t567 = np.identity (3) * 2 * s0

        # Not dependent on vec
        px  = self.pulses.idx
        sl  = self.pulses.seg_len
        pxl = len (px)
        px6 = np.reshape (np.repeat (px [np.newaxis, ...], 6, axis = 1), 6*pxl)
        px3 = np.reshape (np.repeat (px [np.newaxis, ...], 3, axis = 1), 3*pxl)
        # We combine the two first axes to run it through psi helper
        t567px = np.repeat (t567 [np.newaxis, ...], pxl, axis = 0)
        t567px = np.reshape (t567px, (pxl * 3, t567px.shape [-1]))

        for vecno, vec in enumerate (self.near_field_iter ()):
            # Originally X0, Y0, Z0 but only one of them is non-0 in
            # each iteration. This considers both versions of
            # J8 (the values -1,1) in the original code
            v0m = [vec + np.identity (3) * (j8 * s0 / 2)
                   for j8 in (-1, 1)
                  ]
            v0m = np.array (v0m)
            h   = np.zeros (3, dtype = complex)
            # H-field, original Basic variable K!
            # Originally this is 6X2-dimensional in Basic, every two
            # succeeding values are real and imag part, there seem
            # to be two vectors which are indexed using j9 and j8
            # takes the value -1 and 1 in the v0m initialization
            # (originally X0, Y0, Z0)
            kf  = np.zeros ((2, 3, 3), dtype = complex)
            vp  = np.repeat (vec [np.newaxis, ...], pxl, axis = 0)
            u56 = np.zeros ((pxl, 3), dtype = complex)
            u78 = np.zeros (3, dtype = complex)

            # Reshape inputs to run them through nf_helper in one go
            v = np.repeat (v0m [np.newaxis, ...], pxl, axis = 0)
            v = np.reshape (v, (pxl * 2 * 3, v.shape [-1]))

            for k in self.image_iter ():
                cond = np.ones (pxl, dtype = bool)
                if k < 0:
                    cond = np.logical_and \
                        (*np.logical_not (self.pulses.ground.T))
                v35_e = self.nf_helper (k, vp, px)
                v35hf = self.nf_helper (k, v, px6)
                v35_h = np.reshape (v35hf, (pxl, 2, 3, 3))
                # At this point comment notes
                # magnetic field calculation completed
                # and jumps to 1042 if H field
                # We compute both, E and H and continue
                d    = np.sum \
                    (v35_e [..., np.newaxis] * t567, axis = 2) * self.w2
                # compute psi(.5,J,J+1)
                u    = self.psi_near_field_56 (vec, t567px, k, .5, px3, 1)
                # compute psi(-.5,J,J+1)
                tmp  = self.psi_near_field_56 (vec, t567px, k, -.5, px3, 1)
                tmp2 = np.reshape (tmp - u, (pxl, 3))
                u    = tmp2 / sl [:, 1, np.newaxis]
                # compute psi(.5,J-1,J)
                u34  = self.psi_near_field_56 (vec, t567px, k, .5, px3, -1)
                # compute psi(-.5,J-1,J)
                tmp  = self.psi_near_field_56 (vec, t567px, k, -.5, px3, -1)
                tmp2 = np.reshape (u34 - tmp, (pxl, 3)) / sl [:, 0, np.newaxis]
                # gradient of scalar potential
                u56 [cond] += ((u + tmp2 + d) * k) [cond]
                # Here would be a GOTO 1048 (a continue of the K loop)
                # that jumps over the H-field calculation
                # we do both, E and H field
                # components of vector potential A
                curr = self.current [:, np.newaxis, np.newaxis, np.newaxis]
                kf += np.sum ((v35_h * curr) [cond], axis = 0) * k
            # The following code only for E-field (line 1050)
            # Note: We do not sum inside the loop because this leads
            # to a different sum, it seems often image and original
            # cancel.
            u78 += np.sum (u56 * self.current [..., np.newaxis], axis = 0)

            # Here the original code has a conditional backjump to
            # 964 (just before the J loop) re-initializing the
            # X0, Y0, Z0 array (v0m in this implementation).
            # This realizes the computation of both halves of v0m.
            # We do this in one go, see above for j8, j9.

            # This originally is a ON I GOTO
            # Hmm the kf array may not be consecutive real/imag
            # parts after all?
            # The prior variable i is the middle index
            # i == 0
            h [1]  = kf [0][0][2] - kf [1][0][2]
            h [2]  = kf [1][0][1] - kf [0][0][1]
            # i == 1
            h [0]  = kf [1][1][2] - kf [0][1][2]
            h [2] += (  -kf [1][1][0].real + kf [0][1][0].real
                     + (-kf [1][1][0].imag + kf [0][1][0].imag) * 1j
                     )
            # i == 2
            h [0] += (  -kf [1][2][1].real + kf [0][2][1].real
                     + (-kf [1][2][1].imag + kf [0][2][1].imag) * 1j
                     )
            h [1] += (  kf [1][2][0].real - kf [0][2][0].real
                     + (kf [1][2][0].imag - kf [0][2][0].imag) * 1j
                     )

            # imaginary part of electric field
            # real part of electric field
            # Don't know why real- and imag is exchanged here
            u78 *= -1j * self.m / s0
            #np.zeros (3, dtype = complex))
            self.e_field.append (u78 * f_e)
            self.h_field.append (h   * f_h)
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

    def integral_i2_i3 (self, t, vec2, vecv, k, r, exact_kernel = False):
        """ This is the to-be-integrated function called from psi via
            fast_quad.
            Uses variables:
            vec2 (originally (X2, Y2, Z2))
            vecv (originally (V1, V2, V3))
            k, t, exact_kernel
            r: the geo object radius (originally a(p4))
            Starts line 28
            c0 - c9  # Parameter of elliptic integral
            w: 2 * pi * f / c
               (omega / c, frequency-dependent constant in program)
            srm: small radius modification condition
                 0.0001 * c / f
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
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire.r, False)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819959 -0.1414754j
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire.r, True)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819959 -0.1414754j

        # Then a thick wire without exact kernel
        # Original produces
        # 0.2819941 -0.1414753j
        >>> w [0]._r = 0.01
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire.r, False)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.2819941 -0.1414753j

        # Then a thick wire *with* exact kernel
        # Original produces
        # -2.290341 -0.1467051j
        >>> vv = np.array ([ 1.07071418591, 0, 0])
        >>> v2 = np.array ([-1.07071418591, 0, 0])
        >>> t  = np.array ([0.4900725])
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire.r, True)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -2.2903413 -0.1467052j

        # Original produces
        # -4.219833E-02 -4.820928E-02j
        >>> vv = np.array ([16.06072, 0, 0])
        >>> v2 = np.array ([13.91929, 0, 0])
        >>> t  = np.array ([0.7886752])
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire.r, False)
        >>> print ("%.8f %.8fj" % (r.real, r.imag))
        -0.04219836 -0.04820921j

        # Original produces
        # -7.783058E-02 -.1079738j
        # But *ADDED TO THE PREVIOUS RESULT*
        >>> vv = np.array ([16.06072, 0, 0])
        >>> v2 = np.array ([13.91929, 0, 0])
        >>> t  = np.array ([0.2113249])
        >>> r, = m.integral_i2_i3 (t, v2, vv, 1, wire.r, False)
        >>> print ("%.8f %.8fj" % (r.real, r.imag))
        -0.03563231 -0.05976447j

        # Now compute for two vecs
        >>> vv  = np.array ([[16.06072, 0, 0], [ 1.07071418591, 0, 0]])
        >>> v2  = np.array ([[13.91929, 0, 0], [-1.07071418591, 0, 0]])
        >>> t   = np.array ([[0.7886752, 0.2113249], [0.4900725, 0.4900725]])
        >>> r   = np.array ([wire.r, wire.r])
        >>> xct = np.array ([False, True], dtype = bool)
        >>> res = m.integral_i2_i3 (t, v2, vv, 1, r, xct)
        >>> for k in res.flat:
        ...     print ("%.7f %.7fj" % (k.real, k.imag))
        -0.0421984 -0.0482092j
        -0.0356323 -0.0597645j
        -2.2903413 -0.1467052j
        -2.2903413 -0.1467052j
        >>> vecv = np.array ([1.070714, 0, 0])
        >>> vec2 = np.zeros ((2, 2, 3))
        >>> vecv = np.tile (vecv, (2, 2, 1))
        >>> radi = np.ones ((2, 2), dtype = float) * 0.01
        >>> args = vec2, vecv, 1, radi, np.ones ((2, 2), dtype = bool)
        >>> x, w = legendre_cache [8]
        >>> y = (x + .5)
        >>> res  = np.sum (w * m.integral_i2_i3 (y, *args), axis = -1)
        >>> for k in res.flat:
        ...     print ("%.7f %.7fj" % (k.real, k.imag))
        55.7187703 -0.1465045j
        55.7187703 -0.1465045j
        55.7187703 -0.1465045j
        55.7187703 -0.1465045j
        """
        if k < 0:
            vecv, vec2 = vec2, vecv
        if len (vec2.shape) > 1:
            mul  = t.shape [-1]
            shp  = list (vec2.shape)
            shp.insert (-1, mul)
            dvec = np.repeat \
                ((vecv - vec2) [..., np.newaxis, :], mul, axis = -2)
            v2s  = np.repeat \
                (vec2          [..., np.newaxis, :], mul, axis = -2)
        else:
            dvec = vecv - vec2
            v2s  = vec2
        vec3 = v2s + dvec * t [..., np.newaxis]
        t34  = np.zeros (vec3.shape [:-1], dtype = complex)
        d = d3 = np.linalg.norm (vec3, axis = -1)
        # MOD FOR SMALL RADIUS TO WAVELENGTH RATIO
        cond = r > self.srm
        xact = np.logical_and (cond, exact_kernel)
        a2   = r * r
        d3   = d3 * d3
        if np.isscalar (a2):
            a2 = np.array ([a2])
            r  = np.array ([r])
        else:
            a2 = np.repeat (a2 [..., np.newaxis], d.shape [-1], axis = -1)
            r  = np.repeat (r  [..., np.newaxis], d.shape [-1], axis = -1)
        d [cond] = np.sqrt (a2 [cond] + d3 [cond])
        b    = d3 [xact] / (d3 [xact] + 4 * a2 [xact])
        v0   = ellipk (1 - b) * np.sqrt (1 - b)
        t34 [xact] += \
            ( (v0 + np.log (d3 [xact] / (64 * a2 [xact])) / 2)
            / np.pi / r [xact]
            - 1 / d [xact]
            )
        b1 = d * self.w
        # EXP(-J*K*R)/R
        t34 += np.exp (-1j * b1) / d
        return t34
    #end def integral_i2_i3

    def near_field_iter (self):
        for a in self.near_field_coord.T:
            yield a
    # end def near_field_iter

    def fast_quad (self, a, b, args, n):
        """ This uses the idea of the original pre-scaled gauss
            parameters. Computation of the integrals in psi is a little
            faster that way. Note that fixed_quad from scipy.integrate
            is a pure python implementation which does the necessary
            integral bounds scaling. Since we're always integrating from
            0 to 1/f we can speed things up here and avoid some
            multiplications.  Note that we fall back to fixed_quad if
            the order is not in our pre-computed legendre_cache.  Also
            this is a special case that works only for the lower bound
            being 0.  The test uses a non-cached order of integration to
            test the else part of the if statement.
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, w)
        >>> m.register_source (s, 4)
        >>> vec2 = np.zeros (3)
        >>> vecv = np.array ([1.070714, 0, 0])
        >>> args = vec2, vecv, 1, w [0].r, True
        >>> r = m.fast_quad (0, 1, args, 5)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        55.5802636 -0.1465045j
        >>> r = m.fast_quad (0, 1, args, 8)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        55.7187703 -0.1465045j
        >>> vec2 = np.zeros ((2, 2, 3))
        >>> vecv = np.tile (vecv, (2, 2, 1))
        >>> radi = np.ones ((2, 2), dtype = float) * 0.01
        >>> args = vec2, vecv, 1, radi, np.ones ((2, 2), dtype = bool)
        >>> r = m.fast_quad (0, 1, args, 5)
        >>> for k in r.flat:
        ...     print ("%.7f %.7fj" % (k.real, k.imag))
        55.5802636 -0.1465045j
        55.5802636 -0.1465045j
        55.5802636 -0.1465045j
        55.5802636 -0.1465045j
        >>> r = m.fast_quad (0, 1, args, 8)
        >>> for k in r.flat:
        ...     print ("%.7f %.7fj" % (k.real, k.imag))
        55.7187703 -0.1465045j
        55.7187703 -0.1465045j
        55.7187703 -0.1465045j
        55.7187703 -0.1465045j
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

    def fix_distributed_loads (self):
        """ Loop over *all* pulses and check if some are not yet
            attached to an skin/insulation load
            This can happen when we have a wire with skin/coat load
            where an adjacent pulse belongs to another wire without
            load.
        """
        for p in self.pulses:
            # Only if wires are different and only one wire has a skin load
            g1 = p.segs [0].geobj
            g2 = p.segs [1].geobj
            if  g1 != g2:
                if  (  (g1.coat_load and not g2.coat_load)
                    or (not g1.coat_load and g2.coat_load)
                    ):
                    coat_load = g1.coat_load or g2.coat_load
                    if p not in coat_load.pulses:
                        self.register_load (coat_load, p.idx)
                if  (  (g1.skin_load and not g2.skin_load)
                    or (not g1.skin_load and g2.skin_load)
                    ):
                    skin_load = g1.skin_load or g2.skin_load
                    if p not in skin_load.pulses:
                        self.register_load (skin_load, p.idx)
    # end def fix_distributed_loads

    def psi (self, vec2, vecv, k, scale, pidx, exact = False, fvs = 0):
        """ Common code for entry points at 56, 87, and 102.
            This code starts at line 135.
            The variable fvs is used to distiguish code path at the end.
            Both p2 and p3 used to be floating-point segment indeces.
            We now directly pass the difference, it is always positive
            and can be 1 or 0.5.
            The variable p4 was the index of the wire, we now pass the
            pulse index to compute the geobj parameters from.
            vec2 replaces (X2, Y2, Z2)
            vecv replaces (V1, V2, V3)
            i6: Use reduced kernel if 0, this was I6! (single precision)
                So beware: condition "I6!=0" means variable I6! is == 0
                The exclamation mark is part of the variable name :-(

            Input:
            vec2, vecv
            k: ground index (-1 or 1, always a scalar)
            scale (used to be p3 - p2, this is always a scalar)
            pidx is the list of pulse indeces for which to compute geobj
                parameters from the pulse matrix.
                r: the geobj radius
                seg_len: the segment length of the geobj
                i6: the geobj i6
            fvs: scalar vs. vector potential
            is_near: This originally tested input C$ for "N" which is
                the selection of near field compuation, this forces
                non-exact kernel, see exact flag

        >>> ws = []
        >>> ws.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> w = ws [0]
        >>> s = Excitation (1, 0)
        >>> m = Mininec (7, ws)
        >>> m.register_source (s, 4)

        # Original produces:
        # 5.330494 -0.1568644j
        >>> vec2 = np.zeros (3)
        >>> vecv = np.array ([1.070714, 0, 0])
        >>> r = m.psi (vec2, vecv, 1, 0.5, 0, exact = True)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        5.3304830 -0.1568644j

        # Original produces:
        # -8.333431E-02 -0.1156091j
        >>> vec2 = np.array ([13.91929, 0, 0])
        >>> vecv = np.array ([16.06072, 0, 0])
        >>> r = m.psi (vec2, vecv, 1, 1, 0, fvs = 1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -0.0833344 -0.1156090j

        # The above one two times
        >>> vec2 = np.array ([[13.91929, 0, 0], [13.91929, 0, 0]])
        >>> vecv = np.array ([[16.06072, 0, 0], [16.06072, 0, 0]])
        >>> r = m.psi (vec2, vecv, 1, 1, np.array ([0, 0]), fvs = 1)
        >>> print (r.shape)
        (2,)
        >>> for k in r.flat:
        ...     print ("%.7f %.7fj" % (k.real, k.imag))
        -0.0833344 -0.1156090j
        -0.0833344 -0.1156090j
        """
        r       = self.pulses.radius.T  [int (scale > 0)][pidx]
        seg_len = self.pulses.seg_len.T [int (scale > 0)][pidx]
        i6      = self.pulses.i6.T      [int (scale > 0)][pidx]
        if np.isscalar (r):
            r       = np.array ([r])
            seg_len = np.array ([seg_len])
            i6      = np.array ([i6])
        # magnitude of S(U) - S(M)
        d0 = np.linalg.norm (vec2, axis = -1)
        # magnitude of S(V) - S(M)
        d3 = np.linalg.norm (vecv, axis = -1)
        # magnitude of S(V) - S(U)
        s4 = abs (scale) * seg_len
        # order of integration
        # gauss_n order gaussian quadrature
        t = (d0 + d3) / seg_len
        # CRITERIA FOR EXACT KERNEL
        # Use exact kernel only if t <= 1.1 (and exact was specified)
        exact *= (t <= 1.1)
        f2 = np.ones (r.shape, dtype = int)
        f2 [exact] = 2 * abs (scale)
        # gauss_n order gaussian quadrature
        gauss_n = np.ones (r.shape) * 8
        # i6 is 0 if not using exact kernel
        i6 = i6 * exact
        retval  = np.zeros (r.shape, dtype = complex)
        srm_idx = np.logical_and (exact, r <= self.srm)
        fvs_f   = 1 if fvs != 1 else 2
        retval [srm_idx] = fvs_f * np.log (seg_len [srm_idx] / r [srm_idx]) \
                         - 0.5 * fvs_f * self.w * seg_len [srm_idx] * 1j
        rest_idx = np.logical_not (srm_idx)
        # When exact is set we always use gauss_n == 8
        g_idx    = np.logical_and (rest_idx, np.logical_not (exact))
        gauss_n [np.logical_and (g_idx, t >  6)] = 4
        gauss_n [np.logical_and (g_idx, t > 10)] = 2
        # Integrate, need to distinguish different f2 values (1/f2 is
        # the upper integration bound and fast_quad needs a scalar)
        for quad in (8, 4, 2):
            qidx = np.logical_and (rest_idx, gauss_n == quad)
            for f2val in np.unique (f2):
                i = np.logical_and (qidx, f2 == f2val)
                if i.any ():
                    if len (vecv.shape) == 1:
                        assert r.shape     [-1] == 1
                        assert exact.shape [-1] == 1
                        args = vec2, vecv, k, r [0], exact [0]
                    else:
                        args = vec2 [i], vecv [i], k, r [i], exact [i]
                    retval [i] = self.fast_quad (0, 1/f2val, args, quad)
                    retval [i] = (retval [i] + i6 [i]) * s4 [i]
        if len (vecv.shape) == 1:
            assert retval.shape == (1,)
            return retval [0]
        return retval
    # end def psi

    def psi_near_field_56 (self, vec0, vect, k, ds0, pidx, ds2):
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
        >>> r = m.psi_near_field_56 (vec0, vect, 1, 0.5, 0, 1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.5496335 -0.3002106j

        # Vectorized
        >>> v0 = np.array ([vec0, vec0])
        >>> vt = np.array ([vect, vect])
        >>> ix = np.zeros (2, dtype = int)
        >>> r  = m.psi_near_field_56 (v0, vt, 1, 0.5, ix, 1)
        >>> assert (r [0] == r [1]).all ()
        >>> print ("%.7f %.7fj" % (r [0].real, r [0].imag))
        0.5496335 -0.3002106j
        """
        kvec = np.array ([1, 1, k])
        vec1 = vec0 + ds0 * vect / 2
        dv = self.pulses.dvecs (ds2)
        v2 = dv [:, 0, :]
        vv = dv [:, 1, :]
        v2 = vec1 - kvec * v2 [pidx]
        vv = vec1 - kvec * vv [pidx]
        return self.psi (v2, vv, k, ds2, pidx, exact = False)
    # end def psi_near_field_56

    def register_load (self, load, pulse = None, geo_tag = None):
        """ Default if no pulse is given is to add the load to *all*
            pulses of the given geobj, unless geo_tag is None, then it
            is added to *all* pulses of *all* geo objects. Otherwise if
            no geo_tag is given the pulse is an absolute index,
            otherwise it's the tag of a pulse on the wire given by
            geo_tag.  Indeces are 0-based.
        """
        if pulse is None:
            if geo_tag is None:
                for geobj in self.geo:
                    for p in geobj.pulse_iter ():
                        load.add_pulse (p)
            else:
                geobj = self.geo.by_tag [geo_tag]
                for p in geobj.pulse_iter ():
                    load.add_pulse (p)
            # Avoid adding same load several times
            if load.n is None:
                load.n = len (self.loads)
                self.loads.append (load)
        else:
            if pulse < 0:
                raise ValueError ("Pulse tag must be >= 1")
            if geo_tag is not None:
                err = ( 'Invalid pulse tag %d for geo object %d'
                      % (pulse + 1, geo_tag)
                      )
                if not self.geo.by_tag.get (geo_tag):
                    raise ValueError ('Invalid geo object tag %d' % (geo_tag))
                w = self.geo.by_tag [geo_tag]
                if pulse >= len (w.pulses):
                    raise ValueError (err)
                p = w.pulses [pulse].idx
            elif pulse >= len (self.pulses):
                raise ValueError ('Invalid pulse tag %d' % (pulse + 1))
            else:
                p = pulse
            load.add_pulse (self.pulses [p])
            # Avoid adding same load several times
            if load.n is None:
                load.n = len (self.loads)
                self.loads.append (load)
    # end def register_load

    def register_source (self, source, pulse, geo_tag = None):
        """ Register a source, either with absolute pulse index or with
            a pulse index relative to a geobj. Indeces are 0-based.
        """
        if pulse < 0:
            raise ValueError ("Pulse tag must be >= 1")
        # Check source index
        if geo_tag is not None:
            w = self.geo.by_tag.get (geo_tag)
            if not w:
                raise ValueError ('Invalid geo object: "%s"' % geo_tag)
            if pulse >= len (w.pulses):
                raise ValueError \
                    ( 'Invalid pulse tag %d for geo object %d'
                    % (pulse + 1, geo_tag)
                    )
            self.sources.append (source)
            source.register (self, w.pulses [pulse].idx)
        else:
            if pulse >= len (self.pulses):
                raise ValueError ('Invalid pulse tag %d' % (pulse + 1))
            self.sources.append (source)
            source.register (self, pulse)
    # end def register_source

    def scalar_potential (self, k, pidx, ds1, ds2):
        """ Compute scalar potential
            We're now using a two-dimensional index into the pulse
            matrix. This is used to access all pulse matrix items that
            are created via meshgrid. Note that it is possible to use a
            tuple instead to access a single item in the matrix.
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
        >>> p0 = m.pulses [0]
        >>> p8 = m.pulses [8]
        >>> r = m.scalar_potential (1, (0, 8), 0.5, -1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -0.0833344 -0.1156091j
        >>> w [0]._r = 0.001

        # When changing wire we need to flush the relevant parts of the
        # pulse cache
        >>> m.pulses.reset ()
        >>> r = m.scalar_potential (1, (0, 0), -0.5, 1)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        1.0497691 -0.3085993j

        >>> w [0]._r = 0.01
        >>> m.pulses.reset ()

        # vector case
        # Compute (0, 8) and (8, 0)
        >>> l = len (m.pulses)
        >>> cond = np.zeros ((l, l), dtype = bool)
        >>> cond [0, 8] = True
        >>> cond [8, 0] = True
        >>> r = m.scalar_potential (1, cond, 0.5, -1)
        >>> print (r.shape)
        (2,)
        >>> print ("%.7f %.7fj" % (r [0].real, r [0].imag))
        -0.0833344 -0.1156091j
        >>> print ("%.7f %.7fj" % (r [1].real, r [1].imag))
        -0.1052472 -0.0345366j
        """
        widx   = int (ds2 > 0)
        kvec   = np.array ([1, 1, k])
        plen   = len (self.pulses)
        retval = np.zeros ((plen, plen), dtype = complex)
        cidx   = np.zeros ((plen, plen), dtype = bool)
        cidx [pidx] = True
        midx   = self.pulses.matrix_idx
        mwx    = self.pulses.matrix_geo_idx
        cond   = midx [0] + ds1 != midx [1] + ds2 / 2
        cond   = np.logical_or (cond, mwx [0] != mwx [1])
        cond   = np.logical_or \
            (cond, self.pulses.matrix_radius [1][..., widx] >= self.srm)
        cond   = np.logical_or (cond, k < 1)
        co1    = cidx * cond
        co2    = cidx * np.logical_not (cond)
        if co1.any ():
            v1  = self.pulses.matrix_endseg (ds1) [0][co1]
            dv  = self.pulses.matrix_dvecs (ds2) [1]
            v2  = kvec * dv [..., 0, :][co1] - v1
            vv  = kvec * dv [..., 1, :][co1] - v1
            wd  = self.pulses.matrix_geo_unconnected ()
            xct = np.logical_not (wd)
            px  = midx [1]
            retval [co1] = self.psi \
                (v2, vv, k, ds2, px [co1], fvs = 1, exact = xct [co1])
        if co2.any ():
            wl = self.pulses.matrix_seg_len [1].T [widx].T [co2]
            wr = self.pulses.matrix_radius  [1].T [widx].T [co2]
            t1 = 2 * np.log (wl / wr)
            t2 = -self.w * wl
            retval [co2] = t1 + t2 * 1j
        return retval [pidx]
    # end def scalar_potential

    def vector_potential (self, k, pidx, ds):
        """ Compute vector potential
            We're now using a two-dimensional index into the pulse
            matrix. This is used to access all pulse matrix items that
            are created via meshgrid. Note that it is possible to use a
            tuple instead to access a single item in the matrix.
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
        >>> r = m.vector_potential (1, (0, 1), -0.5)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        0.6747199 -0.1555773j

        # Compute (0, 1) and (1, 0)
        >>> l = len (m.pulses)
        >>> cond = np.zeros ((l, l), dtype = bool)
        >>> cond [0, 1] = True
        >>> cond [1, 0] = True
        >>> r = m.vector_potential (1, cond, -0.5)
        >>> print (r.shape)
        (2,)
        >>> print ("%.7f %.7fj" % (r [0].real, r [0].imag))
        0.6747199 -0.1555773j
        >>> print ("%.7f %.7fj" % (r [1].real, r [1].imag))
        0.3750293 -0.1530220j
        """
        widx   = int (ds > 0)
        kvec   = np.array ([1, 1, k])
        plen   = len (self.pulses)
        retval = np.zeros ((plen, plen), dtype = complex)
        cidx   = np.zeros ((plen, plen), dtype = bool)
        cidx [pidx] = True
        cond   = np.logical_not (np.eye (plen, dtype = bool))
        cond   = np.logical_or (cond, k < 1)
        cond   = np.logical_or \
            (cond, self.pulses.matrix_radius [1][..., widx] >= self.srm)
        co1    = cidx * cond
        co2    = cidx * np.logical_not (cond)
        if co1.any ():
            v1  = self.pulses.matrix_point [0][co1]
            dv  = self.pulses.matrix_dvecs (ds) [1]
            v2  = kvec * dv [..., 0, :][co1] - v1
            vv  = kvec * dv [..., 1, :][co1] - v1
            wd  = self.pulses.matrix_geo_unconnected ()
            xct = np.logical_not (wd)
            px  = self.pulses.matrix_idx [1]
            retval [co1] = self.psi \
                (v2, vv, k, ds, px [co1], exact = xct [co1])
        if co2.any ():
            wl = self.pulses.matrix_seg_len [1].T [widx].T [co2]
            wr = self.pulses.matrix_radius  [1].T [widx].T [co2]
            t1 = np.log (wl / wr)
            t2 = -self.w * wl / 2
            retval [co2] = t1 + t2 * 1j
        return retval [pidx]
    # end def vector_potential

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
        for geobj in self.geo:
            r.append ('WIRE NO.%3d :' % (geobj.tag))
            r.append \
                ( 'PULSE%sREAL%sIMAGINARY%sMAGNITUDE%sPHASE'
                % tuple (' ' * x for x in (9, 10, 5, 5))
                )
            r.append \
                ( ' NO.%s(AMPS)%s(AMPS)%s(AMPS)%s(DEGREES)'
                % tuple (' ' * x for x in (10, 8, 8, 8))
                )
            fmt = ''.join (['%s ' * 2] * 2)
            if not geobj.is_ground [0]:
                if not geobj.conn [0]:
                    r.append ((' ' * 13).join (['E '] + ['0'] * 4))
                else:
                    c = 0+0j
                    for p, s in geobj.conn [0].pulse_iter ():
                        assert p is not None
                        c = s * self.current [p]
                    a = np.angle (c) / np.pi * 180
                    r.append \
                        ( ('J ' + ' ' * 12 + fmt)
                        % format_float
                            ((c.real, c.imag, np.abs (c), a), use_e = True)
                        )
            for k in geobj.pulse_idx_iter (yield_ends = False):
                c = self.current [k]
                a = np.angle (c) / np.pi * 180
                r.append \
                    ( ('%s     ' + fmt)
                    % format_float
                        ((k + 1, c.real, c.imag, np.abs (c), a), use_e = True)
                    )
            if not geobj.is_ground [1]:
                if not geobj.conn [1]:
                    r.append ((' ' * 13).join (['E '] + ['0'] * 4))
                else:
                    c = 0+0j
                    for p, s in geobj.conn [1].pulse_iter ():
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
            r.append (l.as_mininec (self))
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
        for coord, (nf_s, nf_i, nf_c) in zip ('XYZ', self.nf_param):
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
        for geobj in self.geo:
            r.append ('%-4s NO. %d' % (geobj.name, geobj.tag))
            r.append \
                ( '%sCOORDINATES%sEND%sNO. OF'
                % (' ' * 12, ' ' * 33, ' ' * 9)
                )
            r.append \
                ( '   X%sY%sZ%sRADIUS%sCONNECTION%sSEGMENTS'
                % (' ' * 13, ' ' * 13, ' ' * 10, ' ' * 5, ' ' * 5)
                )
            l = []
            l.append (('%-13s ' * 3) % format_float (geobj.p1))
            l.append ('%s%3d' % (' ' * 12, geobj.idx_1))
            r.append (''.join (l))
            l = []
            l.append (('%-13s ' * 3) % format_float (geobj.p2))
            l.append ('%-13s' % format_float ([geobj.r]))
            l.append ('%2d%15d' % (geobj.idx_2, geobj.n_segments))
            r.append (''.join (l))
            r.append ('')
        r.append (' ' * 18 + '**** ANTENNA GEOMETRY ****')
        k = 1
        j = 0
        for geobj in self.geo:
            r.append ('')
            r.append \
                ( '%-4s NO.%3d  COORDINATES%sCONNECTION PULSE'
                % (geobj.name, geobj.tag, ' ' * 32)
                )
            r.append \
                (('%-13s ' * 4 + 'END1 END2  NO.') % ('X', 'Y', 'Z', 'RADIUS'))
            if geobj.end_segs [0] is None and geobj.end_segs [1] is None:
                r.append \
                    ( ('%-13s ' * 3 + '    %-10s %-4s %-4s %-4s')
                    % (('-',) * 6 + ('0',))
                    )
            for p in geobj.pulse_iter ():
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
    >>> args.extend (['--medium=0,0,0', '--excitation-pulse=1'])
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
    >>> args.extend (['--medium=0,0,0', '--excitation-pulse=1'])
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
    >>> args.extend (['--medium=0,0,0', '--excitation-pulse=1'])
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

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127,extra,ex2']
    >>> r = main (args, sys.stdout)
    Invalid number of parameters for wire 1
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', 'a,5,0,0,0,0,0,10.0838,0.0127']
    >>> r = main (args, sys.stdout)
    Invalid wire tag "a": invalid literal for int() with base 10: 'a'
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,extra']
    >>> r = main (args, sys.stdout)
    Invalid wire 1: could not convert string to float: 'extra'
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--excitation-pulse=1', '--excitation-pulse=2'])
    >>> r = main (args, sys.stdout)
    Number of excitation pulses must match voltages
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
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Load index 5 out of range
    >>> r
    23

    >>> args = ['--load=1+1j', '--attach-load=1,1', '--attach-load=1,7']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Error attaching load: Invalid pulse tag 7
    >>> r
    23

    >>> args = ['--load=1+1j', '--attach-load=1,1,7']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Error attaching load: Invalid geo object tag 7
    >>> r
    23

    >>> args = ['--load=1+1j', '--attach-load=1,7,1']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Error attaching load: Invalid pulse tag 7 for geo object 1
    >>> r
    23

    >>> args = ['-w', '2,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--excitation-pulse=1,5')
    >>> r = main (args, sys.stdout)
    Invalid source: 1,5: Invalid geo object: "5"
    >>> r
    23

    >>> args = ['-w', '2,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--excitation-pulse=7,1')
    >>> r = main (args, sys.stdout)
    Invalid source: 7,1: Invalid pulse tag 7 for geo object 1
    >>> r
    23

    >>> args = ['--load=1+1j', '--attach-load=1,0']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Error attaching load: Pulse tag must be >= 1
    >>> r
    23

    >>> args = ['--laplace-load-b=1', '--laplace-load-b=1']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Error in Laplace load: At least one denominator parameter required
    >>> r
    23

    >>> args = ['--laplace-load-a=1', '--laplace-load-b=1']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Error: Not all loads were used
    >>> r
    23

    >>> args = ['--laplace-load-a=1', '--laplace-load-b=1']
    >>> args = ['--laplace-load-a=1,2']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Error: Not all loads were used
    >>> r
    23

    >>> args = ['--laplace-load-a=1', '--laplace-load-b=1,b']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
    >>> r = main (args, sys.stdout)
    Error in Laplace load B: could not convert string to float: 'b'
    >>> r
    23

    >>> args = ['--laplace-load-a=1,a', '--laplace-load-b=1']
    >>> args.extend (['-w', '2,0,0,0,0,0,10.0838,0.0127'])
    >>> args.extend (['--excitation-pulse=1'])
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
    >>> args.extend (['--medium=0,0,5'])
    >>> r = main (args, sys.stdout)
    First medium must have height 0
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--excitation-pulse=1'])
    >>> args.extend (['--theta=0,45,3', '--phi=0,180'])
    >>> r = main (args, sys.stdout)
    Invalid phi angle, need three comma-separated values
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--excitation-pulse=1'])
    >>> args.extend (['--medium=1,1,0', '--medium=1,1,0,5'])
    >>> args.extend (['--theta=0,45,3', '--phi=0,180,nonint'])
    >>> r = main (args, sys.stdout)
    Invalid phi angle, need float, float, int
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.extend (['--excitation-pulse=1', '--radial-count=8'])
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

    >>> args = ['-f', '7.15']
    >>> args.extend (['--excitation-pulse=1'])
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

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--skin-effect-conductivity=2.5e+06,1')
    >>> args.append ('--skin-effect-resistivity=4e-07,1')
    >>> r = main (args, sys.stdout)
    Error in skin-effect resistivity: Only one skin-effect load per geo object
    >>> r
    23

    >>> args = ['-w', '4711,10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--skin-effect-resistivity=4e-07,4712')
    >>> r = main (args, sys.stdout)
    Error in skin-effect resistivity: Invalid tag: 4712
    >>> r
    23

    >>> args = ['--wire=10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--skin-effect-resistivity=4e-07,1,2')
    >>> r = main (args, sys.stdout)
    Error in skin-effect-resistivity: Invalid number of parameters
    >>> r
    23

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--skin-effect-resistivity=4e-07,1')
    >>> args.append ('--skin-effect-resistivity=4e-07,1')
    >>> r = main (args, sys.stdout)
    Error in skin-effect resistivity: Only one skin-effect load per geo object
    >>> r
    23

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--skin-effect-conductivity=2.5e+06,1')
    >>> args.append ('--skin-effect-conductivity=2.5e+06,1')
    >>> r = main (args, sys.stdout)
    Error in skin-effect conductivity: Only one skin-effect load per geo object
    >>> r
    23

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--skin-effect-conductivity=2.5e+06,1,e')
    >>> r = main (args, sys.stdout)
    Error in skin-effect-conductivity: Invalid number of parameters
    >>> r
    23

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--insulation-load=0.0047625,3.2,1,5')
    >>> r = main (args, sys.stdout)
    Error in insulation-load: Invalid number of parameters
    >>> r
    23

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--insulation-load=0.0047625,3.2,1')
    >>> r = main (args, sys.stdout)
    Error in insulation-load: Insulation radius must be > geo object radius
    >>> r
    23

    >>> args = ['-w', '10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--insulation-load=0.02,3.2,1')
    >>> args.append ('--insulation-load=0.02,3.2,1')
    >>> r = main (args, sys.stdout)
    Error in insulation-load: Only one insulation-load per geo object
    >>> r
    23

    >>> args = ['-w', '0,10,0,0,0,0,0,10.0838,0.0127']
    >>> r = main (args, sys.stdout)
    Error in Geo: Tag "0" not allowed
    >>> r
    23

    >>> args = ['--wire=4711,10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--wire=4711,10,0,0,0,0,0,-10.0838,0.0127')
    >>> r = main (args, sys.stdout)
    Error in Geo: Duplicate tag "4711" in geo object
    >>> r
    23

    >>> args = ['--wire=4711,10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-translate=1,1,1,1,1,4711')
    >>> r = main (args, sys.stdout)
    Invalid geo-translate option: 1,1,1,1,1,4711, invalid number of parameters
    >>> r
    23

    >>> args = ['--wire=4711,10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-rotate=1,1,1,1,1,4711')
    >>> r = main (args, sys.stdout)
    Invalid geo-rotate option: 1,1,1,1,1,4711, invalid number of parameters
    >>> r
    23

    >>> args = ['--wire=4711,10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--taper=4711,5,1,1,42')
    >>> r = main (args, sys.stdout)
    Invalid taper option: 4711,5,1,1,42, invalid number of parameters
    >>> r
    23

    >>> args = ['--wire=4711,10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--taper=4711,5,a,1')
    >>> r = main (args, sys.stdout)
    Invalid taper option: 4711,5,a,1, could not convert string to float: 'a'
    >>> r
    23

    >>> args = ['--wire=4711,10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--taper=4711,5,1,1')
    >>> r = main (args, sys.stdout)
    Invalid taper option: unknown taper 5
    >>> r
    23

    >>> args = ['--wire=4711,10,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--taper=4712,1,1,1')
    >>> r = main (args, sys.stdout)
    Invalid wire in taper option: 4712
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--excitation-pulse=a')
    >>> r = main (args, sys.stdout)
    Invalid pulse for excitation: "a"
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--excitation-pulse=1,1,5')
    >>> r = main (args, sys.stdout)
    Invalid number of pulse index parameters: "1,1,5"
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-scale=a,b')
    >>> r = main (args, sys.stdout)
    Invalid geo-scale option: a,b, invalid literal for int() with base 10: 'b'
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-scale=1,2,3')
    >>> r = main (args, sys.stdout)
    Invalid geo-scale option: 1,2,3, invalid number of parameters
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-translate=1,1,2,3,4711')
    >>> r = main (args, sys.stdout)
    Invalid geo-transformation: 1,1,2,3,4711, KeyError(4711)
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-rotate=1,1,2,3,4711')
    >>> r = main (args, sys.stdout)
    Invalid geo-transformation: 1,1,2,3,4711, KeyError(4711)
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-scale=2,4711')
    >>> r = main (args, sys.stdout)
    Invalid geo-scale option: 2,4711, KeyError(4711)
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-translate=1,1,2,1.1.')
    >>> r = main (args, sys.stdout)
    Invalid geo-translate option: 1,1,2,1.1., could not convert string to float: '1.1.'
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-translate=1,1,2,3,4,5')
    >>> r = main (args, sys.stdout)
    Invalid geo-translate option: 1,1,2,3,4,5, invalid number of parameters
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-rotate=1,1,2,1.1.')
    >>> r = main (args, sys.stdout)
    Invalid geo-rotate option: 1,1,2,1.1., could not convert string to float: '1.1.'
    >>> r
    23

    >>> args = ['-f', '7.15', '-w', '5,0,0,0,0,0,10.0838,0.0127']
    >>> args.append ('--geo-rotate=1,1,2,3,4,5')
    >>> r = main (args, sys.stdout)
    Invalid geo-rotate option: 1,1,2,3,4,5, invalid number of parameters
    >>> r
    23

    >>> args = ['-w', '1,0,0,0,0,0.5,0,0.001', '--excitation-pulse=1']
    >>> args.extend (['--load=2e-06', '--attach-load=1,1,1'])
    >>> r = main (args, sys.stdout)
    Invalid source: 1: Invalid pulse tag 1
    >>> r
    23

    >>> args = ['-a', '2,2,1,0,90,0.001,bla']
    >>> r = main (args, sys.stdout)
    Invalid number of parameters for arc 1

    >>> args = ['-a', '2,1,0,90']
    >>> r = main (args, sys.stdout)
    Invalid number of parameters for arc 1

    >>> args = ['-a', '2,1,0,90,0.001']
    >>> r = main (args, sys.stdout)
    Invalid arc 1: Arc needs at least three segments

    >>> args = ['-a', '3,0,0,90,0.001']
    >>> r = main (args, sys.stdout)
    Invalid arc 1: Arc radius must be > 0

    >>> args = ['-a', '3,1,0,0,0.001']
    >>> r = main (args, sys.stdout)
    Invalid arc 1: Arc angles must be different

    >>> args = ['-a', '3,1,0,361,0.001']
    >>> r = main (args, sys.stdout)
    Invalid arc 1: Arcs must not exceed a full circle

    >>> args = '-a 3,1,0,90,0.001 --medium=0,0,0 --geo-rotate=1,0,90,0'
    >>> args = args.split ()
    >>> r = main (args, sys.stdout)
    Invalid config: Geo object 1: height cannot not be negative with ground

    >>> args = '-a 3,1,0,90,0.001 --medium=0,0,0 --geo-rotate=1,90,0,0'
    >>> args = args.split ()
    >>> r = main (args, sys.stdout)
    Invalid config: Arc: No two adjacent segments may be grounded

    """
    from argparse import ArgumentParser
    cmd = ArgumentParser ()
    cmd.add_argument \
        ( '-a', '--arc'
        , help    = 'Arc definition, 5-6 values delimited with ",":'
                    " Optional tag, Number of segments, arc radius,"
                    " arc start angle, arc end angle, wire radius;"
                    " can be specified more than once."
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--attach-load'
        , help    = 'Attach load with given tag to pulse, needs'
                    ' load-tag, pulse-tag, optional geo object tag. If'
                    ' a geo object tag is given, pulse tag is relative to'
                    ' the geo object. To attach a load to all pulses use'
                    ' "all" for the pulse tag, this attaches the load to'
                    ' all pulses if no geo tag is given, otherwise to all'
                    ' pulses of the geo object'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--boundary'
        , help    = 'Boundary between different media'
        , default = 'linear'
        , choices = ('linear', 'circular')
        )
    cmd.add_argument \
        (  '--excitation-pulse'
        , help    = "Pulse tag for excitation, either an absolute pulse"
                    " tag or pulse tag and geo object tag separated by a"
                    " comma; can be specified "
                    "more than once, default is the single pulse 5"
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
        , help    = "Distance used for absolute (V/m) far-field computation"
        , type    = float
        )
    cmd.add_argument \
        ( '--ff-power'
        , help    = "Power used for absolute (V/m) far-field computation"
        , type    = float
        )
    cmd.add_argument \
        ( '--geo-rotate'
        , help    = "Gets 4-5 comma-separated parameters,"
                    " a (numeric) key for sorting specifying the order of geo"
                    " transformations, the angle for X- Y- and Z-axis"
                    " and an optional fifth parameter the geo object"
                    " tag to rotate"
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--geo-scale'
        , help    = "Gets one or two comma-separated parameters,"
                    " the first is the scale factor the second the geo"
                    " object tag to scale"
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--geo-translate'
        , help    = "Gets 4-5 comma-separated parameters,"
                    " a (numeric) key for sorting specifying the order of geo"
                    " transformations, the X- Y- and Z-translation"
                    " and an optional fifth parameter the geo object"
                    " tag to translate"
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--laplace-load-a'
        , help    = 'Laplace load, A (denominator) parameters (comma-separated)'
                    ' if multiple load-types are given, Laplace loads'
                    ' are tagged last.'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--laplace-load-b'
        , help    = 'Laplace load, B (numerator) parameters (comma-separated)'
                    ' if multiple load-types are given, Laplace loads'
                    ' are tagged last.'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '-l', '--load'
        , type    = complex
        , help    = 'Complex load, specify complex impedance, e.g.  50+3j,'
                    ' complex loads are tagged first'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--rlc-load'
        , help    = 'RLC (series) load, specify R,L,C in Ohm, Henry, Farad,'
                    ' RLC loads are tagged second'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--skin-effect-conductivity'
        , help    = 'Wire conductivity load, give conductivity and'
                    ' optionally the geometry tag, if no tag is given, it'
                    ' applies to all geometry objects'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--skin-effect-resistivity'
        , help    = 'Wire resistivity load, give resistivity and'
                    ' optionally the geometry tag, if no tag is given, it'
                    ' applies to all geometry objects'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--insulation-load'
        , help    = 'Wire insulation load, radius, epsilon_r and'
                    ' optionally the geometry tag, if no tag is given, it'
                    ' applies to all geometry objects'
        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--trap-load'
        , help    = 'Trap load, R+L in series parallel to C, '
                    'specify R,L,C in Ohm, Henry, Farad,'
                    ' Trap loads are tagged third'
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
        ( '--mininec-version'
        , help    = "Version of MININEC for which to produce the input"
                    " for the Basic implementation"
        , choices = ('9', '12', '13')
        , default = '9'
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
    cmd.add_argument \
        ( '--option'
        , help    = "Computation/printing options, option can be repeated."
                    " If none are given, far field "
                    " is printed if no near-field option is present, "
                    " otherwise near field is printed. The none option"
                    " can be used to inhibit far/near field computation"
                    " when only other output (or only command-line check)"
                    " is desired"
        , action  = "append"
        , choices = ['far-field', 'near-field', 'far-field-absolute', 'none']
        , default = []
        )
    cmd.add_argument \
        ( '--output-basic-input'
        , help    = "Output a file that can be used to compute the"
                    " antenna via the old mininec basic program."
        )
    cmd.add_argument \
        ( '--output-cmdline'
        , help    = "Output a file that contains the command line parameters"
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
        ( '--taper-wire'
        , help    = "Taper a wire from end 1, 2 or both (3) and"
                    " optionally set min and max segment length, gets"
                    " 2-4 parameters: wire tag, taper (1-3)"
                    " and optionally min and max segment length"

        , action  = 'append'
        , default = []
        )
    cmd.add_argument \
        ( '--theta'
        , help    = "Theta angle: start, increment, count"
        , default = '0,10,10'
        )
    cmd.add_argument \
        ( '-T', '--timing'
        , help    = 'Measure the time for certain parts of the algorithm'
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '-w', '--wire'
        , help    = 'Wire definition 8-9 values delimited with ",":'
                    " Optional tag, Number of segments,"
                    " x,y,z coordinates of wire endpoints plus wire radius,"
                    " can be specified more than once, default is a"
                    " single wire with length 21.414285"
        , action  = 'append'
        , default = []
        )
    args = cmd.parse_args (argv)
    if not args.wire and not args.arc:
        args.wire = ['10, 0, 0, 0, 21.414285, 0, 0, 0.001']
    default_excitation = False
    if not args.excitation_pulse:
        args.excitation_pulse = ['5']
        default_excitation = True
    if not args.excitation_voltage:
        args.excitation_voltage = [1]
    geo = Geo_Container ()
    for n, arc in enumerate (args.arc):
        aparams = arc.strip ().split (',')
        if not 5 <= len (aparams) <= 6:
            print \
                ( "Invalid number of parameters for arc %d" % (n + 1)
                , file = f_err
                )
            return 23
        tag = None
        if len (aparams) == 6:
            tag = aparams.pop (0)
            try:
                tag = int (tag)
            except ValueError as err:
                print \
                    ( 'Invalid arc tag "%s": %s'
                    % (tag, str (err))
                    , file = f_err
                    )
                return 23
        try:
            seg = int (aparams [0])
            r = [float (x) for x in aparams [1:]]
            geo.append (Arc (seg, *r, tag = tag))
        except ValueError as err:
            print ("Invalid arc %d: %s" % (n + 1, str (err)), file = f_err)
            return 23

    for n, wire in enumerate (args.wire):
        wparams = wire.strip ().split (',')
        if not 8 <= len (wparams) <= 9:
            print \
                ( "Invalid number of parameters for wire %d" % (n + 1)
                , file = f_err
                )
            return 23
        tag = None
        if len (wparams) == 9:
            tag = wparams.pop (0)
            try:
                tag = int (tag)
            except ValueError as err:
                print \
                    ( 'Invalid wire tag "%s": %s'
                    % (tag, str (err))
                    , file = f_err
                    )
                return 23
        try:
            seg = int (wparams [0])
            r = [float (x) for x in wparams [1:]]
            geo.append (Wire (seg, *r, tag = tag))
        except ValueError as err:
            print ("Invalid wire %d: %s" % (n + 1, str (err)), file = f_err)
            return 23

    try:
        geo.compute_tags ()
    except ValueError as err:
        print ("Error in Geo: %s" % str (err))
        return 23

    geo_transforms = []

    for rot in args.geo_rotate:
        rparam = rot.split (',')
        if not 4 <= len (rparam) <= 5:
            print \
                ( "Invalid geo-rotate option: %s, invalid number of parameters"
                % rot
                )
            return 23
        tag = None
        try:
            key = float (rparam [0])
            if len (rparam) == 5:
                tag = int (rparam [-1])
            rotation = [float (x) for x in rparam [1:4]]
            rotation = np.array (rotation)
            geo_transforms.append ((key, geo.rotate, rotation, tag, rot))
        except ValueError as err:
            print ("Invalid geo-rotate option: %s, %s" % (rot, err))
            return 23

    for tr in args.geo_translate:
        rparam = tr.split (',')
        if not 4 <= len (rparam) <= 5:
            print \
                ( "Invalid geo-translate option: %s, "
                  "invalid number of parameters"
                % tr
                )
            return 23
        tag = None
        try:
            key = float (rparam [0])
            if len (rparam) == 5:
                tag = int (rparam [-1])
            translation = [float (x) for x in rparam [1:4]]
            translation = np.array (translation)
            geo_transforms.append ((key, geo.translate, translation, tag, tr))
        except ValueError as err:
            print ("Invalid geo-translate option: %s, %s" % (tr, err))
            return 23

    for t in sorted (geo_transforms, key = lambda x: x [0]):
        try:
            t [1] (t [2], t [3])
        except (ValueError, KeyError) as err:
            print ("Invalid geo-transformation: %s, %s" % (t [-1], repr (err)))
            return 23

    for scl in args.geo_scale:
        rparam = scl.split (',')
        if not 1 <= len (rparam) <= 2:
            print \
                ( "Invalid geo-scale option: %s, "
                  "invalid number of parameters"
                % scl
                )
            return 23
        tag = None
        try:
            if len (rparam) == 2:
                tag = int (rparam [-1])
            factor = float (rparam [0])
            geo.scale (factor, tag)
        except ValueError as err:
            print ("Invalid geo-scale option: %s, %s" % (scl, err))
            return 23
        except KeyError as err:
            print ("Invalid geo-scale option: %s, %s" % (scl, repr (err)))
            return 23

    for t in args.taper_wire:
        tparam = t.split (',')
        taper_min = taper_max = None
        if not 2 <= len (tparam) <= 4:
            print ("Invalid taper option: %s, invalid number of parameters" % t)
            return 23
        try:
            tag, taper = (int (x) for x in tparam [:2])
            if len (tparam) > 2:
                taper_min = float (tparam [2])
            if len (tparam) > 3:
                taper_max = float (tparam [3])
        except ValueError as err:
            print ("Invalid taper option: %s, %s" % (t, err))
            return 23

        if not 0 <= taper <= 3:
            print ("Invalid taper option: unknown taper %s" % taper)
            return 23
        try:
            wire = geo.by_tag [tag]
        except KeyError as err:
            print ("Invalid wire in taper option: %s" % err)
            return 23
        if not isinstance (wire, Wire):
            # This can currently never happen, so it's not covered by
            # the tests. It will become possible when we have more
            # geometry objects in addition to Wire
            print ('Invalid wire in taper option: "%s" is no wire' % tag)
            return 23
        wire.segtype = taper
        wire.taper_min = taper_min
        wire.taper_max = taper_max

    if len (args.excitation_pulse) != len (args.excitation_voltage):
        print ("Number of excitation pulses must match voltages", file = f_err)
        return 23

    media = []
    rad = {}
    if args.radial_count:
        rad = dict \
            ( nradials = args.radial_count
            , radius   = args.radial_radius
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
        if n == 0 and p [2] != 0:
            print ("First medium must have height 0")
            return 23
        media.append (Medium (*p, **d))
    media = media or None

    try:
        m = Mininec (args.frequency, geo, media = media, t = args.timing)
    except ValueError as err:
        print ("Invalid config: %s" % str (err), file = f_err)
        return 23

    for p, v in zip (args.excitation_pulse, args.excitation_voltage):
        try:
            ep = [int (x) for x in p.split (',')]
        except ValueError as err:
            print ('Invalid pulse for excitation: "%s"' % p)
            return 23
        if not 1 <= len (ep) <= 2:
            print ('Invalid number of pulse index parameters: "%s"' % p)
            return 23
        if len (ep) > 1:
            s = Excitation (cvolt = v, geo_tag = ep [1], geo_idx = ep [0] - 1)
        else:
            s = Excitation (cvolt = v)
        if default_excitation:
            assert len (args.excitation_pulse) == 1
            s.is_default = True
        try:
            if len (ep) > 1:
                m.register_source (s, ep [0] - 1, ep [1])
            else:
                m.register_source (s, ep [0] - 1)
        except ValueError as err:
            print ('Invalid source: %s: %s' % (p, err))
            return 23
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
        if len (att) >= 2 and att [1] == 'all':
            att [1] = None
        try:
            att = [(None if a is None else int (a)) for a in att]
        except ValueError as err:
            print ("Attach-load: %s" % err, file = f_err)
            return 23
        lidx = att [0] - 1
        if lidx >= len (loads) or lidx < 0:
            print ("Load index %d out of range" % att [0], file = f_err)
            return 23
        used_loads.add (lidx)
        # Pulse index is 0-based
        if len (att) > 1 and att [1] is not None:
            att [1] = att [1] - 1
        try:
            m.register_load (loads [lidx], *att [1:])
        except ValueError as err:
            print ("Error attaching load: %s" % err, file = f_err)
            return 23
    if len (used_loads) != len (loads):
        print ("Error: Not all loads were used", file = f_err)
        return 23
    # Skin-effect loads are prior-to-last and attached automagically
    for l in args.skin_effect_conductivity:
        ld = l.split (',')
        if not 1 <= len (ld) <= 2:
            print \
                ( 'Error in skin-effect-conductivity: '
                  'Invalid number of parameters'
                )
            return 23
        tag = None
        try:
            if len (ld) == 2:
                tag = int (ld [-1])
            cond = float (ld [0])
            if tag is None:
                for w in m.geo:
                    ld = Skin_Effect_Load (w, cond)
                    m.register_load (ld, None, w.tag)
            else:
                w  = m.geo.by_tag [tag]
                ld = Skin_Effect_Load (w, cond)
                m.register_load (ld, None, w.tag)
        except (KeyError, ValueError) as err:
            print ("Error in skin-effect conductivity: %s" % err, file = f_err)
            return 23
    for l in args.skin_effect_resistivity:
        ld = l.split (',')
        if not 1 <= len (ld) <= 2:
            print \
                ( 'Error in skin-effect-resistivity: '
                  'Invalid number of parameters'
                )
            return 23
        tag = None
        try:
            if len (ld) == 2:
                tag = int (ld [-1])
            res = float (ld [0])
            if tag is None:
                for w in m.geo:
                    ld = Skin_Effect_Load (w, 1 / res)
                    m.register_load (ld, None, w.tag)
            else:
                w  = m.geo.by_tag [tag]
                ld = Skin_Effect_Load (w, 1 / res)
                m.register_load (ld, None, w.tag)
        except ValueError as err:
            print ("Error in skin-effect resistivity: %s" % err, file = f_err)
            return 23
        except KeyError as err:
            print \
                ("Error in skin-effect resistivity: Invalid tag: %s" % err
                , file = f_err
                )
            return 23

    # Insulation-loads are last and attached automagically
    for l in args.insulation_load:
        try:
            ld = l.split (',')
            if not 2 <= len (ld) <= 3:
                print ("Error in insulation-load: Invalid number of parameters")
                return 23
            tag = None
            if len (ld) == 3:
                tag = int (ld [-1])
            r, eps = (float (x) for x in ld [:2])
            if tag is None:
                for w in m.geo:
                    ld = Insulation_Load (w, r, eps)
                    m.register_load (ld, None, w.tag)
            else:
                w  = m.geo.by_tag [tag]
                ld = Insulation_Load (w, r, eps)
                m.register_load (ld, None, w.tag)
        except (KeyError, ValueError) as err:
            print ("Error in insulation-load: %s" % err, file = f_err)
            return 23

    m.fix_distributed_loads ()

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
        options.add (opt)
        if opt.startswith ('far'):
            far_field = True
    if not options:
        if nf_count:
            options.add ('near-field')
        else:
            options.add ('far-field')
            far_field = True
    if args.output_basic_input:
        with open (args.output_basic_input, 'w') as f:
            f.write (m.as_basic_input (args, azi = azimuth, zen = zenith))
    if args.output_cmdline:
        with open (args.output_cmdline, 'w') as f:
            f.write (m.as_cmdline (azi = azimuth, zen = zenith))
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
    main () # pragma: no cover

__all__ = \
    [ 'Angle'
    , 'Arc'
    , 'Excitation'
    , 'Far_Field_Pattern'
    , 'Gauge_Wire'
    , 'Geo_Container'
    , 'Impedance_Load'
    , 'Medium'
    , 'Mininec'
    , 'Laplace_Load'
    , 'Wire'
    , 'ideal_ground'
    ]
