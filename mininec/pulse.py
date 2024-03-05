#!/usr/bin/python3 
# Copyright (C) 2024 Ralf Schlatterbeck. All rights reserved
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

import numpy as np
from functools import cached_property
from mininec.util import format_float

class Pulse_Container:

    def __init__ (self):
        self.pulses    = []
        self.pulse_idx = 0
    # end def __init__

    def add (self, pulse):
        self.pulses.append (pulse)
        pulse.idx = self.pulse_idx
        self.pulse_idx += 1
        assert len (self.pulses) == self.pulse_idx
    # end def add

    def __iter__ (self):
        for p in self.pulses:
            yield p
    # end def __iter__

    def __len__ (self):
        return self.pulse_idx
    # end def __len__

    def __getitem__ (self, idx):
        return self.pulses [idx]
    # end def __getitem__

    @cached_property
    def seg_len (self):
        return np.array \
            ([[w.seg_len for w in p.wires] for p in self.pulses])
    # end def seg_len

    @cached_property
    def dirvec (self):
        return np.array \
            ([[w.dirvec for w in p.wires] for p in self.pulses])
    # end def dirvec

    @cached_property
    def sign (self):
        return np.array ([p.sign for p in self.pulses])
    # end def sign

    @cached_property
    def ground (self):
        return np.array ([p.ground for p in self.pulses])
    # end def ground

    @cached_property
    def inv_ground (self):
        return np.array ([p.inv_ground for p in self.pulses])
    # end def inv_ground

    @cached_property
    def gnd_sgn (self):
        return np.array ([p.gnd_sgn for p in self.pulses])
    # end def gnd_sgn

    @cached_property
    def point (self):
        return np.array ([p.point for p in self.pulses])
    # end def point

# end class Pulse_Container

class Pulse:
    """ This models a pulse, the endpoint of two (or more) segments. So
        at the end of an unconnected wire there is no pulse.
        The Basic code doesn't have a data structure for this. Instead
        it had a list of segment endpoints in the variables X, Y, Z
        which was implemented as the 'seg' list in this implementation.
        There was an index of pulses to wires in W% (w_per) and a
        computation of indeces into the seg (X, Y, Z) list (here
        0-based, in the Basic code it was 1-based):
        idx = 2 * W% [i] + i + 1
        where i is the pulse index. This takes care of the fact that
        pulses overlapping from one wire to another take an additional
        segment end at the start and end of a wire.
        In addition there was a C% (c_per) index (with two integers per
        entry) that indexes each end of a pulse to the respective wire.
        We no longer have these, instead a Pulse now has links to other
        objects where necessary.
        The gnd parameter can specify wire1 or wire2 to be grounded
        using the index 0 or 1.
        The parameter sgn applies only to end segments and specifies the
        direction of the connected wire. It is applied to the dirvec of
        the wire.
        We compute self.sign from self.ground and self.sgn which also
        takes the sign due to grounding into account.
    """

    def __init__ \
        ( self, container, point, end1, end2
        , wire1, wire2, gnd = None, sgn = None
        ):
        self.container = container
        self.container.add (self)
        self.point   = point
        self.ends    = [end1, end2]
        self.wires   = [wire1, wire2]
        # The original implementation uses the sign of the wire index
        # for marking a ground connection *and* for marking a direction
        # reversal of a wire (when connected end1-end1 or end2-end2).
        # This results in ugly tests in the code which we can avoid by
        # keeping both signs separate.
        self.gnd_sgn = np.ones (2)
        self.dir_sgn = sgn
        if sgn is None:
            self.dir_sgn = [1, 1]
        assert not gnd or gnd == 1
        self.ground = np.array ([False, False])
        if gnd is not None:
            self.ground [gnd] = True
        # Sometimes needed for ground special cases
        # Semantics is "The other end is grounded"
        self.inv_ground = np.array ([self.ground [1], self.ground [0]])
        # The main wire is the one with the larger index
        self.wire_idx = np.argmax ([w.n for w in self.wires])
        self.wire     = self.wires [self.wire_idx]
        self.sign     = self.dir_sgn
        idx = np.array ([w.n for w in self.wires]) + 1
        if idx [0] == idx [1]:
            self.gnd_sgn [self.ground] = -1
            self.sign    = self.sign * self.gnd_sgn
        idx = idx * self.sign
        self.c_per = idx
    # end def __init__

    def as_mininec (self):
        l = []
        l.append (('%-13s ' * 3) % format_float (self.point))
        l.append ('%-12s' % format_float ([self.wire.r]))
        l.append ('%4d %4d' % tuple (self.c_per))
        l.append ('%4d' % (self.idx + 1))
        return  ''.join (l)
    # end def as_mininec

    def endseg (self, ds):
        """ Return segment to the negative direction or to the
            positive direction depending on ds: If ds <= 0 return
            negative, otherwise return positive. Scale with absolute
            value of ds.
        """
        return (self.ends [ds > 0] - self.point) * abs (ds) + self.point
    # end def endseg

    def dvecs (self, ds):
        """ Return two vectors, one of them always self.point, the other
            is the halfseg below or above (in wire direction),
            appropriately sorted, the negative direction segment is
            always first while the positive direction segment is last.
        """
        if ds < 0:
            return self.endseg (ds), self.point
        return self.point, self.endseg (ds)
    # end def dvecs

# end class Pulse

