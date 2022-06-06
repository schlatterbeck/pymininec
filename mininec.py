#!/usr/bin/python3

import numpy as np

class Medium:
    """ This encapsulates the media (e.g. ground screen etc.)
        Note that it seems only the first medium can have a
        ground screen of radials
    """
    def __init__ (self, dieel, cond, height = 0, nradials = 0, radius = 0):
        self.dieel    = dieel    # dielectric constant T(I)
        self.cond     = cond     # conductivity V(I)
        self.nradials = nradials # number of radials (NR)
        self.radius   = radius   # radial wire radius (RR)
        self.coord    = 1e6      # U(I)
        self.height   = height   # H(I)
    # end def __init__
# end class Medium

class Wire:
    """ A NEC-like wire
        The original variable names are
        x1, y1, z1, x2, y2, z2 (X1, Y1, Z1, X2, Y2, Z2)
        n_segments (S1)
        wire_len (D)
        seg_len  (S)
        dirs (CA, CB, CG)
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
            raise ValueError ("Zero length wire: %s %s" % (wire.p1, wire.p2))
        self.wire_len = np.linalg.norm (diff)
        self.seg_len  = self.wire_len / self.n_segments
        # Original comment: compute direction cosines
        self.dirs = diff / self.wire_len
        self.seg_start = None
        self.seg_end   = None
        # Wire end is grounded if Z coordinate is 0
        # In the original implementation this is kept in J1
        # with: 0: not grounded -1: start grounded 1: end grounded
        self.is_ground_start = (self.p1 [-1] == 0)
        self.is_ground_end   = (self.p2 [-1] == 0)
        if self.is_ground_start and self.is_ground_end:
            raise ValueError ("Both ends of a wire may not be grounded")
        self.j2 = [None, None]
    # end def __init__

    def __str__ (self):
        return 'Wire %s-%s, r=%s, seg_start=%s, seg_end=%s' \
            % (self.p1, self.p2, self.r, self.seg_start, self.seg_end)
    __repr__ = __str__

# end class Wire

class Mininec:
    """ A mininec implementation in Python
    >>> w = []
    >>> w.append (Wire (5, 0, 0, 7, 1, 0, 7, 0.001))
    >>> w.append (Wire (5, 1, 0, 7, 1, 1, 7, 0.001))
    >>> w.append (Wire (5, 1, 1, 7, 0, 1, 7, 0.001))
    >>> w.append (Wire (5, 0, 1, 7, 0, 0, 7, 0.001))
    >>> w.append (Wire (5, 0, 0, 7, 0, 0, 0, 0.001))
    >>> w.append (Wire (5, 1, 0, 7, 1, 0, 0, 0.001))
    >>> w.append (Wire (5, 1, 1, 7, 1, 1, 0, 0.001))
    >>> w.append (Wire (5, 0, 1, 7, 0, 1, 0, 0.001))
    >>> w.append (Wire (5, 0, 0, 7, 0, 0, 14, 0.001))
    >>> w.append (Wire (5, 1, 0, 7, 1, 0, 14, 0.001))
    >>> w.append (Wire (5, 1, 1, 7, 1, 1, 14, 0.001))
    >>> w.append (Wire (5, 0, 1, 7, 0, 1, 14, 0.001))
    >>> m = Mininec (20, w)
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

    def __init__ (self, f, geo, media = None):
        """ Initialize, no interactive input is done here
            f:   Frequency in MHz, (F)
            media: sequence of Medium objects, if empty use perfect ground
                   if None (the default) use free space
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
            tb:  Type of boundary (1: linear, 2: circular) (TB)
                 only if nm > 1
                 See class Medium above, if we have radials it's
                 circular
            Computed:
            w:   Wavelength in m, (W)
            s0:  virtual dipole lenght for near field calculation (S0)
            w:   ?  Comment: 1 / (4 * PI * OMEGA * EPSILON)
            srm: SMALL RADIUS MODIFICATION CONDITION
        """
        self.f       = f
        self.media   = media
        self.geo     = geo
        self.wavelen = w = 299.8 / f
        # virtual dipole length for near field calculation:
        self.s0      = .001 * w
        # 1 / (4 * PI * OMEGA * EPSILON)
        self.m       = 4.77783352 * w
        # set small radius modification condition:
        self.srm     = .0001 * w
        self.w       = 2 * np.pi / w
        self.w2      = w ** 2 / 2
        self.flg     = 0
        self.check_geo ()
        self.compute_connectivity ()
        self.compute_impedance_matrix ()
    # end __init__

    def check_geo (self):
        for wire in self.geo:
            if self.media is not None:
                # If we are in free space, negative coordinates are allowed
                if wire.z1 < 0 or wire.z2 < 0:
                    raise ValueError ("Z cannot not be negative with ground")
    # end def check_geo

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
            w.j2 [:] = (-i - 1, -i - 1)
            # check for ground connection
            if self.media is not None:
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
            n1 = n + 1
            self.geo [i].seg_start = n1
            if w.n_segments == 1 and i1 == 0:
                self.geo [i].seg_start = None
            n = n1 + w.n_segments
            if i1 == 0:
                n = n - 1
            if i2 == 0:
                n = n - 1
            self.geo [i].seg_end = n
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
                seg [i4] = w.p1 + j * w.p2 / w.seg_len
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
    # end def compute_connectivity

    def compute_impedance_matrix (self):
        """ Trick: Indeces in c_per are 1-based. A 0 in the index seems
            to mean to get a 0, so we insert a 0 into the 0th position
            of all arrays we index.
        """
        n = len (self.w_per)
        s = np.insert (np.array ([w.seg_len for w in self.geo]), 0, [0.0])
        self.Z = np.zeros ((n, n), dtype=complex)
        i1 = np.abs (self.c_per.T [0])
        i2 = np.abs (self.c_per.T [1])
        f4 = np.sign (self.c_per.T [0]) * s [i1]
        f5 = np.sign (self.c_per.T [1]) * s [i2]
        import pdb; pdb.set_trace ()
        d  = np.array  ([w.dirs for w in self.geo])
        d  = np.insert (d, 0, [0.0, 0.0, 0.0], axis = 0)
        # The t matrix replaces vectors t5, t6, t7
        t  = (np.tile (f4, (3, 1)).T * d [i1] + np.tile (f5, (3, 1)).T * d [i2])
        # Compute the special case in line 220 in one go
        ix = self.c_per.T [0] == -self.c_per.T [1]
        t.T [-1][ix] = (s [i1] * (d.T [-1][i1] + d.T [-1][i2])) [ix]
    # end def matrix_fill

# end class Mininec
