#!/usr/bin/python3

import sys
import copy
import numpy as np

def format_float (*floats):
    """ Reproduce floating-point formatting of the Basic code
    """
    r = []
    e = 1e-9 # An epsilon to avoid python rounding down on exactly .5
    for f in floats:
        if f == 0:
            fmt = '%.1f'
        else:
            fmt = '%%.%df' % (6 - int (np.log (f) / np.log (10)))
        f += e
        s = fmt % f
        s = s [:8]
        s = s.rstrip ('0')
        s = s.rstrip ('.')
        if s.startswith ('0.'):
            s = s [1:]
        s = '%-8s' % s
        r.append (s)
    return tuple (r)
# end def format_float

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

    def mininec_output (self):
        return 'FIXME'
    # end def mininec_output

# end class Wire

class Mininec:
    """ A mininec implementation in Python
#    >>> w = []
#    >>> w.append (Wire (5, 0, 0, 7, 1, 0, 7, 0.001))
#    >>> w.append (Wire (5, 1, 0, 7, 1, 1, 7, 0.001))
#    >>> w.append (Wire (5, 1, 1, 7, 0, 1, 7, 0.001))
#    >>> w.append (Wire (5, 0, 1, 7, 0, 0, 7, 0.001))
#    >>> w.append (Wire (5, 0, 0, 7, 0, 0, 0, 0.001))
#    >>> w.append (Wire (5, 1, 0, 7, 1, 0, 0, 0.001))
#    >>> w.append (Wire (5, 1, 1, 7, 1, 1, 0, 0.001))
#    >>> w.append (Wire (5, 0, 1, 7, 0, 1, 0, 0.001))
#    >>> w.append (Wire (5, 0, 0, 7, 0, 0, 14, 0.001))
#    >>> w.append (Wire (5, 1, 0, 7, 1, 0, 14, 0.001))
#    >>> w.append (Wire (5, 1, 1, 7, 1, 1, 14, 0.001))
#    >>> w.append (Wire (5, 0, 1, 7, 0, 1, 14, 0.001))
#    >>> m = Mininec (20, w)
    >>> w = []
    >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.001))
    >>> m = Mininec (20, w)
    >>> m.print_wires ()
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
            m:   ?  Comment: 1 / (4 * PI * OMEGA * EPSILON)
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
        for n, wire in enumerate (self.geo):
            wire.compute_ground (n, self.media)
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
            w.j2 [:] = (-(i + 1), -(i + 1))
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
        self.c_per_fixed = copy.deepcopy (self.c_per)
        for idx in range (2):
            cnull = self.c_per.T [idx] == 0
            self.c_per_fixed.T [idx][cnull] = self.w_per [cnull]
    # end def compute_connectivity

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
        >>> m = Mininec (7, w)
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

    def psi_near_field_56 (self, vec0, vect, k, p1, p2, p3, p4, i, j):
        """ Compute psi used several times during computation of near field
            Original entry point in line 56
            vec0 originally is (X0, Y0, Z0)
            vect originally is (T5, T6, T7)
            Note that p1 is the only non-zero-based variable, it's not
            used as an index but as a factor.
        >>> w = []
        >>> w.append (Wire (10, 0, 0, 0, 21.414285, 0, 0, 0.01))
        >>> m = Mininec (7, w)

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
        >>> m = Mininec (7, w)

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
        >>> m = Mininec (7, w)

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
        >>> m = Mininec (7, w)

        # Original produces:
        # -8.333431E-02 -0.1156091j
        >>> method = m.scalar_potential
        >>> r = method (k=1, p1=1.5, p2=8, p3=9, p4=0, i=0, j=8)
        >>> print ("%.7f %.7fj" % (r.real, r.imag))
        -0.0833344 -0.1156091j
        """
        wire = self.geo [p4]
        if  (  k < 1
            or wire.r > self.srm
            or p3 != p2 + 1
            or p1 == (p2 + p3) / 2
            ):
            i4 = int (p1)
            i5 = i4 + 1
            vec1 = (self.seg [i4] + self.seg [i5]) / 2
            vec2, vecv = self.common_vec1_vecv (vec1, k, p2, p3)
            return self.psi (vec1, vec2, vecv, k, p2, p3, p4, i, j, fvs = 1)
        t1 = 2 * np.log (wire.seg_len / wire.r)
        t2 = -self.w * wire.seg_len
        return t1, t2 * 1j
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
        >>> m = Mininec (7, w)

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
            vec2, vecv = self.common_vec1_vecv (vec1, k, p2, p3)
            return self.psi (vec1, vec2, vecv, k, p2, p3, p4, i, j, fvs = 0)
        t1 = np.log (wire.seg_len / wire.r)
        t2 = -self.w * wire.seg_len / 2
        return t1, t2 * 1j
    # end def vector_potential

    def common_vec1_vecv (self, vec1, k, p2, p3):
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
            vecv = kvec * (self.seg [i4] + self.seg [i5]) / 3 - vec1
        return vec2, vecv
    # end def common_vec1_vecv

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
        >>> m = Mininec (7, w)

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
        assert self.w_per [i] - 1 >= 0
        assert self.w_per [j] - 1 >= 0
        wire_i_j2 = self.geo [self.w_per [i] - 1].j2
        wire_j_j2 = self.geo [self.w_per [j] - 1].j2
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
                return t1, t2 * 1j
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

    def compute_impedance_matrix (self):
        """ This starts at line 195 (with entry-point for gosub at 196)
            in the original basic code.
            Trick: Indeces in c_per are 1-based. A 0 in the index seems
            to mean to get a 0, so we insert a 0 into the 0th position
            of all arrays we index.
        """
        n = len (self.w_per)
        s = np.insert (np.array ([w.seg_len for w in self.geo]), 0, [0.0])
        #self.Z = np.zeros ((n, n), dtype=complex)
        i1 = np.abs (self.c_per.T [0])
        i2 = np.abs (self.c_per.T [1])
        f4 = np.sign (self.c_per.T [0]) * s [i1]
        f5 = np.sign (self.c_per.T [1]) * s [i2]
        d  = np.array  ([w.dirs for w in self.geo])
        d  = np.insert (d, 0, [0.0, 0.0, 0.0], axis = 0)
        # The t matrix replaces vectors t5, t6, t7
        t  = (np.tile (f4, (3, 1)).T * d [i1] + np.tile (f5, (3, 1)).T * d [i2])
        # Compute the special case in line 220 in one go
        ix = self.c_per.T [0] == -self.c_per.T [1]
        t.T [-1][ix] = (s [i1] * (d.T [-1][i1] + d.T [-1][i2])) [ix]
#        j loop
#            # compute j1 same as i1 above
#            # compute j2 same as i2 above
#            # compute f4 same as f4 above without s [i1] factor
#            # compute f5 same as f5 above without s [i1] factor
#            f6 = 1
#            f7 = 1
#            # IMAGE LOOP
#            k FIXME loop : # FOR K=1 TO G STEP -2
#                # IF C%(J,1)<>-C%(J,2) THEN 235
#                # So use inverse comparison and put in an if statement
#                if ix [j]:
#                    if k < 0:
#                        FIXME goto 332
#                    f6 = f4
#                    f7 = f5
#                f8 = 0
#                # Was inverse comparison with goto 248
#                if k >= 0:
#                    # A bunch of IF statements that jumped over each other
#                    if i1 == i2:
#                        if  (  ca(i1) + cb(i1) == 0
#                            or self.c_per [j][0] == c_per [j][1]
#                            ) and j1 == j2:
#                            if i1 == j1:
#                                f8 = 1
#                            if i == j:
#                                f8 = 2
#                    # line 246
#                    if zr [i, j] != 0 FIXME goto 317
#                # COMPUTE PSI(M,N,N+1/2)
    # end def compute_impedance_matrix

    def print_wires (self, file = None):
        if file is None:
            file = sys.stdout
        print (' ' * 18 + '**** ANTENNA GEOMETRY ****', file = file)
        for i, wire in enumerate (self.geo):
            print (file = file)
            print \
                ( 'WIRE NO.%3d  COORDINATES%sCONNECTION PULSE'
                % (i + 1, ' ' * 32)
                , file = file
                )
            print \
                ( ('%-13s ' * 4 + 'END1 END2  NO.') % ('X', 'Y', 'Z', 'RADIUS')
                , file = file
                )
            if wire.seg_start is None and wire.seg_end is None:
                print \
                    ( ('%-13s ' * 5 + '%-4s %-4s') % tuple (['-'] * 5 + ['0'])
                    , file = file
                    )
            for k in range (wire.seg_start, wire.seg_end + 1):
                seg = tuple (self.seg [k + 1])
                print \
                    ( (' ' + '%-13s ' * 3) % format_float (*seg)
                    , file = file, end = ''
                    )
                print ('%-12s' % format_float (wire.r), file = file, end = '')
                print \
                    ( '%4d %4d' % tuple (self.c_per [k])
                    , file = file, end = ' '
                    )
                print ('%4d' % (k + 1), file = file)
    # end def print_wires

# end class Mininec
