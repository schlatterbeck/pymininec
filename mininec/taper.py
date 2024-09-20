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

# Helper function for printing tapers
def _fmt (x):
    return ('%.3f' % x).rstrip ('0').rstrip ('.')
# end def _fmt

def _taper_fmt (x):
    if np.isscalar (x):
        return _fmt (x)
    return '(' + ', '.join (('%s' % _fmt (a)) for a in x) + ')'
# end def _taper_fmt

def taper_print (taper, dlm = ', '):
    l = list (taper)
    fmt = _taper_fmt
    print (dlm.join ('(%s, %s)' % (fmt (a), fmt (b)) for a, b in l))
# end def taper_print

class Taper_Error (ValueError):
    pass

# Taper something into power-of-two segments
# The 'something' can be two points (wire start / wire end) or just
# numbers (dimension 1). It also can handle more than 3 dimensions.
# If more than one dimension is used the points must be np.array like
# data structures.

def taper2 (p1, p2, n, r, min_t = 0, max_t = None):
    """ For now taper from both ends
    >>> p = taper_print
    >>> p (taper2 (0.0, 1.0, 5, 0.001))
    (0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1)
    >>> p (taper2 (0.0, 1.0, 5, 0.001, max_t = 0.3))
    (0, 0.117), (0.117, 0.35), (0.35, 0.65), (0.65, 0.883), (0.883, 1)
    >>> p (taper2 (0.0, 1.0, 5, 0.001, max_t = 0.2))
    (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)
    >>> p (taper2 (0.0, 1.0, 5, 0.04))
    (0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1)
    >>> p (taper2 (0.0, 1.0, 5, 0.05))
    (0, 0.125), (0.125, 0.375), (0.375, 0.625), (0.625, 0.875), (0.875, 1)
    >>> p (taper2 (0.0, 1.0, 5, 0.06))
    (0, 0.15), (0.15, 0.383), (0.383, 0.617), (0.617, 0.85), (0.85, 1)
    >>> p (taper2 (0.0, 1.0, 5, 0.08))
    (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)
    >>> p (taper2 (0.0, 7.0, 6, 0.001))
    (0, 0.5), (0.5, 1.5), (1.5, 3.5), (3.5, 5.5), (5.5, 6.5), (6.5, 7)
    >>> p (taper2 (0.0, 7.0, 6, 0.001, max_t = 1.5))
    (0, 0.667), (0.667, 2), (2, 3.5), (3.5, 5), (5, 6.333), (6.333, 7)
    >>> p (taper2 (0.0, 7.0, 6, 0.4))
    (0, 1), (1, 2.25), (2.25, 3.5), (3.5, 4.75), (4.75, 6), (6, 7)
    >>> p (taper2 (0.0, 7.0, 6, 0.001, min_t = 1))
    (0, 1), (1, 2.25), (2.25, 3.5), (3.5, 4.75), (4.75, 6), (6, 7)
    >>> p (taper2 (0.0, 3.0, 8, 0.001), '\\n')
    (0, 0.1)
    (0.1, 0.3)
    (0.3, 0.7)
    (0.7, 1.5)
    (1.5, 2.3)
    (2.3, 2.7)
    (2.7, 2.9)
    (2.9, 3)
    >>> p (taper2 (0, 10, 10, 4e-4, 0.32, 4), '\\n')
    (0, 0.32)
    (0.32, 0.96)
    (0.96, 2.24)
    (2.24, 3.62)
    (3.62, 5)
    (5, 6.38)
    (6.38, 7.76)
    (7.76, 9.04)
    (9.04, 9.68)
    (9.68, 10)
    >>> p (taper2 (0, 0.25, 10, 1e-5, 0.008, 0.1), '\\n')
    (0, 0.008)
    (0.008, 0.024)
    (0.024, 0.056)
    (0.056, 0.09)
    (0.09, 0.125)
    (0.125, 0.16)
    (0.16, 0.194)
    (0.194, 0.226)
    (0.226, 0.242)
    (0.242, 0.25)
    >>> p1 = np.ones (3)
    >>> p2 = np.array ([1.0, 2.0, 2.0]) + p1
    >>> p (taper2 (p1, p2, 5, 0.12), '\\n')
    ((1, 1, 1), (1.1, 1.2, 1.2))
    ((1.1, 1.2, 1.2), (1.3, 1.6, 1.6))
    ((1.3, 1.6, 1.6), (1.7, 2.4, 2.4))
    ((1.7, 2.4, 2.4), (1.9, 2.8, 2.8))
    ((1.9, 2.8, 2.8), (2, 3, 3))
    >>> p (taper2 (p1, p2, 5, 0.15), '\\n')
    ((1, 1, 1), (1.125, 1.25, 1.25))
    ((1.125, 1.25, 1.25), (1.375, 1.75, 1.75))
    ((1.375, 1.75, 1.75), (1.625, 2.25, 2.25))
    ((1.625, 2.25, 2.25), (1.875, 2.75, 2.75))
    ((1.875, 2.75, 2.75), (2, 3, 3))
    >>> p (taper2 (p1, p2, 5, 0.24), '\\n')
    ((1, 1, 1), (1.2, 1.4, 1.4))
    ((1.2, 1.4, 1.4), (1.4, 1.8, 1.8))
    ((1.4, 1.8, 1.8), (1.6, 2.2, 2.2))
    ((1.6, 2.2, 2.2), (1.8, 2.6, 2.6))
    ((1.8, 2.6, 2.6), (2, 3, 3))

    """
    lv = p2 - p1
    l  = np.linalg.norm (p2 - p1)
    uv = lv / l
    min_t = max (2.5 * r, min_t)
    assert n > 1
    if l / n < min_t:
        raise Taper_Error ('Segment size too short: %g < %g' % (l / n, min_t)) \
            # pragma: no cover
    assert max_t is None or min_t <= max_t
    assert max_t is None or l / n <= max_t
    if n & 1:
        npieces = 2 * ((1 << (n // 2)) - 1) + (1 << (n // 2))
        if max_t is not None:
            maxl = max_t / (1 << (n // 2))
    else:
        npieces = 2 * ((1 << (n // 2)) - 1)
        if max_t is not None:
            maxl = max_t / (1 << (n // 2 - 1))
    minl = l  / npieces
    if minl < min_t:
        minl = min_t
    eps  = minl / 10
    if max_t is not None and maxl < l / npieces:
        d = 1 if n & 1 else 2
        # m is minl, x is the length of the constant-length part,
        # ideally max_t
        # 2 * ((1 << (k // 2)) - 1) * m + (n - k) * x = l
        # l - 2 * ((1 << (k // 2)) - 1) * m = (n - k) * x
        # x <= max_t
        # l - 2 * ((1 << (k // 2)) - 1) * m = (n - k) * max_t
        # l - (n - k) * max_t <= 2 * ((1 << (k // 2)) - 1) * m
        # (l - (n - k) * max_t) / (2 * ((1 << (k // 2)) - 1)) <= m
        for k in range (n - d, 0, -2):
            nminl = (l - (n - k) * max_t) / (2 * ((1 << (k // 2)) - 1))
            assert nminl > 0
            # The length of one equal-sized middle part:
            x = (l - 2 * ((1 << (k // 2)) - 1) * nminl) / (n - k)
            assert x - eps <= max_t
            # The last (largest) part before the equal sized middle parts
            last = (1 << (k // 2 - 1)) * nminl
            # The length of the whole tapered part
            vlen = 2 * ((1 << (k // 2)) - 1) * nminl
            if last <= x <= 2 * last and (n - k) * x + vlen + eps >= l:
                break
            # We've already subtracted too much, should not happen
            assert x <= 2 * last
        else:
            assert 0 # pragma: no cover
        assert nminl > maxl
        if nminl > minl:
            minl = nminl
    minc  = lv * (minl / l)
    state = 0 # 0: increase, 1: steady, 2: decrease
    p = p1
    for i in range (n):
        rem  = n - 2 * i
        inc  = (1 << i) * minc
        if state == 0:
            inc1 = (lv - 2 * (p - p1)) / rem
        incdif = np.linalg.norm (inc1) - np.linalg.norm (inc) - eps
        if state == 0 and incdif < 0:
            bound = n - i
            state = 1
        if state == 1 and i >= bound:
            state = 2
        if state == 1:
            inc = inc1
        elif state == 2:
            inc = (1 << (n - i - 1)) * minc
        if i == (n - 1):
            yield (p, p2)
        else:
            yield (p, p + inc)
        p = p + inc
# end def taper2

def taper1 (p1, p2, n, r, min_t = 0, max_t = None, end = 0):
    """ Taper single end, 0=start, 1=end
    >>> p = taper_print
    >>> p (taper1 (0.0, 31.0, 5, 0.001, end = 0))
    (0, 1), (1, 3), (3, 7), (7, 15), (15, 31)
    >>> p (taper1 (0.0, 31.0, 5, 0.001, max_t = 7))
    (0, 3.333), (3.333, 10), (10, 17), (17, 24), (24, 31)
    >>> p (taper1 (0.0, 31.0, 5, 0.001, max_t = 8))
    (0, 2.333), (2.333, 7), (7, 15), (15, 23), (23, 31)
    >>> p (taper1 (0.0, 31.0, 5, 0.8))
    (0, 2), (2, 6), (6, 14), (14, 22.5), (22.5, 31)
    >>> p (taper1 (0.0, 31.0, 5, 0.001, min_t = 2.0))
    (0, 2), (2, 6), (6, 14), (14, 22.5), (22.5, 31)
    >>> p (taper1 (0.0, 31.0, 5, 0.8, end = 1))
    (0, 8.5), (8.5, 17), (17, 25), (25, 29), (29, 31)
    >>> p (taper1 (0.0, 31.0, 5, 0.001, end = 1))
    (0, 16), (16, 24), (24, 28), (28, 30), (30, 31)
    >>> p (taper1 (0, 0.25, 10, 1e-5, 0.008, 0.1, 1), '\\n')
    (0, 0.028)
    (0.028, 0.056)
    (0.056, 0.085)
    (0.085, 0.113)
    (0.113, 0.141)
    (0.141, 0.169)
    (0.169, 0.198)
    (0.198, 0.226)
    (0.226, 0.242)
    (0.242, 0.25)
    >>> p1 = np.ones (3)
    >>> p2 = np.array ([1, 2, 2]) * 31 + p1
    >>> p (taper1 (p1, p2, 5, 0.001), '\\n')
    ((1, 1, 1), (2, 3, 3))
    ((2, 3, 3), (4, 7, 7))
    ((4, 7, 7), (8, 15, 15))
    ((8, 15, 15), (16, 31, 31))
    ((16, 31, 31), (32, 63, 63))
    >>> p (taper1 (p1, p2, 5, 0.001, end = 1), '\\n')
    ((1, 1, 1), (17, 33, 33))
    ((17, 33, 33), (25, 49, 49))
    ((25, 49, 49), (29, 57, 57))
    ((29, 57, 57), (31, 61, 61))
    ((31, 61, 61), (32, 63, 63))

    # Example from Lewallen 'The Other Edge of The Sword'
    # Note that the segmentation scheme he proposes in Fig 4 is invalid
    # because it has a jump of more than factor five in segment length.
    # Also using this segmentation scheme does not reproduce his experiment.
    # So we're using about the same maximum segment length but this will
    # start with the shortest segment about 1/100 (not 1/200) and
    # doubles 4 times not 3.
    >>> p (taper1 (0, .25, 7, 0.001), '\\n')
    (0, 0.003)
    (0.003, 0.007)
    (0.007, 0.018)
    (0.018, 0.038)
    (0.038, 0.078)
    (0.078, 0.158)
    (0.158, 0.25)
    >>> p (taper1 (0, .25, 7, 0.001, end = 2), '\\n')
    (0, 0.092)
    (0.092, 0.172)
    (0.172, 0.212)
    (0.212, 0.232)
    (0.232, 0.242)
    (0.242, 0.247)
    (0.247, 0.25)
    >>> p (taper1 (0, 0.5, 7, 0.001, min_t = 1 / 200), '\\n')
    (0, 0.005)
    (0.005, 0.015)
    (0.015, 0.035)
    (0.035, 0.075)
    (0.075, 0.155)
    (0.155, 0.315)
    (0.315, 0.5)
    >>> max_t = (0.5 - 15 / 100) / 3
    >>> p (taper1 (0, 0.5, 7, 0.001, max_t = max_t), '\\n')
    (0, 0.01)
    (0.01, 0.03)
    (0.03, 0.07)
    (0.07, 0.15)
    (0.15, 0.267)
    (0.267, 0.383)
    (0.383, 0.5)
    >>> p (taper1 (0, 0.5, 7, 0.001, min_t = 1 / 200, max_t = max_t), '\\n')
    (0, 0.01)
    (0.01, 0.03)
    (0.03, 0.07)
    (0.07, 0.15)
    (0.15, 0.267)
    (0.267, 0.383)
    (0.383, 0.5)
    """
    if end:
        tp = list (taper1 (p2, p1, n, r, min_t, max_t, 0))
        for x1, x2 in reversed (tp):
            yield (x2, x1)
        return

    lv = p2 - p1
    l  = np.linalg.norm (p2 - p1)
    uv = lv / l
    min_t = max (2.5 * r, min_t)
    assert n > 1
    if l / n < min_t:
        raise Taper_Error ('Segment size too short: %g < %g' % (l / n, min_t)) \
            # pragma: no cover
    assert max_t is None or min_t <= max_t
    assert max_t is None or l / n <= max_t
    npieces = (1 << n) - 1
    minl = l  / npieces
    if minl < min_t:
        minl = min_t
    eps  = minl / 10
    if max_t is not None:
        maxl = max_t  / (1 << (n - 1))
    if max_t is not None and maxl < minl:
        # m is nminl, x is the length of the constant-length part,
        # ideally max_t
        # ((1 << k) - 1) * m + (n - k) * x = l
        # l - ((1 << k) - 1) * m = (n - k) * x
        # x <= max_t
        # l - ((1 << k) - 1) * m <= (n - k) * max_t
        # l / (n - k) - ((1 << k) - 1) / (n - k) * m <= max_t
        # (l - (n - k) * max_t) / ((1 << k) - 1) <= m
        for k in range (n - 1, 0, -1):
            nminl = (l - (n - k) * max_t) / ((1 << k) - 1)
            x = (l - ((1 << k) - 1) * nminl) / (n - k)
            assert x - eps <= max_t
            last = (1 << (k - 1)) * nminl
            if last <= x <= 2 * last:
                break
        else:
            assert 0 # pragma: no cover
        assert nminl > maxl
        if nminl > minl:
            minl = nminl
    minc  = lv * (minl / l)
    state = 0 # 0: increase, 1: steady
    p = p1
    for i in range (n):
        rem  = n - i
        inc  = (1 << i) * minc
        if state == 0:
            inc1 = (lv - (p - p1)) / rem
        incdif = np.linalg.norm (inc1) - np.linalg.norm (inc) - eps
        if state == 0 and incdif < 0:
            state = 1
        if state == 1:
            inc = inc1
        mt = max_t or l
        if i == (n - 1):
            assert min_t - eps <= np.linalg.norm (p2 - p) <= mt + eps
            yield (p, p2)
        else:
            assert min_t - eps <= np.linalg.norm (p + inc - p) <= mt + eps
            yield (p, p + inc)
        p = p + inc
# end def taper1
