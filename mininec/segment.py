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

class Segment:

    def __init__ (self, p1, p2, geobj, idx):
        diff         = p2 - p1
        self.p1      = p1
        self.p2      = p2
        self.geobj   = geobj
        self.idx     = idx
        self.seg_len = np.linalg.norm (diff)
        self.dirvec  = diff / self.seg_len
        # This is Integral I1 in the paper(s), in the implementation
        # used in Basic this boils down to only wire-dependent
        # Parameters. So we can avoid re-computing this for each
        # combination of segments.
        # My original note was:
        # Hmm, dividing by f2 here removes scale = (p3-p2) from logarithm
        # So i6 depends only on wire/seg parameters?!
        # Moved to wire. Original formula was:
        # i6 = (1 - np.log (s4 / f2 / 8 / wire.r)) / np.pi / wire.r
        # But s4 / f2 reduces to wire.seg_len / 2
        # See above, i6 is now passed as a parameter which is set to
        # 0 if not using the exact kernel.
        self.i6 = (1 + np.log (16 * geobj.r / self.seg_len)) / np.pi / geobj.r
    # end def __init__

    def __len__ (self):
        return self.seg_len
    # end def __len__

# end class Segment
