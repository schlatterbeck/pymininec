#!/usr/bin/python3

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
