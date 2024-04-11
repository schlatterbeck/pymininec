#!/usr/bin/python3

import sys
import numpy as np
from argparse import ArgumentParser
from plotly import graph_objects as go
from mininec.mininec import Mininec, Wire, Excitation
from plotly_default import Plotly_Default

class Convergence_Plot (Plotly_Default):
    """ This is the dipole example from [1] in the Validation section
        starting on page 21. We try to reproduce the three Graphics in
        Figures 4-6. Note that the Figure 6 has a different radius a
        *and* a different range of segments!
        To reproduce it:
        - Figure 4: python3 ada121535.py
        - Figure 5: python3 ada121535.py -H 0.25
        - Figure 6: python3 ada121535.py -H .38197 -a .00636 \
          --min-segs=26 --max-segs=53
        Also see that the recomputation of Figure 5 has lower B (so the
        dipole is seen to be more capacitive). This *may* be an
        indication that an error has been introduced for MININEC 3 that
        makes outputs more capacitive. But that effect does not occur
        for the other examples.
        [1] Alfredo J. Julian, James C. Logan, and John W. Rockway.
            Mininec: A mini-numerical electromagnetics code. Technical
            Report NOSC TD 516, Naval Ocean Systems Center (NOSC),
            San Diego, California, September 1982.
    """

    #      h          range G     range B
    ranges = dict \
        (( (0.159155, ((0.3, 0.4),     (3.65, 3.9)))
         , (0.25,     ((9.0, 11.5),    (-6.5, -4)))
         #, (0.38197,  ((1.25, 1.55),   (-0.3, 0.3)))
        ))

    def __init__ (self, args):
        super ().__init__ ()
        self.h = args.halfwave
        self.a = args.wire_radius
        self.min_segs = args.min_segs or  4
        self.max_segs = args.max_segs or 30
        if not args.max_segs and self.h > .36:
            self.max_segs = 72
        if not args.min_segs and self.h > .36:
            self.min_segs = 26
        if self.min_segs & 1:
            self.min_segs += 1
        self.segnos = np.arange (self.min_segs, self.max_segs, 2, dtype = int)
        self.compute ()
    # end def __init__

    def compute (self):
        imp    = []
        for segno in (self.segnos):
            imp.append (self.get_impedance (segno))
        self.imp = np.array (imp)
        self.inv_seg = 1.0 / self.segnos
        self.g_m_mho = 1000.0 / self.imp
    # end def compute

    def get_impedance (self, segno):
        segs = int (segno)
        assert not segs & 1
        w = [Wire (segs, 0.0, 0.0, 0.0, 2 * self.h, 0.0, 0.0, self.a)]
        s = Excitation (1.0, 0.0)
        m = Mininec (299.8, w)
        m.register_source (s, int (segs / 2) - 1)
        m.compute ()
        return s.impedance
    # end def get_impedance

    def plot (self):
        layout = self.plotly_line_default
        y   = layout ['layout']['yaxis']
        y  ['title'].update (text = 'G (mmho)')
        y2  = layout ['layout']['yaxis2']
        y2 ['title'].update (text = 'B (mmho)')
        r = self.ranges.get (self.h)
        if r:
            y.update  (range = r [0])
            y2.update (range = r [1])
        fig = go.Figure ()
        d   = dict (x = self.inv_seg)
        d.update (name = 'G (mmho)', y = self.g_m_mho.real, yaxis = 'y')
        fig.add_trace (go.Scatter (**d))
        d.update (name = 'B (mmho)', y = self.g_m_mho.imag, yaxis = 'y2')
        fig.add_trace (go.Scatter (**d))
        fig.update (layout)
        fig.show ()
    # end def plot

# end class Convergence_Plot

def main (argv = sys.argv [1:]):
    cmd = ArgumentParser ()
    cmd.add_argument \
        ( '-H', '--halfwave'
        , type = float
        , help = 'Half length of wire'
        , default = 0.159155
        )
    cmd.add_argument \
        ( '-a', '--wire-radius'
        , type = float
        , help = 'Wire radius a'
        , default = 0.001588
        )
    cmd.add_argument \
        ( '--min-segs'
        , type = int
        , help = 'Minimum number of segments'
        )
    cmd.add_argument \
        ( '--max-segs'
        , type = int
        , help = 'Minimum number of segments'
        )
    args = cmd.parse_args (argv)
    p = Convergence_Plot (args)
    p.plot ()
# end def main

if __name__ == '__main__':
    main (sys.argv [1:])
