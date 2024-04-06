#!/usr/bin/python3

import sys
import numpy as np
from bisect import bisect_right
from plotly import graph_objects as go
from mininec.mininec import Mininec, Wire, Excitation
from plotly_default import Plotly_Default

class Diff_Plot (Plotly_Default):
    """ This is an example from [1] in the Validation chapter, section
        5.2 starting on page 100. It uses a dipole with half-length
        0.25m and radius a=0.00699m and compares it to the tables in [2]
        for a set of frequencies on that dipole. He uses three different
        segmentation schemes with 10 segments, 20 segments and a
        combination of a tighter segmentation in the vicinity of the
        feed point also resulting in 20 segments overall. We plot --
        like in the dissertation -- the computed values for the three
        segmentation schemes and the tabulated (linear interpolation)
        values from [2],
        [1] Kevin Paul Murray. The Design of Antenna Systems on
            Complex Structures Using Characteristic Modes. Dissertation,
            University of Liverpool, March 1993.
        [2] Ronold W. P. King and Charles W. Harrison. Antennas and
            Waves: A Modern Approach. The MIT Press, Cambridge,
            Massachusetts, 1969.
    """

    def __init__ (self):
        super ().__init__ ()
        self.h0   = 0.25
        self.a    = 0.00699
        self.k0h  = np.round (np.arange (0.5, 2.25, 0.1), 1)
        self.segs = 10
        self.td   = Thin_Cylindrical_Dipole ()
        w = [Wire (10, 0.0, 0.0, 0.0, 2 * self.h0, 0.0, 0.0, self.a)]
        self.s10 = Excitation (1.0, 0.0)
        m = self.m10 = Mininec (299.8, w)
        m.register_source (self.s10, int (10 / 2) - 1)
        w = [Wire (20, 0.0, 0.0, 0.0, 2 * self.h0, 0.0, 0.0, self.a)]
        m = self.m20 = Mininec (299.8, w)
        self.s20 = Excitation (1.0, 0.0)
        m.register_source (self.s20, int (20 / 2) - 1)
        l1 = 4 * self.h0 * .175
        l2 = 4 * self.h0 * .15
        w1 = Wire (5,  0.0,     0.0, 0.0, l1,          0.0, 0.0, self.a)
        w2 = Wire (10, l1,      0.0, 0.0, l1 + l2,     0.0, 0.0, self.a)
        w3 = Wire (5,  l1 + l2, 0.0, 0.0, 2 * l1 + l2, 0.0, 0.0, self.a)
        w = [w1, w2, w3]
        m = self.m202 = Mininec (299.8, w)
        self.s202 = Excitation (1.0, 0.0)
        m.register_source (self.s202, int (20 / 2) - 1)
    # end def __init__

    def compute (self):
        hl  = self.k0h / (2 * np.pi) # h / lambda
        l   = 1 / hl * self.h0       # lambda
        c   = 299.8
        frq = self.frq = c / l
        al  = self.a / l
        imp10  = []
        imp20  = []
        imp202 = []
        for f in frq:
            imp10.append  (self.get_impedance (self.m10, f))
            imp20.append  (self.get_impedance (self.m20, f))
            imp202.append (self.get_impedance (self.m202, f))
        imp10  = self.imp10  = np.array (imp10)
        imp20  = self.imp20  = np.array (imp20)
        imp202 = self.imp202 = np.array (imp202)
        self.king = np.array \
            ([self.td.get (a, b) for a, b in zip (al, self.k0h)])
    # end def compute

    def plot (self):
        layout = self.plotly_line_default
        y   = layout ['layout']['yaxis']
        y  ['title'].update (text = 'R (ohm)')
        y2  = layout ['layout']['yaxis2']
        y2 ['title'].update (text = 'X (ohm)')
        fig = go.Figure ()
        d   = dict (x = self.k0h, yaxis = 'y')
        d.update (name = 'R (ohm) [King]', y = self.king.real)
        fig.add_trace (go.Scatter (**d))
        d.update (name = 'R (ohm) [10 segs]', y = self.imp10.real)
        fig.add_trace (go.Scatter (**d))
        d.update (name = 'R (ohm) [20 segs]', y = self.imp20.real)
        fig.add_trace (go.Scatter (**d))
        d.update (name = 'R (ohm) [special]', y = self.imp202.real)
        fig.add_trace (go.Scatter (**d))
        d   = dict (x = self.k0h, yaxis = 'y2')
        d.update (name = 'X (ohm) [King]', y = self.king.imag)
        fig.add_trace (go.Scatter (**d))
        d.update (name = 'X (ohm) [10 segs]', y = self.imp10.imag)
        fig.add_trace (go.Scatter (**d))
        d.update (name = 'X (ohm) [20 segs]', y = self.imp20.imag)
        fig.add_trace (go.Scatter (**d))
        d.update (name = 'X (ohm) [special]', y = self.imp202.imag)
        fig.add_trace (go.Scatter (**d))
        fig.update (layout)
        fig.show ()
    # end def plot

    def get_impedance (self, m, f):
        m.f = f
        m.compute ()
        return m.sources [0].impedance
    # end def get_impedance

# end class Diff_Plot

#k0h:
# [0.5 0.6 0.7 0.8 0.9 1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2]
#Lambda:
# [3.14159265 2.61799388 2.24399475 1.96349541 1.74532925 1.57079633
# 1.42799666 1.30899694 1.20830487 1.12199738 1.04719755 0.9817477
# 0.92399784 0.87266463 0.82673491 0.78539816 0.74799825 0.71399833]
#Frequency:
# [ 95.42930388 114.51516465 133.60102543 152.6868862  171.77274698
# 190.85860776 209.94446853 229.03032931 248.11619008 267.20205086
# 286.28791163 305.37377241 324.45963318 343.54549396 362.63135474
# 381.71721551 400.80307629 419.88893706]
#[0.00222499 0.00266998 0.00311498 0.00355998 0.00400497 0.00444997
# 0.00489497 0.00533997 0.00578496 0.00622996 0.00667496 0.00711996
# 0.00756495 0.00800995 0.00845495 0.00889994 0.00934494 0.00978994]

class Thin_Cylindrical_Dipole:
    """ In 'Table of Antenna Characteristics [1] (starting p.39) and
        'Antennas and Waves [2] (starting p. 730) we have tabulated
        impedance Z by a/lambda (al) and k0h. We make a list by al with
        dictionaries with entries for k0h. That way we can do a linear
        interpolation by al values.
        [1] Ronold W. P. King. Tables of Antenna Characteristics.
            IFI/Plenum Data Corporation, New York, Washington, London,
            1971.
        [2] Ronold W. P. King and Charles W. Harrison. Antennas and
            Waves: A Modern Approach. The MIT Press, Cambridge,
            Massachusetts, 1969.
    >>> td = Thin_Cylindrical_Dipole ()
    >>> def p (c):
    ...     s = '+-' [int (np.sign (c.imag))]
    ...     r = s.join ('{:#.4g}'.format (abs (x))
    ...                 for x in (c.real, c.imag)) + 'j'
    ...     print (r)
    >>> p (td.get (0.001588, 0.5))
    4.838-619.1j
    >>> p (td.get ((0.001588 + 0.003175)/2, 0.5))
    4.684-540.9j
    >>> p (td.get (0.00978994, 2.2))
    464.5-37.62j
    """
    Z_by_al_k0h = \
    [
    # 0.5:0.00222499 0.6:0.00266998 0.7:0.00311498
      (0.001588, dict
	(( (0.5, 4.838-619.1j)
	,  (0.6, 7.167-523.1j)
	,  (0.7, 10.07-443.4j)
	))
       )
    # 0.8:0.00355998 0.9:0.00400497 1.0:0.00444997
    ,  (0.003175, dict
	(( (0.5, 4.530-462.7j)
	,  (0.6, 6.774-397.5j)
	,  (0.7, 9.600-341.5j)
	,  (0.8, 13.10-291.4j)
	,  (0.9, 17.41-245.3j)
	,  (1.0, 22.67-201.9j)
	))
       )
    # 1.1:0.00489497 1.2:0.00533997 1.3:0.00578496 1.4:0.00622996
    ,  (0.004763, dict
	(( (0.8, 12.52-242.6j)
	,  (0.9, 16.77-205.7j)
	,  (1.0, 22.02-170.2j)
	,  (1.1, 28.50-135.3j)
	,  (1.2, 36.51-100.4j)
	,  (1.3, 46.45-65.11j)
	,  (1.4, 58.83-28.83j)
	))
      )
    # 1.5:0.00667496
    , (0.006350, dict
       (( (1.1,  27.49-117.4j)
        , (1.2,  35.64-87.47j)
        , (1.3,  45.89-56.73j)
        , (1.4,  58.86-24.77j)
        , (1.5,  75.35+8.675j)
        , (1.6,  96.47+43.82j)
        , (1.7, 124.3+80.66j)
        , (1.8, 161.6+118.5j)
        , (1.9, 212.0+155.3j)
        , (2.0, 280.4+186.7j)
        , (2.1, 371.0+203.2j)
        , (2.2, 483.7+188.3j)
       ))
      )
    # 1.6:0.00711996 1.7:0.00756495 1.8:0.00800995 1.9:0.00845495
    # 2.0:0.00889994 2.1:0.00934494
    # 2.2:0.00978994
    , (0.007022, dict
       (( (1.5,  75.75+8.289j)
        , (1.6,  97.59+31.83j)
        , (1.7, 126.5+76.73j)
        , (1.8, 165.5+112.1j)
        , (1.9, 218.4+145.2j)
        , (2.0, 289.5+170.3j)
        , (2.1, 381.8+176.4j)
        , (2.2, 490.8+145.6j)
       ))
      )
    # 2.2:0.00978994
    , (0.009525, dict
        (( (1.6, 101.7+33.15j)
         , (1.7, 135.1+60.65j)
         , (1.8, 180.4+85.20j)
         , (1.9, 240.7+100.9j)
         , (2.0, 316.4+97.16j)
         , (2.1, 399.3+59.75j)
         , (2.2, 467.0-20.08j)
        ))
      )
    ]

    def get (self, alambda, k0h):
        """ Get value for a/lambda, k0h """
        v   = self.Z_by_al_k0h
        pos = bisect_right (v, alambda, 0, len (v), key = self.key)
        if pos <= 0:
            raise ValueError ('Too small: %s' % alambda)
        if pos >= len (v) and alambda > 0.0099:
            raise ValueError ('Too large: %s' % alambda)
        # We extrapolate in that case
        if alambda > 0.009525:
            pos = len (v) - 1
        #print (v [pos-1][0], alambda, v [pos][0])
        return self.interpolate \
            ( alambda, v [pos - 1][0], v [pos][0]
            , v [pos - 1][1][k0h], v [pos][1][k0h]
            )
    # end def get

    def interpolate (self, x, x1, x2, y1, y2):
        dx = x2 - x1
        dy = y2 - y1
        return (x - x1) / dx * dy + y1
    # end def interpolate

    @staticmethod
    def key (k):
        if isinstance (k, tuple):
            return k [0]
        return k
    # end def key

# end class Thin_Cylindrical_Dipole

def main (argv):
    dp = Diff_Plot ()
    dp.compute ()
    dp.plot ()

if __name__ == '__main__':
    main (sys.argv [1:])
