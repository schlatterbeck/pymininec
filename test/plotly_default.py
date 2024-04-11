#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.colors as mcolors
from argparse import ArgumentParser
from plotly import graph_objects as go
from mininec.mininec import Mininec, Wire, Excitation

class Plotly_Default:

    def __init__ (self):
        self.colormap = []
        for cn in mcolors.TABLEAU_COLORS:
            self.colormap.append (mcolors.TABLEAU_COLORS [cn])
        self.linecolor = self.colormap [0]
        self.c_real = '#AE4141'
        self.c_imag = '#FFB329'
    # end def __init__

    @property
    def plotly_line_default (self):
        d = dict \
            ( layout = dict
                ( showlegend = True
                , colorway   = self.colormap
                , xaxis = dict
                    ( linecolor   = "#B0B0B0"
                    , gridcolor   = "#B0B0B0"
                    , domain      = [0, 0.9]
                    #, ticksuffix  = ' MHz'
                    , tickformat  = '.3f'
                    , zeroline    = False
                    )
                , yaxis = dict
                    ( color       = self.linecolor
                    , linecolor   = self.linecolor
                    , showgrid    = True
                    #, gridcolor   = blend (self.linecolor)
                    , title       = {}
                    , anchor      = "x"
                    , side        = "left"
                    , hoverformat = '.3f'
                    , zeroline    = False
                    )
                , yaxis2 = dict
                    ( color       = self.c_real
                    , linecolor   = self.c_real
                    , showgrid    = False
                    #, gridcolor   = blend (self.c_real)
                    , title       = {}
                    , overlaying  = "y"
                    , side        = "right"
                    , anchor      = "x2"
                    , hoverformat = '.3f'
                    , zeroline    = False
                    )
                , yaxis3 = dict
                    ( color       = self.c_imag
                    , linecolor   = self.c_imag
                    , showgrid    = False
                    #, gridcolor   = blend (self.c_imag)
                    , title       = {}
                    , overlaying  = "y"
                    , side        = "right"
                    , position    = 0.96
                    , anchor      = "free"
                    , hoverformat = '.1f'
                    , zeroline    = False
                    )
                , paper_bgcolor = 'white'
                , plot_bgcolor  = 'white'
                , hovermode     = 'x unified'
                )
            )
        return d
    # end def plotly_line_default

# end class Plotly_Default
