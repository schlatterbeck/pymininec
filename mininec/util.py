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

import numpy as np

def format_float (floats, use_e = 0):
    """ Reproduce floating-point formatting of the Basic code
    Test special case with large number, should usually set 'use_e':
    >>> print ('%s' % format_float ((3e7,)))
     30000000
    """
    r = []
    for f in floats:
        if f == 0:
            fmt = '% .1f'
        else:
            prec = 6 - int (np.log (abs (f)) / np.log (10))
            if prec < 0:
                prec = 0
            fmt = '%% .%df' % prec
        if use_e and abs (f) < 1e-1:
            fmt = '% e'
            if abs (f) == 0:
                fmt = '% .0f'
        s = fmt % f
        if fmt != '% e':
            if '.' in s:
                s = s [:9]
                s = s.rstrip ('0')
                s = s.rstrip ('.')
                if s.startswith (' 0.') or s.startswith ('-0.'):
                    s = s [0] + s [2:]
                s = '%-9s' % s
        else:
            s = s.upper ()
        if s.strip () == '-0':
            s = ' ' + s [1:]
        r.append (s)
    return tuple (r)
# end def format_float
