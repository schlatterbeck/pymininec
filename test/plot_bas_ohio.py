#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

X   = [[], []]
Y   = [[[], []], [[], []]]
idx = 0
flg = 0
with open ('test/ohio.bout', 'r') as f:
    for line in f:
        line = line.strip ()
        if not line:
            continue
        if flg == 0:
            if line.startswith ('FIELD POINT:'):
                x = float (line.split () [4])
                if idx == 0 and x > 3000:
                    idx += 1
                X [idx].append (x)
            if line.startswith ('MAXIMUM OR PEAK FIELD'):
                v = float (line.split () [5])
                Y [0][idx].append (v)
            if 'PATTERN DATA' in line:
                flg = 1
                idx = 0
        else:
            if line.startswith ('RADIAL DISTANCE'):
                d = float (line.split () [3])
                if idx == 0 and d > 3000:
                    idx += 1
            if line [0].isdigit ():
                v = float (line.split () [2])
                Y [1][idx].append (v)
Y = np.array (Y)
for i in (0, 1):
    fig = plt.figure ()
    ax  = plt.subplot (111)
    ax.plot (X [i], Y [0][i])
    ax.plot (X [i], Y [1][i])
    plt.show ()
    fig = plt.figure ()
    ax  = plt.subplot (111)
    ax.plot (X [i], (Y [0][i] - Y [1][i]) / Y [1][i] * 100)
    plt.show ()

