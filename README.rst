MININEC in Python
=================

:Author: Ralf Schlatterbeck <rsc@runtux.com>

.. |--| unicode:: U+2013   .. en dash
.. |__| unicode:: U+2013   .. en dash without spaces
    :trim:
.. |_| unicode:: U+00A0 .. Non-breaking space
    :trim:
.. |-| unicode:: U+202F .. Thin non-breaking space
    :trim:
.. |numpy.linalg.solve| replace:: ``numpy.linalg.solve``

This is an attempt to rewrite the original MININEC3 basic sources in
Python. Standard use-case like computation of feed impedance and far
field are implemented and are quite well tested. There is only a command
line interface.

An example of using the command-line interface uses the following
12-element Yagi/Uda antenna originally presented by Cebik [6]_ without
the resistive element loading and with slight corrections to the element
lengths to make the antenna symmetric (the errors in the display in
Cebik's article may be due to a display issue of the program used that
displays negative numbers with less accuracy than positive numbers).
You can get further help by calling ``pymininec`` with the ``--help``
option.  The command-line options for the 12-element antenna example
(also available in the file ``test/12-el.pym``) are::

    pymininec -f 148 \
    -w 22,-.51943,0.00000,0,.51943,0.00000,0,.00238 \
    -w 22,-.50165,0.22331,0,.50165,0.22331,0,.00238 \
    -w 22,-.46991,0.34215,0,.46991,0.34215,0,.00238 \
    -w 22,-.46136,0.64461,0,.46136,0.64461,0,.00238 \
    -w 22,-.46224,1.03434,0,.46224,1.03434,0,.00238 \
    -w 22,-.45989,1.55909,0,.45989,1.55909,0,.00238 \
    -w 22,-.44704,2.19682,0,.44704,2.19682,0,.00238 \
    -w 22,-.43561,2.94640,0,.43561,2.94640,0,.00238 \
    -w 22,-.42672,3.72364,0,.42672,3.72364,0,.00238 \
    -w 22,-.41783,4.53136,0,.41783,4.53136,0,.00238 \
    -w 22,-.40894,5.33400,0,.40894,5.33400,0,.00238 \
    -w 22,-.39624,6.0452,0,.39624,6.0452,0,.00238 \
    --theta=0,5,37 --phi=0,5,73 \
    --excitation-segment=33 > 12-el.pout

Users on Linux can run this using (I'm sure Windows users can come up
with a similar command-line on Windows)::

    pymininec $(cat test/12-el.pym) > 12-el.pout

The resulting output file contains currents, impedance at the feedpoint
and antenna far field in dBi as tables. The output tries to reproduce
the format of the original Basic implementation of Mininec. Now these
tables are not very useful to get an idea of the far field behaviour of
an antenna. Therefore there is a small companion program ``plot-antenna``
that can plot the antenna pattern. The default is to plot all available
graphics, including an interactive 3d view. In addition with the
``--azimuth`` or ``--elevation`` options you can get an Azimuth
diagram::

    plot-antenna --azimuth test/12-el-1deg.pout

.. figure:: https://raw.githubusercontent.com/schlatterbeck/pymininec/master/test/12-el-azimuth.png
    :align: center

or an elevation diagram::

    plot-antenna --elevation test/12-el-1deg.pout

.. figure:: https://raw.githubusercontent.com/schlatterbeck/pymininec/master/test/12-el-elevation.png
    :align: center

respectively. Note that I used an output file with 1-degree resolution
in elevation and azimuth angles not with 5 degrees as in the example
above. The pattern look smoother but a 3D-view will be very slow due to
the large number of points. The plot program also has a ``--help``
option for further information. In particular the scaling of the antenna
plot can be ``linear``, ``linear_db``, and ``linear_voltage`` in
addition to the default of ``arrl`` scaling. You may consult Cebik's [6]_
article for explanation of the different diagrams.

The latest version accepts several plot parameters, ``--elevation``,
``--azimuth``, ``--plot3d``, ``--plot-vswr`` which are plotted into one
diagram. The default is to plot all four graphs. With the ``--output``
option pictures can directly be saved without displaying the graphics on
the screen.

Test coverage: Making sure it is consistent with original Mininec
-----------------------------------------------------------------

There are several tests against the `original Basic source code`_, for
the test cases see the subdirectory ``test``. One of the test cases is
a simple 7MHz wire dipole with half the wavelength and 10 segments.
In one case the wire is 0.01m (1cm) thick, we use such a thick wire to
make the mininec code work harder because it cannot use the thin wire
assumptions. Another test is for the thin wire case. Also added are the
inverted-L and the T antenna from the original Mininec reports. All
these may also serve as examples.  Tests statement coverage is currently
at 100%.

There is a line that is flagged as not covered by the ``pytest``
framework if the Python version is below 3.10. This is a ``continue``
statement in ``compute_impedance_matrix`` near the end (as of this
writing line 1388). This is a bug in Python in versions below 3.10:
When setting a breakpoint in the python debugger on the continue
statement, the breakpoint is never reached although the continue
statement is correctly executed. A workaround would be to put a dummy
assignment before the continue statement and verify the test coverage
now reports the continue statement as covered.
I've `reported this as a bug in the pytest project`_ and `as a bug in
python`_, the bugs are closed now because Python3.9 does no longer get
maintenance.

For all the test examples it was carefully verified that the results are
close to the original results in Basic (see `Running examples in Basic`_
to see how you can run the original Basic code in the 21th century). The
differences are due to rounding errors in the single precision
implementation in Basic compared to a double precision implementation in
Python. I'm using numeric code from `numpy`_ where possible to speed up
computation, e.g. solving the impedance matrix is done using
|numpy.linalg.solve|_ instead of a line-by-line translation from Basic.
You can verify the differences yourself. In the ``test`` directory there
are input files with extension ``.mini`` which are intended (after
conversion to carriage-return convention) to be used as input to the
original Basic code. The output of the Basic code is in files with the
extension ``.bout`` while the output of the Python code is in files
with the extension ``.pout``. The ``.pout`` files are compared in the
regression tests. The ``.pym`` files in the ``test`` directory are the
command-line arguments to recreate the ``.pout`` files with
``mininec.py``.

In his thesis [5]_, Zeineddin investigates numerical instabilities when
comparing near and far field. He solves this by doing certain
computations for the near field in double precision arithmetics.
I've tried to replicate these experiments and the numerical
instabilities are reproduceable in the Basic version. In the Python
version the instabilities are not present (because everything is in
double precision). But the absolute field values computed in Python are
lower than the ones reported by Zeineddin (and the Basic code *does*
reproduce Zeineddins values).

It doesn't look like there is a problem in the computations of the
currents in the Python code, the computed currents are lower than in
Basic which leads to lower field values. But the computed impedance
matrix when comparing both versions has very low error, see the test
``test_matrix_fill_ohio_example`` in ``test/test_mininec.py`` and the
routine ``plot_z_errors`` to plot the errors (in percent) in
``test/ohio.py``. Compared to the values computed by NEC [5]_, the Basic
code produces slightly higher values for near and far field while the
Python code produces slightly lower values than NEC. I've not tried to
simulate this myself in NEC yet.

You can find the files in
``test/ohio*`` (the thesis was at Ohio University). This time there is a
python script ``ohio.py`` to compute the near and far field values
without recomputing the impedance matrix. This script can show the near
and far field values in a plot and the difference in a second plot.
There are two distances for which these are computed, so the code
produces four plots. There is a second script to plot the Basic near and
far field differences ``plot_bas_ohio.py``.

The current Python code is still hard to understand |--| it's the
result of a line-by-line translation from Basic, especially where I
didn't (yet) understand the intention of the code. The same holds for
Variable names which might not (yet) reflect the intention of the code.
I *did* move things like computation of the angle of a complex number,
or the computation of the absolute value, or multiplication/division of
complex numbers to the corresponding complex arithmetic in python where
I detected the pattern.

So the *de-spaghettification* was not successful in some parts of the
code yet :-) My notes from the reverse-engineering can be found in the
file ``basic-notes.txt`` which has explanations of some of the variables
used in mininec and some sub routines with descriptions (mostly taken
from ``REM`` statements) of the Basic code.

The code is also still quite slow: An example of a 12 element Yagi/Uda
antenna used in modeling examples by Cebik [6]_ takes about 50 seconds
on my PC (this has 264 segments, more than the original Mininec ever
supported) when I'm using 5 degree increments for theta and phi angles
and about 11 minutes (!) for 1 degree angles. The reason is that
everything currently is implemented (like in Basic) as nested loops.
This could (and should) be changed to use vector and matrix operations
in `numpy`_. In the inner loop of the matrix fill operation there are
several integrals computed using `gaussian quadrature`_ or a numeric
solution to an `elliptic integral`_. These could be implemented by
scipy_ library functions.

Notes on Elliptic Integral Parameters
-------------------------------------

The Mininec code uses the implementation of an elliptic integral when
computing the impedance matrix and in several other places. The integral
uses a set of E-vector coefficients that are cited differently in
different places. In the latest version of the open source Basic code
these parameters are in lines 1510 |__| 1512. They are also
reprinted in the publication [2]_ about that version of Mininec which
has a listing of the Basic source code (slightly different from the
version available online) where it is on p. |-| C-31 in lines
1512 |__| 1514.

+---------------+--------------+--------------+--------------+--------------+
| 1.38629436112 | .09666344259 | .03590092383 | .03742563713 | .01451196212 |
+---------------+--------------+--------------+--------------+--------------+
|            .5 | .12498593397 | .06880248576 | .0332835346  | .00441787012 |
+---------------+--------------+--------------+--------------+--------------+

In one of the first publications on Mininec [1]_ the authors give the
parameters on p. |-| 13 as:

+---------------+--------------+--------------+--------------+--------------+
| 1.38629436112 | .09666344259 | .03590092383 | .03742563713 | .01451196212 |
+---------------+--------------+--------------+--------------+--------------+
|            .5 | .1249859397  | .06880248576 | .03328355346 | .00441787012 |
+---------------+--------------+--------------+--------------+--------------+

This is consistent with the later Mininec paper [2]_ on version |-| 3 of
the Mininec code on p. |-| 9, but large portions of that paper are copy
& paste from the earlier paper.

The first paper [1]_ has a listing of the Basic code of that version and
on p.  |-| 48 the parameters are given as:

+---------------+--------------+--------------+--------------+--------------+
| 1.38629436    | .09666344    | .03590092    | .03742563713 | .01451196    |
+---------------+--------------+--------------+--------------+--------------+
|            .5 | .12498594    | .06880249    | .0332836     | .0041787     |
+---------------+--------------+--------------+--------------+--------------+

In each case the first line are the *a* parameters, the second line are
the *b* parameters. The *a* parameters are consistent in all versions
but notice how in the *b* parameters (2nd line) the current Basic code
has one more *3* in the second column. The rounding of the earlier Basic
code suggests that the second *3* is a typo in the later Basic version.
Also notice that in the 4th column the later Basic code has a *5* less
than the version in the papers. The rounding in the earlier Basic code
also suggests that the later Basic code is in error.

The errors in the elliptic integral parameters do not have much effect
on the computed values of the Mininec code. There are some minor
differences but these are below the differences between Basic and Python
implementation (single vs. double precision arithmetics). I had hoped
that this has something to do with the well known fact that Mininec
finds a resonance point of an antenna some percent too high which means
that usually in practice the computed wire lengths are a little too
long. This is apparently not the case. The resonance point is also wrong
for very thin wires below the *small radius modification condition*
which happens when the wire radius is below 1e-4 of the wavelength.
Even in that case --  where the elliptic integral is not used -- the
resonance is slightly wrong.

The reference for the elliptic integral parameters [3]_ cited in both
reports lists the following table on p. |-| 591:

+---------------+--------------+--------------+--------------+--------------+
| 1.38629436112 | .09666344259 | .03590092383 | .03742563713 | .01451196212 |
+---------------+--------------+--------------+--------------+--------------+
|            .5 | .12498593597 | .06880248576 | .03328355346 | .00441787012 |
+---------------+--------------+--------------+--------------+--------------+

Note that I could only locate the 1972 version of the Handbook, not the
1980 version cited by the reports. So there is a small chance that these
parameters were corrected in a later version. It turns out that the
reports are correct in the fourth column and the Basic program is wrong.
But the second column contains still *another* version, note that there
is a *5* in the 9th position after the comma, not a *3* like in the
Basic program and not a missing digit like in the Mininec reports [1]_
[2]_.

Since I could not be sure that there was a typo in the handbook [3]_, I
dug deeper: The handbook cites *Approximations for Digital Computers* by
Hastings (without giving a year) [4]_. The version of that book I found
is from 1955 and lists the coefficients on p. |-| 172:

+---------------+--------------+--------------+--------------+--------------+
| 1.38629436112 | .09666344259 | .03590092383 | .03742563713 | .01451196212 |
+---------------+--------------+--------------+--------------+--------------+
|            .5 | .12498593597 | .06880248576 | .03328355346 | .00441787012 |
+---------------+--------------+--------------+--------------+--------------+

So apparently the handbook [3]_ is correct. And the Basic version and
*both* Mininec reports have at least one typo.

Running examples in Basic
-------------------------

The original Basic source code can still be run today, thanks to Rob
Hagemans `pcbasic`_ project. It is written in Python and can be
installed with pip. It is also packaged in some Linux distributions,
e.g. in Debian_.

Since Mininec reads all inputs for an antenna simulation from the
command-line in Basic, I'm creating input files that contain
reproduceable command-line input for an antenna simulation. An example
of such a script is in ``dipole-01.mini``, the suffix ``mini``
indicating a Mininec file.

Of course the input files only make sense if you actually run them with
the mininec basic code as this displays all the prompts.
Note that I had to change the dimensions of some arrays in the Basic
code to not run into an out-of-memory condition with the Basic
interpreter.

You can run `pcbasic`_ with the command-line option ``--input=`` to specify
an input file. Note that the input file has to be converted to carriage
return line endings (no newlines). I've described how I'm debugging the
Basic code using the Python debugger in a `contribution to pcbasic`_,
this has been moved to the `pcbasic wiki`_.

In the file ``debug-basic.txt`` you can find my notes on how to debug
mininec using the python debugger. This is more or less a random
cut&paste buffer.

The `original basic source code`_ can be obtained from the `unofficial
NEC archive`_ by PA3KJ or from a `Mininec github project`_, I'm using
the version from the `unofficial NEC archive`_ and have not verified if
the two links I've given contain the same code.

Testing math:

.. math::
    \frac{1}{n^{2}}

Block math $$ \frac{1}{n^{2}} $$ in the text

.. _`original basic source code`: http://nec-archives.pa3kj.com/mininec3.zip
.. _`unofficial NEC archive`: http://nec-archives.pa3kj.com/
.. _`Mininec github project`: https://github.com/Kees-PA3KJ/MiniNec
.. _`numpy`: https://numpy.org/
.. _`pcbasic`: https://github.com/robhagemans/pcbasic
.. _`Debian`: https://packages.debian.org/stable/python3-pcbasic
.. _`contribution to pcbasic`: https://github.com/robhagemans/pcbasic/pull/183
.. _`pcbasic wiki`:
    https://github.com/robhagemans/pcbasic/wiki/Debugging-Basic-with-the-Python-Debugger

.. [1] Alfredo J. Julian, James C. Logan, and John W. Rockway.
    Mininec: A mini-numerical electromagnetics code. Technical Report
    NOSC TD 516, Naval Ocean Systems Center (NOSC), San Diego,
    California, September 1982. Available as ADA121535_ from the Defense
    Technical Information Center.
.. [2] J. C. Logan and J. W. Rockway. The new MININEC (version |-| 3): A
    mini-numerical electromagnetic code. Technical Report NOSC TD 938,
    Naval Ocean Systems Center (NOSC), San Diego, California, September
    1986. Available as ADA181682_ from the Defense Technical Information
    Center. Note: The scan of that report is *very* bad. If you have
    access to a better version, please make it available!
.. [3] Milton Abramowitz and Irene A. Stegun, editors. Handbook of
    Mathematical Functions With Formulas, Graphs, and Mathematical
    Tables.  Number 55 in Applied Mathematics Series.  National Bureau
    of Standards, 1972.
.. [4] Cecil Hastings, Jr. Approximations for Digital Computers.
    Princeton University Press, 1955.
.. [5] Rafik Paul Zeineddin. Numerical electromagnetics codes: Problems,
    solutions and applications. Master’s thesis, Ohio University, March 1993.
    Available from the `OhioLINK Electronic Theses & Dissertations Center`_
.. [6] L. B. Cebik. Radiation plots: Polar or rectangular; log or linear.
    In Antenna Modeling Notes [7], chapter 48, pages 366–379. Available
    in Cebik's `Antenna modelling notes episode 48`_
.. [7] L. B. Cebik. Antenna Modeling Notes, volume 2. antenneX Online
    Magazine, 2003. Available with antenna models from the `Cebik
    collection`_.

.. _ADA121535: https://apps.dtic.mil/sti/pdfs/ADA121535.pdf
.. _ADA181682: https://apps.dtic.mil/sti/pdfs/ADA181682.pdf
.. _`numpy.linalg.solve`:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
.. _`OhioLINK Electronic Theses & Dissertations Center`:
    https://etd.ohiolink.edu/apexprod/rws_etd/send_file/send?accession=ohiou1176315682
.. _`reported this as a bug in the pytest project`:
    https://github.com/pytest-dev/pytest/issues/10152
.. _`as a bug in python`:
    https://github.com/python/cpython/issues/94974
.. _`Cebik collection`:
    http://on5au.be/Books/allmodnotes.zip
.. _`Antenna modelling notes episode 48`:
    http://on5au.be/content/amod/amod48.html
.. _`gaussian quadrature`: https://en.wikipedia.org/wiki/Gaussian_quadrature
.. _`elliptic integral`: https://en.wikipedia.org/wiki/Elliptic_integral
.. _`scipy`: https://scipy.org/
