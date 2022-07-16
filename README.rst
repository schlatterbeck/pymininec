MININEC in Python
=================

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

There are several tests against the `original Basic source code`_, for
the test cases see the subdirectory ``test``. One of the test cases is
a simple 7MHz wire dipole with half the wavelength and 10 segments.
In one case the wire is 0.01m (1cm) thick, we use such a thick wire to
make the mininec code work harder because it cannot use the thin wire
assumptions. Another test is for the thin wire case. Also added are the
inverted-L and the T antenna from the original Mininec reports. All
these may also serve as examples.  Tests currently cover all except for
two statements, one of them the line after::

 if __name__ == '__main__':

For the second one in ``compute_impedance_matrix`` near the end (as of
this writing line 1367) I've not yet found a test case. I've not yet run
the current tests through the basic interpreter with a breakpoint on
that statement in Basic. It may very well be that this is still a bug
and that it is not reachable in the current Python code.

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
double precision). But the absolute values computed in Python are lower
than the ones reported by Zeineddin (and the Basic code *does* reproduce
Zeineddins values). So it may well be that there is still a problem in
the computations of the currents in the Python code, the computed
currents are too low in that example. You can find the files in
``test/ohio*`` (the thesis was at Ohio University). This time there is a
python script ``ohio.py`` to compute the near and far field values
without recomputing the impedance matrix. This script can plot the near
and far field values into a plot and the difference into a second plot.
There are two distances for which these are computed. There is a second
script to plot the Basic near and far field patterns
``plot_bas_ohio.py``.

Note that the current code is still hard to understand |--| it's the
result of a line-by-line translation from Basic, especially where I
didn't (yet) understand the intention of the code. The same holds for
Variable names which might not (yet) reflect the intention of the code.
I *did* move things like computation of the angle of a complex number,
or the computation of the absolute value, or multiplication/division of
complex numbers to the corresponding complex arithmetic in python where
I detected the pattern.

So the *de-spaghettification* was not successful in some parts of the
code yet :-) My notes from the reverse-engineering can be found in the
file ``mininec-done`` which has explanations of some of the variables
used in mininec and some sub routines with descriptions (mostly taken
from ``REM`` statements) of the Basic code.

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

I've not investigated yet how these errors affect the computed values of
the later Mininec code. Now Mininec is known to find a resonance point
of an antenna some percent too high which means that usually in practice
the computed wire lengths are a little too long. It may well be that the
elliptic integral parameters have an influence there.

I've not investigated how to derive the elliptic integral parameters to
correct possible errors in the elliptic integral implementation.

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
the file can be found `in my pcbasic fork`_ on github.

In the file ``debug-basic.txt`` you can find my notes on how to debug
mininec using the python debugger. This is more or less a random
cut&paste buffer.

The `original basic source code`_ can be obtained from the `unofficial
NEC archive`_ by PA3KJ or from a `Mininec github project`_, I'm using
the version from the `unofficial NEC archive`_ and have not verified if
the two links I've given contain the same code.

.. _`original basic source code`: http://nec-archives.pa3kj.com/mininec3.zip
.. _`unofficial NEC archive`: http://nec-archives.pa3kj.com/
.. _`Mininec github project`: https://github.com/Kees-PA3KJ/MiniNec
.. _`numpy`: https://numpy.org/
.. _`pcbasic`: https://github.com/robhagemans/pcbasic
.. _`Debian`: https://packages.debian.org/stable/python3-pcbasic
.. _`contribution to pcbasic`: https://github.com/robhagemans/pcbasic/pull/183
.. _`in my pcbasic fork`:
    https://github.com/schlatterbeck/pcbasic/blob/pydebug/debugging.rst

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

.. _ADA121535: https://apps.dtic.mil/sti/pdfs/ADA121535.pdf
.. _ADA181682: https://apps.dtic.mil/sti/pdfs/ADA181682.pdf
.. _`numpy.linalg.solve`:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
