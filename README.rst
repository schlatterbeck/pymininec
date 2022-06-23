MININEC in Python
=================

.. |--| unicode:: U+2013   .. en dash

This is an attempt to rewrite the original MININEC3 basic sources in
Python. Currently implemented is the computation of the impedance
matrix, the computation of currents resulting from solving that matrix,
and the computation of the far field.

There are several tests against the `original Basic source code`_, for
the test cases see the subdirectory ``test``. One of the test cases is
a simple 7MHz wire dipole with half the wavelength and 10 segments.
In one case the wire is 0.01m (1cm) thick, we use such a thick wire to
make the mininec code work harder because it cannot use the thin wire
asumptions. Another test is for the thin wire case.

For all the test examples it was carefully verified that the results are
close to the original results in Basic (see `Running examples in Basic`_
to see how you can run the original Basic code in the 21th century). The
differences are due to rounding errors in the single precision
implementation in Basic compared to a double precision implementation in
Python. I'm using numeric code from `numpy`_ where possible to speed up
computation, e.g. solving the impedance matrix is done using
``numpy.linalg.solve`` instead of a line-by-line translation from Basic.
You can verify the differences yourself. In the ``test`` directory there
are input files with extension ``.mini`` which are intended (after
conversion to carriage-return convention) to be used as input to the
original Basic code. The output of the Basic code is in files with the
extension ``.bout`` while the output of the Python code is iin files
with the extension ``.pout``. The ``.pout`` files are compared in the
regression tests.

Note that the current code is still hard to understand |--| it's the
result of a line-by-line translation from Basic, especially where I
didn't (yet) understand the intention of the code. The same holds for
Variable names which might not (yet) reflect the intention of the code.
So the *de-spaghettification* was not successful in some parts of the
code yet :-) My notes from the reverse-engineering can be found in the
file ``mininec-done`` which has explanations of some of the variables
used in mininec and some sub routines with descriptions (mostly taken
from ``REM`` statements) of the Basic code.

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
