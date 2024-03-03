Daf.py 0.1.0 - Data in Axes in Formats
======================================

`Daf.jl <https://github.com/tanaylab/Daf.jl>`_ is a Julia package which provides a uniform generic interface for
accessing 1D and 2D data arranged along some set of axes. This is a much-needed generalization of the `AnnData
<https://github.com/scverse/anndata>`_ functionality. This package (``Daf.py``) is a wrapper around the Julia package
that allows accessing ``Daf`` data from Python, using the `JuliaCall <https://github.com/JuliaPy/PythonCall.jl>`_
package.

Installation
------------

Just ``pip install daf``, like installing any other Python package.

Usage
-----

The Python package provides the same API as the Julia package, with the following modifications:

- (Most) functions are exposed as member functions of the ``DafReader`` and ``DafWriter`` classes (e.g., write
  ``reader.get_scalar("version")`` in Python instead of ``get_scalar(reader, "version")`` in Julia).

- There is no ``!`` suffix for functions that modify the data (e.g., write ``writer.set_scalar("version", "1.0")`` in
  Python instead of ``set_scalar!(writer, "version", "1.0")`` in Julia.

- ``Daf.jl`` supports importing and exporting ``AnnData`` objects. However, since it is a Julia package, these objects
  are **not** Python ``anndata`` objects; instead they use the implementation provides by the `Muon.jl
  <https://github.com/scverse/Muon.jl>`_ Julia package. That is, to import/export ``AnnData`` between ``Daf`` and
  Python, your best bet is to go through ``h5ad`` files.

See the Python API documentation (latest published: TODO, head revision: `here
<https://tanaylab.github.io/Daf.py/v0.1.0/html>`_) for details.

License (MIT)
-------------

Copyright © 2024 Weizmann Institute of Science

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
