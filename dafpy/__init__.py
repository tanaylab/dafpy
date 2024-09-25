"""
Data in Axes in Files.

This is a thin wrapper for the ``DataAxesFormats.jl`` Julia package, with the following adaptations:

* In Julia, the API is defined as a set of functions, which take the ``Daf`` object as the 1st parameter. In Python,
  this is implemented as member functions of the ``DafReader`` and ``DafWriter`` classes that wrap the matching Julia
  objects (e.g., Julia's ``get_vector(data, "gene", "is_marker")`` becomes
  Python's ``data.get_vector("gene", "is_marker")``.

* Python doesn't support the ``!`` trailing character in function names (to indicate modifying the object), so it is
  removed from the wrapper Python method names (e.g., Julia's ``set_vector!(data, "gene", "is_marker", mask)`` becomes
  Python's ``data.set_vector("gene", "is_marker", mask)``.

* Python wrappers for Julia functions that take a callback as their first parameter should be invoked using the ``with``
  statement (e.g., Julia's ``empty_dense_vector(data, "gene", "is_marker") do empty_vector ... end`` becomes Python's
  ``with data.empty_dense_vector("gene", "is_marker") as empty_vector: ...``. Since Python only allows a single
  parameter to ``with`` blocks, if Julia provides multiple such parameters (e.g. ``empty_sparse_*``), they are packed in
  a tuple.

* Dense numeric data (that is, ``np.ndarray``) is passed as a zero-copy between Python and Julia. Passing anything else
  creates an in-memory copy. This, alas, includes sparse matrices, because Julia uses 1-based indices in the internal
  sparse representation, while Python uses zero-based indices (e.g Julia's ``colptr`` is equal to 1 plus Python's
  ``indptr``). This forces us to copy and increment/decrement these indices when passing the data between the languages.
  Also note that when invoking ``empty_sparse_*``, you should fill the empty indices array(s) with Julia (1-based)
  values.

* Julia "likes" column-major matrices. All the matrices given to and returned by the API are therefore in column-major
  layout. In contrast, Python "likes" row-major matrices. To convert between the two, flip the order of the axes, and
  ``transpose`` the data. This is a zero-copy view so is very efficient (e.g., to get a row-major UMIs matrix with a row
  per cell and a column per gene, write ``data.get_np_matrix("gene", "cell", "UMIs").transpose()``).

* Python has no notion of sparse vectors because "reasons", while Julia (and R) sensibly do support them. Therefore
  reading a sparse Julia ``Daf`` vector automatically converts it to a dense Python ``numpy`` one. You can however still
  store sparse vectors from Python into ``Daf`` by passing a sparse matrix with a single row or column, or by calling
  ``empty_sparse_vector``.

* Sparse matrices are not supported inside ``pandas``. Therefore reading sparse ``Daf`` data into a ``pandas``
  ``DataFrame`` automatically converts it to a dense Python ``numpy`` matrix. Therefore, you should take care not to ask
  for a data frame of a large sparse matrix (e.g., the UMIs matrix of a large data set), as this will consume a *lot* of
  memory.

Otherwise, the API works "just the same" :-) The documentation therefore mostly just links to the relevant entry in the
Julia `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/index.html>`__.
"""

__author__ = "Oren Ben-Kiki"
__email__ = "oren@ben-kiki.org"
__version__ = "0.1.1"

# pylint: disable=wildcard-import,unused-wildcard-import

from .julia_import import *  # isort: skip
from .generic_functions import *  # isort: skip
from .generic_logging import *  # isort: skip
from .storage_types import *  # isort: skip
from .operations import *  # isort: skip
from .queries import *  # isort: skip
from .data import *  # isort: skip
from .formats import *  # isort: skip
from .views import *  # isort: skip
from .copies import *  # isort: skip
from .adapters import *  # isort: skip
from .concat import *  # isort: skip
from .anndata_format import *  # isort: skip
from .reconstruction import *  # isort: skip
