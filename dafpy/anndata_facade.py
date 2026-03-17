"""
Facade that presents a ``DafReader`` or ``DafWriter`` as an ``AnnData``-like object.

Given two axis names and a primary matrix name, :class:`DafAnnData` exposes the
``DataAxesFormats`` data through the standard ``AnnData`` API so that existing code
written against ``AnnData`` can consume a ``Daf`` data set with minimal changes.

Key mappings
------------
* ``X`` — the named matrix over ``(obs_axis, var_axis)``.
* ``obs`` / ``var`` — dict-like proxies over vector properties of each axis,
  behave like ``pandas`` ``DataFrame`` for the common ``[]`` get/set/delete,
  ``.columns``, ``.index``, and ``.to_df()`` operations.
* ``layers`` — dict-like proxy over additional ``(obs_axis, var_axis)`` matrices
  (all except ``X``).
* ``uns`` — flat dict-like proxy over Daf scalars (strings and numbers only,
  nested dicts are not supported).
* ``obsp`` / ``varp`` — dict-like proxies over square ``(axis, axis)`` matrices.
* ``obsm`` / ``varm`` — dict-like proxies over ``(main_axis, other_axis)`` matrices
  addressed by the key ``"other_axis:matrix_name"``.  Setting requires
  ``other_axis`` to already exist in the Daf data set.

Slicing
-------
``adata[obs_index, var_index]`` returns a read-only ``DafAnnData`` backed by a
``DafView``.  Each index may be a Boolean ``numpy`` array, a list of entry names
or integer positions, a single name or integer, or a ``slice``.

The extension methods :meth:`DafAnnData.query_obs` and :meth:`DafAnnData.query_var`
accept any Daf query fragment that would fit between the ``[`` and ``]`` in
``@ axis [ ... ]``.

Limitations
-----------
* Categorical values assigned to ``obs`` or ``var`` columns are automatically
  converted to strings.
* ``uns`` only supports flat scalar values (strings and numbers).  Assigning a
  non-scalar raises ``TypeError``.
* ``obs_names`` and ``var_names`` are read-only; axis entries cannot be renamed
  through this facade.
* Concatenation and other operations that create new ``AnnData`` objects are not
  supported.
"""

import re
from typing import AbstractSet
from typing import Any
from typing import FrozenSet
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore

from .data import DafReader
from .formats import chain_writer
from .formats import memory_daf
from .storage_types import StorageScalar
from .views import viewer

__all__ = ["DafAnnData"]

_MASK_NAME = "__mask__"

_AXIS_MATRIX_KEY_RE = re.compile(r"^([^:]+):(.+)$")

# pylint: disable=missing-function-docstring


def _is_storage_scalar(value: Any) -> bool:
    """
    Return True if ``value`` is a string or number acceptable by Daf.
    """
    return isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_))


def _prepare_vector(value: Any) -> Any:
    """
    Normalise a vector value before writing it to Daf.

    * ``pd.Series`` is converted to the underlying numpy array (categoricals are converted to a string array).
    * ``pd.Categorical`` is concerted to a string numpy array.
    * Object-dtype numpy arrays are converted to strings.
    """
    if isinstance(value, pd.Series):
        if isinstance(value.dtype, pd.CategoricalDtype):
            return np.array(value.astype(str).tolist())
        return value.values
    if isinstance(value, pd.Categorical):
        return np.array(list(value.astype(str)))
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        return value.astype(str)
    return value


def _to_bool_mask(  # pylint: disable=too-many-return-statements
    index: Any, entries: np.ndarray
) -> Optional[np.ndarray]:
    """
    Convert ``index`` to a Boolean mask aligned with *entries*.

    Returns ``None`` when all entries are selected (no filtering needed). Accepts Boolean arrays, lists of names or
    integer positions, single name or integer, and slices.
    """
    n = len(entries)

    if index is None or index is ...:
        return None

    if isinstance(index, slice):
        if index == slice(None):
            return None
        mask = np.zeros(n, dtype=bool)
        mask[index] = True
        return None if mask.all() else mask

    if isinstance(index, np.ndarray):
        if index.dtype == bool:
            return None if index.all() else index.copy()
        # Integer array
        mask = np.zeros(n, dtype=bool)
        mask[index] = True
        return None if mask.all() else mask

    if isinstance(index, list):
        if not index:
            return np.zeros(n, dtype=bool)
        if isinstance(index[0], (str, np.str_)):
            entry_set = set(index)
            mask = np.array([e in entry_set for e in entries], dtype=bool)
        else:
            mask = np.zeros(n, dtype=bool)
            mask[np.array(index)] = True
        return None if mask.all() else mask

    if isinstance(index, str):
        return entries == index

    if isinstance(index, (int, np.integer)):
        mask = np.zeros(n, dtype=bool)
        mask[int(index)] = True
        return mask

    raise IndexError(f"unsupported index type: {type(index).__name__}")


class _DafAxisFrame:
    """
    Dict-like proxy over the vector properties of one Daf axis.

    Mimics a ``pandas`` ``DataFrame`` for the operations most commonly used on
    ``adata.obs`` and ``adata.var``:

    * ``frame["col"]`` — returns the vector as a ``pd.Series``.
    * ``frame[["col1", "col2"]]`` — returns a ``pd.DataFrame``.
    * ``frame["col"] = values`` — writes the vector back to Daf.
    * ``del frame["col"]`` — deletes the vector from Daf.
    * ``frame.columns`` — ``pd.Index`` of column names.
    * ``frame.index`` — ``pd.Index`` of axis entry names.
    * ``len(frame)`` — number of rows (axis length), matching ``pandas`` semantics.
    * ``"col" in frame`` — membership test.
    * ``for col in frame:`` — iterates column names.
    * ``frame.to_df()`` — returns a real ``pd.DataFrame``.
    """

    def __init__(self, daf: DafReader, axis: str, hidden: FrozenSet[str]) -> None:
        self._daf = daf
        self._axis = axis
        self._hidden = hidden

    @property
    def index(self) -> pd.Index:
        """
        Entry names of the axis as a ``pd.Index``.
        """
        return pd.Index(self._daf.axis_np_vector(self._axis), name=self._axis)

    @property
    def columns(self) -> pd.Index:
        """
        Names of the vector properties (columns) as a ``pd.Index``.
        """
        return pd.Index(sorted(self._visible()))

    def _visible(self) -> Set[str]:
        return {v for v in self._daf.vectors_set(self._axis) if v not in self._hidden}

    def __contains__(self, key: str) -> bool:
        return key not in self._hidden and self._daf.has_vector(self._axis, key)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._visible()))

    def __len__(self) -> int:
        # Match pandas DataFrame: len(df) is the number of *rows*.
        return self._daf.axis_length(self._axis)

    def __getitem__(self, key: Union[str, list]) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(key, str):
            if key in self._hidden:
                raise KeyError(key)
            return self._daf.get_pd_vector(self._axis, key)
        if isinstance(key, list):
            for k in key:
                if k in self._hidden:
                    raise KeyError(k)
            return pd.DataFrame({k: self._daf.get_np_vector(self._axis, k) for k in key}, index=self.index)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._daf.set_vector(self._axis, key, _prepare_vector(value), overwrite=True)  # type: ignore

    def __delitem__(self, key: str) -> None:
        if key in self._hidden:
            raise KeyError(key)
        self._daf.delete_vector(self._axis, key)  # type: ignore

    def keys(self) -> Set[str]:
        """
        Return the set of visible column names.
        """
        return self._visible()

    def to_df(self) -> pd.DataFrame:
        """
        Return all columns as a real ``pandas`` ``DataFrame``.
        """
        cols = sorted(self._visible())
        return pd.DataFrame(
            {col: self._daf.get_np_vector(self._axis, col) for col in cols},
            index=self.index,
        )

    def __repr__(self) -> str:
        return self.to_df().__repr__()


class _DafLayersMapping:
    """
    Dict-like proxy over ``(obs_axis, var_axis)`` matrices, excluding ``X``.
    """

    def __init__(self, daf: DafReader, obs_axis: str, var_axis: str, x_name: str) -> None:
        self._daf = daf
        self._obs_axis = obs_axis
        self._var_axis = var_axis
        self._x_name = x_name

    def _names(self) -> Set[str]:
        return {name for name in self._daf.matrices_set(self._obs_axis, self._var_axis) if name != self._x_name}

    def keys(self) -> Set[str]:
        return self._names()

    def __contains__(self, key: str) -> bool:
        return key != self._x_name and self._daf.has_matrix(self._obs_axis, self._var_axis, key)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._names()))

    def __len__(self) -> int:
        return len(self._names())

    def __getitem__(self, key: str) -> Union[np.ndarray, sp.csc_matrix]:
        if key == self._x_name:
            raise KeyError(key)
        return self._daf.get_np_matrix(self._obs_axis, self._var_axis, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._daf.set_matrix(self._var_axis, self._obs_axis, key, value.T, overwrite=True)  # type: ignore

    def __delitem__(self, key: str) -> None:
        if key == self._x_name:
            raise KeyError(key)
        self._daf.delete_matrix(self._obs_axis, self._var_axis, key)  # type: ignore

    def items(self):
        return ((k, self[k]) for k in self)

    def __repr__(self) -> str:
        return f"layers: {sorted(self._names())}"


class _DafUns:
    """
    Dict-like proxy over Daf scalars; values must be strings or numbers.
    """

    def __init__(self, daf: DafReader) -> None:
        self._daf = daf

    def keys(self) -> AbstractSet[str]:
        return self._daf.scalars_set()  # type: ignore

    def __contains__(self, key: str) -> bool:
        return self._daf.has_scalar(key)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._daf.scalars_set()))

    def __len__(self) -> int:
        return len(self._daf.scalars_set())

    def __getitem__(self, key: str) -> StorageScalar:
        return self._daf.get_scalar(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not _is_storage_scalar(value):
            raise TypeError(
                f"uns values must be strings or numbers, got {type(value).__name__}; "
                "nested dicts and other structured values are not supported"
            )
        self._daf.set_scalar(key, value, overwrite=True)  # type: ignore

    def __delitem__(self, key: str) -> None:
        self._daf.delete_scalar(key)  # type: ignore

    def get(self, key: str, default: Any = None) -> Any:
        """
        Return the value for ``key`` if it exists, otherwise ``default``.
        """
        return self[key] if key in self else default

    def update(self, mapping: Mapping) -> None:
        """
        Set multiple scalars at once.
        """
        for k, v in mapping.items():
            self[k] = v

    def items(self):
        return ((k, self[k]) for k in self)

    def __repr__(self) -> str:
        return f"{{{', '.join(f'{k!r}: {self[k]!r}' for k in self)}}}"


class _DafPairwiseMapping:
    """
    Dict-like proxy over square ``(axis, axis)`` matrices (``obsp`` / ``varp``).
    """

    def __init__(self, daf: DafReader, axis: str) -> None:
        self._daf = daf
        self._axis = axis

    def keys(self) -> AbstractSet[str]:
        return self._daf.matrices_set(self._axis, self._axis)

    def __contains__(self, key: str) -> bool:
        return self._daf.has_matrix(self._axis, self._axis, key)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.keys()))

    def __len__(self) -> int:
        return len(self.keys())

    def __getitem__(self, key: str) -> Union[np.ndarray, sp.csc_matrix]:
        return self._daf.get_np_matrix(self._axis, self._axis, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._daf.set_matrix(self._axis, self._axis, key, value.T, overwrite=True)  # type: ignore

    def __delitem__(self, key: str) -> None:
        self._daf.delete_matrix(self._axis, self._axis, key)  # type: ignore

    def items(self):
        return ((k, self[k]) for k in self)

    def __repr__(self) -> str:
        return f"pairwise({self._axis!r}): {sorted(self.keys())}"


class _DafEmbeddingMapping:
    """
    Dict-like proxy over ``(main_axis, other_axis)`` matrices (``obsm`` / ``varm``).

    Keys use the naming convention ``"other_axis:matrix_name"``.  The ``other_axis`` must not be either of the two
    primary axes (those belong to ``layers``, ``obsp``, or ``varp``).  Setting a matrix whose ``other_axis`` does not
    already exist in the Daf data set raises ``KeyError``.
    """

    def __init__(self, daf: DafReader, main_axis: str, exclude_axes: FrozenSet[str]) -> None:
        self._daf = daf
        self._main_axis = main_axis
        self._exclude_axes = exclude_axes

    def _other_axes(self) -> Set[str]:
        return {ax for ax in self._daf.axes_set() if ax not in self._exclude_axes}

    def _all_keys(self) -> Set[str]:
        result: Set[str] = set()
        for other_axis in self._other_axes():
            for mat_name in self._daf.matrices_set(self._main_axis, other_axis):
                result.add(f"{other_axis}:{mat_name}")
        return result

    @staticmethod
    def _parse_key(key: str) -> Tuple[str, str]:
        m = _AXIS_MATRIX_KEY_RE.match(key)
        if not m:
            raise KeyError(f"obsm/varm key must be 'other_axis:matrix_name', got {key!r}")
        return m.group(1), m.group(2)

    def keys(self) -> Set[str]:
        return self._all_keys()

    def __contains__(self, key: str) -> bool:
        try:
            other_axis, mat_name = self._parse_key(key)
        except KeyError:
            return False
        return (
            other_axis not in self._exclude_axes
            and self._daf.has_axis(other_axis)
            and self._daf.has_matrix(self._main_axis, other_axis, mat_name)
        )

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._all_keys()))

    def __len__(self) -> int:
        return len(self._all_keys())

    def __getitem__(self, key: str) -> np.ndarray:
        other_axis, mat_name = self._parse_key(key)
        if other_axis in self._exclude_axes:
            raise KeyError(key)
        return self._daf.get_np_matrix(self._main_axis, other_axis, mat_name)

    def __setitem__(self, key: str, value: Any) -> None:
        other_axis, mat_name = self._parse_key(key)
        if other_axis in self._exclude_axes:
            raise KeyError(f"axis {other_axis!r} is a primary axis; " "use 'layers', 'obsp', or 'varp' instead")
        if not self._daf.has_axis(other_axis):
            raise KeyError(
                f"axis {other_axis!r} does not exist in the Daf data set; " "create it before assigning an embedding"
            )
        self._daf.set_matrix(other_axis, self._main_axis, mat_name, value.T, overwrite=True)  # type: ignore

    def __delitem__(self, key: str) -> None:
        other_axis, mat_name = self._parse_key(key)
        if other_axis in self._exclude_axes:
            raise KeyError(key)
        self._daf.delete_matrix(self._main_axis, other_axis, mat_name)  # type: ignore

    def items(self):
        return ((k, self[k]) for k in self)

    def __repr__(self) -> str:
        return f"embeddings({self._main_axis!r}): {sorted(self._all_keys())}"


# pylint: enable=missing-function-docstring


class DafAnnData:  # pylint: disable=too-many-instance-attributes
    """
    Facade that presents a :class:`~dafpy.DafReader` or :class:`~dafpy.DafWriter` as an ``AnnData``-like object.

    Parameters
    ----------
    daf:
        The underlying Daf data set (reader or writer).
    obs_axis:
        Name of the Daf axis that corresponds to AnnData's observations (rows of ``X``).
    var_axis:
        Name of the Daf axis that corresponds to AnnData's variables (columns of ``X``).
    x_matrix:
        Name of the ``(obs_axis, var_axis)`` matrix that is exposed as ``X``.
    """

    def __init__(
        self,
        daf: DafReader,
        *,
        obs_axis: str,
        var_axis: str,
        x_matrix: str,
    ) -> None:
        self._daf = daf
        self._obs_axis = obs_axis
        self._var_axis = var_axis
        self._x_name = x_matrix

        _hidden = frozenset({_MASK_NAME})
        self._obs_proxy = _DafAxisFrame(daf, obs_axis, _hidden)
        self._var_proxy = _DafAxisFrame(daf, var_axis, _hidden)
        self._layers_proxy = _DafLayersMapping(daf, obs_axis, var_axis, x_matrix)
        self._uns_proxy = _DafUns(daf)
        self._obsp_proxy = _DafPairwiseMapping(daf, obs_axis)
        self._varp_proxy = _DafPairwiseMapping(daf, var_axis)
        self._obsm_proxy = _DafEmbeddingMapping(daf, obs_axis, frozenset({obs_axis, var_axis}))
        self._varm_proxy = _DafEmbeddingMapping(daf, var_axis, frozenset({obs_axis, var_axis}))

    @property
    def daf(self) -> DafReader:
        """
        Access the wrapped ``Daf`` repository.
        """
        return self._daf

    @property
    def obs_names(self) -> pd.Index:
        """
        Names of the observations as a ``pd.Index``.
        """
        return self._obs_proxy.index

    @property
    def var_names(self) -> pd.Index:
        """
        Names of the variables as a ``pd.Index``.
        """
        return self._var_proxy.index

    @property
    def n_obs(self) -> int:
        """
        Number of observations.
        """
        return self._daf.axis_length(self._obs_axis)

    @property
    def n_vars(self) -> int:
        """
        Number of variables.
        """
        return self._daf.axis_length(self._var_axis)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of the data matrix ``(n_obs, n_vars)``.
        """
        return (self.n_obs, self.n_vars)

    @property
    def X(self) -> Union[np.ndarray, sp.csc_matrix]:
        """
        Primary data matrix of shape ``(n_obs, n_vars)``.
        """
        return self._daf.get_np_matrix(self._obs_axis, self._var_axis, self._x_name)

    @X.setter
    def X(self, value: Any) -> None:
        self._daf.set_matrix(self._var_axis, self._obs_axis, self._x_name, value.T, overwrite=True)  # type: ignore

    @property
    def obs(self) -> _DafAxisFrame:
        """
        Observation annotations — dict-like proxy over obs-axis vectors.
        """
        return self._obs_proxy

    @obs.setter
    def obs(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            self._daf.set_vector(self._obs_axis, col, _prepare_vector(df[col]), overwrite=True)  # type: ignore

    @property
    def var(self) -> _DafAxisFrame:
        """
        Variable annotations — dict-like proxy over var-axis vectors.
        """
        return self._var_proxy

    @var.setter
    def var(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            self._daf.set_vector(self._var_axis, col, _prepare_vector(df[col]), overwrite=True)  # type: ignore

    @property
    def layers(self) -> _DafLayersMapping:
        """
        Additional ``(obs, var)`` matrices, excluding ``X``.
        """
        return self._layers_proxy

    @layers.setter
    def layers(self, value: Mapping) -> None:
        for k, v in value.items():
            self._layers_proxy[k] = v

    @property
    def uns(self) -> _DafUns:
        """
        Unstructured annotations as a flat dict of Daf scalars.
        """
        return self._uns_proxy

    @uns.setter
    def uns(self, value: Mapping) -> None:
        self._uns_proxy.update(value)

    @property
    def obsp(self) -> _DafPairwiseMapping:
        """
        Square ``(obs × obs)`` pairwise matrices.
        """
        return self._obsp_proxy

    @property
    def varp(self) -> _DafPairwiseMapping:
        """
        Square ``(var × var)`` pairwise matrices.
        """
        return self._varp_proxy

    @property
    def obsm(self) -> _DafEmbeddingMapping:
        """
        Observation embeddings, keyed as ``"other_axis:matrix_name"``.
        """
        return self._obsm_proxy

    @property
    def varm(self) -> _DafEmbeddingMapping:
        """
        Variable embeddings, keyed as ``"other_axis:matrix_name"``.
        """
        return self._varm_proxy

    def to_df(self, layer: Optional[str] = None) -> pd.DataFrame:
        """
        Return ``X`` (or a named ``layer``) as a ``pandas`` ``DataFrame``.

        The result has ``obs_names`` as the row index and ``var_names`` as columns.  Sparse matrices are densified.
        """
        matrix = self._daf.get_np_matrix(
            self._obs_axis,
            self._var_axis,
            self._x_name if layer is None else layer,
        )
        if sp.issparse(matrix):
            matrix = matrix.toarray()  # type: ignore
        return pd.DataFrame(matrix, index=self.obs_names, columns=self.var_names)

    def __getitem__(self, index: Any) -> "DafAnnData":
        """
        Subset observations and/or variables: ``adata[obs_index, var_index]``.

        Each index may be a Boolean ``numpy`` array, a list of entry names or integer positions, a single name or
        integer, or a ``slice``.

        Returns a new *read-only* :class:`DafAnnData` backed by a ``DafView``.
        """
        if not isinstance(index, tuple) or len(index) != 2:
            raise IndexError("DafAnnData slicing requires two indices: adata[obs_index, var_index]")
        obs_index, var_index = index

        obs_entries = self._daf.axis_np_vector(self._obs_axis)
        var_entries = self._daf.axis_np_vector(self._var_axis)

        obs_bool = _to_bool_mask(obs_index, obs_entries)
        var_bool = _to_bool_mask(var_index, var_entries)

        if obs_bool is None and var_bool is None:
            # No filtering — wrap self without creating any new objects.
            return DafAnnData(
                self._daf.read_only(name=f"{self._daf.name}.sliced"),
                obs_axis=self._obs_axis,
                var_axis=self._var_axis,
                x_matrix=self._x_name,
            )

        tmp = memory_daf(name=f"{self._daf.name}.mask")
        chained = chain_writer([self._daf, tmp])
        axes_view = {"*": "="}

        if obs_bool is not None:
            chained.set_vector(self._obs_axis, _MASK_NAME, obs_bool, overwrite=True)
            axes_view[self._obs_axis] = f"@ {self._obs_axis} [{_MASK_NAME}]"

        if var_bool is not None:
            chained.set_vector(self._var_axis, _MASK_NAME, var_bool, overwrite=True)
            axes_view[self._var_axis] = f"@ {self._var_axis} [{_MASK_NAME}]"
        view = viewer(chained, axes=axes_view)
        return DafAnnData(
            view,
            obs_axis=self._obs_axis,
            var_axis=self._var_axis,
            x_matrix=self._x_name,
        )

    def query_obs(self, query_fragment: str) -> "DafAnnData":
        """
        Return a read-only view with observations filtered by a Daf query.

        Parameters
        ----------
        query_fragment:
            Anything that would fit between the ``[`` and ``]`` of a Daf axis query, e.g. ``donor = D1 & age > 30``.

        Returns
        -------
        DafAnnData
            A read-only facade wrapping a ``DafView`` filtered on the obs axis.
        """
        view = viewer(self._daf, axes={"*": "=", self._obs_axis: f"@ {self._obs_axis} [{query_fragment}]"})
        return DafAnnData(view, obs_axis=self._obs_axis, var_axis=self._var_axis, x_matrix=self._x_name)

    def query_var(self, query_fragment: str) -> "DafAnnData":
        """
        Return a read-only view with variables filtered by a Daf query.

        Parameters
        ----------
        query_fragment:
            Anything that would fit between the ``[`` and ``]`` of a Daf axis query, e.g.
            ``highly_variable & ! is_lateral``.

        Returns
        -------
        DafAnnData
            A read-only facade wrapping a ``DafView`` filtered on the var axis.
        """
        view = viewer(self._daf, axes={"*": "=", self._var_axis: f"@ {self._var_axis} [{query_fragment}]"})
        return DafAnnData(view, obs_axis=self._obs_axis, var_axis=self._var_axis, x_matrix=self._x_name)

    def __repr__(self) -> str:
        obs_cols = sorted(self._obs_proxy.keys())
        var_cols = sorted(self._var_proxy.keys())
        layer_names = sorted(self._layers_proxy.keys())
        uns_keys = sorted(self._uns_proxy.keys())
        obsm_keys = sorted(self._obsm_proxy.keys())
        varm_keys = sorted(self._varm_proxy.keys())
        obsp_keys = sorted(self._obsp_proxy.keys())
        varp_keys = sorted(self._varp_proxy.keys())
        lines = [
            f"DafAnnData: {self._daf.name!r}",
            f"  {self.n_obs} obs ({self._obs_axis!r}) × {self.n_vars} vars ({self._var_axis!r})",
            f"  X: {self._x_name!r}",
        ]
        if obs_cols:
            lines.append(f"  obs:    {obs_cols}")
        if var_cols:
            lines.append(f"  var:    {var_cols}")
        if layer_names:
            lines.append(f"  layers: {layer_names}")
        if uns_keys:
            lines.append(f"  uns:    {uns_keys}")
        if obsm_keys:
            lines.append(f"  obsm:   {obsm_keys}")
        if varm_keys:
            lines.append(f"  varm:   {varm_keys}")
        if obsp_keys:
            lines.append(f"  obsp:   {obsp_keys}")
        if varp_keys:
            lines.append(f"  varp:   {varp_keys}")
        return "\n".join(lines)
