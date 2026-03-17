"""
Test the ``DafAnnData`` AnnData facade.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring

import numpy as np
import pandas as pd

import dafpy as dp
from dafpy.anndata_facade import DafAnnData

from .utilities import assert_raises


def _cells():
    return dp.example_cells_daf()


def _metacells():
    return dp.example_metacells_daf()


def _chain():
    return dp.example_chain_daf()


def _cells_adata():
    return DafAnnData(_cells(), obs_axis="cell", var_axis="gene", x_matrix="UMIs")


def _metacells_adata():
    return DafAnnData(_metacells(), obs_axis="metacell", var_axis="gene", x_matrix="fraction")


def _chain_adata():
    return DafAnnData(_chain(), obs_axis="cell", var_axis="gene", x_matrix="UMIs")


def test_n_obs() -> None:
    assert _cells_adata().n_obs == 856


def test_n_vars() -> None:
    assert _cells_adata().n_vars == 683


def test_shape() -> None:
    assert _cells_adata().shape == (856, 683)


def test_obs_names_length() -> None:
    assert len(_cells_adata().obs_names) == 856


def test_obs_names_type() -> None:
    assert isinstance(_cells_adata().obs_names, pd.Index)


def test_var_names_length() -> None:
    assert len(_cells_adata().var_names) == 683


def test_var_names_type() -> None:
    assert isinstance(_cells_adata().var_names, pd.Index)


def test_daf_property() -> None:
    daf = _cells()
    adata = DafAnnData(daf, obs_axis="cell", var_axis="gene", x_matrix="UMIs")
    assert adata.daf is daf


def test_as_anndata() -> None:
    daf = _cells()
    adata = daf.as_anndata(obs_axis="cell", var_axis="gene", x_matrix="UMIs")
    assert adata.daf is daf
    assert adata.n_obs == 856
    assert adata.n_vars == 683


def test_X_shape() -> None:  # pylint: disable=invalid-name
    assert _cells_adata().X.shape == (856, 683)


def test_X_dtype() -> None:  # pylint: disable=invalid-name
    assert _cells_adata().X.dtype == np.uint8


def test_X_setter() -> None:  # pylint: disable=invalid-name
    adata = _cells_adata()
    new_x = np.zeros((856, 683), dtype=np.float32)
    adata.X = new_x
    assert adata.X.shape == (856, 683)
    assert adata.X.dtype == np.float32


def test_obs_columns_present() -> None:
    cols = _cells_adata().obs.columns
    assert "donor" in cols
    assert "experiment" in cols


def test_obs_columns_hides_mask() -> None:
    assert "__mask__" not in _cells_adata().obs.columns


def test_obs_index_length() -> None:
    assert len(_cells_adata().obs.index) == 856


def test_obs_getitem_str() -> None:
    series = _cells_adata().obs["donor"]  # type: ignore
    assert isinstance(series, pd.Series)
    assert len(series) == 856


def test_obs_getitem_list() -> None:
    df = _cells_adata().obs[["donor", "experiment"]]
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"donor", "experiment"}


def test_obs_contains_present() -> None:
    assert "donor" in _cells_adata().obs


def test_obs_contains_absent() -> None:
    assert "nonexistent" not in _cells_adata().obs


def test_obs_contains_mask_hidden() -> None:
    assert "__mask__" not in _cells_adata().obs


def test_obs_iter() -> None:
    cols = list(_cells_adata().obs)
    assert "donor" in cols
    assert "__mask__" not in cols


def test_obs_len() -> None:
    # len(obs) matches the number of observations (rows), like a DataFrame.
    assert len(_cells_adata().obs) == 856


def test_obs_to_df() -> None:
    df = _cells_adata().obs.to_df()
    assert isinstance(df, pd.DataFrame)
    assert "donor" in df.columns
    assert "__mask__" not in df.columns


def test_obs_setitem() -> None:
    adata = _cells_adata()
    adata.obs["new_col"] = np.zeros(856, dtype=np.int32)
    assert adata.obs["new_col"].iloc[0] == 0


def test_obs_setitem_categorical() -> None:
    adata = _cells_adata()
    cat = pd.Categorical(["A", "B"] * 428)
    adata.obs["cat_col"] = cat
    result = adata.obs["cat_col"]
    assert result.iloc[0] in ("A", "B")


def test_obs_delitem() -> None:
    adata = _cells_adata()
    adata.obs["to_delete"] = np.zeros(856, dtype=np.int32)
    assert "to_delete" in adata.obs
    del adata.obs["to_delete"]
    assert "to_delete" not in adata.obs


def test_var_columns_present() -> None:
    assert "is_lateral" in _cells_adata().var.columns


def test_var_getitem() -> None:
    series = _cells_adata().var["is_lateral"]
    assert isinstance(series, pd.Series)
    assert len(series) == 683


def test_var_setitem() -> None:
    adata = _cells_adata()
    adata.var["new_var"] = np.ones(683, dtype=np.float32)
    assert adata.var["new_var"].iloc[0] == 1.0


def test_var_delitem() -> None:
    adata = _cells_adata()
    adata.var["to_delete"] = np.zeros(683, dtype=np.int32)
    del adata.var["to_delete"]
    assert "to_delete" not in adata.var


def test_layers_empty_for_cells() -> None:
    assert len(_cells_adata().layers) == 0


def test_layers_x_excluded() -> None:
    assert "UMIs" not in _cells_adata().layers


def test_layers_x_getitem_raises() -> None:
    with assert_raises("UMIs"):
        _ = _cells_adata().layers["UMIs"]


def test_layers_set_get_delete() -> None:
    adata = _cells_adata()
    mat = np.zeros((856, 683), dtype=np.float32)
    adata.layers["zeros"] = mat
    assert "zeros" in adata.layers
    assert adata.layers["zeros"].shape == (856, 683)
    del adata.layers["zeros"]
    assert "zeros" not in adata.layers


def test_uns_get_organism() -> None:
    assert _cells_adata().uns["organism"] == "human"


def test_uns_get_reference() -> None:
    assert _cells_adata().uns["reference"] == "test"


def test_uns_len() -> None:
    assert len(_cells_adata().uns) == 2


def test_uns_contains_present() -> None:
    assert "organism" in _cells_adata().uns


def test_uns_contains_absent() -> None:
    assert "nonexistent" not in _cells_adata().uns


def test_uns_iter() -> None:
    keys = list(_cells_adata().uns)
    assert "organism" in keys
    assert "reference" in keys


def test_uns_set_get_delete() -> None:
    adata = _cells_adata()
    adata.uns["answer"] = 42
    assert adata.uns["answer"] == 42
    del adata.uns["answer"]
    assert "answer" not in adata.uns


def test_uns_set_string() -> None:
    adata = _cells_adata()
    adata.uns["note"] = "hello"
    assert adata.uns["note"] == "hello"


def test_uns_set_nested_dict_raises() -> None:
    adata = _cells_adata()
    with assert_raises("uns values must be strings or numbers"):
        adata.uns["nested"] = {"key": "value"}


def test_uns_set_list_raises() -> None:
    adata = _cells_adata()
    with assert_raises("uns values must be strings or numbers"):
        adata.uns["lst"] = [1, 2, 3]


def test_uns_get_with_default_present() -> None:
    assert _cells_adata().uns.get("organism") == "human"


def test_uns_get_with_default_absent() -> None:
    assert _cells_adata().uns.get("nonexistent", "default_val") == "default_val"


def test_uns_get_absent_no_default() -> None:
    assert _cells_adata().uns.get("nonexistent") is None


def test_obsp_has_edge_weight() -> None:
    adata = _metacells_adata()
    assert "edge_weight" in adata.obsp
    mat = adata.obsp["edge_weight"]
    assert mat.shape == (7, 7)


def test_varp_empty_for_cells() -> None:
    assert len(_cells_adata().varp) == 0


def test_obsp_set_get_delete() -> None:
    adata = _metacells_adata()
    mat = np.ones((7, 7), dtype=np.float32)
    adata.obsp["similarity"] = mat
    assert "similarity" in adata.obsp
    assert adata.obsp["similarity"].shape == (7, 7)
    del adata.obsp["similarity"]
    assert "similarity" not in adata.obsp


def test_obsm_empty_for_cells() -> None:
    assert len(_cells_adata().obsm) == 0


def test_varm_has_fraction_in_chain() -> None:
    adata = _chain_adata()
    assert "metacell:fraction" in adata.varm
    mat = adata.varm["metacell:fraction"]
    assert mat.shape == (683, 7)


def test_obsm_set_get_delete() -> None:
    # "type" axis (4 entries) already exists in metacells; not in exclude set {metacell, gene}
    adata = _metacells_adata()
    mat = np.zeros((7, 4), dtype=np.float32)
    adata.obsm["type:weights"] = mat
    assert "type:weights" in adata.obsm
    assert adata.obsm["type:weights"].shape == (7, 4)
    del adata.obsm["type:weights"]
    assert "type:weights" not in adata.obsm


def test_obsm_bad_key_format_raises() -> None:
    with assert_raises("obsm/varm key must be"):
        _ = _cells_adata().obsm["no_colon"]


def test_obsm_primary_axis_raises() -> None:
    adata = _cells_adata()
    mat = np.zeros((856, 683), dtype=np.float32)
    with assert_raises("primary axis"):
        adata.obsm["gene:embedding"] = mat


def test_obsm_missing_axis_raises() -> None:
    adata = _cells_adata()
    mat = np.zeros((856, 3), dtype=np.float32)
    with assert_raises("does not exist in the Daf data set"):
        adata.obsm["fake_axis:embedding"] = mat


def test_to_df_shape() -> None:
    df = _cells_adata().to_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (856, 683)


def test_to_df_index() -> None:
    adata = _cells_adata()
    df = adata.to_df()
    assert list(df.index) == list(adata.obs_names)


def test_to_df_columns() -> None:
    adata = _cells_adata()
    df = adata.to_df()
    assert list(df.columns) == list(adata.var_names)


def test_to_df_layer() -> None:
    adata = _cells_adata()
    mat = np.ones((856, 683), dtype=np.float32)
    adata.layers["ones"] = mat
    df = adata.to_df(layer="ones")
    assert df.shape == (856, 683)
    assert float(df.iloc[0, 0]) == 1.0


def test_slice_bool_obs() -> None:
    adata = _cells_adata()
    mask = np.zeros(856, dtype=bool)
    mask[:100] = True
    sliced = adata[mask, :]
    assert sliced.n_obs == 100
    assert sliced.n_vars == 683


def test_slice_bool_var() -> None:
    adata = _cells_adata()
    mask = np.zeros(683, dtype=bool)
    mask[:50] = True
    sliced = adata[:, mask]
    assert sliced.n_obs == 856
    assert sliced.n_vars == 50


def test_slice_list_of_names() -> None:
    adata = _cells_adata()
    names = list(adata.obs_names[:10])
    sliced = adata[names, :]
    assert sliced.n_obs == 10


def test_slice_list_of_ints() -> None:
    sliced = _cells_adata()[[0, 1, 2, 3, 4], :]
    assert sliced.n_obs == 5


def test_slice_single_name() -> None:
    adata = _cells_adata()
    name = str(adata.obs_names[0])
    sliced = adata[name, :]
    assert sliced.n_obs == 1


def test_slice_single_int() -> None:
    sliced = _cells_adata()[0, :]
    assert sliced.n_obs == 1


def test_slice_python_slice() -> None:
    sliced = _cells_adata()[10:20, :]
    assert sliced.n_obs == 10


def test_slice_both_axes() -> None:
    sliced = _cells_adata()[:50, :100]
    assert sliced.n_obs == 50
    assert sliced.n_vars == 100


def test_slice_all_entries() -> None:
    adata = _cells_adata()
    sliced = adata[:, :]
    assert sliced.n_obs == 856
    assert sliced.n_vars == 683


def test_slice_mask_hidden_in_result() -> None:
    sliced = _cells_adata()[:50, :]
    assert "__mask__" not in sliced.obs


def test_slice_obs_vectors_preserved() -> None:
    sliced = _cells_adata()[:10, :]
    assert "donor" in sliced.obs
    assert len(sliced.obs["donor"]) == 10


def test_slice_wrong_index_count_raises() -> None:
    with assert_raises("DafAnnData slicing requires two indices"):
        _ = _cells_adata()[:50]


def test_query_obs() -> None:
    adata = _metacells_adata()
    # metacell axis has a "type" vector; filter by the first observed type value
    first_type = str(adata.obs["type"].iloc[0])
    filtered = adata.query_obs(f"type = {first_type}")
    assert 1 <= filtered.n_obs < adata.n_obs
    assert filtered.n_vars == adata.n_vars


def test_query_var() -> None:
    adata = _cells_adata()
    # 438 out of 683 genes are lateral; use boolean filter
    filtered = adata.query_var("is_lateral")
    assert filtered.n_vars == 438
    assert filtered.n_obs == adata.n_obs


def test_readonly_obs_write_raises() -> None:
    sliced = _cells_adata()[:10, :]
    with assert_raises("DafReadOnly"):
        sliced.obs["new_col"] = np.zeros(10)


def test_readonly_var_write_raises() -> None:
    sliced = _cells_adata()[:, :10]
    with assert_raises("DafReadOnly"):
        sliced.var["new_col"] = np.zeros(10)


def test_readonly_uns_write_raises() -> None:
    sliced = _cells_adata()[:10, :]
    with assert_raises("DafReadOnly"):
        sliced.uns["new_key"] = "value"


def test_readonly_layers_write_raises() -> None:
    sliced = _cells_adata()[:10, :]
    with assert_raises("DafReadOnly"):
        sliced.layers["new"] = np.zeros((10, 683))


def test_readonly_obsp_write_raises() -> None:
    sliced = _metacells_adata()[:3, :]
    with assert_raises("DafReadOnly"):
        sliced.obsp["sim"] = np.zeros((3, 3))


def test_readonly_obsm_write_raises() -> None:
    sliced = _metacells_adata()[:3, :]
    with assert_raises("DafReadOnly"):
        sliced.obsm["type:w"] = np.zeros((3, 4))


def test_repr_does_not_raise() -> None:
    r = repr(_cells_adata())
    assert isinstance(r, str)


def test_repr_metacells_does_not_raise() -> None:
    r = repr(_metacells_adata())
    assert isinstance(r, str)


def test_repr_chain_does_not_raise() -> None:
    # exercises the varm branch in __repr__
    r = repr(_chain_adata())
    assert isinstance(r, str)


def test_repr_with_layers_does_not_raise() -> None:
    # exercises the layers branch in __repr__
    adata = _cells_adata()
    adata.layers["extra"] = np.zeros((856, 683), dtype=np.float32)
    r = repr(adata)
    assert "extra" in r


def test_slice_ellipsis_obs() -> None:
    # Ellipsis treated as "all" → no filtering
    adata = _cells_adata()
    sliced = adata[..., :]
    assert sliced.n_obs == 856


def test_slice_all_true_bool_array() -> None:
    # Bool array that's all-True → _to_bool_mask returns None → no-filter path
    adata = _cells_adata()
    all_true = np.ones(856, dtype=bool)
    sliced = adata[all_true, :]
    assert sliced.n_obs == 856


def test_slice_int_numpy_array() -> None:
    # Integer numpy array index
    adata = _cells_adata()
    idx = np.array([0, 1, 2, 3, 4])
    sliced = adata[idx, :]
    assert sliced.n_obs == 5


def test_slice_empty_list() -> None:
    # Empty list → zero observations
    adata = _cells_adata()
    sliced = adata[[], :]
    assert sliced.n_obs == 0


def test_slice_unsupported_type_raises() -> None:
    with assert_raises("unsupported index type"):
        _ = _cells_adata()[{"set": "index"}, :]


def test_slice_none_obs() -> None:
    # None index → no filtering
    adata = _cells_adata()
    sliced = adata[None, :]
    assert sliced.n_obs == 856


def test_obs_setitem_series_noncategorical() -> None:
    # pd.Series (non-categorical) goes through .values path
    adata = _cells_adata()
    series = pd.Series(np.arange(856, dtype=np.int32))  # type: ignore
    adata.obs["int_col"] = series
    assert adata.obs["int_col"].iloc[0] == 0


def test_obs_setitem_series_categorical() -> None:
    # pd.Series with CategoricalDtype goes through astype(str) path
    adata = _cells_adata()
    cat_series = pd.Series(pd.Categorical(["X", "Y"] * 428))  # type: ignore
    adata.obs["cat_series"] = cat_series
    assert adata.obs["cat_series"].iloc[0] in ("X", "Y")


def test_obs_setitem_object_array() -> None:
    # ndarray with object dtype → converted to string array
    adata = _cells_adata()
    obj_arr = np.array(["a", "b"] * 428, dtype=object)
    adata.obs["obj_col"] = obj_arr
    assert adata.obs["obj_col"].iloc[0] in ("a", "b")


def test_obs_getitem_hidden_key_raises() -> None:
    # Accessing the mask vector via obs[] must raise KeyError
    with assert_raises("__mask__"):
        _ = _cells_adata().obs["__mask__"]


def test_obs_getitem_list_with_hidden_key_raises() -> None:
    with assert_raises("__mask__"):
        _ = _cells_adata().obs[["donor", "__mask__"]]


def test_obs_getitem_invalid_key_type_raises() -> None:
    with assert_raises(""):
        _ = _cells_adata().obs[42]


def test_obs_delitem_hidden_key_raises() -> None:
    with assert_raises("__mask__"):
        del _cells_adata().obs["__mask__"]


def test_layers_iter() -> None:
    adata = _cells_adata()
    adata.layers["layer_a"] = np.zeros((856, 683), dtype=np.float32)
    adata.layers["layer_b"] = np.zeros((856, 683), dtype=np.float32)
    names = list(adata.layers)
    assert "layer_a" in names
    assert "layer_b" in names


def test_layers_items() -> None:
    adata = _cells_adata()
    adata.layers["lx"] = np.ones((856, 683), dtype=np.float32)
    items = dict(adata.layers.items())
    assert "lx" in items
    assert items["lx"].shape == (856, 683)


def test_layers_repr() -> None:
    adata = _cells_adata()
    adata.layers["ly"] = np.zeros((856, 683), dtype=np.float32)
    r = repr(adata.layers)
    assert "ly" in r


def test_layers_delete_x_raises() -> None:
    with assert_raises("UMIs"):
        del _cells_adata().layers["UMIs"]


def test_uns_update() -> None:
    adata = _cells_adata()
    adata.uns.update({"key1": 1, "key2": "two"})
    assert adata.uns["key1"] == 1
    assert adata.uns["key2"] == "two"


def test_uns_items() -> None:
    items = dict(_cells_adata().uns.items())
    assert items["organism"] == "human"
    assert items["reference"] == "test"


def test_uns_repr() -> None:
    r = repr(_cells_adata().uns)
    assert "organism" in r
    assert "human" in r


def test_uns_setter_bulk() -> None:
    # DafAnnData.uns = mapping bulk-sets all entries
    adata = _cells_adata()
    adata.uns = {"batch": "run1", "version": 2}
    assert adata.uns["batch"] == "run1"
    assert adata.uns["version"] == 2


def test_obsp_items() -> None:
    items = dict(_metacells_adata().obsp.items())
    assert "edge_weight" in items
    assert items["edge_weight"].shape == (7, 7)


def test_obsp_repr() -> None:
    r = repr(_metacells_adata().obsp)
    assert "edge_weight" in r


def test_varm_items() -> None:
    items = dict(_chain_adata().varm.items())
    assert "metacell:fraction" in items


def test_varm_repr() -> None:
    r = repr(_chain_adata().varm)
    assert "metacell:fraction" in r


def test_obsm_contains_bad_key_format() -> None:
    # Invalid key format → __contains__ returns False (not raises)
    assert "no_colon" not in _cells_adata().obsm


def test_obsm_contains_excluded_axis() -> None:
    # gene is an excluded axis for cells obsm
    assert "gene:UMIs" not in _cells_adata().obsm


def test_obsm_delitem_excluded_axis_raises() -> None:
    adata = _metacells_adata()
    with assert_raises(""):
        del adata.obsm["gene:fraction"]


def test_obs_bulk_setter() -> None:
    adata = _cells_adata()
    df = pd.DataFrame({"score": np.arange(856, dtype=np.float32)}, index=adata.obs_names)
    adata.obs = df
    assert adata.obs["score"].iloc[0] == 0.0


def test_var_bulk_setter() -> None:
    adata = _cells_adata()
    df = pd.DataFrame({"weight": np.ones(683, dtype=np.float32)}, index=adata.var_names)
    adata.var = df
    assert float(adata.var["weight"].iloc[0]) == 1.0


def test_layers_bulk_setter() -> None:
    adata = _cells_adata()
    adata.layers = {"bulk_layer": np.zeros((856, 683), dtype=np.float32)}
    assert "bulk_layer" in adata.layers
