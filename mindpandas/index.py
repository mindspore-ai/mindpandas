# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like, is_bool, is_scalar, is_integer

import mindpandas as mpd

def _compute_ndim(row_loc, col_loc):
    """
    Compute the number of dimensions of result from locators.
    """

    row_scalar = is_scalar(row_loc) or isinstance(row_loc, tuple)
    col_scalar = is_scalar(col_loc) or isinstance(col_loc, tuple)
    if row_scalar and col_scalar:
        ndim = 0
    elif row_scalar ^ col_scalar:
        ndim = 1
    else:
        ndim = 2
    return ndim

def is_range_like(obj):
    """
    Check if the object is range-like.
    """
    return (
        hasattr(obj, "__iter__")
        and hasattr(obj, "start")
        and hasattr(obj, "stop")
        and hasattr(obj, "step")
    )

def compute_sliced_len(slc, sequence_len):
    """
    Compute length of sliced object.
    """
    return len(range(*slc.indices(sequence_len)))

def get_qc_axis(input_dataframe, axis):
    """
    Get the corresponding dataframe axis, index or columns.
    """
    if axis == 0:
        return input_dataframe.backend_frame.index
    elif axis == 1:
        return input_dataframe.backend_frame.columns
    else:
        raise ValueError("The axis should be 0 or 1.")

class _BaseIndex:
    """
    The base class for indexing operations, e.g. loc and iloc.
    """

    def __init__(self, obj):
        self._df = obj
        self._qc = obj._qc
        self.backend_frame = obj.backend_frame
        self.row_scalar = False
        self.col_scalar = False

    def __getitem__(self, row_ids, col_ids, ndim):
        if isinstance(row_ids, slice):
            if row_ids != slice(None):
                raise TypeError(
                    "Only None-slices are acceptable as a slice argument in masking, got, "
                    + f"received: {type(row_ids)}")
            row_ids = None
        if isinstance(col_ids, slice):
            if col_ids != slice(None):
                raise TypeError(
                    "Only None-slices are acceptable as a slice argument in masking, got, "
                    + f"received: {type(col_ids)}")
            col_ids = None
        need_squeeze = True
        dataframe_result = True
        if ndim == 2:
            need_squeeze = False
        if isinstance(self._df, mpd.Series) and not self.row_scalar:
            dataframe_result = False
            need_squeeze = False
        if isinstance(self._df, mpd.Series):
            axis = 0
            dataframe_result = False
        if ndim == 0:
            axis = None
        else:
            axis = (
                None
                if self.col_scalar and self.row_scalar
                else 1
                if self.col_scalar
                else 0
            )
        result = self.get_index_results(row_ids, col_ids, dataframe_result, need_squeeze, axis)
        return result

    def __setitem__(self, key, row_ids, col_ids, item, axis=None):
        """
        Assign item to located dataset
        Args:
            row_ids (np.ndarray): The array of row ids.
            col_ids (np.ndarray): The array of column ids.
            item: value to assign
            axis: 0, 1 or None
        """
        if isinstance(row_ids, slice):
            row_ids = range(len(get_qc_axis(self._df, 0)))[row_ids]
        if isinstance(col_ids, slice):
            col_ids = range(len(get_qc_axis(self._df, 1)))[col_ids]
        if axis == 0:
            self._df[self._df.columns[col_ids[0]]] = item
        elif axis == 1:
            self._df._set_item(1, self._df.index[row_ids[0]], item)
        else:
            to_shape = len(row_ids), len(col_ids)
            if not is_scalar(item):
                item = self._broadcast_value(row_ids, col_ids, item, to_shape)
            self._qc.setitem_elements(self._df, 'iloc', col_ids, row_ids, item)

    def _broadcast_value(self, row_lookup, col_lookup, value, target_shape):
        """
        This is used to reshape or broadcast value.
        """
        if isinstance(value, (pd.Series, pd.DataFrame, mpd.Series, mpd.DataFrame)):
            reindex_axes = {}
            target_index_values = self._df.index[row_lookup]
            if not target_index_values.equals(value.index):
                reindex_axes["index"] = target_index_values
            if hasattr(value, "columns"):
                target_column_values = self._df.columns[col_lookup]
                if not target_column_values.equals(value.columns):
                    reindex_axes["columns"] = target_column_values
            if reindex_axes:
                value = value.reindex(**reindex_axes)
        try:
            value = np.array(value)
            if np.prod(target_shape) == np.prod(value.shape):
                return value.reshape(target_shape)
            else:
                return np.broadcast_to(value, target_shape)
        except ValueError:
            original_shape = np.array(value).shape
            raise ValueError(
                f"could not broadcast input array from shape {original_shape} into shape "
                + f"{target_shape}"
            )

    def get_index_results(self,
                          row_ids,
                          col_ids,
                          dataframe_result=False,
                          need_squeeze=False,
                          axis=None):
        """
        Get the indexing results.

        Args:
            row_ids (np.ndarray): The array of row ids.
            col_ids (np.ndarray): The array of column ids.

        Returns:
            EagerFrame, the dataframe results.
        """
        if not need_squeeze:
            result = self._qc.view(self, row_ids, col_ids, dataframe_result)
        else:
            result = self._qc.view(self, row_ids, col_ids, dataframe_result).squeeze(axis=axis)
        return result

    def _get_row_and_column_loc(self, key):
        """
        Parse the key to get the row and column index separately.

        Args:
            key (Union[scalar, tuple, callable]): The index key for the dataframe.

        Returns:
            Tuple, the row and column locator.
        """
        row_loc, col_loc = slice(None), slice(None)

        if isinstance(key, tuple):
            if len(key) > 2:
                raise IndexError(f'Too many indexers, expected 1 or 2, got {len(key)}.')

            row_loc = key[0]
            if len(key) == 2:
                col_loc = key[1]
        else:
            row_loc = key

        if callable(row_loc):
            row_loc = row_loc(self._df)
        if callable(col_loc):
            col_loc = col_loc(self._df)
        return row_loc, col_loc, _compute_ndim(row_loc, col_loc)

    def _is_boolean_array(self, x):
        """
        Check whether the input is a boolean array.

        Args:
            x (Union[scalar, list, slice]): The input to check.

        Returns:
            Boolean, whether the input is a boolean array or not.
        """
        return is_list_like(x) and all(map(is_bool, x))

    def _is_integer_slice(self, x):
        """Check if x is an integer slice

        Args:
            x (object): The object to check

        Returns:
            boolean: True if x is a slice object of int, else False
        """
        if not isinstance(x, slice):
            return False
        for val in [x.start, x.stop, x.step]:
            if not (is_integer(val) or val is None):
                return False
        return True

    def _is_integer_array(self, x):
        """Check if x is an array-like object with all elements are integer

        Args:
            x (object): The object to check

        Returns:
            boolean: True if x is an array-like object with all elements are
                     integer, otherwise False
        """
        if not is_list_like(x):
            return False
        if not all([is_integer(i) for i in x]):
            return False
        return True

    def _get_setitem_axis(self, row_ids, col_ids, row_scalar, col_scalar):
        """
        get the axis along which we should do the assignment
        Args:
            row_ids: slice or list.
            col_ids: slice or list.
            row_scalar: if row indexer is scalar
            col_scalar: if col indexer is scalar
        Return:
        int(0/1) or None
        """
        if self._df.shape == (1, 1):
            return None if not (row_scalar ^ col_scalar) else 1 if row_scalar else 0
        def get_axis(axis):
            return get_qc_axis(self._df, 0) if axis == 0 else get_qc_axis(self._df, 1)
        row_ids_len, col_ids_len = [
            len(ids)
            if not isinstance(ids, slice)
            else compute_sliced_len(ids, len(get_axis(i)))
            for i, ids in enumerate([row_ids, col_ids])
        ]

        if col_ids_len == 1 and row_ids_len == 1:
            axis = None
        elif (
            row_ids_len == len(get_qc_axis(self._df, 0))
            and col_ids_len == 1
            and isinstance(self._df, mpd.DataFrame)
        ):
            axis = 0
        elif col_ids_len == len(get_qc_axis(self._df, 1)) and row_ids_len == 1:
            axis = 1
        else:
            axis = None
        return axis


class _Loc(_BaseIndex):
    """
    The class for the loc function.

    For loc, the index for row and column can be one of the following types:
        - A single label, e.g. 'a'
        - A list or array of labels, e.g. ['a', 'b', 'c'] or [True, False, True]
        - A slice object with labels, e.g. 'a':'f'
        - A callable function
    """

    def __getitem__(self, key):
        """
        Access a group of rows and columns by label(s) or boolean array.

        Args:
            key (Union[scalar, tuple, callable]): The row and column index.

        Returns:
            EagerFrame, the dataframe results.
        """
        row_loc, col_loc, ndim = self._get_row_and_column_loc(key)
        self.row_scalar = is_scalar(row_loc)
        self.col_scalar = is_scalar(col_loc)
        row_ids, col_ids = self._get_row_and_column_ids(row_loc, col_loc)
        result = super(_Loc, self).__getitem__(row_ids, col_ids, ndim)
        return result

    def __setitem__(self, key, item):
        """
        Assign value to dataset located by 'key'
        Args:
            key: (Union[scalar, tuple, callable]): The row and column index.
            item: scalar, list, DataFrame or Series
        """
        row_loc, col_loc, _ = self._get_row_and_column_loc(key)
        # when row_loc is scalar and not in existing rows
        if is_scalar(row_loc) and row_loc not in self._df.index:
            index = self._df.index.insert(len(self._df.index), row_loc)
            self._qc.reindex(self._df, labels=index, axis=0)
        # when col_loc is scalar and not in existing columns
        if is_scalar(col_loc) and col_loc not in self._df.columns:
            if isinstance(item, (pd.DataFrame, mpd.DataFrame)):
                new_col = pd.Series(data=item.squeeze(), index=self._df.index)
            else:
                new_col = pd.Series(data=item, index=self._df.index)
            self._df.insert(loc=len(self._df.columns), column=col_loc, value=new_col)
        else:
            row_ids, col_ids = self._get_row_and_column_ids(row_loc, col_loc)
            axis = self._get_setitem_axis(row_ids, col_ids, is_scalar(row_loc), is_scalar(col_loc))
            super(_Loc, self).__setitem__(key, row_ids, col_ids, item, axis)


    def _get_row_and_column_ids(self, row_loc, col_loc):
        """
        Convert the row/column index to a list of row/column numeric ids.
        Args:
            row_loc: The row locator.
            col_loc: The column locator.

        Returns:
            Tuple, the row and column indexing ids.
        """
        row_ids = self._get_ids_from_loc(axis=0, loc=row_loc)
        if isinstance(self._df, mpd.Series):
            col_ids = [0]
        else:
            col_ids = self._get_ids_from_loc(axis=1, loc=col_loc)
        return row_ids, col_ids

    def _get_ids_from_loc(self, axis, loc):
        """
        Get the indexing ids from the loc index.

        Args:
            axis (int): The index axis.
            loc (scalar, list, slice, array): The index value.

        Returns:
            np.ndarray, the loc indexing ids.
        """
        axis_labels = get_qc_axis(self._df, axis)
        if is_scalar(loc):
            loc = np.array([loc])
        if self._is_boolean_array(loc):
            ids = np.flatnonzero(loc)
        elif isinstance(loc, slice):
            # If the index is a slice, get all the index ids for the slice
            if isinstance(loc, slice) and loc == slice(None):
                ids = loc
            else:
                ids = axis_labels.slice_indexer(loc.start, loc.stop, loc.step)
                ids = pd.RangeIndex(
                    start=(
                        ids.start
                        if ids.start >= 0
                        else ids.start + len(axis_labels)
                    ),
                    stop=(
                        ids.stop
                        if ids.stop >= 0
                        else ids.stop + len(axis_labels)
                    ),
                    step=ids.step,
                )
        else:
            if is_list_like(loc) and not isinstance(loc, (np.ndarray, pd.Index)):
                loc = np.array(loc, dtype=axis_labels.dtype)
            ids = axis_labels.get_indexer_for(loc)
            for i in range(len(ids)):
                if ids[i] == -1:
                    raise KeyError(f"label {loc[i]} not found in index")
        if isinstance(ids, pd.Index) and not is_range_like(ids):
            ids = ids.values
        return ids

class _ILoc(_BaseIndex):
    """
    The class for the iloc function.

    For iloc, the index for row and column can be one of the following types:
        - An integer, e.g. 1
        - A list of integers, e.g. [1, 2, 3]
        - A slice object with integers, e.g. 1:3
        - A boolean array, e.g. [True, False, True]
        - A callable function
    """
    def __getitem__(self, key):
        """
        Access a group of rows and columns by integers or boolean array.

        Args:
            key (Union[int, tuple, callable]): The row and column index.

        Returns:
            EagerFrame, the dataframe results.
        """
        row_loc, col_loc, ndim = self._get_row_and_column_loc(key)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)
        self.row_scalar = is_scalar(row_loc)
        self.col_scalar = is_scalar(col_loc)
        row_ids, col_ids = self._get_row_and_column_ids(row_loc, col_loc)
        result =  super(_ILoc, self).__getitem__(row_ids, col_ids, ndim)
        return result

    def __setitem__(self, key, item):
        """
        Assign value to dataset located by 'key'
        Args:
            key: (Union[scalar, tuple, callable]): The row and column index.
            item: scalar, list, DataFrame or Series
        """
        row_loc, col_loc, _ = self._get_row_and_column_loc(key)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)
        row_ids, col_ids = self._get_row_and_column_ids(row_loc, col_loc)
        axis = self._get_setitem_axis(row_ids, col_ids, is_scalar(row_loc), is_scalar(col_loc))
        super(_ILoc, self).__setitem__(key, row_ids, col_ids, item, axis)

    def _get_row_and_column_ids(self, row_loc, col_loc):
        """
        Convert the row/column index to a list of row/column numeric ids.

        Args:
            row_loc (Union[int, list, slice]): The row index.
            col_loc (Union[int, list, slice]): The column index.

        Returns:
            Tuple, the iloc row and column indexing ids.
        """
        row_ids = self._get_ids_from_index(axis=0, loc=row_loc)
        col_ids = self._get_ids_from_index(axis=1, loc=col_loc)

        return row_ids, col_ids

    def _get_ids_from_index(self, axis, loc):
        """
        Get the indexing ids from the iloc index.

        Args:
            axis (int): The index axis.
            loc (int, List, slice, array): The index value.

        Returns:
            np.ndarray, the iloc index ids.
        """
        if is_scalar(loc):
            loc = np.array([loc])
        if isinstance(loc, slice):
            ids = (
                loc
                if loc == slice(None)
                else pd.RangeIndex(
                    *loc.indices(len(get_qc_axis(self._df, axis)))
                )
            )
        elif is_range_like(loc):
            ids = pd.RangeIndex(loc.start, loc.stop, loc.step)
        elif self._is_boolean_array(loc):
            ids = np.flatnonzero(loc)
        else:
            if isinstance(loc, pd.Index):
                loc = loc.values
            elif is_list_like(loc) and not isinstance(loc, np.ndarray):
                loc = np.array(loc, dtype=np.int64)
            if isinstance(loc, np.ndarray) and not (loc < 0).any():
                ids = loc
            else:
                ids = pd.RangeIndex(len(get_qc_axis(self._df, axis)))[loc]
        if isinstance(ids, pd.Index) and not is_range_like(ids):
            ids = ids.values
        return ids

    def _check_dtypes(self, locator):
        """
        Check that `locator` is an integer scalar, integer slice, integer list or array of booleans.
        """
        is_int = is_integer(locator)
        is_int_slice = self._is_integer_slice(locator)
        is_int_arr = self._is_integer_array(locator)
        is_bool_arr = self._is_boolean_array(locator)

        if not any([is_int, is_int_slice, is_int_arr, is_bool_arr]):
            raise TypeError("Cannot index by location index with a non-integer key")
