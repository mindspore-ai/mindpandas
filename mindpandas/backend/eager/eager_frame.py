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
"""
Module that provides an EagerFrame to execution eager operations on contained partitions.
"""
import copy as copy_module
import math
from collections import OrderedDict, defaultdict

import numpy as np
import pandas
from pandas._libs.lib import is_list_like
from pandas.core.indexes.api import ensure_index

import mindpandas as mpd
import mindpandas.internal_config as i_config
from mindpandas.backend.base_frame import BaseFrame
from mindpandas.backend.eager.multiprocess_operators import MultiprocessOperator as mp_ops
from mindpandas.backend.eager.multithread_operators import MultithreadOperator as mt_ops
from mindpandas.backend.eager.partition_operators import SinglethreadOperator as st_ops
from mindpandas.index import is_range_like, compute_sliced_len
from mindpandas.util import hashable, is_full_grab_slice
# from .eager_backend import Partition
from .partition import Partition as MultithreadPartition
from .ds_partition import DSPartition


class EagerFrame(BaseFrame):
    """
    Backend frame for executing eager operations on internal pandas partitions.
    """
    def __init__(
            self,
            partitions=None,
            index=None,
            columns=None,
            row_lengths=None,
            column_widths=None,
            dtypes=None,
    ):
        """
        Parameters:
        ----------
        partitions: numpy 2D array
        """
        if i_config.get_adaptive_concurrency() and partitions is not None:
            if isinstance(partitions[0][0], DSPartition):
                self.ops = mp_ops
                self.default_partition = DSPartition
                self.default_partition_shape = mpd.config.get_adaptive_partition_shape('multiprocess')
            else:
                self.ops = mt_ops
                self.default_partition = MultithreadPartition
                self.default_partition_shape = mpd.config.get_adaptive_partition_shape('multithread')
        else:
            if i_config.get_concurrency_mode() == "multiprocess":
                self.ops = mp_ops
                self.default_partition = DSPartition
            elif i_config.get_concurrency_mode() == "multithread":
                self.ops = mt_ops
                self.default_partition = MultithreadPartition
            else:
                self.ops = st_ops
                self.default_partition = MultithreadPartition
            self.default_partition_shape = i_config.get_partition_shape()

        # create an empty frame
        if partitions is None:
            part = self.default_partition.put(data=pandas.DataFrame(), coord=(0, 0))
            self.partitions = np.array([[part]])
        else:
            self.partitions = partitions
        self._index_cache = index
        self._columns_cache = columns
        self._partition_rows_cache = row_lengths
        self._partition_cols_cache = column_widths
        self._dtypes = dtypes


    @property
    def dtypes(self):
        if self._dtypes is None:
            self._dtypes = self.dtypes_from_partitions()
        return self._dtypes

    @property
    def _partition_rows(self):
        """
        get rows lengths for partitions
        """
        if self._partition_rows_cache is None:
            self._partition_rows_cache = self._get_row_lengths_of_partitions()
        return self._partition_rows_cache

    @property
    def _partition_cols(self):
        """
        get columns width for partitions.
        """
        if self._partition_cols_cache is None:
            self._partition_cols_cache = self._get_column_widths_of_partitions()
        return self._partition_cols_cache

    @property
    def _partition_axes(self):
        return [self._partition_rows, self._partition_cols]

    def _validate_axis(self, new_labels, old_labels):
        """
        validate new labels
        """
        new_labels = ensure_index(new_labels)
        original_length, new_length = len(old_labels), len(new_labels)
        if original_length != new_length:
            raise ValueError(
                "Length does not match: Expected axis has %d elements, "
                "new values have %d elements" % (original_length, new_length)
            )
        return new_labels

    _index_cache = None
    _columns_cache = None

    def _get_eagerframe_index(self):
        """
        Get the index from the cache object.
        """
        if self._index_cache is not None:
            return self._index_cache
        return self._index_from_partitions()

    def _get_eagerframe_columns(self):
        """
        Get the columns from the cache object.
        """
        if self._columns_cache is not None:
            return self._columns_cache
        return self._columns_from_partitions()

    def _set_eagerframe_index(self, new_index):
        """
        Replace the current row labels with new labels, and synchronize partition labels lazily
        """
        if self._index_cache is None:
            self._index_cache = ensure_index(new_index)
        else:
            new_index = self._validate_axis(new_index, self._index_cache)
            self._index_cache = new_index
        self.synchronize_axes(axis=0)

    def _set_eagerframe_columns(self, new_columns):
        """
        Replace the current column labels with new labels,  and synchronize partition labels lazily
        """
        if self._columns_cache is None:
            self._columns_cache = ensure_index(new_columns)
        else:
            new_columns = self._validate_axis(new_columns, self._columns_cache)
            self._columns_cache = new_columns
            if self._dtypes is not None:
                self._dtypes.index = new_columns
        self.synchronize_axes(axis=1)

    columns = property(_get_eagerframe_columns, _set_eagerframe_columns)
    index = property(_get_eagerframe_index, _set_eagerframe_index)

    @property
    def num_rows(self):
        '''Gets number of rows in frame.'''
        row_count = len(self.index)
        return row_count

    @property
    def num_cols(self):
        '''Gets number of columns in frame.'''
        col_count = len(self.columns)
        return col_count

    @property
    def shape(self):
        '''Get shape of frame.'''
        return self.num_rows, self.num_cols

    @property
    def partition_shape(self):
        '''Get the shape of partitions.'''
        return self.partitions.shape

    @property
    def axes(self):
        '''Get the axes of the frame.'''
        return [self.index, self.columns]

    def _remove_empty_partitions(self):
        """Remove empty partitions from `self.partitions` to avoid triggering excess computation."""
        if not self.axes[0].size or not self.axes[1].size:
            return
        self.partitions = np.array(
            [
                [
                    self.partitions[r][c]
                    for c in range(len(self.partitions[r]))
                    if c < len(self._partition_cols) and self._partition_cols[c] != 0
                ]
                for r in range(len(self.partitions))
                if r < len(self._partition_rows) and self._partition_rows[r] != 0
            ]
        )
        self._partition_cols_cache = [w for w in self._partition_cols if w != 0]
        self._partition_rows_cache = [r for r in self._partition_rows if r != 0]

    def synchronize_axes(self, axis=None):
        """
        Synchronize labels by applying the index object lazily.
        Append `set_axis` function to func_queue of each partition

        """
        self._remove_empty_partitions()
        if axis is None or axis == 0:
            cumulative_row_len = np.cumsum([0] + self._partition_rows)
        if axis is None or axis == 1:
            cumulative_column_wid = np.cumsum([0] + self._partition_cols)

        if axis is None:
            def set_axis_func(df, index, columns):
                return df.set_axis(index, axis="index", inplace=False).set_axis(
                    columns, axis="columns", inplace=False)
            self.partitions = np.array(
                [[self.partitions[i][j].append_func(
                    set_axis_func,
                    index=self.index[slice(cumulative_row_len[i],
                                           cumulative_row_len[i + 1])],
                    columns=self.columns[slice(cumulative_column_wid[j],
                                               cumulative_column_wid[j + 1])],)
                  for j in range(len(self.partitions[i]))]
                 for i in range(len(self.partitions))])

        elif axis == 0:
            def set_axis_func(df, index):
                return df.set_axis(index, axis="index", inplace=False)
            self.partitions = np.array(
                [[self.partitions[i][j].append_func(
                    set_axis_func,
                    index=self.index[slice(cumulative_row_len[i],
                                           cumulative_row_len[i + 1])],)
                  for j in range(len(self.partitions[i]))]
                 for i in range(len(self.partitions))])

        elif axis == 1:
            def set_axis_func(df, columns):
                return df.set_axis(columns, axis="columns", inplace=False)
            self.partitions = np.array(
                [[self.partitions[i][j].append_func(
                    set_axis_func,
                    columns=self.columns[slice(cumulative_column_wid[j],
                                               cumulative_column_wid[j + 1])],)
                  for j in range(len(self.partitions[i]))]
                 for i in range(len(self.partitions))])

    def _index_from_partitions(self, partitions=None):
        if partitions is None:
            partitions = self.partitions

        index0 = partitions[0, 0].index
        index = index0
        for part in partitions[1:, 0]:
            index = index.append(part.index)
        return index

    def set_index(self, new_labels):
        '''Set index of frame.'''
        output_partitions = self.ops.set_index(self.partitions, new_labels)
        return EagerFrame(output_partitions)

    def setitem_elements(self, set_item_fn, *args):
        """
        pick up the specific partitions and update these partitions only to improve performance
        """
        assert len(args) == 3
        row_positions = args[1]
        col_positions = args[0]
        value = args[2]
        # if row_positions is not in order, need to re-order value as well
        if (not isinstance(row_positions, range) and len(row_positions) >= 1
                and not np.all(row_positions[1:] > row_positions[:-1])):
            if not isinstance(row_positions, np.ndarray):
                row_positions = np.array(row_positions)
            rows_index_argsort = row_positions.argsort()
            value_sorted = value
            if not np.isscalar(value):
                value_sorted = value[rows_index_argsort]
        else:
            value_sorted = value
        part_col_locs = self.get_dict_of_partition_index(1, col_positions)
        part_row_locs = self.get_dict_of_partition_index(0, row_positions)
        item = value_sorted
        ## if item is not scalar or NoneType, batch item to match rows/cols in each partition
        if not np.isscalar(item) and not isinstance(item, type(None)):
            pre_r = 0
            item = [[0]*len(part_col_locs) for _ in range(len(part_row_locs))]
            ##TODO: need optimization  if item is dataframe/series having same partitions
            for i, value_r in enumerate(list(part_row_locs.values())):
                pre_c = 0
                for j, value_c in enumerate(list(part_col_locs.values())):
                    if isinstance(value_r, slice):
                        len_value_r = compute_sliced_len(value_r, self.partitions[i][j].num_rows)
                    else:
                        len_value_r = len(value_r)
                    if isinstance(value_c, slice):
                        len_value_c = compute_sliced_len(value_c, self.partitions[i][j].num_cols)
                    else:
                        len_value_c = len(value_c)
                    item[i][j] = value_sorted[pre_r:pre_r + len_value_r, pre_c:pre_c + len_value_c]
                    pre_c += len_value_c
                pre_r += len_value_r
        output_partitions = self.ops.setitem_elements(self.partitions, set_item_fn, part_row_locs, part_col_locs, item)
        return EagerFrame(output_partitions)

    def append(self, other):
        '''Append other partitions to frame.'''
        df_row_split_points = self.get_axis_split_points(axis=1)
        df_in = self.axis_repartition(axis=1, mblock_size=i_config.get_min_block_size(),
                                      by='split_pos', by_data=df_row_split_points)
        df_other = other.axis_repartition(axis=1, mblock_size=i_config.get_min_block_size(),
                                          by='split_pos', by_data=df_row_split_points)
        new_part = np.append(df_in.partitions, df_other.partitions, axis=0)
        # update the coord of the appended new partitions
        self.ops.reset_coord(new_part)
        return EagerFrame(new_part)

    def _columns_from_partitions(self, partitions=None):
        """Get columns from partitions"""
        if partitions is None:
            partitions = self.partitions
        column0 = partitions[0, 0].columns
        if column0 is None:  # Series has no column
            return None
        columns = column0
        for _, part in enumerate(partitions[0, 1:]):
            columns = columns.append(part.columns)
        return columns

    def index_names_from_partitions(self):
        '''Get index_names from partitions.'''
        index = self.partitions[0, 0].index
        return index.names if isinstance(index, pandas.MultiIndex) else index.name

    def get_columns_with_pred_2(self, columns, lhs, op, rhs, func=None):
        """
        This code is not enable. It implements an alternative of `get_columns_with_pred` by applying
        the predicate in parallel. We did not see any performance benefit from running in multi-threading
        environment.
        """
        all_columns = self._columns_from_partitions()
        if all_columns is None:
            column_ids = None
        else:
            column_ids = []
            try:
                if isinstance(columns, (list, np.ndarray, pandas.Index, pandas.Series, mpd.Series)):
                    column_ids = all_columns.get_indexer_for(columns)
                    column_ids.sort()
                elif hashable(columns):
                    all_columns_list = all_columns.to_list()
                    column_ids = [all_columns_list.index(columns)]
            except TypeError as exc:
                raise TypeError(f'type {type(columns)} is not a supported column type') from exc

        # Input: predicate
        # Output: N partitions of an 1-column DataFrame containing row ids
        pred_eval_func = PredEval(lhs, op, rhs)
        row_id_parts: self.default_partition = self.ops.map(self.partitions, pred_eval_func)
        # Consolidate the row ids from all partitions into a single 1-column DataFrame
        row_ids = []
        for row_part in row_id_parts:
            for part in row_part:
                # N rows, 1 column (N >= 0)
                row_ids += list(part.data.values[:, 0])

        output_eagerframe = self.view(rows_index_numeric=row_ids,
                                      columns_index_numeric=column_ids,
                                      func=func)
        return output_eagerframe

    def apply_predicate(self, lhs, op, rhs):
        '''Apply a predicate op to frame.'''
        rows_ids = []
        expr = 'df.' + lhs + '.values.__' + op + '__(' + str(rhs) + ')'
        for row_part in self.partitions:
            for part in row_part:
                df = part.data
                if lhs in df.columns:
                    np_array = eval(expr)
                    rows_ids += list(df.index[np_array])
        return rows_ids

    def get_columns(self, columns, lhs=None, op=None, rhs=None, func=None):
        '''Get columns by a predicate op.'''
        all_columns = self._columns_from_partitions()
        if all_columns is None:
            columns_ids = None
        else:
            columns_ids = []
            try:
                if isinstance(columns, (list, np.ndarray, pandas.Index, pandas.Series, mpd.Series)):
                    columns_ids = all_columns.get_indexer_for(columns)
                    columns_ids = np.array(columns_ids)
                elif hashable(columns):
                    all_columns_list = all_columns.to_list()
                    columns_ids = [all_columns_list.index(columns)]
            except TypeError as exc:
                raise TypeError(f'type {type(columns)} is not a supported column type') from exc
        if lhs is None:
            row_ids = None
        else:
            # Evaluate the predicate
            row_ids = self.apply_predicate(lhs, op, rhs)
        if isinstance(columns_ids, list):
            columns_ids = np.array(columns_ids)
        # change from self.mask() to self.view() to keep the user_defined order
        output_eagerframe = self.view(rows_index_numeric=row_ids,
                                      columns_index_numeric=columns_ids,
                                      func=func)
        return output_eagerframe

    def get_rows(self, indices, indices_numeric=False, func=None):
        '''Get rows by indices.'''
        all_indices = self._index_from_partitions()
        indices_ids = []
        if indices_numeric:
            indices_ids = indices
        elif isinstance(indices, (list, np.ndarray, pandas.Index, pandas.Series, mpd.Series)):
            indices_ids = all_indices.get_indexer_for(indices)
            indices_ids = np.sort(indices_ids)
        elif hashable(indices):
            indices_ids = all_indices.get_indexer_for(indices)
        elif isinstance(indices, slice):
            indices_ids = indices
        # change from self.mask() to self.view() to keep the user_defined order
        output_eagerframe = self.view(rows_index_numeric=indices_ids, columns_index_numeric=None,
                                      func=func)
        return output_eagerframe

    def dtypes_from_partitions(self):
        '''Get data types from partitions.'''
        dtypes = pandas.concat([part.dtypes for part in self.partitions[0, :]])
        return dtypes

    def map(self, map_func, fuse=False, pass_coord=False, repartition=False, **kwargs):
        '''Perform map operation on the frame with a map function.'''
        if self.partitions.shape == (1, 1) and repartition:
            frame = self.repartition(self.default_partition_shape, i_config.get_min_block_size())
        else:
            frame = self
        if frame.ops is mp_ops:
            output_partitions = frame.ops.map(frame.partitions, map_func, fuse=fuse, pass_coord=pass_coord, **kwargs)
        else:
            output_partitions = frame.ops.map(frame.partitions, map_func, pass_coord, **kwargs)
        return EagerFrame(output_partitions)

    def injective_map(self, cond, other, func, is_scalar=False, need_repartition=False):
        '''Perform injective map operation on the frame.'''
        if cond is None:
            cond_partitions = None
        else:
            if need_repartition:
                df_row_split_points = self.get_axis_split_points(axis=0)
                cond_row_split_points = cond.get_axis_split_points(axis=0)
                if not np.array_equal(df_row_split_points, cond_row_split_points):
                    cond = cond.axis_repartition(axis=0, mblock_size=i_config.get_min_block_size(),
                                                 by='split_pos', by_data=df_row_split_points)
            cond_partitions = cond.partitions
        if is_scalar:
            output_partitions = self.ops.injective_map(self.partitions, cond_partitions, other, func, is_scalar)
        else:
            output_partitions = self.ops.injective_map(self.partitions, cond_partitions, other.partitions, func,
                                                       is_scalar)
        return EagerFrame(output_partitions)

    def reduce(self, func, axis=0, concat_axis=None):
        """Apply the given function along axis

        Args:
            func: callable, the function to apply on the dataframe.
            axis: int {0, 1}, the axis to apply the func along.

        Return:
            An EagerFrame.
        """
        output_partitions = self.ops.reduce(self.partitions, func, axis=axis, concat_axis=concat_axis)
        if output_partitions.size == 1 and np.isscalar(output_partitions[0][0].get()):
            return output_partitions[0][0].get()
        if axis == 0 and output_partitions[0, 0].container_type is pandas.Series:
            output_partitions = output_partitions.T
            for row_part in output_partitions:
                for part in row_part:
                    part.coord = (part.coord[1], part.coord[0])
        return EagerFrame(output_partitions)

    def map_reduce(self, map_func, reduce_func, axis=0, concat_axis=None):
        '''Perform map reduce operation with map function and reduce function on the frame.'''
        mapped_partitions = self.ops.map(self.partitions, map_func)
        reduced_partitions = self.ops.reduce(mapped_partitions, reduce_func, axis=axis, concat_axis=concat_axis)

        if axis == 0 and reduced_partitions[0, 0].container_type is pandas.Series:
            reduced_partitions = reduced_partitions.T
            for row_part in reduced_partitions:
                for part in row_part:
                    part.coord = (part.coord[1], part.coord[0])
         # compute new axes information
        new_axes, new_axes_lengths = [0, 0], [0, 0]
        new_axes[axis ^ 1] = ["__unsqueeze_series__"]
        new_axes[axis] = self.axes[axis ^ 1]
        new_axes_lengths[axis ^ 1] = [1]
        new_axes_lengths[axis] = self._partition_axes[axis ^ 1]
        return EagerFrame(reduced_partitions, *new_axes, *new_axes_lengths)

    def repartition(self, output_shape, mblock_size):
        '''Repartition the frame according to output_shape.'''
        output_partitions = self.ops.repartition(self.partitions, output_shape, mblock_size)
        return EagerFrame(output_partitions)

    def get_axis_split_points(self, axis):
        '''Get axis split points of the frame's partitions.'''
        axis_lens = self._partition_rows if axis == 0 else self._partition_cols
        axis_lens = [0] + axis_lens
        axis_lens = np.array(axis_lens)
        split_points = axis_lens.cumsum()
        return split_points

    def get_axis_repart_range(self, axis, by, by_data):
        '''Get axis repartition range.'''
        return self.ops.get_axis_repart_range(self.partitions, axis=axis, by=by, by_data=by_data)

    def axis_repartition(self, axis, mblock_size, by, by_data):
        '''Perform repartition along axis.'''
        output_partitions = self.ops.axis_repartition(self.partitions, axis=axis,
                                                      mblock_size=mblock_size, by=by, by_data=by_data)
        return EagerFrame(output_partitions)

    def to_pandas(self, force_series=False):
        '''Convert EagerFrame to pandas frame.'''
        return self.ops.to_pandas(self.partitions, force_series)

    def values(self):
        '''Get values of frame.'''
        return self.ops.values(self.partitions)

    def flush(self):
        '''Peform flush operation on partitions.'''
        self.partitions = self.ops.flush(self.partitions)

    def to_numpy(self, dtype, copy, na_value):
        '''Convert EagerFrame to numpy.'''
        return self.ops.to_numpy(self.partitions, dtype, copy, na_value)

    def to_csv(self, path_or_buf=None, **kwargs):
        '''Save frame to csv file.'''
        return self.ops.to_csv(self.partitions, path_or_buf, **kwargs)

    def _get_column_widths_of_partitions(self):
        '''Get column widths of frame's partitions.'''
        col_widths = [part.num_cols for part in self.partitions[0]]
        return col_widths

    def _get_row_lengths_of_partitions(self):
        '''Get row lengths of frame's partitions.'''
        row_lengths = [part.num_rows for part in self.partitions[:, 0]]
        return row_lengths

    def get_dict_of_partition_index(self, axis, global_indices, are_indices_sorted=False):
        """
        Get a dictionary of the partitions specified by global indices along axis.

        Parameters:
        axis (int, {0 or 1}): get indices along the rows or columns (0 - rows, 1 - columns)
        indices (list-like): the global indices to convert
        """
        # TODO: Support handling of slices with specified 'step'. For now, converting them into a range
        if isinstance(global_indices, slice) and (global_indices.step is not None and global_indices.step != 1):
            global_indices = range(*global_indices.indices(len(self.axes[axis])))

        if isinstance(global_indices, slice) or (is_range_like(global_indices) and global_indices.step == 1):
            def fast_slices(global_indices):
                # Converting range-like indexer to slice
                global_indices = slice(global_indices.start, global_indices.stop, global_indices.step)
                if is_full_grab_slice(global_indices, sequence_len=len(self.axes[axis])):
                    return OrderedDict(zip(range(self.partitions.shape[axis]),
                                           [slice(None)] * self.partitions.shape[axis],))
                # Empty selection case
                if global_indices.start == global_indices.stop and global_indices.start is not None:
                    return OrderedDict()
                if global_indices.start is None or global_indices.start == 0:
                    last_partition, last_index = list(
                        self.get_dict_of_partition_index(axis, [global_indices.stop]).items()
                        )[0]
                    dict_of_slices = OrderedDict(zip(range(last_partition),
                                                     [slice(None)] * last_partition))
                    dict_of_slices.update({last_partition: slice(last_index[0])})
                    return dict_of_slices
                if global_indices.stop is None or global_indices.stop >= len(self.axes[axis]):
                    first_partition, first_index = list(
                        self.get_dict_of_partition_index(axis, [global_indices.start]).items()
                        )[0]
                    dict_of_slices = OrderedDict({first_partition: slice(first_index[0], None)})
                    num_partitions = np.size(self.partitions, axis=axis)
                    partition_list = range(first_partition + 1, num_partitions)
                    dict_of_slices.update(OrderedDict(zip(partition_list, [slice(None)] * len(partition_list))))
                    return dict_of_slices
                first_partition, first_index = list(
                    self.get_dict_of_partition_index(axis, [global_indices.start]).items()
                    )[0]
                last_partition, last_index = list(
                    self.get_dict_of_partition_index(axis, [global_indices.stop]).items()
                    )[0]
                if first_partition == last_partition:
                    return OrderedDict({first_partition: slice(first_index[0], last_index[0])})
                if last_partition - first_partition == 1:
                    return OrderedDict({first_partition: slice(first_index[0], None),
                                        last_partition: slice(None, last_index[0]),})
                dict_of_slices = OrderedDict({first_partition: slice(first_index[0], None)})
                partition_list = range(first_partition + 1, last_partition)
                dict_of_slices.update(
                    OrderedDict(zip(partition_list, [slice(None)] * len(partition_list)))
                )
                dict_of_slices.update({last_partition: slice(None, last_index[0])})
                return dict_of_slices
            return fast_slices(global_indices)
        if isinstance(global_indices, list):
            # Converting python list to numpy for faster processing
            global_indices = np.array(global_indices, dtype=np.int64)
        negative_mask = np.less(global_indices, 0)
        has_negative = np.any(negative_mask)
        if has_negative:
            global_indices = (
                global_indices.copy()
                if isinstance(global_indices, np.ndarray)
                else np.array(global_indices, dtype=np.int64)
            )
            global_indices[negative_mask] = global_indices[negative_mask] % len(self.axes[axis])

        if has_negative or not are_indices_sorted:
            global_indices = np.sort(global_indices)
        if axis == 0:
            bins = np.array(self._get_row_lengths_of_partitions())
        else:
            bins = np.array(self._get_column_widths_of_partitions())
        cumulative_len = np.append(bins[:-1].cumsum(), np.iinfo(bins.dtype).max)
        partition_ids = np.digitize(global_indices, cumulative_len)
        each_partition_count = np.array([(partition_ids == i).sum() for i in range(len(cumulative_len))]).cumsum()

        def transform_global_index_to_internal(block_index, global_index):
            """Transform global index to internal one for given block (identified by its index)."""
            if not block_index:
                return global_index
            return np.subtract(global_index, cumulative_len[min(block_index, len(cumulative_len) - 1) - 1])

        if each_partition_count[0] > 0:
            first_partition_indices = [
                (0, transform_global_index_to_internal(0, global_indices[slice(each_partition_count[0])]))
            ]
        else:
            first_partition_indices = []
        partition_ids_with_indices = first_partition_indices + [
            (
                i,
                transform_global_index_to_internal(
                    i,
                    global_indices[slice(each_partition_count[i - 1], each_partition_count[i])]
                    )
            )
            for i in range(1, len(each_partition_count))
            if each_partition_count[i] > each_partition_count[i - 1]
        ]

        return OrderedDict(partition_ids_with_indices)

    ##TODO: check the difference with func get_dict_of_partition_index
    def get_dict_of_internal_index(self, part_sizes, indices):
        """Get internal index of input indices.

        Args:
            part_sizes (np.ndarray): size of each partition on either axis 0 or 1.
            indices (np.ndarray): array of integers to be transformed to internal indices.

        Returns:
            result_dict (dict): dictionary where the key is partition id and the value is
            np.ndarray of internal indices in that partition.
        """
        result_dict = defaultdict(list)
        if not indices.size:
            result_dict[0] = []
            return result_dict
        part_num = len(part_sizes)
        part_sizes = np.append(0, part_sizes)
        part_sizes_cumsum = part_sizes.cumsum()
        part_sizes_cumsum = np.append(part_sizes_cumsum[:-1], np.iinfo(part_sizes_cumsum.dtype).max)
        part_global_start = part_sizes_cumsum[:-1]
        part_global_end = part_sizes_cumsum[1:]
        partition_ids = np.digitize(indices, part_global_end)
        count_arr = np.bincount(partition_ids, minlength=part_num)
        part_counts = np.repeat(np.arange(part_num), count_arr)
        part_bounds = np.repeat(part_global_start, count_arr)
        internal_ids = np.subtract(indices, part_bounds)
        partition_ids_with_indices = list(zip(part_counts, internal_ids))
        for k, v in partition_ids_with_indices:
            result_dict[k].append(v)
        return result_dict

    def mask(self, rows_index, columns_index, is_series=False, keep_index=False):
        '''Return masked frame specified by rows_index and columns_index.'''
        if np.array_equal(rows_index, []) or np.array_equal(columns_index, []):
            # edge case: if we are removing all partitions, we need to create a new empty partition
            if keep_index:
                return EagerFrame(self.partitions, self.index, self.columns)
            part = self.default_partition.put(data=pandas.DataFrame(), coord=(0, 0))
            arr = np.array([[part]])
            return EagerFrame(arr)

        num_rows, num_cols = self.partitions.shape
        if rows_index is not None:
            rows_partition_index_dict = self.get_dict_of_partition_index(0, rows_index)
        else:
            rows_partition_index_dict = {i: slice(None) for i in range(num_rows)}

        if columns_index is not None:
            columns_partition_index_dict = self.get_dict_of_partition_index(1, columns_index)
        else:
            columns_partition_index_dict = {i: slice(None) for i in range(num_cols)}

        output_partitions = self.ops.mask(self.partitions, rows_index, columns_index, rows_partition_index_dict,
                                          columns_partition_index_dict, is_series)
        return EagerFrame(output_partitions)

    def rolling_map(self, map_func, **kwargs):
        '''Perform rolling window map operation on frame with map function.'''
        axis = kwargs.get('axis')
        output_partitions = self.axis_partition(axis)
        mapped_partitions = self.ops.map(output_partitions, map_func, **kwargs)
        return EagerFrame(mapped_partitions)

    def axis_partition(self, axis):
        '''Perform partition along axis.'''
        output_partitions = self.ops.axis_partition(self.partitions, axis=axis)
        return output_partitions

    def filter_dict_func(self, agg_func):
        '''Filter aggregation function of type dictionary.'''
        if not isinstance(agg_func, dict):
            return agg_func
        func_lst = []
        for v in agg_func.values():
            func_lst.append(v)
        return func_lst[0]

    def groupby_reduce(self,
                       axis,
                       keys,
                       as_index,
                       map_func,
                       reduce_func,
                       reset_index_func,
                       apply_indices):
        '''Performs groupby operation on EagerFrame.'''
        if apply_indices is not None:
            numeric_indices = self.axes[axis^1].get_indexer_for(apply_indices)
            apply_indices = list(self.get_dict_of_partition_index(axis^1, numeric_indices).keys())
            partitions = (self.partitions[apply_indices] if axis else self.partitions[:, apply_indices])
            ###update partitions coord
            st_ops.reset_coord(partitions)
        else:
            partitions = self.partitions

        map_func = self.filter_dict_func(map_func)
        reduce_func = self.filter_dict_func(reduce_func)
        mapped_partitions = self.ops.groupby_map(partitions, axis, getattr(keys, "partitions", None), map_func)
        reduced_partitions = self.ops.reduce(mapped_partitions, reduce_func, axis)

        output_frame = None
        if as_index:
            output_frame = EagerFrame(reduced_partitions)
        else:
            # Put index into columns
            rows, cols = reduced_partitions.shape
            output_partitions = np.ndarray(reduced_partitions.shape, dtype=object)
            l_kwargs = {'drop': True}
            for i in range(rows):
                output_partitions[i, 0] = reduced_partitions[i, 0].apply(reset_index_func)
                for j in range(1, cols):
                    output_partitions[i, j] = reduced_partitions[i, j].apply(reset_index_func, **l_kwargs)
            output_frame = EagerFrame(output_partitions)

        return output_frame

    def combine_fold(self, other_dataframe, orig_df_cols, orig_other_cols, fold_func, df_shapes):
        '''Performs fold operation on dataframe from a combine call.'''
        # Repartitions one of the dfs if they do not have the same number of columns
        if df_shapes[0][1] > df_shapes[1][1]:
            other_part_col_size = other_dataframe.partitions[0][0].get().shape[1]
            output_cols = math.ceil(df_shapes[0][1] / other_part_col_size)

            output_shape = (self.partitions.shape[0], output_cols)
            self.partitions = self.ops.repartition(self.partitions, output_shape, i_config.get_min_block_size())
        elif df_shapes[0][1] < df_shapes[1][1]:
            this_part_col_size = self.partitions[0][0].get().shape[1]
            output_cols = math.ceil(df_shapes[1][1] / this_part_col_size)

            output_shape = (other_dataframe.partitions.shape[0], output_cols)
            other_dataframe.partitions = self.ops.repartition(other_dataframe.partitions, output_shape,
                                                              i_config.get_min_block_size())

        output_partitions = self.ops.combine_reduce(self.partitions, other_dataframe.partitions, orig_df_cols,
                                                    orig_other_cols, fold_func)
        return EagerFrame(output_partitions)

    def get_internal_loc(self, loc, axis, allow_append=False):
        """Get the partition location and location inside the partition given a global location.

        Args:
            loc (int): The location in the frame on axis.
            axis (int {0, 1}): The axis to calculate internal location on.
            allow_append (bool, default False): Determines if loc can be the length of index/columns.

        Returns:
            part_loc (int): the indice of row/column partition that loc falls into.
            internal_loc (int): the indice of item inside the partition.

        Raises:
            ValueError: When loc is out of range.
        """
        if loc < 0:
            raise ValueError(f"loc should be 0 or a positive integer, got {loc}")

        part_loc = -1
        part_start = -1
        part_end = 0
        if axis == 0:
            for part in self.partitions[:, 0]:
                part_loc += 1
                part_start = part_end
                part_end += len(part.index)
                if part_start <= loc < part_end:
                    return part_loc, loc - part_start
            if allow_append and loc == part_end:
                return part_loc, loc - part_start
            raise ValueError(f"loc {loc} out of range")
        for part in self.partitions[0]:
            part_loc += 1
            part_start = part_end
            part_end += len(part.columns)
            if part_start <= loc < part_end:
                return part_loc, loc - part_start
        if allow_append and loc == part_end:
            return part_loc, loc - part_start
        raise ValueError(f"loc {loc} out of range")

    def apply_select_indice_axis(self, func, func_concat, indice, axis):
        """Apply func to the row or column partition specified by the indice.

        Args:
            func (callable): The function to apply.
            indice (int): The indice of the partition on this given axis.
            axis (int, {0, 1}): 0 for column partition, 1 for row partition.

        Returns:
            EagerFrame.

        Raises:
            ValueError: If the output has a different number of rows/columns with the input.
        """
        partition_shape = self.partitions.shape
        if axis == 0:
            if self.ops == mp_ops:
                column_df_parts = self.ops.get_select_partitions(self.partitions, indice, axis)
            else:
                column_df_parts = self.partitions[:, indice]
            column_df_parts = [part.get() for part in column_df_parts]
            column_df_lens = self._partition_rows
            column_df = func_concat(column_df_parts, axis=axis)
            len_rows = len(column_df.index)
            output_df = func(column_df)
            # If func is an inplace operation
            if output_df is None:
                output_df = column_df
            if len(output_df.index) != len_rows:
                raise ValueError(f"Index length mismatch, should be {len_rows}, got {len(output_df.index)}")
            pos = 0
            for i in range(partition_shape[0]):
                data = output_df.iloc[pos:pos + column_df_lens[i], :]
                self.partitions[i, indice] = self.default_partition.put(data=data, coord=(i, indice))
                pos += column_df_lens[i]
        else:
            if self.ops == mp_ops:
                row_df_parts = self.ops.get_select_partitions(self.partitions, indice, axis)
            else:
                row_df_parts = self.partitions[indice, :]
            row_df_parts = [part.get() for part in row_df_parts]
            row_df_lens = self._partition_cols
            row_df = func_concat(row_df_parts, axis=axis)
            len_cols = len(row_df.columns)
            output_df = func(row_df)
            # If func is an inplace operation
            if output_df is None:
                output_df = row_df
            if len(output_df.index) != len_cols:
                raise ValueError(f"Column length mismatch, should be {len_cols}, got {len(output_df.columns)}")
            pos = 0
            for i in range(partition_shape[1]):
                data = output_df.iloc[:, pos:pos + row_df_lens[i]]
                self.partitions[indice, i] = self.default_partition.put(data=data, coord=(indice, i))
                pos += row_df_lens[i]
        return self

    def apply_select_indice(self, axis, func, indices, labels, new_index, new_columns, keep_reminding=False):
        """
        Apply a function across an entire axis for a subset of the data

        Parameters:
        axis (int, {0,1}): 0 for column partition, 1 for row partition
        func (callable): The function to apply
        indices (list-like): the numeric indices to apply over, default: None
        labels (list-like): the index/columns keys to apply over, default: None
        new_index (list-like): the new index of the result
        new_columns (list-like): the new columns of the result
        keep_reminding (boolean): whetheror not to drop the data that is not applied over, default: False

        Returns:
            EagerFrame: a new EagerFrame

        """
        assert indices is not None or labels is not None

        if new_index is None:
            new_index = self.index if axis == 1 else None
        if new_columns is None:
            new_columns = self.columns if axis == 0 else None

        old_index = self.index if axis else self.columns
        if labels is not None:
            indices = old_index.get_indexer_for(labels)
        dict_indices = self.get_dict_of_partition_index(axis ^ 1, indices)
        output_partitions = self.ops.axis_partition_to_selected_indices(self.partitions, axis, func, dict_indices,
                                                                        keep_reminding=keep_reminding)
        # overwrite self.partitions
        self.partitions = output_partitions
        lengths_objs = {
            axis: [len(labels)] if not keep_reminding else [self._partition_rows, self._partition_cols][axis],
            axis ^ 1: [self._partition_rows, self._partition_cols][axis ^ 1]}
        return EagerFrame(self.partitions, new_index, new_columns, lengths_objs[0], lengths_objs[1])

    def to_labels(self, column_list: list, func=None):
        '''Get columns specified by column_list as index labels.'''
        extracted_columns = self.get_columns(column_list, func=func).to_pandas()

        if extracted_columns.columns.to_list() != column_list:
            extracted_columns = extracted_columns.reindex(columns=column_list)

        if len(column_list) == 1:
            new_labels = pandas.Index(extracted_columns.squeeze(axis=1))
        else:
            new_labels = pandas.MultiIndex.from_frame(extracted_columns)

        return new_labels

    def squeeze(self, axis=None):
        '''Perform squeeze operation on frame.'''
        output_partitions = self.ops.squeeze(self.partitions, axis=axis)
        return EagerFrame(output_partitions)

    def view(self, rows_index=None, columns_index=None, rows_index_numeric=None,
             columns_index_numeric=None, func=None):
        '''lazily get view of frame specified by rows_index and columns_index.'''
        indexers = []
        if rows_index is not None:
            rows_index_numeric = self.index.get_indexer_for(rows_index)
        if columns_index is not None:
            columns_index_numeric = self.columns.get_indexer_for(columns_index)
        rows_index_numeric_sorted, columns_index_numeric_sorted = None, None
        for axis, idx in enumerate((rows_index_numeric, columns_index_numeric)):
            if is_range_like(idx):
                if idx.step == 1 and len(idx) == len(self.axes[axis]):
                    idx = None
                elif idx is not None and not isinstance(idx, pandas.RangeIndex):
                    idx = pandas.RangeIndex(idx.start, idx.stop, idx.step)
            else:
                if not(idx is None or is_list_like(idx)):
                    raise TypeError(f"View takes list-like numeric indexers, but received: {type(idx)}")
            indexers.append(idx)
        row_positions, col_positions = indexers
        if (col_positions is None and row_positions is None
                and rows_index is None and columns_index is None):
            return self.copy()
        partition_row_lengths = self._get_row_lengths_of_partitions()
        partition_column_lengths = self._get_column_widths_of_partitions()

        row_ascending = True
        def get_row_partitions_list(row_positions, rows_index_numeric):
            rows_index_numeric_sorted = None
            rows_index_numeric_argsort_argsort = None
            row_partitions_list = None
            row_ascending = True
            if is_range_like(row_positions) and row_positions.step > 0 or len(row_positions) == 0:
                rows_index_numeric_sorted = row_positions
            else:
                if len(row_positions) >= 1 and not np.all(row_positions[1:] > row_positions[:-1]):
                    rows_index_numeric_argsort = row_positions.argsort()
                    rows_index_numeric_sorted = row_positions[rows_index_numeric_argsort]
                    rows_index_numeric_argsort_argsort = rows_index_numeric_argsort.argsort()
                    row_ascending = False
                else:
                    rows_index_numeric_sorted = row_positions

            if len(rows_index_numeric) == 0:
                row_partitions_list = self.get_dict_of_internal_index(partition_row_lengths, rows_index_numeric_sorted)
            else:
                row_partitions_list = self.get_dict_of_partition_index(0, rows_index_numeric_sorted,
                                                                       are_indices_sorted=True)
            new_row_lengths = [len(range(*sub_indexer.indices(partition_row_lengths[sub_idx]))
                                   if isinstance(sub_indexer, slice) else sub_indexer)
                               for sub_idx, sub_indexer in row_partitions_list.items()]
            try:
                new_index = self.index[
                    slice(row_positions.start, row_positions.stop, row_positions.step)
                    if is_range_like(row_positions) and row_positions.step > 0
                    else rows_index_numeric_sorted
                ]
            except IndexError:
                raise IndexError(f"positional indexers are out-of-bounds")
            else:
                return (rows_index_numeric_sorted, rows_index_numeric_argsort_argsort,
                        row_partitions_list, row_ascending, new_row_lengths, new_index)

        if row_positions is not None:
            rows_get_out = get_row_partitions_list(row_positions, rows_index_numeric)
            rows_index_numeric_sorted = rows_get_out[0]
            rows_index_numeric_argsort_argsort = rows_get_out[1]
            row_partitions_list = rows_get_out[2]
            row_ascending = rows_get_out[3]
            new_row_lengths = rows_get_out[4]
            new_index = rows_get_out[5]
        else:
            row_partitions_list = {
                i: slice(None) for i in range(len(partition_row_lengths))
            }
            new_row_lengths = partition_row_lengths
            new_index = self.index

        col_ascending = True
        def get_cols_partitions_list(col_positions, columns_index_numeric):
            columns_index_numeric_sorted = None
            columns_index_numeric_argsort_argsort = None
            col_partitions_list = None
            col_ascending = True
            if is_range_like(col_positions) and col_positions.step > 0 or len(col_positions) == 0:
                columns_index_numeric_sorted = col_positions
            else:
                if len(col_positions) >= 1 and not np.all(col_positions[1:] > col_positions[:-1]):
                    columns_index_numeric_argsort = col_positions.argsort()
                    columns_index_numeric_sorted = col_positions[columns_index_numeric_argsort]
                    columns_index_numeric_argsort_argsort = columns_index_numeric_argsort.argsort()
                    col_ascending = False
                else:
                    columns_index_numeric_sorted = col_positions
            if len(columns_index_numeric) == 0:
                col_partitions_list = self.get_dict_of_internal_index(partition_column_lengths,
                                                                      columns_index_numeric_sorted)
            else:
                col_partitions_list = self.get_dict_of_partition_index(1, columns_index_numeric_sorted,
                                                                       are_indices_sorted=True)

            new_col_widths = [len(range(*sub_indexer.indices(partition_column_lengths[sub_idx]))
                                  if isinstance(sub_indexer, slice) else sub_indexer)
                              for sub_idx, sub_indexer in col_partitions_list.items()]
            if is_range_like(col_positions) and col_positions.step > 0:
                col_idx = slice(col_positions.start, col_positions.stop, col_positions.step)
            else:
                col_idx = columns_index_numeric_sorted
            try:
                new_columns = self.columns[col_idx]
            except IndexError:
                raise IndexError(f"positional indexers are out-of-bounds")
            else:
                if self._dtypes is not None:
                    new_dtypes = self.dtypes.iloc[col_idx]
                else:
                    new_dtypes = None
                return (columns_index_numeric_sorted, columns_index_numeric_argsort_argsort,
                        col_partitions_list, col_ascending, new_col_widths, new_columns, new_dtypes)

        if col_positions is not None:
            cols_get_out = get_cols_partitions_list(col_positions, columns_index_numeric)
            columns_index_numeric_sorted = cols_get_out[0]
            columns_index_numeric_argsort_argsort = cols_get_out[1]
            col_partitions_list = cols_get_out[2]
            col_ascending = cols_get_out[3]
            new_col_widths = cols_get_out[4]
            new_columns = cols_get_out[5]
            new_dtypes = cols_get_out[6]
        else:
            col_partitions_list = {
                i: slice(None) for i in range(len(partition_column_lengths))
            }

            new_col_widths = partition_column_lengths
            new_columns = self.columns
            if self._dtypes is not None:
                new_dtypes = self.dtypes
            else:
                new_dtypes = None
        view_parts = self.ops.mask(self.partitions, rows_index_numeric_sorted,
                                   columns_index_numeric_sorted, row_partitions_list,
                                   col_partitions_list, func=func)
        if not row_ascending:
            view_parts = self.ops.reduce(view_parts,
                                         reduce_func=lambda df: df.iloc[rows_index_numeric_argsort_argsort],
                                         axis=0)
        if not col_ascending:
            view_parts = self.ops.reduce(view_parts,
                                         reduce_func=lambda df: df.iloc[:, columns_index_numeric_argsort_argsort],
                                         axis=1)
        return EagerFrame(view_parts, new_index, new_columns, new_row_lengths, new_col_widths, new_dtypes)

    def transpose(self):
        '''Perform transpose operation on EagerFrame.'''
        self.partitions = self.partitions.T
        for row_part in self.partitions:
            for part in row_part:
                part.coord = (part.coord[1], part.coord[0])

    def update(self, new_dataframe):
        '''Perform update operation on EagerFrame.'''
        partition = self.default_partition.put(data=new_dataframe, coord=(0, 0), container_type=type(new_dataframe))
        partition_array = np.array([[partition]])  # single partition
        new_partitions = self.ops.repartition(partition_array,
                                              self.partitions.shape,
                                              i_config.get_min_block_size())
        if self.partitions.shape != new_partitions.shape:
            raise ValueError("Cannot update backend partitions with different shape")
        self.ops.update(self.partitions, new_partitions)
        return EagerFrame(new_partitions, new_dataframe.index, new_dataframe.columns)

    def copy(self, deep=True):
        '''Perform copy operation on EagerFrame.'''
        if not deep:
            output_partitions = copy_module.copy(self.partitions)
        else:
            if (
                    self.partitions is not None and
                    not self.partitions.size and
                    isinstance(self.partitions[0][0], DSPartition)
            ):
                self.ops.flush(self.partitions)
            output_partitions = copy_module.deepcopy(self.partitions)
        return EagerFrame(output_partitions)

    @property
    def series_like(self):
        '''Return whether the EagerFrame contains series-like data.'''
        return self.num_rows == 1 or self.num_cols == 1

    def axis_frame(self, axis):
        '''Get frame partitioned along axis.'''
        output_partitions = self.ops.axis_partition(self.partitions, axis)
        return EagerFrame(output_partitions)

    def remove_empty_rows(self):
        '''Remove empty rows of frame.'''
        output_partitions = self.ops.remove_empty_rows(self.partitions)
        return EagerFrame(output_partitions)

    def validate_partitions(self):
        """validate partitions and remove invalid partitions."""
        output_partitions = self.ops.remove_empty_partitions(self.partitions)
        if output_partitions.size > 0:
            return EagerFrame(output_partitions)
        # empty dataframe
        return EagerFrame(None)

    def append_column(self, appending_frame):
        '''Append column to frame.'''
        df_row_split_points = self.get_axis_split_points(axis=0)
        append_row_split_points = appending_frame.get_axis_split_points(axis=0)
        if not np.array_equal(df_row_split_points, append_row_split_points):
            appending_frame = appending_frame.axis_repartition(axis=0, mblock_size=i_config.get_min_block_size(),
                                                               by='split_pos', by_data=df_row_split_points)
        if self.ops is mp_ops and appending_frame.ops is mt_ops:
            appending_frame.partitions = mt_ops.convert_partitions(appending_frame.partitions)
        new_partitions = np.concatenate([self.partitions, appending_frame.partitions], axis=1)
        # update the coord of the appended new partitions
        self.ops.reset_coord(new_partitions)
        return EagerFrame(new_partitions)

    def broadcast_container_type(self, new_type):
        """Set the container_type of all partitions in the frame.

        Args:
            new_type: The input container type.
        """
        for row in self.partitions:
            for part in row:
                part.container_type = new_type

    @staticmethod
    def _join_index(axis, indices, how, sort):
        """
        Join two indices
        """
        assert isinstance(indices, list)

        # Define a helper function to combine two indices
        def combine_index(left_index, right_index):
            if axis == 1 and how == "outer" and not sort:
                return left_index.union(right_index, sort=False)
            return left_index.join(right_index, how=how, sort=sort)

        is_equal = all(indices[0].equals(index) for index in indices[1:])
        need_join = how is not None and not is_equal

        check_need_indexers = (
            axis == 0
            and not is_equal
            and any(not index.is_unique for index in indices)
        )
        used_indexers = None

        if need_join:
            if len(indices) == 2 and check_need_indexers:
                used_indexers = [None, None]
                joined_index, used_indexers[0], used_indexers[1] = indices[0].join(
                    indices[1], how=how, sort=sort, return_indexers=True
                )
            else:
                joined_index = indices[0]
                for index in indices[1:]:
                    joined_index = combine_index(joined_index, index)
        else:
            joined_index = indices[0].copy()

        if check_need_indexers and used_indexers is None:
            used_indexers = [index.get_indexer_for(joined_index) for index in indices]
        # Create a helper function to reindex df
        def create_reindexer(need_reindex, frame_id):
            if not need_reindex:
                return lambda df: df

            if check_need_indexers:
                assert used_indexers is not None

                return lambda df: df._reindex_with_indexers(
                    {0: [joined_index, used_indexers[frame_id]]},
                    copy=True,
                    allow_dups=True,
                )
            return lambda df: df.reindex(joined_index, axis=axis)
        return joined_index, create_reindexer


    def _copartition(self, axis, other, how, sort, force_repartition=False):
        """
        Copartition two Mindpandas dataframe
        """
        if isinstance(other, type(self)):
            other = [other]

        def get_axis_lengths(partitions, axis):
            if axis:
                return [len(obj.columns) for obj in partitions[0]]
            return [len(obj.index) for obj in partitions.T[0]]

        self_index = self.axes[axis]
        others_index = [o.axes[axis] for o in other]
        # Join self.index with other index.
        # Get reindex function which will be appiled to axis partitions.
        joined_index, create_reindexer = self._join_index(
            axis, [self_index] + others_index, how, sort
        )
        frames = [self] + other
        non_empty_frames_id = [
            i for i, o in enumerate(frames) if o.partitions.size != 0
        ]

        # If all frames are empty, return directly.
        if non_empty_frames_id is None:
            return self.partitions, [o.partitions for o in other], joined_index

        base_frame_id = non_empty_frames_id[0]
        other_frames = frames[base_frame_id + 1 :]

        base_frame = frames[non_empty_frames_id[0]]
        base_index = base_frame.axes[axis]
        # Check if we need to reindex or repartition base frame
        need_reindex_base = not base_index.equals(joined_index)
        do_repartition_base = force_repartition or need_reindex_base

        # If we need to repartition base frame, apply reindex function to axis partitions in base frame first
        # and then repartition base frame
        if do_repartition_base:
            reindexed_base_partitions = base_frame.ops.map_axis_partitions(
                axis,
                create_reindexer(need_reindex_base, base_frame_id),
                base_frame.partitions,
            )
            reindexed_base_partitions = self.ops.repartition(reindexed_base_partitions,
                                                             i_config.get_partition_shape(),
                                                             i_config.get_min_block_size())
        else:
            reindexed_base_partitions = base_frame.partitions

        base_lengths = get_axis_lengths(reindexed_base_partitions, axis)
        others_lengths = [o._partition_axes[axis] for o in other_frames]
        # Check if we need to reindex or repartition other frames
        need_reindex_others = [
            not o.axes[axis].equals(joined_index) for o in other_frames
        ]

        do_repartition_others = [None] * len(other_frames)
        for i in range(len(other_frames)):
            do_repartition_others[i] = (
                force_repartition
                or need_reindex_others[i]
                or others_lengths[i] != base_lengths
            )
        new_list_other = [None] * len(other_frames)
        # If we need to repartition other frames, apply reindex function to axis partitions in other frames first
        # and then repartition other frames
        for i in range(len(other_frames)):
            if not do_repartition_others[i]:
                new_list_other[i] = other_frames[i].partitions
            else:
                new_list_other[i] = other_frames[
                    i
                ].ops.map_axis_partitions(
                    axis,
                    create_reindexer(do_repartition_others[i], base_frame_id + i + 1),
                    other_frames[i].partitions
                    )
                new_list_other[i] = self.ops.repartition(new_list_other[i],
                                                         i_config.get_partition_shape(),
                                                         i_config.get_min_block_size())

        new_frames = (
            [frames[i].partitions for i in range(base_frame_id)]
            + [reindexed_base_partitions]
            + new_list_other
        )
        return new_frames[0], new_frames[1:], joined_index

    def injective_map_with_join(self, func, right_frame):
        """
        Perform operations which need to join frames
        """
        # if two frames have same index, skip copartiton.
        if self.index.equals(right_frame.index):
            left_parts = self.partitions
            right_parts = right_frame.partitions
            new_frame = self.ops.injective_map(left_parts, None, right_parts, func, False)
            return EagerFrame(new_frame)

        # if one of dataframes has index which is mixed type, use original join method
        if (
                'mixed' in self.index.inferred_type or
                'mixed' in right_frame.index.inferred_type or
                self.index.inferred_type != right_frame.index.inferred_type
            ):
            left_parts, right_parts, _ = self._copartition(0, right_frame, "outer", sort=False)
            right_parts = right_parts[0]
            new_frame = self.ops.injective_map(left_parts, None, right_parts, func, False)
            return EagerFrame(new_frame)

        # choose different methods to copartition based on if both frames have range_like index
        # if one of the dataframes has non range index
        row_parts_num = self.partitions.shape[0]
        if  not is_range_like(self.index) or not is_range_like(right_frame.index):
            is_range = False
            left_output_partitions = self.ops.copartition(self.partitions, is_range,
                                                          num_output=row_parts_num)
            right_output_partitions = self.ops.copartition(right_frame.partitions, is_range,
                                                           num_output=row_parts_num)
            new_frame = self.ops.injective_map(left_output_partitions, None, right_output_partitions, func, False)

        # if both left index and right index are range_like index
        else:
            is_range = True
            left_index = self.index
            right_index = right_frame.index
            # if range index step is negative, convert it to positive firstly
            if left_index.step < 0:
                left_index = slice(left_index.stop+left_index.step, left_index.start+left_index.step, -left_index.step)
            if right_index.step < 0:
                right_index = slice(right_index.stop+right_index.ste,
                                    right_index.start+right_index.step, -right_index.step)

            min_index = max(left_index.start, right_index.start)
            max_index = min(left_index.stop, right_index.stop)

            common_index = max_index - min_index
            if row_parts_num < 3:
                index_gap = common_index//(min(left_index.step, right_index.step)*row_parts_num)
            else:
                index_gap = common_index//(min(left_index.step, right_index.step)*(row_parts_num-2))
            # no common index, skip copartition
            if min_index >= max_index:
                left_output_partitions = self.partitions
                right_output_partitions = right_frame.partitions
                new_frame = self.ops.injective_map(left_output_partitions, None, right_output_partitions, func, False)
            else:
                def range_index_included(large_index):
                    small_index_slices = []
                    large_index_slices = []

                    # copartition small index
                    small_index_slices.append(None)
                    if row_parts_num > 3:
                        for i in range(row_parts_num-3):
                            small_index_slices.append(slice(index_gap*i+min_index, min_index+index_gap*(i+1)-1))
                        small_index_slices.append(slice(index_gap*(i+1)+min_index, max_index))
                    else:
                        small_index_slices.append(slice(min_index, max_index))
                    small_index_slices.append(None)

                    # copartition large index
                    large_index_slices.append(slice(large_index.start, min_index-1))
                    if row_parts_num > 3:
                        for i in range(row_parts_num-3):
                            large_index_slices.append(slice(index_gap*i+min_index, min_index+index_gap*(i+1)-1))
                        large_index_slices.append(slice(index_gap*(i+1)+min_index, max_index))
                    else:
                        large_index_slices.append(slice(min_index, max_index))
                    large_index_slices.append(slice(max_index+1, large_index.stop))
                    return small_index_slices, large_index_slices

                def range_index_intersected(down_index, up_index):
                    down_index_slices = []
                    up_index_slices = []
                    down_index_slices.append(None)
                    #copartiton  the "down" index
                    if row_parts_num > 3:
                        for i in range(row_parts_num-3):
                            down_index_slices.append(slice(index_gap*i+min_index, min_index+index_gap*(i+1)-1))
                        down_index_slices.append(slice(index_gap*(i+1)+min_index, max_index))
                    else:
                        down_index_slices.append(slice(min_index, max_index))
                    down_index_slices.append(slice(max_index+1, down_index.stop))
                    # copartition the "up" index
                    up_index_slices.append(slice(up_index.start, min_index-1))
                    if row_parts_num > 3:
                        for i in range(row_parts_num-3):
                            up_index_slices.append(slice(index_gap*i+min_index, min_index+index_gap*(i+1)-1))
                        up_index_slices.append(slice(index_gap*(i+1)+min_index, max_index))
                    else:
                        up_index_slices.append(slice(min_index, max_index))
                    up_index_slices.append(None)
                    return down_index_slices, up_index_slices

                #left index is included by right index
                if min_index == left_index.start and max_index == left_index.stop:
                    left_index_slices, right_index_slices = range_index_included(right_index)
                #right index is included by left index
                elif min_index == right_index.start and max_index == right_index.stop:
                    right_index_slices, left_index_slices = range_index_included(left_index)
                # left index and right index have common index, and left index is the down index
                elif min_index == left_index.start and max_index == right_index.stop:
                    left_index_slices, right_index_slices = range_index_intersected(left_index, right_index)
                # left index and right index have common index, and right index is the down index
                elif min_index == right_index.start and max_index == left_index.stop:
                    right_index_slices, left_index_slices = range_index_intersected(right_index, left_index)
                left_output_partitions = self.ops.copartition(self.partitions,
                                                              is_range, index_slices=left_index_slices)
                right_output_partitions = self.ops.copartition(right_frame.partitions,
                                                               is_range, index_slices=right_index_slices)
                new_frame = self.ops.injective_map(left_output_partitions,
                                                   None, right_output_partitions, func, False)
        return EagerFrame(new_frame)




class PredEval:
    """
    Class for handling predicate evaluations.
    """
    def __init__(self, lhs, op, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = str(rhs)

    def __call__(self, df):
        if self.lhs not in df.columns:
            return pandas.DataFrame({'rowid': []})
        np_array = eval('df.' + self.lhs + '.values.__' + self.op + '__(' + self.rhs + ')')
        row_ids = df.index[np_array]
        result = pandas.DataFrame({'rowid': row_ids})
        return result
