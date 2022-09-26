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

import mindpandas as mpd
import mindpandas.iternal_config as config
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
    def __init__(self, partitions=None):
        """
        Parameters:
        ----------
        partitions: numpy 2D array
        """
        if config.get_adaptive_concurrency() and partitions is not None:
            if isinstance(partitions[0][0], DSPartition):
                self.ops = mp_ops
                self.default_partition = DSPartition
                self.default_partition_shape = config.get_adaptive_partition_shape(config.get_multiprocess_backend())
            else:
                self.ops = mt_ops
                self.default_partition = MultithreadPartition
                self.default_partition_shape = config.get_adaptive_partition_shape('multithread')
        else:
            if config.get_concurrency_mode() == "yr":
                self.ops = mp_ops
                self.default_partition = DSPartition
            elif config.get_concurrency_mode() == "multithread":
                self.ops = mt_ops
                self.default_partition = MultithreadPartition
            else:
                self.ops = st_ops
                self.default_partition = MultithreadPartition
            self.default_partition_shape = config.get_partition_shape()

        # create an empty frame
        if partitions is None:
            part = self.default_partition.put(data=pandas.DataFrame(), coord=(0, 0))
            self.partitions = np.array([[part]])
        else:
            self.partitions = partitions

    @property
    def index(self):
        '''Gets frame's index.'''
        return self.index_from_partitions()

    @property
    def columns(self):
        '''Gets frame's columns.'''
        return self.columns_from_partitions()


    @property
    def _partition_rows(self):
        row_lengths = []
        for first_col in self.partitions.T[0]:
            if first_col.num_rows is None:
                break
            row_lengths.append(first_col.num_rows)
        return row_lengths

    @property
    def _partition_columns(self):
        col_widths = []
        for first_row in self.partitions[0]:
            if first_row.num_cols is None:
                break
            col_widths.append(first_row.num_cols)
        return col_widths

    @property
    def axes_lengths(self):
        return [self._partition_rows, self._partition_columns]

    @property
    def num_rows(self):
        '''Gets number of rows in frame.'''
        row_count = 0
        for first_col in self.partitions.T[0]:
            if first_col.num_rows is None:
                break
            row_count += first_col.num_rows
        return row_count

    @property
    def num_cols(self):
        '''Gets number of columns in frame.'''
        col_count = 0
        for first_row in self.partitions[0]:
            if first_row.num_cols is None:
                break
            col_count += first_row.num_cols
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

    @property
    def dtypes(self):
        '''Get data types in frame.'''
        return self.dtypes_from_partitions()

    def index_from_partitions(self):
        '''Get index from partitions.'''
        index0 = self.partitions[0, 0].index
        index = index0
        for part in self.partitions[1:, 0]:
            index = index.append(part.index)
        return index

    def set_index(self, new_labels):
        '''Set index of frame.'''
        self.ops.set_index(self.partitions, new_labels)

    def setitem_elements(self, set_item_fn, *args):
        """
        pick up the specific partitions and update these partitions only to improve performance
        """
        assert len(args) == 3
        row_positions = args[1]
        col_positions = args[0]
        value = args[2]
        # if row_positions is not in order, need to re-order value as well
        if len(row_positions) >= 1 and not np.all(row_positions[1:] > row_positions[:-1]):
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
                    if isinstance(value_c, slice):
                        len_value_c = compute_sliced_len(value_c, self.partitions[i][j].num_cols)
                    else:
                        len_value_r, len_value_c = len(value_r), len(value_c)
                    item[i][j] = value_sorted[pre_r:pre_r + len_value_r, pre_c:pre_c + len_value_c]
                    pre_c += len_value_c
                pre_r += len_value_r
        output_partitions = self.ops.setitem_elements(self.partitions, set_item_fn, part_row_locs, part_col_locs, item)
        return EagerFrame(output_partitions)

    def append(self, other):
        '''Append other partitions to frame.'''
        df_row_split_points = self.get_axis_split_points(axis=1)
        df_in = self.axis_repartition(axis=1, by='split_pos', by_data=df_row_split_points)
        df_other = other.axis_repartition(axis=1, by='split_pos', by_data=df_row_split_points)
        new_part = np.append(df_in.partitions, df_other.partitions, axis=0)
        # update the coord of the appended new partitions
        self.ops.reset_coord(new_part)
        return EagerFrame(new_part)

    def columns_from_partitions(self):
        '''Get columns from partitions'''
        column0 = self.partitions[0, 0].columns
        if column0 is None:  # Series has no column
            return None
        columns = column0
        for _, part in enumerate(self.partitions[0, 1:]):
            columns = columns.append(part.columns)
        return columns

    def index_names_from_partitions(self):
        '''Get index_names from partitions.'''
        index = self.partitions[0, 0].index
        return index.names if isinstance(index, pandas.MultiIndex) else index.name

    def get_columns_with_pred_2(self, columns, lhs, op, rhs):
        """
        This code is not enable. It implements an alternative of `get_columns_with_pred` by applying
        the predicate in parallel. We did not see any performance benefit from running in multi-threading
        environment.
        """
        all_columns = self.columns_from_partitions()
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

        result = self.mask(row_ids, column_ids)
        return result

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

    def get_columns(self, columns, lhs=None, op=None, rhs=None):
        '''Get columns by a predicate op.'''
        all_columns = self.columns_from_partitions()
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
        output_eagerframe = self.view(row_ids, columns_ids)
        return output_eagerframe

    def get_rows(self, indices, indices_numeric=False):
        '''Get rows by indices.'''
        all_indices = self.index_from_partitions()
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
        output_eagerframe = self.view(indices_ids, None)
        return output_eagerframe

    def dtypes_from_partitions(self):
        '''Get data types from partitions.'''
        dtype0 = self.partitions[0, 0].dtypes
        if dtype0 is None:  # Series has no dtypes
            return None
        dtypes = dtype0
        for _, part in enumerate(self.partitions[0, 1:]):
            dtypes = dtypes.append(part.dtypes)
        return dtypes

    def map(self, map_func, fuse=False, pass_coord=False, repartition=False, **kwargs):
        '''Perform map operation on the frame with a map function.'''
        if self.partitions.shape == (1, 1) and repartition:
            frame = self.repartition(self.default_partition_shape)
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
                    cond = cond.axis_repartition(axis=0, by='split_pos', by_data=df_row_split_points)
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
        return EagerFrame(reduced_partitions)

    def repartition(self, output_shape, mblock_size=config.get_min_block_size()):
        '''Repartition the frame according to output_shape.'''
        output_partitions = self.ops.repartition(self.partitions, output_shape, mblock_size)
        return EagerFrame(output_partitions)

    def get_axis_split_points(self, axis):
        '''Get axis split points of the frame's partitions.'''
        return self.ops.get_axis_split_points(self.partitions, axis)

    def get_axis_repart_range(self, axis, by, by_data):
        '''Get axis repartition range.'''
        return self.ops.get_axis_repart_range(self.partitions, axis=axis, by=by, by_data=by_data)

    def axis_repartition(self, axis, by, by_data):
        '''Perform repartition along axis.'''
        output_partitions = self.ops.axis_repartition(self.partitions, axis=axis, by=by, by_data=by_data)
        return EagerFrame(output_partitions)

    def to_pandas(self, force_series=False):
        '''Convert EagerFrame to pandas frame.'''
        self.ops.flush(self.partitions)
        return self.ops.to_pandas(self.partitions, force_series)

    def values(self):
        '''Get values of frame.'''
        self.ops.flush(self.partitions)
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

    def get_column_widths_of_partitions(self):
        '''Get column widths of frame's partitions.'''
        self.ops.flush(self.partitions)
        col_widths = [part.num_cols for part in self.partitions[0]]
        return col_widths

    def get_row_lengths_of_partitions(self):
        '''Get row lengths of frame's partitions.'''
        self.ops.flush(self.partitions)
        row_lengths = [part.num_rows for part in self.partitions[:, 0]]
        return row_lengths

    def get_dict_of_partition_index(self, axis, indices, are_indices_sorted=False):
        """
        Get a dictionary of the partitions specified by global indices along axis.

        Parameters:
        axis (int, {0 or 1}): get indices along the rows or columns (0 - rows, 1 - columns)
        indices (list-like): the global indices to convert
        """
        # NOTE: reference of Modin source code
        # TODO: Support handling of slices with specified 'step'. For now, converting them into a range
        if isinstance(indices, slice) and (
                indices.step is not None and indices.step != 1
        ):
            indices = range(*indices.indices(len(self.axes[axis])))
        # Fasttrack slices
        if isinstance(indices, slice) or (is_range_like(indices) and indices.step == 1):
            def fast_slices(indices):
                # Converting range-like indexer to slice
                indices = slice(indices.start, indices.stop, indices.step)
                if is_full_grab_slice(indices, sequence_len=len(self.axes[axis])):
                    return OrderedDict(
                        zip(
                            range(self.partitions.shape[axis]),
                            [slice(None)] * self.partitions.shape[axis],
                        )
                    )
                # Empty selection case
                if indices.start == indices.stop and indices.start is not None:
                    return OrderedDict()
                if indices.start is None or indices.start == 0:
                    last_part, last_idx = list(
                        self.get_dict_of_partition_index(axis, [indices.stop]).items()
                    )[0]
                    dict_of_slices = OrderedDict(
                        zip(range(last_part), [slice(None)] * last_part)
                    )
                    dict_of_slices.update({last_part: slice(last_idx[0])})
                    return dict_of_slices
                if indices.stop is None or indices.stop >= len(self.axes[axis]):
                    first_part, first_idx = list(
                        self.get_dict_of_partition_index(axis, [indices.start]).items()
                    )[0]
                    dict_of_slices = OrderedDict({first_part: slice(first_idx[0], None)})
                    num_partitions = np.size(self.partitions, axis=axis)
                    part_list = range(first_part + 1, num_partitions)
                    dict_of_slices.update(
                        OrderedDict(zip(part_list, [slice(None)] * len(part_list)))
                    )
                    return dict_of_slices
                first_part, first_idx = list(
                    self.get_dict_of_partition_index(axis, [indices.start]).items()
                )[0]
                last_part, last_idx = list(
                    self.get_dict_of_partition_index(axis, [indices.stop]).items()
                )[0]
                if first_part == last_part:
                    return OrderedDict({first_part: slice(first_idx[0], last_idx[0])})
                if last_part - first_part == 1:
                    return OrderedDict(
                        # TODO: it might not maintain the order
                        {
                            first_part: slice(first_idx[0], None),
                            last_part: slice(None, last_idx[0]),
                        }
                    )
                dict_of_slices = OrderedDict(
                    {first_part: slice(first_idx[0], None)}
                )
                part_list = range(first_part + 1, last_part)
                dict_of_slices.update(
                    OrderedDict(zip(part_list, [slice(None)] * len(part_list)))
                )
                dict_of_slices.update({last_part: slice(None, last_idx[0])})
                return dict_of_slices
            return fast_slices(indices)
        if isinstance(indices, list):
            # Converting python list to numpy for faster processing
            indices = np.array(indices, dtype=np.int64)
        negative_mask = np.less(indices, 0)
        has_negative = np.any(negative_mask)
        if has_negative:
            # We're going to modify 'indices' inplace in a numpy way, so doing a copy/converting indices to numpy.
            indices = (
                indices.copy()
                if isinstance(indices, np.ndarray)
                else np.array(indices, dtype=np.int64)
            )
            indices[negative_mask] = indices[negative_mask] % len(self.axes[axis])
        # If the `indices` array was modified because of the negative indices conversion
        # then the original order was broken and so we have to sort anyway:
        if has_negative or not are_indices_sorted:
            indices = np.sort(indices)
        if axis == 0:
            bins = np.array(self.get_row_lengths_of_partitions())
        else:
            bins = np.array(self.get_column_widths_of_partitions())
        cumulative = np.append(bins[:-1].cumsum(), np.iinfo(bins.dtype).max)
        partition_ids = np.digitize(indices, cumulative)
        count_for_each_partition = np.array([(partition_ids == i).sum() for i in range(len(cumulative))]).cumsum()

        def internal(block_idx, global_index):
            """Transform global index to internal one for given block (identified by its index)."""
            return (
                global_index
                if not block_idx
                else np.subtract(
                    global_index, cumulative[min(block_idx, len(cumulative) - 1) - 1]
                )
            )

        if count_for_each_partition[0] > 0:
            first_partition_indices = [
                (0, internal(0, indices[slice(count_for_each_partition[0])]))
            ]
        else:
            first_partition_indices = []
        partition_ids_with_indices = first_partition_indices + [
            (
                i, internal(i, indices[slice(count_for_each_partition[i - 1],
                                             count_for_each_partition[i])])
            )
            for i in range(1, len(count_for_each_partition))
            if count_for_each_partition[i] > count_for_each_partition[i - 1]
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
                return EagerFrame(self.partitions)  # Keep the attributes for old partitions
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
            self.partitions = self.ops.repartition(self.partitions, output_shape, config.get_min_block_size())
        elif df_shapes[0][1] < df_shapes[1][1]:
            this_part_col_size = self.partitions[0][0].get().shape[1]
            output_cols = math.ceil(df_shapes[1][1] / this_part_col_size)

            output_shape = (other_dataframe.partitions.shape[0], output_cols)
            other_dataframe.partitions = self.ops.repartition(other_dataframe.partitions, output_shape,
                                                              config.get_min_block_size())

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
            column_df_parts = [part.get() for part in self.partitions[:, indice]]
            column_df_lens = [len(df.index) for df in column_df_parts]
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
            row_df_parts = [part.get() for part in self.partitions[indice, :]]
            row_df_lens = [len(df.columns) for df in row_df_parts]
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

    def apply_select_indice(self, axis, func, labels, indices, new_index, new_columns, keep_reminding=False):
        """
        Apply a function across an entire axis for a subset of the data

        Parameters:
        axis (int, {0,1}): 0 for column partition, 1 for row partition
        func (callable): The function to apply
        labels (list-like): the index/columns keys to apply over, default: None
        indices (list-like): the numeric indices to apply over, default: None
        new_index (list-like): the new index of the result
        new_columns (list-like): the new columns of the result
        keep_reminding (boolean): whetheror not to drop the data that is not applied over, default: False

        Returns:
            EagerFrame: a new EagerFrame

        """
        assert labels is not None or indices is not None
        old_index = self.index if axis else self.columns
        if labels is not None:
            indices = old_index.get_indexer_for(labels)
        dict_indices = self.get_dict_of_partition_index(axis ^ 1, indices)
        output_partitions = self.ops.axis_partition_to_selected_indices(self.partitions, axis, func, dict_indices,
                                                                        keep_reminding=keep_reminding)
        # overwrite self.partitions
        self.partitions = output_partitions
        ## need to be added when we can initialize eager_frame with index and column
        if new_index is None:
            new_index = self.index if axis == 1 else None
        if new_columns is None:
            new_columns = self.columns if axis == 0 else None
        return EagerFrame(self.partitions)

    def to_labels(self, column_list: list):
        '''Get columns specified by column_list as index labels.'''
        extracted_columns = self.get_columns(column_list).to_pandas()

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

    def view(self, rows_index, columns_index):
        '''Get view of frame specified by rows_index and columns_index.'''
        indexers = []
        rows_index_sorted, columns_index_sorted = None, None
        for axis, indexer in enumerate((rows_index, columns_index)):
            if is_range_like(indexer):
                if indexer.step == 1 and len(indexer) == len(self.axes[axis]):
                    indexer = None
                elif indexer is not None and not isinstance(indexer, pandas.RangeIndex):
                    indexer = pandas.RangeIndex(
                        indexer.start, indexer.stop, indexer.step
                    )
            else:
                if not(indexer is None or is_list_like(indexer)):
                    raise TypeError(f"Mask takes only list-like numeric indexers, received: {type(indexer)}")
            indexers.append(indexer)
        row_positions, col_positions = indexers
        if col_positions is None and row_positions is None:
            return self.copy()
        partition_row_lengths = self.get_row_lengths_of_partitions()
        partition_column_lengths = self.get_column_widths_of_partitions()

        row_ascending = True
        def get_row_partitions_list(row_positions, rows_index):
            rows_index_sorted = None
            rows_index_argsort_argsort = None
            row_partitions_list = None
            row_ascending = True
            if is_range_like(row_positions) and row_positions.step > 0 or not row_positions.size:
                rows_index_sorted = row_positions
            else:
                if row_positions.size >= 1 and not np.all(row_positions[1:] > row_positions[:-1]):
                    rows_index_argsort = row_positions.argsort()
                    rows_index_sorted = row_positions[rows_index_argsort]
                    rows_index_argsort_argsort = rows_index_argsort.argsort()
                    row_ascending = False
                else:
                    rows_index_sorted = row_positions

            if not rows_index.size:
                row_partitions_list = self.get_dict_of_internal_index(partition_row_lengths, rows_index_sorted)
            else:
                row_partitions_list = self.get_dict_of_partition_index(0, rows_index_sorted, are_indices_sorted=True)
            return rows_index_sorted, rows_index_argsort_argsort, row_partitions_list, row_ascending

        if row_positions is not None:
            rows_get_out = get_row_partitions_list(row_positions, rows_index)
            rows_index_sorted = rows_get_out[0]
            rows_index_argsort_argsort = rows_get_out[1]
            row_partitions_list = rows_get_out[2]
            row_ascending = rows_get_out[3]
        else:
            row_partitions_list = {
                i: slice(None) for i in range(len(partition_row_lengths))
            }

        col_ascending = True
        def get_cols_partitions_list(col_positions, columns_index):
            columns_index_sorted = None
            columns_index_argsort_argsort = None
            col_partitions_list = None
            col_ascending = True
            if is_range_like(col_positions) and col_positions.step > 0 or not col_positions.size:
                columns_index_sorted = col_positions
            else:
                if col_positions.size >= 1 and not np.all(col_positions[1:] > col_positions[:-1]):
                    columns_index_argsort = col_positions.argsort()
                    columns_index_sorted = col_positions[columns_index_argsort]
                    columns_index_argsort_argsort = columns_index_argsort.argsort()
                    col_ascending = False
                else:
                    columns_index_sorted = col_positions

            if not columns_index.size:
                col_partitions_list = self.get_dict_of_internal_index(partition_column_lengths, columns_index_sorted)
            else:
                col_partitions_list = self.get_dict_of_partition_index(1,
                                                                       columns_index_sorted,
                                                                       are_indices_sorted=True)
            return columns_index_sorted, columns_index_argsort_argsort, col_partitions_list, col_ascending

        if col_positions is not None:
            cols_get_out = get_cols_partitions_list(col_positions, columns_index)
            columns_index_sorted = cols_get_out[0]
            columns_index_argsort_argsort = cols_get_out[1]
            col_partitions_list = cols_get_out[2]
            col_ascending = cols_get_out[3]
        else:
            col_partitions_list = {
                i: slice(None) for i in range(len(partition_column_lengths))
            }

        view_parts = self.ops.mask(self.partitions, rows_index_sorted, columns_index_sorted, row_partitions_list,
                                   col_partitions_list)
        if not row_ascending:
            view_parts = self.ops.reduce(view_parts,
                                         reduce_func=lambda df: df.iloc[rows_index_argsort_argsort],
                                         axis=0)
        if not col_ascending:
            view_parts = self.ops.reduce(view_parts,
                                         reduce_func=lambda df: df.iloc[:, columns_index_argsort_argsort],
                                         axis=1)
        return EagerFrame(view_parts)

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
        new_partitions = self.ops.repartition(partition_array, self.partitions.shape, config.get_min_block_size())
        if self.partitions.shape != new_partitions.shape:
            raise ValueError("Cannot update backend partitions with different shape")
        self.ops.update(self.partitions, new_partitions)

    def copy(self, deep=True):
        '''Perform copy operation on EagerFrame.'''
        if not deep:
            output_partitions = copy_module.copy(self.partitions)
        else:
            # temporary fix, will remove when yr fixs the error
            if (
                    self.partitions is not None and
                    not self.partitions.size and
                    isinstance(self.partitions[0][0], DSPartition) and
                    config.get_multiprocess_backend() == "yr"
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

    def append_column(self, appending_frame):
        '''Append column to frame.'''
        df_row_split_points = self.get_axis_split_points(axis=0)
        append_row_split_points = appending_frame.get_axis_split_points(axis=0)
        if not np.array_equal(df_row_split_points, append_row_split_points):
            appending_frame = appending_frame.axis_repartition(axis=0, by='split_pos', by_data=df_row_split_points)
        if self.ops is mp_ops and appending_frame.ops is mt_ops:
            appending_frame.partitions = mt_ops.convert_partitions(appending_frame.partitions)
        self.partitions = np.concatenate([self.partitions, appending_frame.partitions], axis=1)
        # update the coord of the appended new partitions
        self.ops.reset_coord(self.partitions)

    def broadcast_container_type(self, new_type):
        """Set the container_type of all partitions in the frame.

        Args:
            new_type: The input container type.
        """
        for row in self.partitions:
            for part in row:
                part.container_type = new_type

    @staticmethod
    def _join_index(axis, indexes, how, sort):
        """
        Join two indices
        """
        assert isinstance(indexes, list)

        # Define a helper function to combine two indices
        def combine_index(left_index, right_index):
            if axis == 1 and how == "outer" and not sort:
                return left_index.union(right_index, sort=False)
            return left_index.join(right_index, how=how, sort=sort)

        is_equal = all(indexes[0].equals(index) for index in indexes[1:])
        need_join = how is not None and not is_equal
        # Check if indexers are needed
        need_indexers = (
            axis == 0
            and not is_equal
            and any(not index.is_unique for index in indexes)
        )
        indexers = None

        if need_join:
            if len(indexes) == 2 and need_indexers:
                indexers = [None, None]
                joined_index, indexers[0], indexers[1] = indexes[0].join(
                    indexes[1], how=how, sort=sort, return_indexers=True
                )
            else:
                joined_index = indexes[0]
                for index in indexes[1:]:
                    joined_index = combine_index(joined_index, index)
        else:
            joined_index = indexes[0].copy()

        if need_indexers and indexers is None:
            indexers = [index.get_indexer_for(joined_index) for index in indexes]
        # Create a helper function to reindex df
        def create_reindexer(need_reindex, frame_id):
            if not need_reindex:
                return lambda df: df

            if need_indexers:
                assert indexers is not None

                return lambda df: df._reindex_with_indexers(
                    {0: [joined_index, indexers[frame_id]]},
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
                                                             config.get_partition_shape(),
                                                             config.get_min_block_size())
        else:
            reindexed_base_partitions = base_frame.partitions

        base_lengths = get_axis_lengths(reindexed_base_partitions, axis)
        others_lengths = [o.axes_lengths[axis] for o in other_frames]
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
        reindexed_other_list = [None] * len(other_frames)
        # If we need to repartition other frames, apply reindex function to axis partitions in other frames first
        # and then repartition other frames
        for i in range(len(other_frames)):
            if do_repartition_others[i]:
                reindexed_other_list[i] = other_frames[
                    i
                ].ops.map_axis_partitions(
                    axis,
                    create_reindexer(do_repartition_others[i], base_frame_id + 1 + i),
                    other_frames[i].partitions
                    )
                reindexed_other_list[i] = self.ops.repartition(reindexed_other_list[i],
                                                               config.get_partition_shape(),
                                                               config.get_min_block_size())
            else:
                reindexed_other_list[i] = other_frames[i].partitions

        reindexed_frames = (
            [frames[i].partitions for i in range(base_frame_id)]
            + [reindexed_base_partitions]
            + reindexed_other_list
        )
        return reindexed_frames[0], reindexed_frames[1:], joined_index

    def injective_map_with_join(self, func, right_frame, join_type="outer"):
        """
        Perform operations which need to join frames
        """
        # If self.index equals to right_frame.index, skip copartition and repartition
        if self.index.equals(right_frame.index):
            left_parts = self.partitions
            right_parts = right_frame.partitions
        else:
            left_parts, right_parts, _ = self._copartition(0, right_frame, join_type, sort=False)
            right_parts = right_parts[0]
        new_frame = self.ops.injective_map(left_parts, None, right_parts, func, False)
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
