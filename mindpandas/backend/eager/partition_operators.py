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
Module for performing operations on partitions using Python backend.
"""
import warnings
import numpy as np
import pandas

from .partition import Partition


class SinglethreadOperator:
    """
    Class for providing operations in single thread mode.
    """
    @classmethod
    def map(cls, partitions, map_func, pass_coord=False, **kwargs):
        '''Performs singlethreaded map.'''
        if pass_coord:
            output_partitions = np.array(
                [
                    [part.apply(map_func, coord=(row_idx, col_idx), **kwargs) for col_idx, part in row_parts]
                    for row_idx, row_parts in enumerate(partitions)
                ]
            )
        else:
            output_partitions = np.array(
                [
                    [part.apply(map_func, **kwargs) for part in row_parts]
                    for row_parts in partitions
                ]
            )
        return output_partitions

    @classmethod
    def injective_map(cls, partitions, cond_partitions, other, func, is_scalar):
        '''Performs singlethreaded injective mapping.'''
        def injective_map_no_cond(partitions, other, func, is_scalar):
            if is_scalar:
                output_partitions = np.array(
                    [
                        [partitions[i, j].apply(func, other) for j in range(lcols)]
                        for i in range(lrows)
                    ]
                )
            else:
                if lcols > 1 and other.shape[1] == 1:
                    output_partitions = np.array(
                        [
                            [partitions[i, j].apply(func, other[i][0].get()) for j in range(lcols)]
                            for i in range(lrows)
                        ]
                    )
                else:
                    output_partitions = np.array(
                        [
                            [partitions[i, j].apply(func, other[i][j].get()) for j in range(lcols)]
                            for i in range(lrows)
                        ]
                    )
            return output_partitions

        lrows, lcols = partitions.shape
        if cond_partitions is None:
            return injective_map_no_cond(partitions, other, func, is_scalar)

        is_broadcast = lcols > 1 and cond_partitions.shape[1] == 1
        def injective_map_with_cond(partitions, cond_partitions, other, func, is_scalar):
            if is_scalar:
                if is_broadcast:
                    output_partitions = np.array(
                        [
                            [partitions[i, j].apply(func, cond_partitions[i, 0].get(), other) for j in range(lcols)]
                            for i in range(lrows)
                        ]
                    )
                else:
                    output_partitions = np.array(
                        [
                            [partitions[i, j].apply(func, cond_partitions[i, j].get(), other) for j in range(lcols)]
                            for i in range(lrows)
                        ]
                    )
            else:
                output_partitions = np.array(
                    [
                        [partitions[i, j].apply(func, cond_partitions[i, j].get(),
                                                other[i, j].get()) for j in range(lcols)]
                        for i in range(lrows)
                    ]
                )
            return output_partitions
        return injective_map_with_cond(partitions, cond_partitions, other, func, is_scalar)

    @classmethod
    def reduce(cls, partitions, reduce_func, axis=0, concat_axis=None):
        '''Peform reduce operation in singlethreaded mode.'''
        num_rows, num_cols = partitions.shape
        container_type = partitions[0, 0].container_type
        if concat_axis is None:
            concat_axis = axis
        if axis == 0:
            output_parts = np.ndarray((1, num_cols), dtype=object)
            for j in range(num_cols):
                if container_type is pandas.Series:
                    if concat_axis == 0:
                        data = pandas.concat([part.get() for part in partitions[:, j]], axis=axis)
                    else:
                        data = pandas.concat([part.get().T for part in partitions[:, j]], axis=axis)
                elif container_type is pandas.DataFrame:
                    data = pandas.concat([part.get() for part in partitions[:, j]], axis=axis)
                else:
                    data = [part.get() for part in partitions[:, j]]
                coord = (0, j)
                if isinstance(data, pandas.Series):
                    data = data.to_frame()
                data = reduce_func(data)
                output_parts[0][j] = Partition.put(data=data, coord=coord)
        else:
            output_parts = np.ndarray((num_rows, 1), dtype=object)
            for i in range(num_rows):
                data = pandas.concat([part.get() for part in partitions[i, :]], axis=axis)
                coord = (i, 0)
                if isinstance(data, pandas.Series):
                    data = data.to_frame()
                data = reduce_func(data)
                output_parts[i][0] = Partition.put(data=data, coord=coord)
        return output_parts

    @classmethod
    def combine_reduce(cls, partitions, other_df_partitions, orig_df_cols, orig_other_cols, reduce_func):
        '''Peforms reduce for a combine call in singlethreaded mode.'''
        num_cols = partitions.shape[1]
        other_cols = other_df_partitions.shape[1]

        max_cols = max(num_cols, other_cols)

        def combine_partitions(idx):
            """
            No axis argument since pandas df.combine only applies to columns i.e. axis=0

            Since one of the dfs may have more partitions columns, check if idx surpasses the smaller df

            If yes, apply reduce_func with the smaller df as empty df
            """
            if idx >= num_cols:
                data = pandas.DataFrame([])
                data2 = pandas.concat([part.get() for part in other_df_partitions[:, idx]], axis=0)
                if isinstance(data, pandas.Series):
                    data = data.to_frame()
                if isinstance(data2, pandas.Series):
                    data2 = data2.to_frame()
            elif idx >= other_cols:
                data = pandas.concat([part.get() for part in partitions[:, idx]], axis=0)
                data2 = pandas.DataFrame([])
                if isinstance(data, pandas.Series):
                    data = data.to_frame()
                if isinstance(data2, pandas.Series):
                    data2 = data2.to_frame()
            else:
                data = pandas.concat([part.get() for part in partitions[:, idx]], axis=0)
                data2 = pandas.concat([part.get() for part in other_df_partitions[:, idx]], axis=0)
                if isinstance(data, pandas.Series):
                    data = data.to_frame()
                if isinstance(data2, pandas.Series):
                    data2 = data2.to_frame()

            if idx >= num_cols:
                data = reduce_func(data2, data, orig_other_cols, orig_df_cols)
            else:
                data = reduce_func(data, data2, orig_df_cols, orig_other_cols)

            return Partition.put(data=data)

        output_parts = np.ndarray((1, max_cols), dtype=object)
        for j in range(max_cols):
            output_parts[0][j] = combine_partitions(j)

        return output_parts

    @classmethod
    def to_pandas(cls, partitions, expect_series):
        '''Convert partitions to pandas frame in singlethreaded mode.'''
        if expect_series:
            if partitions.shape[0] == 1:
                pandas_ser = pandas.concat([part.get().squeeze(1) for part in partitions[0, :]])
            else:
                pandas_ser = pandas.concat([part.get().squeeze(1) for part in partitions[:, 0]])
            if not isinstance(pandas_ser, pandas.DataFrame) and pandas_ser.name == '__unsqueeze_series__':
                pandas_ser.name = None
            return pandas_ser

        row_dfs = []
        for row_parts in partitions:
            row_df = []
            for part in row_parts:
                # if the partition is a series, then the column doesn't really exist, we should transpose it back
                if part.container_type is pandas.Series:
                    data = part.get().T
                else:
                    data = part.get()
                row_df.append(data)
            row_dfs.append(pandas.concat(row_df, axis=1))
        pandas_df = pandas.concat(row_dfs)
        return pandas_df

    @classmethod
    def values(cls, partitions):
        '''Get values of partitions in singlethreaded mode.'''
        container_type = partitions[0, 0].container_type
        if container_type is pandas.DataFrame:
            row_dfs = [pandas.concat([part.get() for part in row_parts], axis=1) for row_parts in partitions]
            output_df = pandas.concat(row_dfs)
            return output_df
        output_ser = pandas.concat([part.get() for part in partitions[:, 0]])
        if output_ser.name == '__unsqueeze_series__':
            output_ser.name = None
        return output_ser

    @classmethod
    def axis_partition(cls, partitions, axis):
        """Perform repartitioning along given axis.

        Args:
            partitions: the input partitions.
            axis: {0, 1}
                0 for column partitions, 1 for row partitions.

        Return:
            output_partitions: a numpy array of partitions.
        """
        num_rows, num_cols = partitions.shape
        if axis == 0:
            output_parts = np.ndarray((1, num_cols), dtype=object)
            for j in range(num_cols):
                data = pandas.concat([part.get() for part in partitions[:, j]], axis=axis)
                coord = (0, j)
                output_parts[0][j] = Partition.put(data=data, coord=coord)
        else:
            output_parts = np.ndarray((num_rows, 1), dtype=object)
            for i in range(num_rows):
                data = pandas.concat([part.get() for part in partitions[i, :]], axis=axis)
                coord = (i, 0)
                output_parts[i][0] = Partition.put(data=data, coord=coord)

        return output_parts

    @classmethod
    def to_numpy(cls, partitions, dtype, copy, na_value):
        '''Convert partitions to numpy in singlethreaded mode.'''
        if isinstance(partitions[0, 0].get(), pandas.Series):
            return np.block([part.get().to_numpy(dtype, copy, na_value) for part in partitions[:, 0]])
        return np.block([[part.get().to_numpy(dtype, copy, na_value) for part in row_parts]
                         for row_parts in partitions])

    @classmethod
    def get_column_widths(cls, partitions):
        '''Get column widths in singlethreaded mode.'''
        col_widths = [part.num_cols for part in partitions[0]]
        return col_widths

    @classmethod
    def get_row_lengths(cls, partitions):
        '''Get row lengths in singlethreaded mode.'''
        row_lengths = [part.num_rows for part in partitions[:, 0]]
        return row_lengths

    @classmethod
    def mask(cls,
             partitions,
             rows_index,
             columns_index,
             rows_partition_index_dict,
             columns_partition_index_dict,
             is_series=False):
        '''Mask operation in singlethreaded mode.'''
        # optimize the case where we select all rows
        # when we select all rows or columns the partition_index_dict is
        # {i: slice(None)for i in range(num_rows or num_columns)}
        if rows_index is not None:
            warnings.warn("rows_index not used for single- or multi-threaded mask.")

        if columns_index is not None:
            warnings.warn("columns_index not used for single- or multi-threaded mask.")

        output_partitions = np.array(
            [
                [
                    partitions[row_idx][col_idx].mask(row_internal_indices, col_internal_indices, is_series)
                    for col_idx, col_internal_indices in columns_partition_index_dict.items()
                ]
                for row_idx, row_internal_indices in rows_partition_index_dict.items()
            ]
        )
        ## Since whole partitions could be dropped above, need to make sure coord index is correct now
        cls.reset_coord(output_partitions)

        return output_partitions

    @classmethod
    def groupby_map(cls, partitions, axis, keys_partitions, map_func):
        """Perform repartitioning along given axis

        Args:
            partitions: the masked partitions
            axis: {0 for ‘index’, 1 for ‘columns’}, @classmethod
    default 0
                Split along rows (0) or columns (1).
            keys_partitions: the keys partitions
            map_func:

        Return:
            output_partitions: a numpy array of partitions
        """

        keys_partitions = cls.axis_partition(keys_partitions, axis ^ 1)

        if axis == 0:
            output_partitions = np.array(
                [
                    [part.group_apply(map_func, key_parts[0]) for part in row_parts]
                    for row_parts, key_parts in zip(partitions, keys_partitions)
                ]
            )
        else:
            output_partitions = np.array(
                [
                    [part.group_apply(map_func, key_parts[0]) for part, key_parts in zip(row_parts, keys_partitions)]
                    for row_parts in partitions
                ]
            )
        return output_partitions

    @classmethod
    def reset_coord(cls, partitions):
        '''Reset coordinates in singlethreaded mode.'''
        num_rows, num_cols = partitions.shape
        for r in range(num_rows):
            for c in range(num_cols):
                partitions[r][c].coord = (r, c)

    @classmethod
    def squeeze(cls, partitions, axis=None):
        '''Perform squeeze operation in singlethreaded mode.'''
        output_partitions = np.array([
            [part.squeeze(axis) for part in row_partitions]
            for row_partitions in partitions
        ])
        return output_partitions

    @classmethod
    def partition_pandas(cls, df, row_slices, col_slices, container_type=None):
        '''Derive partitions from pandas df in singlethreaded mode.'''
        if row_slices is None and col_slices is None:
            output_parts = np.array([[Partition.put(data=df, coord=(0, 0), container_type=container_type)]])
            return output_parts

        row_size = len(row_slices) if row_slices is not None else 1
        col_size = len(col_slices) if col_slices is not None else 1
        if col_slices is None:
            output_parts = np.array(
                [[Partition.put(data=df.iloc[rs].copy(), coord=(i, 0), container_type=container_type)] for i, rs in
                 zip(range(row_size), row_slices)]
            )
        elif row_slices is None:
            output_parts = np.array(
                [[Partition.put(data=df.iloc[:, cs].copy(), coord=(0, j), container_type=container_type) for j, cs in
                  zip(range(col_size), col_slices)]]
            )
        else:
            output_parts = np.array(
                [[Partition.put(data=df.iloc[rs, cs].copy(), coord=(i, j), container_type=container_type) for j, cs in
                  zip(range(col_size), col_slices)]
                 for i, rs in zip(range(row_size), row_slices)]
            )
        return output_parts

    @classmethod
    def repartition(cls, parts, output_shape, mblock_size):
        """Perform repartition on partitions, experimental.

        Args:
            parts (np.ndarray): numpy array of Partition objects.
            output_shape (tuple(int, int)): the expected output partition shape.
            mblock_size (int): the minimum block size.

        Returns:
            result (np.ndarray): numpy array of Partition objects.
        """
        if parts.shape == (1, 1):
            input_part = parts[0, 0]
            row_len, col_len = input_part.num_rows, input_part.num_cols

            row_slices, col_slices = cls.get_slicing_plan(row_len, col_len, output_shape, mblock_size)
            if row_slices is None and col_slices is None:
                return parts

            df = input_part.get()
            output_parts = cls.partition_pandas(df, row_slices, col_slices, input_part.container_type)
            return output_parts
        row_size, col_size = output_shape
        cur_rp, cur_cp = parts.shape
        if cur_rp != row_size:
            parts = cls.axis_repartition(parts, axis=0, mblock_size=mblock_size, by_data=output_shape[0])
        if cur_cp != col_size:
            parts = cls.axis_repartition(parts, axis=1, mblock_size=mblock_size, by_data=output_shape[1])
        return parts

    @classmethod
    def get_axis_split_points(cls, parts, axis):
        """Get current partition slicing positions along the axis.

        Args:
            parts (np.ndarray): numpy array of Partition objects.
            axis {0, 1}: the axis to calculate on, 0 for row, 1 for column.

        Returns:
            split_points (np.ndarray): array of slicing positions.
        """
        axis_lens = cls.get_row_lengths(parts) if axis == 0 else cls.get_column_widths(parts)
        axis_lens.insert(0, 0)
        axis_lens = np.array(axis_lens)
        split_points = axis_lens.cumsum()
        return split_points

    @classmethod
    def calc_axis_split_points(cls, axis_len, num_partitions, mblock_size):
        """Calculate partition slicing positions along the axis.

        Args:
            axis_len (int): number of elements along the axis.
                            If axis_len is less than num_partitions, no partition will be performed
            num_partitions (int): the expected number of output partitions.
            mblock_size (int): minimum block size. Only works if axis_len > num_partitions

        Returns:
            new_split_points (np.ndarray): array of slicing positions.
        """
        if axis_len < num_partitions:
            # not enough elements to partition
            split_points = np.append(0, axis_len)
        else:
            quotient = axis_len // num_partitions
            remainder = axis_len % num_partitions
            block_size = max(quotient, mblock_size)
            stop_point = min(block_size * num_partitions, axis_len)
            split_points = np.append(np.arange(start=0, stop=stop_point, step=block_size), axis_len)
            if len(split_points) - 1 == num_partitions and remainder != 0:
                # Balance the size of all partitions
                resize_part = np.arange(start=0, stop=remainder + 1)
                shift_part = np.full(num_partitions - (remainder + 1), fill_value=remainder)
                split_points += np.concatenate([resize_part, shift_part, (0,)], axis=0)
        return split_points

    @classmethod
    def get_axis_repart_range(cls, parts, axis, mblock_size=32, by='size', by_data=None):
        """Calculate the repartition range.

        Args:
            parts (np.ndarray): numpy array of Partition objects.
            axis {0, 1}: axis to repartition on, 0 for row, 1 for column.
            mblock_size (int): minimum block size.
            by {'size', 'split_pos'}: specify whether to compute the split points or directly use given data.
            by_data Union(int, np.ndarray): an integer value that represents repartition size, or a numpy
            array represents split points.

        Returns:
            repart_range_dict (dict{int: dict{int: slice}): A dictionary of dictionary, where key is new partition.
            indice, value is a dictionary of old partition indices and corresponding slice objects.
            axis_size (int): size of the axis after repartition.

        Raises:
            AttributeError: when neither size or new_split_points is provided.
        """
        old_split_points = cls.get_axis_split_points(parts, axis)
        axis_len = old_split_points[-1]
        if by_data is None:
            raise AttributeError("Must provide new_split_points or size")
        if by == 'size':
            new_split_points = cls.calc_axis_split_points(axis_len, by_data, mblock_size)
        else:
            new_split_points = by_data
        repart_range_dict = {}
        p_old = 0
        for p_new in range(len(new_split_points) - 1):
            slices = {}
            new_start = new_split_points[p_new]
            new_stop = new_split_points[p_new + 1]
            while p_old < len(old_split_points) - 1:
                old_start = old_split_points[p_old]
                old_stop = old_split_points[p_old + 1]
                if old_stop <= new_start:
                    p_old += 1
                    continue
                elif old_start >= new_stop:
                    break

                if old_start < new_start:
                    if old_stop <= new_stop:
                        slices[p_old] = slice(new_start - old_start, None, None)
                        p_old += 1
                    else:
                        slices[p_old] = slice(new_start - old_start, new_stop - old_start, None)
                        break
                else:
                    if old_stop <= new_stop:
                        slices[p_old] = slice(None)
                        p_old += 1
                    else:
                        slices[p_old] = slice(0, new_stop - old_start, None)
                        break
            repart_range_dict[p_new] = slices

        axis_size = len(new_split_points) - 1
        return repart_range_dict, axis_size

    @classmethod
    def get_slicing_plan(cls, num_rows, num_cols, output_shape, mblock_size):
        '''Get slicing plan in singlethreaded mode.'''
        row_size, col_size = output_shape
        if num_rows == 0 or num_cols == 0:
            # Empty DataFrame
            return None, None

        row_quot = num_rows // row_size
        col_quot = num_cols // col_size
        if (row_quot == 0 and col_quot == 0) or (row_size <= mblock_size and col_size <= mblock_size):
            # small dataframe, there is no need to repartition it
            return None, None

        if row_quot == 0:
            row_slices = None
        else:
            row_split_points = cls.calc_axis_split_points(num_rows, row_size, mblock_size)
            row_slices = [slice(row_split_points[i],
                                row_split_points[i + 1]) for i in range(len(row_split_points) - 1)]

        if col_quot == 0:
            col_slices = None
        else:
            col_split_points = cls.calc_axis_split_points(num_cols, col_size, mblock_size)
            col_slices = [slice(col_split_points[i],
                                col_split_points[i + 1]) for i in range(len(col_split_points) - 1)]

        return row_slices, col_slices

    @classmethod
    def axis_repartition(cls, parts, axis=0, mblock_size=1, by='size', by_data=None):
        '''Perform repartition along axis in singlethreaded mode.'''
        repart_range_dict, axis_size = cls.get_axis_repart_range(parts,
                                                                 axis=axis,
                                                                 mblock_size=mblock_size,
                                                                 by=by,
                                                                 by_data=by_data)
        if axis == 0:
            new_shape_row = axis_size
            new_shape_col = len(parts[0])
        else:
            new_shape_row = len(parts)
            new_shape_col = axis_size
        result = np.ndarray((new_shape_row, new_shape_col), dtype=object)
        if axis == 0:
            for j in range(len(result[0])):
                for i, d in repart_range_dict.items():
                    segments = []
                    for m, s in d.items():
                        segments.append(parts[m, j].data.iloc[s])
                    data = pandas.concat(segments, axis=0)
                    result[i, j] = Partition.put(data=data, coord=(i, j))
        else:
            for i in range(len(result)):
                for j, d in repart_range_dict.items():
                    segments = []
                    for n, s in d.items():
                        segments.append(parts[i, n].data.iloc[:, s])
                    data = pandas.concat(segments, axis=1)
                    result[i, j] = Partition.put(data=data, coord=(i, j))

        return result

    @classmethod
    def set_index(cls, partitions, labels):
        '''Set index in singlethreaded mode.'''
        def set_index_1d():
            if labels is None:
                sub_new_labels = pandas.RangeIndex(start=0, stop=partitions[0, 0].num_rows)
            else:
                sub_new_labels = labels

            for row_parts in partitions:
                for part in row_parts:
                    part.set_index(sub_new_labels)

        def set_index_2d():
            start = 0
            for row_parts in partitions:
                end = start + row_parts[0].num_rows
                if labels is None:
                    sub_new_labels = pandas.RangeIndex(start, end)
                else:
                    sub_new_labels = labels[start:end]
                for part in row_parts:
                    part.set_index(sub_new_labels)
                start = end

        if partitions.shape[0] == 1:
            set_index_1d()
        else:
            set_index_2d()

    @classmethod
    def axis_partition_to_selected_indices(cls, partitions, axis, func, indices, keep_reminding=False):
        '''Partition to selected indices along axis in singlethreaded mode.'''
        if partitions.size == 0:
            return np.array([[]])
        if keep_reminding:
            selected_partitions = partitions
        else:
            selected_partitions = partitions if axis else partitions.T
            selected_partitions = np.array([selected_partitions[i] for i in indices])
            selected_partitions = selected_partitions if axis else selected_partitions.T
        if axis == 0:
            reminded_partitions = cls.axis_partition(partitions, axis)[0]
            applied_partitions = cls.axis_partition(selected_partitions, axis)[0]
        else:
            reminded_partitions = cls.axis_partition(partitions, axis)[:, 0]
            applied_partitions = cls.axis_partition(selected_partitions, axis)[:, 0]
        if not keep_reminding:
            result = np.array(
                [
                    [part.apply(func, internal_indices=indices[i])]
                    for i, part in zip(indices, applied_partitions)
                ]
            )
        else:
            result = np.array(
                [
                    [reminded_partitions[i]
                     if i not in indices
                     else applied_partitions[i].apply(func, internal_indices=indices[i])]
                    for i in range(len(reminded_partitions))
                ]
            )
        return result if axis else result.T

    @classmethod
    def update(cls, partitions, new_partitions):
        '''Perform update operation in singlethreaded mode.'''
        for row_parts, new_row_parts in zip(partitions, new_partitions):
            for part, new_part in zip(row_parts, new_row_parts):
                part.update(new_part)

    @classmethod
    def flush(cls, partitions):
        '''Only return partitions in singlethreaded mode.'''
        return partitions

    @classmethod
    def remove_empty_rows(cls, partitions):
        '''Remove empty rows in singlethreaded mode.'''
        del_list = []
        for i, row_part in enumerate(partitions):
            if row_part[0].get().index.empty:
                del_list.append(i)
        output_partitions = np.delete(partitions, del_list, axis=0)
        cls.reset_coord(output_partitions)
        return output_partitions

    @classmethod
    def setitem_elements(cls, partitions, func, part_row_locs, part_col_locs, item):
        """Setting item to specific rows/cols in specific partitions

        Args:
            partitions: the input partitions
            func: setitem function
            part_row_locs: ordered dictionary of indices for partitions and row/col index in each partition
            item: value to set which can be a scalar or a list

        Return:
            output_partitions: a numpy array of partitions
        """
        if not np.isscalar(item) and not isinstance(item, type(None)):
            for j, part_col in enumerate(list(part_col_locs.keys())):
                for i, part_row in enumerate(list(part_row_locs.keys())):
                    partitions[part_row][part_col].update(
                        partitions[part_row][part_col].apply(func, part_row_locs[part_row], part_col_locs[part_col],
                                                             item[i][j]))

        else:
            for part_col in list(part_col_locs.keys()):
                for part_row in list(part_row_locs.keys()):
                    partitions[part_row][part_col].update(
                        partitions[part_row][part_col].apply(func, part_row_locs[part_row], part_col_locs[part_col],
                                                             item))
        cls.reset_coord(partitions)
        return partitions


    @classmethod
    def map_axis_partitions(cls, axis, apply_func, base):
        """Apply map function along given axis."""
        base_partitions = cls.axis_partition(base, axis)
        result_blocks = np.array([
            [
                base_partitions[row, col].apply(
                    apply_func
                )
                for row in range(base_partitions.shape[0])
            ]
            for col in range(base_partitions.shape[1])
        ])
        return result_blocks.T if not axis else result_blocks
