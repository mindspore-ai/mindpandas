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
Module for performing multithreaded operations on partitions.
"""
import concurrent.futures

from collections import defaultdict
import hashlib
import numpy as np
import pandas
from pandas.api.types import is_numeric_dtype

from .partition import Partition
from .ds_partition import DSPartition
from .partition_operators import SinglethreadOperator


class MultithreadOperator(SinglethreadOperator):
    """
    Class for providing operations in multithread mode.
    """
    @classmethod
    def map(cls, partitions, map_func, pass_coord=False, **kwargs):
        '''Perform map operation onto partitions in multithreaded mode.'''
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if pass_coord:
                future_parts = {executor.submit(part.apply, map_func, coord=(row_idx, col_idx), **kwargs): part for
                                row_idx, row_parts in enumerate(partitions) for col_idx, part in enumerate(row_parts)}
            else:
                future_parts = {executor.submit(part.apply, map_func, **kwargs): part for row_parts in partitions for
                                part in row_parts}
            for future in concurrent.futures.as_completed(future_parts):
                output_part = future.result()
                output_partitions[output_part.coord] = output_part
        return output_partitions

    @classmethod
    def injective_map(cls, partitions, cond_partitions, other, func, other_is_scalar):
        '''Perform injective map operation onto partitions in multithreaded mode.'''
        def injective_map_no_cond(partitions, other, func, is_scalar):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if is_scalar:
                    future_parts = {executor.submit(partitions[i, j].apply, func, other): (i, j)
                                    for i in range(lrows) for j in range(lcols)}
                else:
                    if lcols > 1 and other.shape[1] == 1:
                        future_parts = {executor.submit(partitions[i, j].apply, func, other[i][0].get()): (i, j)
                                        for i in range(lrows) for j in range(lcols)}
                    else:
                        future_parts = {executor.submit(partitions[i, j].apply, func, other[i][j].get()): (i, j)
                                        for i in range(lrows) for j in range(lcols)}
                for future in concurrent.futures.as_completed(future_parts):
                    output_part = future.result()
                    output_partitions[output_part.coord] = output_part
            return output_partitions

        lrows, lcols = partitions.shape
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        if cond_partitions is None:
            return injective_map_no_cond(partitions, other, func, other_is_scalar)

        is_broadcast = lcols > 1 and cond_partitions.shape[1] == 1

        def injective_map_with_cond(partitions, cond_partitions, other, func, is_scalar):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if is_scalar:
                    if is_broadcast:
                        future_parts = {
                            executor.submit(partitions[i, j].apply, func, cond_partitions[i, 0].get(), other): (i, j)
                            for i in range(lrows) for j in range(lcols)}
                    else:
                        future_parts = {
                            executor.submit(partitions[i, j].apply, func, cond_partitions[i, j].get(), other): (i, j)
                            for i in range(lrows) for j in range(lcols)}
                else:
                    future_parts = {
                        executor.submit(partitions[i, j].apply, func, cond_partitions[i, j].get(), other[i, j].get()): (
                            i, j)
                        for i in range(lrows) for j in range(lcols)}
                for future in concurrent.futures.as_completed(future_parts):
                    output_part = future.result()
                    output_partitions[output_part.coord] = output_part
            return output_partitions
        return injective_map_with_cond(partitions, cond_partitions, other, func, other_is_scalar)

    @classmethod
    def reduce(cls, partitions, reduce_func, axis=0, concat_axis=None):
        '''Perform reduce operation on partitions in multithreaded mode.'''
        num_rows, num_cols = partitions.shape
        container_type = partitions[0, 0].container_type
        if concat_axis is None:
            concat_axis = axis

        def wrap_partitions(axis, idx):
            '''Helper function to perform reduce function on partitions along axis with specified index.'''
            if axis == 0:
                if container_type is pandas.Series:
                    if concat_axis == 0:
                        data = pandas.concat([part.get() for part in partitions[:, idx]], axis=axis)
                    else:
                        data = pandas.concat([part.get().T for part in partitions[:, idx]], axis=axis)
                elif container_type is pandas.DataFrame:
                    data = pandas.concat([part.get() for part in partitions[:, idx]], axis=axis)
                else:
                    data = [part.get() for part in partitions[:, idx]]
                coord = (0, idx)
            else:
                data = pandas.concat([part.get() for part in partitions[idx, :]], axis=axis)
                coord = (idx, 0)
            data = reduce_func(data)
            return Partition.put(data=data, coord=coord)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if axis == 0:
                output_parts = np.ndarray((1, num_cols), dtype=object)
                future_parts = {executor.submit(wrap_partitions, axis, j): j for j in range(num_cols)}
                for future in concurrent.futures.as_completed(future_parts):
                    idx = future_parts[future]
                    output_parts[0][idx] = future.result()
            elif axis == 1:
                output_parts = np.ndarray((num_rows, 1), dtype=object)
                future_parts = {executor.submit(wrap_partitions, axis, i): i for i in range(num_rows)}
                for future in concurrent.futures.as_completed(future_parts):
                    idx = future_parts[future]
                    output_parts[idx][0] = future.result()

        return output_parts

    @classmethod
    def combine_reduce(cls, partitions, other_df_partitions, orig_df_cols, orig_other_cols, reduce_func):
        '''Perform reduce operation on partitions from a combine call in multithreaded mode.'''
        num_cols = partitions.shape[1]
        other_cols = other_df_partitions.shape[1]

        max_cols = max(num_cols, other_cols)

        def combine_partitions(idx):
            '''
            Helper function to combine partitions on specified index.
            '''
            if idx >= num_cols:
                data = pandas.DataFrame([])
                data2 = pandas.concat([part.get() for part in other_df_partitions[:, idx]], axis=0)
                coord = (0, idx)
                if isinstance(data, pandas.Series):
                    data = data.to_frame()
                if isinstance(data2, pandas.Series):
                    data2 = data2.to_frame()
            elif idx >= other_cols:
                data = pandas.concat([part.get() for part in partitions[:, idx]], axis=0)
                data2 = pandas.DataFrame([])
                coord = (0, idx)
                if isinstance(data, pandas.Series):
                    data = data.to_frame()
                if isinstance(data2, pandas.Series):
                    data2 = data2.to_frame()
            else:
                data = pandas.concat([part.get() for part in partitions[:, idx]], axis=0)
                data2 = pandas.concat([part.get() for part in other_df_partitions[:, idx]], axis=0)
                coord = (0, idx)
                if isinstance(data, pandas.Series):
                    data = data.to_frame()
                if isinstance(data2, pandas.Series):
                    data2 = data2.to_frame()

            if idx >= num_cols:
                data = reduce_func(data2, data, orig_other_cols, orig_df_cols)
            else:
                data = reduce_func(data, data2, orig_df_cols, orig_other_cols)

            return Partition.put(data=data, coord=coord)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            output_parts = np.ndarray((1, max_cols), dtype=object)
            future_parts = {executor.submit(combine_partitions, j): j for j in range(max_cols)}
            for future in concurrent.futures.as_completed(future_parts):
                idx = future_parts[future]
                output_parts[0][idx] = future.result()

        return output_parts

    @classmethod
    def to_pandas(cls, partitions, expect_series):
        '''Convert partitions to pandas frame in multithreaded mode.'''
        num_rows, _ = partitions.shape
        if expect_series:
            if partitions.shape[0] == 1:
                pandas_ser = pandas.concat([part.get().squeeze(1) for part in partitions[0, :]])
            else:
                pandas_ser = pandas.concat([part.get().squeeze(1) for part in partitions[:, 0]])
            if not isinstance(pandas_ser, pandas.DataFrame) and pandas_ser.name == '__unsqueeze_series__':
                pandas_ser.name = None
            if not isinstance(pandas_ser, pandas.Series) and pandas_ser.empty:
                return pandas.Series(None)
            return pandas_ser

        def wrap_partitions(row_idx):
            '''Helper function that concatenates partitions on row_idx.'''
            row_df = []
            for part in partitions[row_idx, :]:
                data = part.get()
                if part.container_type is pandas.Series:
                    data = data.T
                row_df.append(data)
            return pandas.concat(row_df, axis=1)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_parts = {executor.submit(wrap_partitions, i): i for i in range(num_rows)}
            row_dfs = [None] * num_rows
            for future in concurrent.futures.as_completed(future_parts):
                idx = future_parts[future]
                row_dfs[idx] = future.result()
        pandas_df = pandas.concat(row_dfs)
        return pandas_df

    @classmethod
    def axis_partition(cls, partitions, axis):
        """Perform repartitioning along given axis

        Args:
            partitions: the input partitions
            axis: {0, 1}
                0 for column partitions, 1 for row partitions

        Return:
            output_partitions: a numpy array of partitions
        """
        num_rows, num_cols = partitions.shape

        def wrap_partitions(axis, idx):
            '''Helper function to concatenate partitions along axis with specified index.'''
            if axis == 0:
                data = pandas.concat([part.get() for part in partitions[:, idx]], axis=axis)
                coord = (0, idx)
            else:
                data = pandas.concat([part.get() for part in partitions[idx, :]], axis=axis)
                coord = (idx, 0)
            return Partition.put(data=data, coord=coord)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if axis == 0:
                output_parts = np.ndarray((1, num_cols), dtype=object)
                future_parts = {executor.submit(wrap_partitions, axis, j): j for j in range(num_cols)}
                for future in concurrent.futures.as_completed(future_parts):
                    idx = future_parts[future]
                    output_parts[0][idx] = future.result()
            elif axis == 1:
                output_parts = np.ndarray((num_rows, 1), dtype=object)
                future_parts = {executor.submit(wrap_partitions, axis, i): i for i in range(num_rows)}
                for future in concurrent.futures.as_completed(future_parts):
                    idx = future_parts[future]
                    output_parts[idx][0] = future.result()

        return output_parts

    @classmethod
    def axis_partition_to_selected_indices(cls, partitions, axis, func, indices, keep_reminding=False):
        '''Partition to selected indices along axis in multithreaded mode.'''
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
            output_partitions = np.ndarray((len(applied_partitions), 1), dtype=object)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_parts = {executor.submit(part.apply, func, internal_indices=indices[i]): j for i, j, part in
                                zip(indices, range(len(indices)), applied_partitions)}
                for future in concurrent.futures.as_completed(future_parts):
                    idx = future_parts[future]
                    output_partitions[idx] = future.result()
        else:
            output_partitions = np.ndarray((len(reminded_partitions), 1), dtype=object)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_parts = {executor.submit(part.apply, func, internal_indices=indices[i]): i for i, part in
                                zip(indices, applied_partitions[list(indices)])}
                for future in concurrent.futures.as_completed(future_parts):
                    idx = future_parts[future]
                    output_partitions[idx] = future.result()
            # fill up the output_partitions with the reminded_partitions
            for i, part in enumerate(reminded_partitions):
                if i not in indices:
                    output_partitions[i] = part
        result = output_partitions if axis else output_partitions.T
        return result

    @classmethod
    def groupby_map(cls, partitions, axis, keys_partitions, map_func):
        """Perform repartitioning along given axis

        Args:
            partitions: the masked partitions
            axis: {0 for ‘index’, 1 for ‘columns’}, default 0
                Split along rows (0) or columns (1).
            keys_partitions: the keys partitions
            map_func:

        Return:
            output_partitions: a numpy array of partitions
        """
        level = map_func.groupby_kwargs['level']
        if level is not None:
            output_partitions = np.ndarray(partitions.shape, dtype=object)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                if axis == 0:
                    future_parts = {executor.submit(part.group_apply, map_func, None): part
                                    for row_parts in partitions for part in row_parts}
                else:
                    future_parts = {executor.submit(part.group_apply, map_func, None): part
                                    for row_parts in partitions for part in row_parts}
        else:
            keys_partitions = cls.axis_partition(keys_partitions, axis ^ 1)
            output_partitions = np.ndarray(partitions.shape, dtype=object)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                if axis == 0:
                    future_parts = {executor.submit(part.group_apply, map_func, key_parts[0]): part
                                    for row_parts, key_parts in zip(partitions, keys_partitions)
                                    for part in row_parts}
                else:
                    future_parts = {executor.submit(part.group_apply, map_func, key_parts): part
                                    for row_parts in partitions
                                    for part, key_parts in zip(row_parts, keys_partitions[0])}

        for future in concurrent.futures.as_completed(future_parts):
            output_part = future.result()
            output_partitions[output_part.coord] = output_part
        return output_partitions

    @classmethod
    def squeeze(cls, partitions, axis=None):
        '''Perform squeeze operation on partitions in multithreaded mode.'''
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_parts = {executor.submit(part.squeeze, axis): part for
                            row_parts in partitions for part in row_parts}
            for future in concurrent.futures.as_completed(future_parts):
                output_part = future.result()
                output_partitions[output_part.coord] = output_part
        return output_partitions

    @classmethod
    def partition_pandas(cls, df, row_slices, col_slices, container_type=None):
        '''Derive partitions from pandas df in multithreaded mode.'''
        def slice_task(df, coord, row_slice, col_slice):
            '''Helper function to create partitions from slice.'''
            if col_slice is None:
                return Partition.put(data=df.iloc[row_slice].copy(), coord=coord, container_type=container_type)
            if row_slice is None:
                return Partition.put(data=df.iloc[:, col_slice].copy(), coord=coord, container_type=container_type)
            return Partition.put(data=df.iloc[row_slice, col_slice].copy(), coord=coord,
                                 container_type=container_type)

        if row_slices is None and col_slices is None:
            output_parts = np.array([[Partition.put(data=df, coord=(0, 0), container_type=container_type)]])
            return output_parts

        row_size = len(row_slices) if row_slices is not None else 1
        col_size = len(col_slices) if col_slices is not None else 1
        output_parts = np.ndarray((row_size, col_size), dtype=object)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_parts = {}
            if col_slices is None:
                for row_slice, i in zip(row_slices, range(row_size)):
                    future_parts[executor.submit(slice_task, df, (i, 0), row_slice, None)] = (i, 0)
            elif row_slices is None:
                for col_slice, j in zip(col_slices, range(col_size)):
                    future_parts[executor.submit(slice_task, df, (0, j), None, col_slice)] = (0, j)
            else:
                for row_slice, i in zip(row_slices, range(row_size)):
                    for col_slice, j in zip(col_slices, range(col_size)):
                        future_parts[executor.submit(slice_task, df, (i, j), row_slice, col_slice)] = (i, j)

            for future in concurrent.futures.as_completed(future_parts):
                coord = future_parts[future]
                output_parts[coord] = future.result()
        return output_parts

    @classmethod
    def axis_repartition(cls, parts, axis=0, mblock_size=1, by='size', by_data=None):
        '''Perform repartition along axis in multithreaded mode.'''
        def task(p_array, t_size, tid):
            '''Concatenate segments into a partition.'''
            t_res = np.ndarray(t_size, dtype=object)
            if axis == 0:
                for i, range_dict in repart_range_dict.items():
                    segments = []
                    for part_index, part_slice in range_dict.items():
                        segments.append(p_array[part_index].get().iloc[part_slice])
                    data = pandas.concat(segments, axis=axis)
                    t_res[i] = Partition.put(data, coord=(i, tid))
            else:
                for j, range_dict in repart_range_dict.items():
                    segments = []
                    for part_index, part_slice in range_dict.items():
                        segments.append(p_array[part_index].get().iloc[:, part_slice])
                    data = pandas.concat(segments, axis=axis)
                    t_res[j] = Partition.put(data, coord=(tid, j))
            return t_res

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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if axis == 0:
                future_parts = {executor.submit(task, parts[:, j], new_shape_row, j): j for j in range(new_shape_col)}
                for future in concurrent.futures.as_completed(future_parts):
                    idx = future_parts[future]
                    result[:, idx] = future.result()
            else:
                future_parts = {executor.submit(task, parts[i, :], new_shape_col, i): i for i in range(new_shape_row)}
                for future in concurrent.futures.as_completed(future_parts):
                    idx = future_parts[future]
                    result[idx, :] = future.result()
        return result

    @classmethod
    def set_index(cls, partitions, labels):
        '''Set index of partitions with labels.'''
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        def set_index_1d():
            '''Set index on 1D partitions.'''
            if labels is None:
                sub_new_labels = pandas.RangeIndex(start=0, stop=partitions[0, 0].num_rows)
            else:
                sub_new_labels = labels

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_parts = {}
                for row_parts in partitions:
                    for part in row_parts:
                        future_parts[executor.submit(part.set_index, sub_new_labels)] = part
                for future in concurrent.futures.as_completed(future_parts):
                    #continue  # loop through all futures make sure all parts have set their index
                    output_part = future.result()
                    output_partitions[output_part.coord] = output_part

        def set_index_2d():
            '''Set index on 2D partitions.'''
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_parts = {}
                start = 0
                for row_parts in partitions:
                    end = start + row_parts[0].num_rows
                    if labels is None:
                        sub_new_labels = pandas.RangeIndex(start, end)
                    else:
                        sub_new_labels = labels[start:end]
                    for part in row_parts:
                        future_parts[executor.submit(part.set_index, sub_new_labels)] = part
                    start = end
                for future in concurrent.futures.as_completed(future_parts):
                    #continue  # loop through all futures make sure all parts have set their index
                    output_part = future.result()
                    output_partitions[output_part.coord] = output_part

        if partitions.shape[0] == 1:
            set_index_1d()
        else:
            set_index_2d()
        return output_partitions

    def append_func(self, apply_func, *args, **kwargs):
        # add this function to be consistent with multiprocess mode
        return self.apply(apply_func, *args, **kwargs)

    @classmethod
    def update(cls, partitions, new_partitions):
        '''Update partitions with data from new_partitions.'''
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_parts = {executor.submit(part.update, new_part): part
                            for row_parts, new_row_parts in zip(partitions, new_partitions) for part, new_part in
                            zip(row_parts, new_row_parts)}
            for _ in concurrent.futures.as_completed(future_parts):
                continue

    @classmethod
    def flush(cls, partitions):
        '''Returns the partitions.'''
        return partitions

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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if not np.isscalar(item) and not isinstance(item, type(None)):
                future_parts = {executor.submit(partitions[part_row][part_col].apply, func, part_row_locs[part_row],
                                                part_col_locs[part_col], item[i][j]): (part_row, part_col) for
                                i, part_row in enumerate(list(part_row_locs.keys())) for j, part_col in
                                enumerate(list(part_col_locs.keys()))}
            else:
                future_parts = {executor.submit(partitions[part_row][part_col].apply, func, part_row_locs[part_row],
                                                part_col_locs[part_col], item): (part_row, part_col) for part_row in
                                list(part_row_locs.keys()) for part_col in list(part_col_locs.keys())}
            for future in concurrent.futures.as_completed(future_parts):
                input_part = future_parts[future]
                output_part = future.result()
                partitions[input_part].update(output_part)
            cls.reset_coord(partitions)
        return partitions

    @classmethod
    def convert_partitions(cls, partitions):
        '''Convert partitions to DSPartitions.'''
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_parts = {executor.submit(DSPartition.put, part.get(), coord=(row_idx, col_idx)): part for row_idx,
                            row_parts in enumerate(partitions) for col_idx, part in enumerate(row_parts)}
            for future in concurrent.futures.as_completed(future_parts):
                output_part = future.result()
                output_partitions[output_part.coord] = output_part
        return output_partitions

    @classmethod
    def copartition(cls, parts, is_range, **kwargs):
        num_rows, num_cols = parts.shape
        container_type = parts[0, 0].container_type

        def copartition_execution(index_map=None, num_output=None, is_range=False):
            def wrap_partitions(idx, index_map=None):
                def create_index_map_numeric(part_index):
                    output_index = defaultdict(list)
                    for idx in part_index:
                        if not is_numeric_dtype(type(idx)) or (is_numeric_dtype(type(idx)) and idx % 1 != 0):
                            # hash object index to int index
                            hashed_index = int(hashlib.sha512(idx.encode('utf-8')).hexdigest()[:16], 16)
                            hashed_index = abs(hash(hashed_index)) % (10**8)
                            output_index[hashed_index%num_output].append(idx)
                        else:
                            output_index[idx%num_output].append(idx)
                    for i in range(num_output):
                        if output_index[i] is None:
                            output_index[i].append(None)
                    return output_index

                sub_output_parts = []
                if not is_range:
                    concat_data = pandas.concat([part.get() for part in parts[idx, :]], axis=1)
                    index_map = create_index_map_numeric(concat_data.index)
                else:
                    concat_data = pandas.concat([part.get() for part in parts[:, idx]], axis=0)

                for i in range(len(index_map)):
                    coord = (i, idx)
                    if index_map[i] is None:
                        data = pandas.DataFrame()
                    else:
                        if isinstance(index_map[i], list):
                            index_map[i] = set(index_map[i])
                        data = concat_data.loc[index_map[i]]
                    output_part = Partition.put(data=data, coord=coord, container_type=container_type)
                    sub_output_parts.append(output_part)
                return sub_output_parts

            with concurrent.futures.ThreadPoolExecutor() as executor:
                if not is_range:
                    output_parts = np.ndarray((num_rows, num_rows), dtype=object)
                    future_parts = {executor.submit(wrap_partitions, j): j for j in range(num_rows)}
                else:
                    output_parts = np.ndarray((max(3, num_rows), num_cols), dtype=object)
                    future_parts = {executor.submit(wrap_partitions, j, index_map): j for j in range(num_cols)}
                for future in concurrent.futures.as_completed(future_parts):
                    sub_output_parts = future.result()
                    for output_part in sub_output_parts:
                        output_parts[output_part.coord] = output_part
                return output_parts

        def copartition_concat(partitions):
            num_rows, _ = partitions.shape
            container_type = partitions[0, 0].container_type

            def wrap_partitions(row):
                data = pandas.concat([part.get() for part in partitions[row, :]], axis=0)
                coord = (row, 0)
                return Partition.put(data=data, coord=coord, container_type=container_type)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                output_parts = np.ndarray((num_rows, 1), dtype=object)
                future_parts = {executor.submit(wrap_partitions, i): i for i in range(num_rows)}
                for future in concurrent.futures.as_completed(future_parts):
                    output_part = future.result()
                    output_parts[output_part.coord] = output_part

                return output_parts

        if is_range:
            index_slices = kwargs.get("index_slices")
            output_parts = copartition_execution(index_map=index_slices, is_range=is_range)
        else:
            num_output = kwargs.get("num_output")
            output_parts = copartition_execution(num_output=num_output, is_range=is_range)
            output_parts = copartition_concat(output_parts)

        return output_parts
