"""Multiprocess Operator Class"""
import concurrent.futures
import warnings
from functools import partial

import numpy as np
import pandas

from .ds_partition import DSPartition as Partition
from .eager_backend import get_scheduler
from .eager_backend import remote_functions as rf
from .partition_operators import SinglethreadOperator


def determine_if_multithreaded_partitions(partitions, is_scalar=False):
    return not is_scalar and partitions is not None and len(partitions) > 0 and not isinstance(partitions[0][0],
                                                                                               Partition)


class MultiprocessOperator(SinglethreadOperator):
    """Multiprocess Operator Class"""
    @classmethod
    def append_func_queue(cls, partitions, func):
        func_id = get_scheduler().put(func)
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        for row_parts in partitions:
            for part in row_parts:
                output_part = part.append_func(func_id)
                output_partitions[output_part.coord] = output_part
        return output_partitions

    @classmethod
    def apply_func_queue(cls, partitions, pass_coord=False, **kwargs):
        """apply func_queue for partitions"""
        # return immediately if there is no work to do
        if not partitions[0, 0].func_queue:
            return partitions
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        input_futures = {}
        for row_parts in partitions:
            for part in row_parts:
                if part.func_queue:
                    input_futures[part.data_id] = part
                else:
                    output_partitions[part.coord] = part

        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    output_part = input_futures[fid].apply_queue(pass_coord, **kwargs)
                    output_partitions[output_part.coord] = output_part
        return output_partitions

    @classmethod
    def map(cls, partitions, map_func, pass_coord=False, **kwargs):
        fuse = kwargs.pop('fuse', False)
        partitions = cls.append_func_queue(partitions, map_func)
        if not fuse:
            partitions = cls.apply_func_queue(partitions, pass_coord, **kwargs)
        return partitions

    @classmethod
    def injective_map(cls, partitions, cond_partitions, other, func, is_scalar):
        inj_func_id = get_scheduler().put(func)
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        if cond_partitions is None:
            # Inspect whether 'other' consists of dspartitions or partitions
            other_contains_multithreaded_partitions = determine_if_multithreaded_partitions(other, is_scalar)
            input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}
            pending_ids = list(input_futures.keys())
            while pending_ids:
                ready_ids, pending_ids = get_scheduler().wait(pending_ids)
                def injective_map_no_cond():
                    for fid in ready_ids:
                        if fid not in  input_futures:
                            continue
                        part = input_futures[fid]
                        if is_scalar:
                            # Use case: partitions = df or series, other = scalar
                            output_part = part.apply(inj_func_id, other)
                        elif partitions.shape == other.shape:
                            # Use case: partitions = df or series, other = same data type
                            coord = part.coord
                            if other_contains_multithreaded_partitions:
                                output_part = part.apply(inj_func_id, other[coord].get())
                            else:
                                output_part = part.apply(inj_func_id, other[coord].data_id)
                        elif partitions.shape[0] == other.shape[0] and len(other.shape) > 1 and other.shape[1] == 1:
                            # Use case: partitions = df, other = series
                            # Broadcasts series across columns of dataframe
                            coord = (part.coord[0], 0)
                            if other_contains_multithreaded_partitions:
                                output_part = part.apply(inj_func_id, other[coord].get())
                            else:
                                output_part = part.apply(inj_func_id, other[coord].data_id)
                        else:
                            raise ValueError("There are too many partitions or the partitions are misaligned.")
                        output_partitions[output_part.coord] = output_part
                injective_map_no_cond()
            return output_partitions

        # When series pass same cond_partitions to each column partition of partitions
        is_broadcast = bool(cond_partitions.shape[1] == 1 and partitions.shape[1] > 0)
        # Inspect whether 'cond' and 'other' consists of dspartitions or partitions
        cond_contains_multithreaded_partitions = determine_if_multithreaded_partitions(cond_partitions)
        other_contains_multithreaded_partitions = determine_if_multithreaded_partitions(other, is_scalar)

        input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}
        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            def injective_map_with_cond():
                for fid in ready_ids:
                    if fid not in input_futures:
                        continue
                    part = input_futures[fid]
                    if is_broadcast:
                        coord = (part.coord[0], 0)
                    else:
                        coord = part.coord
                    if is_scalar:
                        if cond_contains_multithreaded_partitions:
                            output_part = part.apply(inj_func_id, cond_partitions[coord].get(), other)
                        else:
                            output_part = part.apply(inj_func_id, cond_partitions[coord].data_id, other)
                    else:
                        if cond_contains_multithreaded_partitions:
                            if other_contains_multithreaded_partitions:
                                output_part = part.apply(inj_func_id, cond_partitions[coord].get(), other[coord].get())
                            else:
                                output_part = part.apply(inj_func_id, cond_partitions[coord].get(),
                                                         other[coord].data_id)
                        else:
                            if other_contains_multithreaded_partitions:
                                output_part = part.apply(inj_func_id, cond_partitions[coord].data_id,
                                                         other[coord].get())
                            else:
                                output_part = part.apply(inj_func_id, cond_partitions[coord].data_id,
                                                         other[coord].data_id)
                    output_partitions[output_part.coord] = output_part

            injective_map_with_cond()
        return output_partitions

    @classmethod
    def reduce(cls, partitions, reduce_func, axis=0, concat_axis=None):
        partitions = cls.apply_func_queue(partitions)
        reduce_func_id = get_scheduler().put(reduce_func)

        if concat_axis is None:
            concat_axis = axis
        if axis == 0:
            output_partitions = np.ndarray((1, partitions.shape[1]), dtype=object)
        else:
            output_partitions = np.ndarray((partitions.shape[0], 1), dtype=object)
        input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}

        pending_ids_mask = np.zeros(partitions.shape)
        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    coord = input_futures[fid].coord
                    pending_ids_mask[coord] = 1
                    if axis == 0:
                        idx = coord[1]
                        # start reduction as soon as whole column of partitions are ready
                        if pending_ids_mask[:, idx].all():
                            axis_id_list = [part.data_id for part in partitions[:, idx]]
                            future_id, meta_data_id = get_scheduler().remote(rf()._remote_map_axis, axis_id_list,
                                                                             reduce_func_id, axis, concat_axis)
                            output_part = Partition.put(data_id=future_id, meta_id=meta_data_id, coord=(0, idx))
                            output_partitions[output_part.coord] = output_part
                    elif axis == 1:
                        idx = coord[0]
                        if pending_ids_mask[idx, :].all():
                            axis_id_list = [part.data_id for part in partitions[idx, :]]
                            future_id, meta_data_id = get_scheduler().remote(rf()._remote_map_axis, axis_id_list,
                                                                             reduce_func_id, axis, concat_axis)
                            output_part = Partition.put(data_id=future_id, meta_id=meta_data_id, coord=(idx, 0))
                            output_partitions[output_part.coord] = output_part

        return output_partitions

    @classmethod
    def combine_reduce(cls, partitions, other_df_partitions, orig_df_cols, orig_other_cols, reduce_func):
        def merge_rows(parts):
            num_cols = parts.shape[1]
            return cls.repartition(parts, (1, num_cols), 0)

        def combine_cols(cols, other_cols, other):
            if other is not None:
                raise ValueError("combine_cols method expects other to be None.")
            return reduce_func(cols, other_cols, orig_df_cols, orig_other_cols)

        cols = merge_rows(partitions)
        other_cols = merge_rows(other_df_partitions)

        return cls.injective_map(cols, other_cols, None, combine_cols, is_scalar=True)

    @classmethod
    def partition_pandas(cls, df, row_slices, col_slices, container_type=None):
        def slice_task(df, coord, row_slice, col_slice):
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
    def repartition(cls, parts, output_shape, mblock_size):
        parts = cls.flush(parts)
        return super().repartition(parts, output_shape, mblock_size)

    @classmethod
    def axis_repartition(cls, parts, axis=0, mblock_size=32, by='size', by_data=None):
        repart_range_dict, axis_size = cls.get_axis_repart_range(parts, axis, mblock_size, by=by, by_data=by_data)
        if axis == 0:
            new_shape_row = axis_size
            new_shape_col = parts.shape[1]
        else:
            new_shape_row = parts.shape[0]
            new_shape_col = axis_size
        output_partitions = np.ndarray((new_shape_row, new_shape_col), dtype=object)
        input_futures = {part.data_id: part for row_parts in parts for part in row_parts}
        slice_tasks = {}
        pending_ids_mask = np.zeros(parts.shape)
        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    coord = input_futures[fid].coord
                    pending_ids_mask[coord] = 1
                    if axis == 0:
                        idx = coord[1]
                        # start concat as soon as whole column of partitions are ready
                        if pending_ids_mask[:, idx].all():
                            # If the original data cannot be deserialized with zero-copy,
                            # then each worker needs a copy of the original data,
                            # so limit the parallelism to improve performance and avoid OOM
                            axis_id_list = [part.data_id for part in parts[:, idx]]
                            future_list_id, meta_data_list_id = get_scheduler().remote(rf()._remote_concat_segments,
                                                                                       axis_id_list,
                                                                                       axis,
                                                                                       repart_range_dict)
                            slice_tasks[future_list_id] = (idx, meta_data_list_id)
                    elif axis == 1:
                        idx = coord[0]
                        if pending_ids_mask[idx, :].all():
                            axis_id_list = [part.data_id for part in parts[idx, :]]
                            future_list_id, meta_data_list_id = get_scheduler().remote(rf()._remote_concat_segments,
                                                                                       axis_id_list,
                                                                                       axis,
                                                                                       repart_range_dict)
                            slice_tasks[future_list_id] = (idx, meta_data_list_id)

        pending_ids = list(slice_tasks.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                idx, meta_data_list_id = slice_tasks[fid]
                future_list = get_scheduler().get(fid)[0]
                meta_data_list = get_scheduler().get(meta_data_list_id)[0]
                if axis == 0:
                    for row, (future_id, meta_data_id) in enumerate(zip(future_list, meta_data_list)):
                        output_part = Partition.put(data_id=future_id, meta_id=meta_data_id, coord=(row, idx))
                        output_partitions[output_part.coord] = output_part
                else:
                    for col, (future_id, meta_data_id) in enumerate(zip(future_list, meta_data_list)):
                        output_part = Partition.put(data_id=future_id, meta_id=meta_data_id, coord=(idx, col))
                        output_partitions[output_part.coord] = output_part

        return output_partitions

    @classmethod
    def axis_partition(cls, partitions, axis):
        # similar to reduce, except no apply_func
        partitions = cls.apply_func_queue(partitions)

        if axis == 0:
            output_partitions = np.ndarray((1, partitions.shape[1]), dtype=object)
        else:
            output_partitions = np.ndarray((partitions.shape[0], 1), dtype=object)
        input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}

        pending_ids_mask = np.zeros(partitions.shape)
        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    coord = input_futures[fid].coord
                    pending_ids_mask[coord] = 1
                    if axis == 0:
                        idx = coord[1]
                        # start concat as soon as whole column of partitions are ready
                        if pending_ids_mask[:, idx].all():
                            axis_id_list = [part.data_id for part in partitions[:, idx]]
                            future_id, meta_data_id = get_scheduler().remote(rf()._remote_concat_axis,
                                                                             axis_id_list,
                                                                             axis)
                            output_part = Partition.put(data_id=future_id, meta_id=meta_data_id, coord=(0, idx))
                            output_partitions[output_part.coord] = output_part
                    elif axis == 1:
                        idx = coord[0]
                        if pending_ids_mask[idx, :].all():
                            axis_id_list = [part.data_id for part in partitions[idx, :]]
                            future_id, meta_data_id = get_scheduler().remote(rf()._remote_concat_axis,
                                                                             axis_id_list,
                                                                             axis)
                            output_part = Partition.put(data_id=future_id, meta_id=meta_data_id, coord=(idx, 0))
                            output_partitions[output_part.coord] = output_part

        return output_partitions

    @classmethod
    def mask(cls, partitions, rows_index, columns_index, rows_partition_index_dict, columns_partition_index_dict,
             is_series=False):
        partitions = cls.apply_func_queue(partitions)

        num_rows, num_cols = partitions.shape
        if rows_index is not None:
            num_rows = len(rows_partition_index_dict)

        if columns_index is not None:
            num_cols = len(columns_partition_index_dict)

        output_shape = (num_rows, num_cols)
        output_partitions = np.ndarray(output_shape, dtype=object)
        input_futures = {part.data_id: part for row_parts in partitions for part in row_parts
                         if part.coord[0] in rows_partition_index_dict.keys() and
                         part.coord[1] in columns_partition_index_dict.keys()}

        row_id_mapping = {v: k for k, v in enumerate(rows_partition_index_dict)}
        col_id_mapping = {v: k for k, v in enumerate(columns_partition_index_dict)}

        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    coord = input_futures[fid].coord
                    row_internal_indices = rows_partition_index_dict[coord[0]]
                    col_internal_indices = columns_partition_index_dict[coord[1]]
                    output_part = input_futures[fid].mask(row_internal_indices, col_internal_indices, is_series)
                    output_part.coord = row_id_mapping.get(output_part.coord[0]), col_id_mapping.get(
                        output_part.coord[1])
                    output_partitions[output_part.coord] = output_part

        return output_partitions

    @classmethod
    def groupby_map(cls, partitions, axis, keys_partitions, map_func):
        if map_func.groupby_kwargs['level']:
            # Groupby level, we don't need distribute key_partitions in this case
            partitions = cls.apply_func_queue(partitions)

            map_func_id = get_scheduler().put(map_func)

            output_partitions = np.ndarray(partitions.shape, dtype=object)
            input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}

            pending_ids = list(input_futures.keys())

            while pending_ids:
                ready_ids, pending_ids = get_scheduler().wait(pending_ids)
                for fid in ready_ids:
                    if fid not in input_futures:
                        warnings.warn(f'unexpected future id {fid}')

                    input_part = input_futures[fid]
                    output_part = input_part.group_apply(map_func_id, None)
                    output_partitions[output_part.coord] = output_part

            return output_partitions

        partitions = cls.apply_func_queue(partitions)

        map_func_id = get_scheduler().put(map_func)

        keys_partitions = cls.axis_partition(keys_partitions, axis ^ 1)
        keys_futures = {part.data_id: part for row_parts in keys_partitions for part in row_parts}

        output_partitions = np.ndarray(partitions.shape, dtype=object)
        input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}

        input_pending_mask = np.zeros(partitions.shape)
        key_pending_mask = np.zeros(keys_partitions.shape)
        pending_ids = list(input_futures.keys()) + list(keys_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            def groupby_map_call():
                for fid in ready_ids:
                    if fid not in input_futures and fid not in keys_futures:
                        continue
                    has_input_fid = True
                    if fid in input_futures:
                        coord = input_futures[fid].coord
                        input_pending_mask[coord] = 1
                    elif fid in keys_futures:
                        has_input_fid = False
                        coord = keys_futures[fid].coord
                        key_pending_mask[coord] = 1
                    else:
                        warnings.warn(f'unexpected future id {fid}')

                    if axis == 0:
                        row = coord[0]
                        if has_input_fid and key_pending_mask[row, 0] == 1:
                            input_part = input_futures[fid]
                            key_part = keys_partitions[row, 0]
                            output_part = input_part.group_apply(map_func_id, key_part.data_id)
                            output_partitions[output_part.coord] = output_part
                        elif not has_input_fid:
                            for col in range(partitions.shape[1]):
                                if input_pending_mask[row, col] != 1:
                                    continue
                                input_part = partitions[row, col]
                                key_part = keys_futures[fid]
                                output_part = input_part.group_apply(map_func_id, key_part.data_id)
                                output_partitions[output_part.coord] = output_part
                                input_pending_mask[row, col] = 0  # turn off so don't apply again
                    else:
                        col = coord[1]
                        if has_input_fid and key_pending_mask[0, col] == 1:
                            input_part = input_futures[fid]
                            key_part = keys_partitions[0, col]
                            output_part = input_part.group_apply(map_func_id, key_part.data_id)
                            output_partitions[output_part.coord] = output_part
                        elif not has_input_fid:
                            for row in range(partitions.shape[0]):
                                if input_pending_mask[row, col] != 1:
                                    continue
                                input_part = partitions[row, col]
                                key_part = keys_futures[fid]
                                output_part = input_part.group_apply(map_func_id, key_part.data_id)
                                output_partitions[output_part.coord] = output_part
                                input_pending_mask[row, col] = 0  # turn off so don't apply again
            groupby_map_call()

        return output_partitions

    @classmethod
    def set_index(cls, partitions, labels):
        partitions = cls.apply_func_queue(partitions)

        def set_index_1d():
            if labels is None:
                sub_labels = pandas.RangeIndex(start=0, stop=partitions[0, 0].num_rows)
            else:
                sub_labels = labels

            input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}
            pending_ids = list(input_futures.keys())
            while pending_ids:
                ready_ids, pending_ids = get_scheduler().wait(pending_ids)
                for fid in ready_ids:
                    if fid in input_futures:
                        input_futures[fid].set_index(sub_labels)

        def set_index_2d():
            sub_labels = []
            start = 0
            for row_parts in partitions:
                end = start + row_parts[0].num_rows
                if labels is None:
                    sub_labels.append(pandas.RangeIndex(start, end))
                else:
                    sub_labels.append(labels[start:end])
                start = end

            input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}
            pending_ids = list(input_futures.keys())
            while pending_ids:
                ready_ids, pending_ids = get_scheduler().wait(pending_ids)
                for fid in ready_ids:
                    if fid in input_futures:
                        input_row, _ = input_futures[fid].coord
                        input_futures[fid].set_index(sub_labels[input_row])

        if partitions.shape[0] == 1:
            set_index_1d()
        else:
            set_index_2d()

    @classmethod
    def squeeze(cls, partitions, axis=None):
        partitions = cls.apply_func_queue(partitions)
        output_partitions = np.ndarray(partitions.shape, dtype=object)
        input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}

        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    output_part = input_futures[fid].squeeze(axis)
                    output_partitions[output_part.coord] = output_part

        return output_partitions

    @classmethod
    def to_pandas(cls, partitions, expect_series):
        partitions = cls.flush(partitions)
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
            row_df = []
            for part in partitions[row_idx, :]:
                data = part.get()
                if part.container_type is pandas.Series:
                    data = data.T
                row_df.append(data)
            return pandas.concat(row_df, axis=1)

        # use multithreading for better performance
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_parts = {executor.submit(wrap_partitions, i): i for i in range(num_rows)}
            row_dfs = [None] * num_rows
            for future in concurrent.futures.as_completed(future_parts):
                idx = future_parts[future]
                row_dfs[idx] = future.result()
        pandas_df = pandas.concat(row_dfs)
        return pandas_df

    @classmethod
    def to_numpy(cls, partitions, dtype, copy, na_value):
        # if know container type, can add to_numpy to function queue and then apply all to improve performance
        if partitions[0, 0].container_type not in (pandas.Series, pandas.DataFrame):
            # if not sure what the container type is, apply all functions first to get the type
            partitions = cls.apply_func_queue(partitions)
            if partitions[0, 0].container_type is pandas.Series:
                return np.block([part.get().to_numpy(dtype, copy, na_value) for part in partitions[:, 0]])
            return np.block(
                [[part.get().to_numpy(dtype, copy, na_value) for part in row_parts] for row_parts in partitions])
        if partitions[0, 0].container_type is pandas.Series:
            to_numpy_partial = partial(pandas.DataFrame.to_numpy, dtype=dtype, copy=copy, na_value=na_value)
            partitions = cls.append_func_queue(partitions, to_numpy_partial)
            partitions = cls.apply_func_queue(partitions)
            return np.block([[part.get().squeeze()] for part in partitions[:, 0]])
        to_numpy_partial = partial(pandas.DataFrame.to_numpy, dtype=dtype, copy=copy, na_value=na_value)
        partitions = cls.append_func_queue(partitions, to_numpy_partial)
        partitions = cls.apply_func_queue(partitions)
        return np.block([[part.get().squeeze() for part in row_parts] for row_parts in partitions])

    @classmethod
    def update(cls, partitions, new_partitions):
        for rowpart, newrowpart in zip(partitions, new_partitions):
            for part, new_part in zip(rowpart, newrowpart):
                part.update(new_part)

    @classmethod
    def flush(cls, partitions):
        partitions = cls.apply_func_queue(partitions)

        input_futures = {part.data_id: part for row_parts in partitions for part in row_parts}
        pending_ids = list(input_futures.keys())
        while pending_ids:
            _, pending_ids = get_scheduler().wait(pending_ids)
        return partitions

    @classmethod
    def axis_partition_to_selected_indices(cls, partitions, axis, func, indices, keep_reminding=False):
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

        func_id = get_scheduler().put(func)
        if not keep_reminding:
            output_partitions = np.ndarray((len(applied_partitions), 1), dtype=object)
            input_futures = {part.data_id: (part, j, indices[i]) for i, j, part in
                             zip(indices, range(len(indices)), applied_partitions)}
        else:
            output_partitions = np.ndarray((len(reminded_partitions), 1), dtype=object)
            input_futures = {part.data_id: (part, i, indices[i]) for i, part in
                             zip(indices, applied_partitions[list(indices)])}

        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    input_part = input_futures[fid][0]
                    input_part_id = input_futures[fid][1]
                    input_internal_indices = input_futures[fid][2]
                    output_part = input_part.apply(func_id, internal_indices=input_internal_indices)
                    output_partitions[input_part_id] = output_part

        if keep_reminding:
            # fill up the output_partitions with the reminded_partitions
            for i, part in enumerate(reminded_partitions):
                if i not in indices:
                    output_partitions[i] = part
        result = output_partitions if axis else output_partitions.T
        return result

    @classmethod
    def remove_empty_rows(cls, partitions):
        partitions = cls.apply_func_queue(partitions)
        del_list = []
        for i, row_part in enumerate(partitions):
            if row_part[0].num_rows == 0:
                del_list.append(i)
        output_partitions = np.delete(partitions, del_list, axis=0)

        input_futures = {part.data_id: (part, i, j)
                         for i, row_parts in enumerate(output_partitions)
                         for j, part in enumerate(row_parts)}
        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    input_part = input_futures[fid][0]
                    input_row_id, input_col_id = input_futures[fid][1], input_futures[fid][2]
                    input_part.coord = (input_row_id, input_col_id)

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
        partitions = cls.apply_func_queue(partitions)
        func_id = get_scheduler().put(func)
        output_partitions = partitions
        if not np.isscalar(item) and not isinstance(item, type(None)):
            input_futures = {partitions[part_row][part_col].data_id: (
                partitions[part_row][part_col], part_row_locs[part_row], part_col_locs[part_col], item[i][j])
                             for i, part_row in enumerate(list(part_row_locs.keys())) for j, part_col in
                             enumerate(list(part_col_locs.keys()))}
        else:
            input_futures = {partitions[part_row][part_col].data_id: (
                partitions[part_row][part_col], part_row_locs[part_row], part_col_locs[part_col], item)
                             for part_row in list(part_row_locs.keys()) for part_col in list(part_col_locs.keys())}
        pending_ids = list(input_futures.keys())
        while pending_ids:
            ready_ids, pending_ids = get_scheduler().wait(pending_ids)
            for fid in ready_ids:
                if fid in input_futures:
                    input_part = input_futures[fid][0]
                    input_part_row_locs = input_futures[fid][1]
                    input_part_col_locs = input_futures[fid][2]
                    item = input_futures[fid][3]
                    output_part = input_part.apply(func_id, input_part_row_locs, input_part_col_locs, item)
                    output_partitions[output_part.coord] = output_part
                    input_part.update(output_part)
        cls.reset_coord(partitions)
        return output_partitions

    @classmethod
    def map_axis_partitions(cls, axis, apply_func, base):
        """Apply map function along given axis."""
        apply_func_id = get_scheduler().put(apply_func)
        base_partitions = cls.axis_partition(base, axis)
        result_blocks = np.array([
            [
                base_partitions[row, col].apply(
                    apply_func_id
                )
                for row in range(base_partitions.shape[0])
            ]
            for col in range(base_partitions.shape[1])
        ])
        return result_blocks.T if not axis else result_blocks
