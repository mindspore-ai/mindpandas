# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Remote functions based on yr."""
import pandas
import yr

import mindpandas.backend.eager.ds_partition as partition


# -------------------------------
# Single data remote functions
# -------------------------------


@yr.invoke(return_nums=2)
def _remote_apply_func(data, func, *args, **kwargs):
    """Apply function onto data."""
    output = func(data, *args, **kwargs)
    output_data, output_meta = partition.process_raw_data(output)
    return output_data, output_meta


@yr.invoke(return_nums=2)
def _remote_apply_queue(data, func_id_list, **kwargs):
    """Execute all functions in data's function queue."""
    func_queue = yr.get([*func_id_list])
    output = data
    for func in func_queue:
        output = func(output, **kwargs)
    output_data, output_meta = partition.process_raw_data(output)
    return output_data, output_meta


@yr.invoke(return_nums=2)
def _remote_groupby_apply_func(data, func, keys_data):
    """Apply func on groupby data."""
    output = func(data, keys_data)
    output_data, output_meta = partition.process_raw_data(output)
    return output_data, output_meta


@yr.invoke(return_nums=2)
def _remote_mask(data, row_indices, column_indices, is_series=False):
    """Perform mask operation on data."""
    if isinstance(data, pandas.Series) or column_indices is None:
        output = data.iloc[row_indices]
    elif row_indices is None:
        output = data.iloc[:, column_indices]
    elif isinstance(data, list):
        output = data[row_indices]
    else:
        output = data.iloc[row_indices, column_indices]
    output_shape = output.shape
    if is_series and len(output_shape) > 1 and output_shape[1] == 1:
        output = output.squeeze("columns")
    output_data, output_meta = partition.process_raw_data(output)
    return output_data, output_meta


@yr.invoke(return_nums=2)
def _remote_set_index(data, labels):
    """Set index with labels."""
    data.index = labels
    output_data, output_meta = partition.process_raw_data(data)
    return output_data, output_meta


@yr.invoke(return_nums=2)
def _remote_squeeze(data, axis):
    """Perform squeeze operation on data."""
    if axis is None or axis == 1 and len(data.columns) == 1:
        output = data.squeeze(axis=1)
    elif axis is None or axis == 0 and len(data.index) == 1:
        output = data.squeeze(axis=0)
    else:
        output = data
    output_data, output_meta = partition.process_raw_data(output)
    return output_data, output_meta


# -------------------------------
# Multiple data remote functions
# -------------------------------
@yr.invoke(return_nums=2)
def _remote_map_axis(data_id_list, func, axis, concat_axis):
    """Map func on data along specified axis."""
    data_list = yr.get([*data_id_list])
    container_type = pandas.Series if '__unsqueeze_series__' in data_list[0].columns else pandas.DataFrame
    if axis == 0:
        if container_type is pandas.Series:
            if concat_axis == 0:
                data = pandas.concat([part for part in data_list], axis=axis)
            else:
                data = pandas.concat([part.T for part in data_list], axis=axis)
        elif container_type is pandas.DataFrame:
            data = pandas.concat([part for part in data_list], axis=axis)
        else:
            data = [part for part in data_list]
    else:
        data = pandas.concat([part for part in data_list], axis=axis)
    output = func(data)
    output_data, output_meta = partition.process_raw_data(output)
    return output_data, output_meta


@yr.invoke(return_nums=2)
def _remote_concat_axis(data_id_list, axis):
    """Concat data along specified axis."""
    data_list = yr.get([*data_id_list])
    if not data_list and isinstance(data_list[0], list):
        output = []
        for d in data_list:
            output.extend(d)
    else:
        output = pandas.concat(data_list, axis=axis)
    output_data, output_meta = partition.process_raw_data(output)
    return output_data, output_meta


@yr.invoke(return_nums=2)
def _remote_concat_segments(data_id_list, axis, repart_range_dict):
    """Concat data along specified axis."""
    data_list = yr.get([*data_id_list])
    output_data_list, output_meta_list = [], []
    for _, input_row_ranges in repart_range_dict.items():
        segments = []
        for idx, data_slice in input_row_ranges.items():
            if axis == 0:
                segments.append(data_list[idx].iloc[data_slice])
            else:
                segments.append(data_list[idx].iloc[:, data_slice])
        output = pandas.concat(segments, axis=axis)

        output_data, output_meta_data = partition.process_raw_data(output)

        output_data_id = yr.put(output_data)
        output_meta_id = yr.put(output_meta_data)

        output_data_list.append(output_data_id)
        output_meta_list.append(output_meta_id)
    return output_data_list, output_meta_list
