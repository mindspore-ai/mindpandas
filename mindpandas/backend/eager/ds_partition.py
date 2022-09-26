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
Module to provide datasystem partition class for multiprocess backend.
"""
import pandas

from .eager_backend import get_scheduler
from .eager_backend import remote_functions as rf


class NA:
    '''NA object.'''
    def __init__(self):
        pass


class DSPartition():
    """
    Partition class used by multiprocess backend.
    """
    def __init__(self, data_id, meta_data_id, coord, meta_data):
        # object store ids
        self.data_id = data_id
        self.meta_data_id = meta_data_id
        self.coord = coord
        self.func_queue = []

        # cached data
        self.container_type = meta_data.get('container_type', NA())
        self.num_rows = meta_data.get('num_rows', NA())
        self.num_cols = meta_data.get('num_cols', NA())
        self.index = meta_data.get('index', NA())
        self.columns = meta_data.get('columns', NA())
        self.valid = meta_data.get('valid', NA())
        self.dtypes = meta_data.get('dtypes', NA())

    def append_func(self, func_id):
        '''Appends function to function queue.'''
        output_partition = DSPartition.put(data_id=self.data_id, meta_id=self.meta_data_id, coord=self.coord)
        output_partition.func_queue = self.func_queue + [func_id]
        return output_partition

    def apply_queue(self, pass_coord=False, **kwargs):
        '''Apply function queue.'''
        if self.func_queue:
            if pass_coord:
                kwargs["coord"] = self.coord
            future_id, meta_data_id = get_scheduler().remote(rf()._remote_apply_queue,
                                                             self.data_id,
                                                             self.func_queue,
                                                             **kwargs)
            output_partition = DSPartition.put(data_id=future_id, meta_id=meta_data_id, coord=self.coord)
            return output_partition
        return DSPartition.put(data_id=self.data_id, meta_id=self.meta_data_id, coord=self.coord)

    def _check_apply_queue_internal(self):
        '''Clears function queue if it is not None.'''
        if self.func_queue:
            self.data_id, self.meta_data_id = get_scheduler().remote(rf()._remote_apply_queue,
                                                                     self.data_id,
                                                                     self.func_queue)
            self.func_queue.clear()

    def apply(self, apply_func_id, *args, **kwargs):
        '''Apply function to partitions.'''
        self._check_apply_queue_internal()
        future_id, meta_data_id = get_scheduler().remote(rf()._remote_apply_func,
                                                         self.data_id, apply_func_id,
                                                         *args,
                                                         **kwargs)
        output_partition = DSPartition.put(data_id=future_id, meta_id=meta_data_id, coord=self.coord)
        return output_partition

    def get(self):
        '''Get partition data.'''
        self._check_apply_queue_internal()
        values = get_scheduler().get(self.data_id)
        data = values[0]
        return data

    @classmethod
    def put(cls, data=None, data_id=None, meta_id=None, coord=None, container_type=None):
        '''Puts new data in partition.'''
        if data is not None:
            data, meta_data = process_raw_data(data, container_type)
            data_id = get_scheduler().put(data)
            meta_id = get_scheduler().put(meta_data)
        else:
            meta_data = dict()
        return DSPartition(data_id=data_id, meta_data_id=meta_id, coord=coord, meta_data=meta_data)

    def group_apply(self, apply_func_id, keys_data_id):
        '''Apply function on partitions with key grouping.'''
        self._check_apply_queue_internal()
        future_id, meta_data_id = get_scheduler().remote(rf()._remote_groupby_apply_func,
                                                         self.data_id, apply_func_id,
                                                         keys_data_id)
        output_partition = DSPartition.put(data_id=future_id, meta_id=meta_data_id, coord=self.coord)
        return output_partition

    @property
    def index(self):
        '''Get index of partition.'''
        if isinstance(self._index, NA):
            self._get_meta_data()
        return self._index

    @index.setter
    def index(self, val):
        '''Set index of partition.'''
        self._index = val

    @property
    def columns(self):
        '''Get partition columns.'''
        if isinstance(self._columns, NA):
            self._get_meta_data()
        return self._columns

    @columns.setter
    def columns(self, val):
        '''Set partition columns.'''
        self._columns = val

    @property
    def num_rows(self):
        '''Get number of rows.'''
        if isinstance(self._num_rows, NA):
            self._get_meta_data()
        return self._num_rows

    @num_rows.setter
    def num_rows(self, val):
        '''Set number of rows.'''
        self._num_rows = val

    @property
    def num_cols(self):
        '''Get number of columns.'''
        if isinstance(self._num_cols, NA):
            self._get_meta_data()
        return self._num_cols

    @num_cols.setter
    def num_cols(self, val):
        '''Set number of columns.'''
        self._num_cols = val

    @property
    def container_type(self):
        '''Get container type.'''
        if isinstance(self._container_type, NA):
            self._get_meta_data()
        return self._container_type

    @container_type.setter
    def container_type(self, val):
        '''Set container type.'''
        self._container_type = val

    @property
    def valid(self):
        '''Check partitions valid status.'''
        if isinstance(self._valid, NA):
            self._get_meta_data()
        return self._valid

    @valid.setter
    def valid(self, val):
        '''Set valid status.'''
        self._valid = val

    @property
    def dtypes(self):
        '''Get partition data types.'''
        if isinstance(self._dtypes, NA):
            self._get_meta_data()
        return self._dtypes

    @dtypes.setter
    def dtypes(self, val):
        '''Set partition data types.'''
        self._dtypes = val

    def is_full_axis_mask(self, index, axis_length):
        """Check whether `index` mask grabs `axis_length` amount of elements."""
        if isinstance(index, slice):
            return index == slice(None) or (
                isinstance(axis_length, int)
                and len(range(*index.indices(axis_length))) == axis_length
            )
        return (
            hasattr(index, "__len__")
            and isinstance(axis_length, int)
            and len(index) == axis_length
        )

    def mask(self, row_indices, column_indices, is_series=False):
        '''Return mask of partitions based on provided indices.'''
        self._check_apply_queue_internal()
        new_row_indices = row_indices
        new_column_indices = column_indices
        if self.is_full_axis_mask(row_indices, self.num_rows):
            new_row_indices = None
        elif self.is_full_axis_mask(column_indices, self.num_cols):
            new_column_indices = None
        future_id, meta_data_id = get_scheduler().remote(rf()._remote_mask,
                                                         self.data_id,
                                                         new_row_indices,
                                                         new_column_indices,
                                                         is_series)
        output_partition = DSPartition.put(data_id=future_id, meta_id=meta_data_id, coord=self.coord)
        return output_partition

    def set_index(self, labels):
        '''Set index of partitions with labels.'''
        self._check_apply_queue_internal()
        self.data_id, self.meta_data_id = get_scheduler().remote(rf()._remote_set_index, self.data_id, labels)

    def _get_meta_data(self):
        '''Get partitions meta data.'''
        self._check_apply_queue_internal()
        meta_data = get_scheduler().get(self.meta_data_id)[0]
        self.num_rows = meta_data.get('num_rows')
        self.num_cols = meta_data.get('num_cols')
        self.container_type = meta_data.get('container_type')
        self.index = meta_data.get('index')
        self.columns = meta_data.get('columns')
        self.valid = meta_data.get('valid')
        self.dtypes = meta_data.get('dtypes')

    def squeeze(self, axis):
        '''Perform squeeze operation along axis.'''
        self._check_apply_queue_internal()
        future_id, meta_data_id = get_scheduler().remote(rf()._remote_squeeze, self.data_id, axis)
        output_partition = DSPartition.put(data_id=future_id, meta_id=meta_data_id, coord=self.coord)
        return output_partition

    def update(self, new_part):
        '''Update partitions with new data.'''
        self.data_id = new_part.data_id
        self.meta_data_id = new_part.meta_data_id
        self.coord = new_part.coord
        self.container_type = NA()
        self.num_rows = NA()
        self.num_cols = NA()
        self.index = NA()
        self.columns = NA()
        self.valid = NA()
        self.dtypes = NA()


def process_raw_data(data, container_type=None):
    '''Process raw data, such as from read_csv, into partitions.'''
    valid = data is not None and not isinstance(data, Exception)
    if isinstance(data, pandas.Series):
        container_type = pandas.Series if container_type is None else container_type
        if data.name is None:
            data = data.to_frame('__unsqueeze_series__')
        else:
            data = data.to_frame()
    elif isinstance(data, pandas.DataFrame):
        container_type = pandas.DataFrame
        row, col = data.shape
        if row == 1 and col >= 1 and '__unsqueeze_series__' in data.index:
            data = data.T
        if col == 1 and '__unsqueeze_series__' in data.columns:
            container_type = pandas.Series
    else:
        container_type = type(data)
        data = pandas.DataFrame([[data]])

    num_rows = len(data) if hasattr(data, '__len__') else 0
    num_cols = len(data.columns) if hasattr(data, 'columns') else 0
    index = data.index if hasattr(data, 'index') else None
    columns = data.columns if hasattr(data, 'columns') else None
    dtypes = data.dtypes if hasattr(data, 'dtypes') else None

    cached_meta = {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'index': index,
        'columns': columns,
        'container_type': container_type,
        'valid': valid,
        'dtypes': dtypes
    }
    return data, cached_meta
