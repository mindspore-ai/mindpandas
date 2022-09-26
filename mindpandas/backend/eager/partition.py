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
Module to provide partition class for single- or multithreaded operations using Python backend.
"""
import copy
import pandas


class Partition():
    """
    Partition class used by multithread or singlethread backend.
    """
    def __init__(self, data, coord=None, **kwargs):
        self.data = data
        self.coord = coord
        self.container_type = kwargs.get('container_type', None)
        self.num_rows = kwargs.get('num_rows', 0)
        self.num_cols = kwargs.get('num_cols', 0)
        self.index = kwargs.get('index', None)
        self.columns = kwargs.get('columns', None)
        self.valid = kwargs.get('valid', True)
        self.dtypes = kwargs.get('dtypes', None)

    def apply(self, apply_func, *args, **kwargs):
        '''Apply function to partitions.'''
        output_data = apply_func(self.data, *args, **kwargs)
        # inplace operator returns None
        if output_data is None:
            output_partition = Partition.put(data=self.data, coord=self.coord)
        else:
            output_partition = Partition.put(data=output_data, coord=self.coord)
        return output_partition

    def group_apply(self, apply_func, keys):
        '''Apply function to partitions with key grouping.'''
        output_data = apply_func(self.data, getattr(keys, "data", None))
        output_partition = Partition.put(data=output_data, coord=self.coord)
        return output_partition

    def get(self):
        '''Get partition data.'''
        return self.data

    @classmethod
    def put(cls, data, coord=None, container_type=None):
        '''Put data into partitions.'''
        valid = data is not None and not isinstance(data, Exception)
        if isinstance(data, pandas.Series):
            container_type = pandas.Series if container_type is None else container_type
            if data.name is None:
                data = data.to_frame('__unsqueeze_series__')
            else:
                data = data.to_frame()
        elif isinstance(data, pandas.DataFrame):
            container_type = pandas.DataFrame
            r, c = data.shape
            if r == 1 and c >= 1 and '__unsqueeze_series__' in data.index:
                data = data.T
            if c == 1 and '__unsqueeze_series__' in data.columns:
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

        return Partition(data, coord=coord, **cached_meta)

    @property
    def num_rows(self):
        '''Get number of rows.'''
        return self._num_rows

    @num_rows.setter
    def num_rows(self, val):
        '''Set number of rows.'''
        self._num_rows = val

    @property
    def num_cols(self):
        '''Get number of columns.'''
        return self._num_cols

    @num_cols.setter
    def num_cols(self, val):
        '''Set number of columns.'''
        self._num_cols = val

    @property
    def container_type(self):
        '''Get container type.'''
        return self._container_type

    @container_type.setter
    def container_type(self, val):
        '''Set container type.'''
        self._container_type = val

    @property
    def valid(self):
        '''Get valid status.'''
        return self._valid

    @valid.setter
    def valid(self, val):
        '''Set valid status.'''
        self._valid = val

    @property
    def index(self):
        '''Get index of partitions.'''
        return self._index

    @index.setter
    def index(self, val):
        '''Set index of partitions.'''
        if self.container_type in (pandas.DataFrame, pandas.Series):
            self.data.index = val
            self._index = val

    @property
    def columns(self):
        '''Get partition columns.'''
        return getattr(self, '_columns', None)

    @columns.setter
    def columns(self, val):
        '''Set partition columns.'''
        if self.container_type in (pandas.DataFrame, pandas.Series):
            self.get().columns = val
            self._columns = val

    @property
    def dtypes(self):
        '''Get data types of partitions.'''
        return self._dtypes

    @dtypes.setter
    def dtypes(self, val):
        '''Set data types of partitions.'''
        if self.container_type in (pandas.DataFrame, pandas.Series):
            self._dtypes = val

    def mask(self, row_indices, column_indices, is_series=False):
        '''Mask operation on partitions using specified indices.'''
        if row_indices is None:
            new_data = self.data.iloc[:, column_indices]
        elif column_indices is None:
            if isinstance(self.data, pandas.Series):
                new_data = self.data.iloc[row_indices]
            else:
                new_data = self.data.iloc[row_indices, :]
        else:
            if isinstance(self.data, pandas.Series):
                new_data = self.data.iloc[row_indices]
            else:
                new_data = self.data.iloc[row_indices, column_indices]

        new_data_shape = new_data.shape
        if is_series and len(new_data_shape) > 1 and new_data_shape[1] == 1:
            new_data = new_data.squeeze("columns")

        output_partition = Partition.put(data=new_data, coord=self.coord)
        return output_partition

    def squeeze(self, axis=None):
        '''Squeeze operation on partitions.'''
        if axis is None or axis == 1 and self.num_cols == 1:
            data = self.data.squeeze(axis=1)
            return Partition.put(data, coord=self.coord, container_type=type(data))
        if axis is None or axis == 0 and self.num_rows == 1:
            data = self.data.squeeze(axis=0)
            return Partition.put(data, coord=self.coord, container_type=type(data))
        return copy.copy(self)

    def set_index(self, labels):
        '''Set index of partitions with labels.'''
        self.data.index = labels

    def update(self, new_part):
        '''Update partitions with new data.'''
        self.data = new_part.data
        self.coord = new_part.coord
        self.num_rows = new_part.num_rows
        self.num_cols = new_part.num_cols
        self.index = new_part.index
        self.columns = new_part.columns
        self.valid = new_part.valid
        self.dtypes = new_part.dtypes
        self.container_type = new_part.container_type
