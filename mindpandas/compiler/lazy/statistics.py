# Copyright 2023 Huawei Technologies Co., Ltd
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
"""MindPandas Lazy Mode Statistics Class"""
from enum import Enum
import math


class StatSource(Enum):
    """StatSource class for statistic source"""
    UNKNOWN = 0
    ACTUAL = 1
    INFERRED = 2
    ESTIMATE = 3

    def __str__(self):
        str_map: dict = {
            StatSource.ESTIMATE: 'estimate',
            StatSource.INFERRED: 'inferred',
            StatSource.ACTUAL: 'actual',
            StatSource.UNKNOWN: 'unknown',
        }
        try:
            out = str_map[self]
        except KeyError:
            return 'unknown'
        else:
            return out


class Statistics:
    """
    Statistics() will set nrows and ncolumns to unknown
    Statistics(nrows, ncolumns) will set nrows and ncolumns according and assume these are actual cardinality
    Statistics(nrows) assumes this is from a Series and will set ncolumns to 1.
    """

    def __init__(self, nrows=None, ncolumns=1, shape=(-1, -1)):
        if nrows is None:
            self._nrows = (math.nan, StatSource.UNKNOWN)
            self._ncolumns = (math.nan, StatSource.UNKNOWN)
        else:
            self._nrows = (nrows, StatSource.ACTUAL)
            self._ncolumns = (ncolumns, StatSource.ACTUAL)
        self._part_shape = (shape, StatSource.ESTIMATE)
        #self._part_rows, self._part_columns = config.partitions

    def __str__(self):
        out = "nrows({0})= {1}, ncols({2})= {3}, part_shape({4}) = {5}".format(str(self._nrows[1]), self._nrows[0],
                                                                               str(self._ncolumns[1]),
                                                                               self._ncolumns[0],
                                                                               str(self._part_shape[1]),
                                                                               self._part_shape[0])

        return out

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, nrows):
        self._nrows: tuple = nrows

    @property
    def ncolumns(self):
        return self._ncolumns

    @ncolumns.setter
    def ncolumns(self, ncolumns):
        self._ncolumns: tuple = ncolumns

    @property
    def shape(self):
        return self._part_shape

    @shape.setter
    def shape(self, shape):
        self._part_shape: tuple = shape

    def actual(self, nrows=None, ncolumns=None):
        """actual func to get the number of rows/columns"""
        if nrows is not None:
            self._nrows = (nrows, StatSource.ACTUAL)
        if ncolumns is not None:
            self._ncolumns = (ncolumns, StatSource.ACTUAL)

    def estimate(self, nrows=None, ncolumns=None):
        """estimate func to get the statistic"""
        if nrows is not None:
            self._nrows = (nrows, StatSource.ESTIMATE)
        if ncolumns is not None:
            self._ncolumns = (ncolumns, StatSource.ESTIMATE)

    def infer(self, nrows=None, ncolumns=None, axis=None, shape=None):
        """infer func to get the statistic"""
        if nrows is not None:
            self._nrows = (nrows, StatSource.INFERRED)
        if ncolumns is not None:
            self._ncolumns = (ncolumns, StatSource.INFERRED)
        if shape is not None:
            self._part_shape = (shape, StatSource.INFERRED)
        if axis is not None:
            shape = self._part_shape[0]
            if axis == 0:
                shape = (1, shape[1])
            else:
                shape = (shape[0], 1)
            self._part_shape = (shape, StatSource.INFERRED)

    def infer_from_other(self, child_stats, axis=None):
        """infer func from children nodes to get the statistic"""
        if child_stats is not None:
            self._nrows = (child_stats.nrows[0], StatSource.INFERRED)
            self._ncolumns = (child_stats.ncolumns[0], StatSource.INFERRED)
            self._part_shape = (child_stats.shape[0], StatSource.INFERRED)
        else:
            new_stats = Statistics()
            self.infer_from_other(new_stats, axis)
        if axis is not None:
            shape = child_stats.shape[0]
            shape[axis] = 1
            self._part_shape = (shape, StatSource.INFERRED)
