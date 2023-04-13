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
"""
This module defines Operator and Function for lazy mode.
"""
from enum import Enum


class PlanOpMixin:
    """
    PlanOpMixin Class.
    """
    def __init__(self):
        # Initialize to an invalid node id
        self._node_id = 0

    @property
    def node_id(self):
        """
        return node id
        """
        return self._node_id

    @node_id.setter
    def node_id(self, node_id):
        self._node_id = node_id


class Operator(Enum):
    """
    Defines Operators
    """
    SOURCE = 0
    SINK = 1
    FILTER = 2  # Support both selection and projection
    JOIN = 3
    UNION = 4
    GROUPBY = 5
    SORT = 6
    WINDOW = 7
    SCALARF = 8
    AGGF = 9
    WINDOWF = 10
    PROJECT = 11
    SELECT = 12
    MAP1 = 13
    REDUCE = 14
    REGION = 15
    TASKCHAIN = 16
    REDUCEBYKEY = 17
    SOURCEREP = 18
    MAP2 = 19
    REDUCEPARTITIONS = 20
    SETITEM = 21
    AUX = 22
    VIEW = 23
    DEFAULT_TO_PANDAS = 24

    def __str__(self):
        str_map: dict = {
            Operator.SOURCE: 'Read',
            Operator.SINK: 'Write',
            Operator.FILTER: 'Filter',
            Operator.JOIN: 'Join',
            Operator.UNION: 'Union',
            Operator.GROUPBY: 'Groupby',
            Operator.SORT: 'Sort',
            Operator.WINDOW: 'Window',
            Operator.SCALARF: 'ScalarFn',
            Operator.AGGF: 'AggregateFn',
            Operator.WINDOWF: 'WindowFn',
            Operator.PROJECT: 'Project',
            Operator.SELECT: 'Select',
            Operator.MAP1: 'Map',
            Operator.MAP2: 'Map2',
            Operator.REDUCE: 'Reduce',
            Operator.REDUCEPARTITIONS: 'ReducePartitions',
            Operator.REGION: 'Region',
            Operator.TASKCHAIN: 'TaskChain',
            Operator.REDUCEBYKEY: 'ReduceByKey',
            Operator.SOURCEREP: 'ReadRepartition',
            Operator.SETITEM: 'SetItem',
            Operator.AUX: 'Aux',
            Operator.VIEW: 'View',
            Operator.DEFAULT_TO_PANDAS: 'DefaultToPandas'
        }
        try:
            string = str_map[self]
        except KeyError:
            return "Unknown"
        else:
            return string


class Function(Enum):
    """
    Defines Functions
    """
    DATAFRAME = 1
    SERIES = 2
    READ_CSV = 10
    GROUPBY = 11
    MERGE = 12
    MATH = 13
    SUM = 20
    COUNT = 21
    MIN = 22
    MAX = 23
    SIZE = 24
    MEAN = 25
    MEDIAN = 26
    ISNA = 30
    SUMW = 40
    COUNTW = 41
    RANK = 42
    COMPOP = 50
    PROJECT1 = 60
    PROJECTN = 61
    SELECT = 62
    VIEW = 63
    TO_CSV = 70
    UDF = 100
    NODE = 200
    TASKCHAIN = 201
    GETITEM = 300
    SETITEM = 301
    APPLY = 302
    REPLACE = 303
    DROP = 304
    FILLNA = 305
    APPLYMAP = 306
    NOOP = 307
    AUX = 308
    DEFAULT_TO_PANDAS = 400

    def __str__(self):
        str_map: dict = {
            Function.DATAFRAME: 'DataFrame',
            Function.SERIES: 'Series',
            Function.READ_CSV: 'read_csv',
            Function.GROUPBY: 'groupby',
            Function.MERGE: 'merge',
            Function.MATH: 'math_op',
            Function.SUM: 'sum',
            Function.COUNT: 'count',
            Function.MIN: 'min',
            Function.MAX: 'max',
            Function.SIZE: 'size',
            Function.MEAN: 'mean',
            Function.MEDIAN: 'median',
            Function.ISNA: 'isna',
            Function.SUMW: 'sum',
            Function.COUNTW: 'count',
            Function.RANK: 'rank',
            Function.COMPOP: 'compop',
            Function.PROJECT1: 'get_item',
            # Function.PROJECT1: 'project-1',
            # Function.PROJECTN: 'project-n',
            Function.PROJECTN: 'get_item',
            Function.SELECT: 'select',
            Function.TO_CSV: 'to_csv',
            Function.UDF: 'UDF',
            Function.NODE: 'NODE_REFERENCE',
            Function.TASKCHAIN: 'TASKCHAIN',
            Function.GETITEM: 'get_item',
            Function.SETITEM: 'set_item',
            Function.APPLY: 'apply',
            Function.REPLACE: 'replace',
            Function.DROP: 'drop',
            Function.FILLNA: 'fill_na',
            Function.APPLYMAP: 'applymap',
            Function.NOOP: 'noop',
            Function.AUX: 'AUX',
            Function.VIEW: 'view',
            Function.DEFAULT_TO_PANDAS: 'default_to_pandas'
        }
        try:
            string = str_map[self]
        except KeyError:
            return "Unknown"
        else:
            return string


fn_op_map: dict = {
    # Maintain a dictionary of { opname: run-time function }
    # All "run-time" functions must have the same signature of the node id of
    # the current operator followed by its keyword arguments.
    Function.READ_CSV: Operator.SOURCE,
    Function.DATAFRAME: Operator.SOURCE,
    Function.SERIES: Operator.SOURCE,
    Function.GROUPBY: Operator.GROUPBY,
    Function.SUM: Operator.AGGF,
    Function.COUNT: Operator.AGGF,
    Function.MIN: Operator.AGGF,
    Function.MAX: Operator.AGGF,
    Function.SIZE: Operator.AGGF,
    Function.MEAN: Operator.AGGF,
    Function.MEDIAN: Operator.AGGF,
    Function.ISNA: Operator.SCALARF,
    Function.MERGE: Operator.JOIN,
    Function.MATH: Operator.JOIN,
    Function.COMPOP: Operator.JOIN,
    # Function.PROJECT1: Operator.MAP1,
    # Function.PROJECTN: Operator.MAP1,
    Function.PROJECT1: Operator.PROJECT,
    Function.PROJECTN: Operator.PROJECT,
    Function.SELECT: Operator.SELECT,
    Function.TO_CSV: Operator.SINK,
    Function.NODE: Operator.SOURCE,
    Function.GETITEM: Operator.MAP1,
    # Function.SETITEM: Operator.MAP2,
    Function.APPLY: Operator.REDUCEPARTITIONS,
    Function.SETITEM: Operator.JOIN,
    Function.REPLACE: Operator.MAP1,
    Function.DROP: Operator.MAP1,
    Function.FILLNA: Operator.MAP1,
    Function.APPLYMAP: Operator.MAP1,
    Function.VIEW: Operator.VIEW,
    Function.DEFAULT_TO_PANDAS: Operator.DEFAULT_TO_PANDAS,
}
