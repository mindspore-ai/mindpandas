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
""" Mindpandas Lazy Mode Function Factory Class"""


class FunctionFactoryLazy:
    """ Mindpandas Lazy Mode Function Factory Class"""

    @classmethod
    def sum_reduce(cls, axis=0, skipna=True, numeric_only=False, min_count=0, **kwargs):
        return cls.ff_.ReduceSum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @classmethod
    def task_chain(cls, graph, **kwargs):
        return cls.ff_.task_chain(graph, **kwargs)

    @classmethod
    def csv_iterator(cls, file_path, chunksize, **kwargs):
        return cls.ff_.CSVIterator(file_path, chunksize, **kwargs)

    @classmethod
    def keyby(cls, key_field):
        return cls.ff_.KeyBy(key_field)

    @classmethod
    def to_csv(cls, filepath, chunksize, **kwargs):
        return cls.ff_.ToCSV(filepath, chunksize, **kwargs)

    @classmethod
    def noop(cls, **kwargs):
        return cls.ff_.noop(**kwargs)
