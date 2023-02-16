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
"""pandas factory for lazy mode"""
import pandas


def _task_chain_helper(graph, multi_input, df_list, dest=None):
    import mindpandas.dist.local_executor as ex
    executor = ex.Executor(graph, multi_input, dest=dest)
    return executor.run(df_list)


def task_chain(graph, multi_input=False, dest=None):
    # TODO: tbd if it is better to create executor once or create inside the function.
    # param: dest: the node that its result should be returned. If None, we return the root results.
    if multi_input:
        return lambda df_input_list, df2: _task_chain_helper(graph, multi_input, [df_input_list, df2], dest=dest)
    return lambda df_input_list: _task_chain_helper(graph, multi_input, [df_input_list], dest=dest)


class CSVIterator:
    """CSV iterator"""

    def __init__(self, file_path, chunksize):
        self.reader = None
        self.file_path = file_path
        self.chunksize = chunksize

    def __iter__(self):
        return self

    def __next__(self):
        if self.reader is None:
            self.reader = pandas.read_csv(
                self.file_path, chunksize=self.chunksize)
        chunk = self.reader.get_chunk()
        return chunk


class KeyBy:
    """KeyBy class"""

    def __init__(self, key_field):
        self.key_field = key_field

    def __call__(self, data):
        # return a list of (group key, grouped df)
        result = []
        gb = data.groupby(self.key_field)
        for key, df in gb:
            output = (key, df.set_index(self.key_field))
            result.append(output)
        return result


class ToCSV:
    """ToCSV class"""

    def __init__(self, filepath, chunksize, **kwargs):
        self.timestemp_in_name = kwargs.get('timestamp_in_name', False)
        self.filepath = filepath
        self.chunksize = chunksize
        self.kwargs = kwargs
        self.buffer_df = None

    def __call__(self, data):
        if isinstance(data, dict):
            for _, df in data.items():
                if self.buffer_df is None:
                    self.buffer_df = df
                else:
                    self.buffer_df = self.buffer_df.append(df)

            # TODO: How to write the last chunk?
            if len(self.buffer_df) > self.chunksize:
                self.buffer_df.to_csv(self.filepath, mode='a')
                self.buffer_df = None
        else:
            filename = self.filepath
            data.to_csv(filename)


def noop():
    return lambda x: x
