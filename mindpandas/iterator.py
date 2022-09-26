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
"""Iterator for a mindpandas.DataFrame object."""
from collections.abc import Iterator

class DataFrameIterator(Iterator):
    """
    Iterator for a mindpandas.DataFrame object.

    Parameters
    ----------
    df (mindpandas.DataFrame): The dataframe to iterate over.
    axis {0, 1}: Axis to iterate over.
    """

    def __init__(self, df, axis):
        self.df = df
        self.axis = axis
        self.axis_bf = self.df.backend_frame.axis_frame(self.axis^1)
        self.max_idx = self.axis_bf.partitions.shape[self.axis]
        self.idx = 0
        self.axis_iter = self.axis_bf.partitions[0, 0].get().iterrows()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.axis_iter)
        except StopIteration:
            self.idx += 1
            if self.idx >= self.max_idx:
                raise StopIteration
            if self.axis == 0:
                self.axis_iter = self.axis_bf.partitions[self.idx, 0].get().iterrows()
            else:
                self.axis_iter = self.axis_bf.partitions[0, self.idx].get().iterrows()
            return next(self.axis_iter)
