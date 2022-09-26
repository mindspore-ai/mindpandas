# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
import pytest
import numpy as np

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_sort_index():
    """
    Test sort_index
    Description: tests df.sort_index
    Expectation: same output as pandas df.sort_index
    """

    def create_simple_test_series(module):
        return module.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])

    def create_nan_test_series(module):
        return module.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, np.nan])

    def create_multilevel_test_series(module):
        arrays = [np.array(['qux', 'qux', 'foo', 'foo',
                            'baz', 'baz', 'bar', 'bar']),
                  np.array(['two', 'one', 'two', 'one',
                            'two', 'one', 'two', 'one'])]
        return module.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)

    def test_sort_index_simple(series):
        return series.sort_index()

    def test_sort_index_simple_ascending(series):
        return series.sort_index(ascending=False)

    def test_sort_index_simple_inplace(series):
        return series.sort_index(inplace=True)

    def test_sort_index_simple_nan(series):
        return series.sort_index(na_position='first')

    def test_sort_index_multilevel(series):
        return series.sort_index(level=1)

    def test_sort_index_remaining(series):
        return series.sort_index(level=1, sort_remaining=False)

    TESTUTIL.compare(test_sort_index_simple, create_simple_test_series)
    TESTUTIL.compare(test_sort_index_simple_ascending, create_simple_test_series)
    TESTUTIL.compare(test_sort_index_simple_inplace, create_simple_test_series)
    TESTUTIL.compare(test_sort_index_simple_nan, create_nan_test_series)
    TESTUTIL.compare(test_sort_index_multilevel, create_multilevel_test_series)
    TESTUTIL.compare(test_sort_index_remaining, create_multilevel_test_series)
