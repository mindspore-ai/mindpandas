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

import pandas as pd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_series_groupby():
    """
    Test series groupby with two types of tests:
            1) call func_wrapper function: sum(), count()
            2) default to pandas function: mean()
    Description: tests series groupby
    Expectation: same output as pandas series groupby
    """

    def create_input_series(module):
        return module.Series([390., 350., 30., 20.],
                             index=['Falcon', 'Falcon', 'Parrot', 'Parrot'], name="Max Speed")

    def create_input_series_with_level(module):
        index = pd.Index(['Falcon', 'Falcon', 'Parrot', 'Parrot'], name="animal")
        return module.Series([390., 350., 30., 20.], index=index, name="Max Speed")

    def create_input_multiindex_series(module):
        arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                  ['Captive', 'Wild', 'Captive', 'Wild']]
        index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
        return module.Series([390., 350., 30., 20.], index=index, name="Max Speed")

    def create_input_series_with_na(module):
        return module.Series([1, 2, 3, 3], index=["a", 'a', 'b', np.nan])

    def test_groupby_mapping(ser):
        return ser.groupby(["a", "b", "a", "b"]).sum()

    def test_groupby_mapping_default(ser):
        return ser.groupby(["a", "b", "a", "b"]).mean()

    def test_groupby_level(ser):
        return ser.groupby(level=0).sum()

    def test_groupby_level_label(ser):
        return ser.groupby(level="Type").sum()

    def test_groupby_level_default(ser):
        return ser.groupby(level=0).mean()

    def test_groupby_level_label_default(ser):
        return ser.groupby(level="Type").mean()

    def test_groupby_drop_false(ser):
        return ser.groupby(level=0, dropna=False).sum()

    def test_groupby_drop_true(ser):
        return ser.groupby(level=0, dropna=True).sum()

    def test_groupby_dict_reduced_mapping(ser):
        return ser.groupby({'Falcon': 'Falcon1', 'Parrot': 'Parrot1'}).sum()

    def test_groupby_ser_mapping(ser):
        ser = pd.Series({'Falcon': 'Falcon1', 'Falcon': 'Falcon1',
                         'Parrot': 'Parrot1', 'Parrot': 'Parrot1'})
        return ser.groupby(ser).sum()

    TESTUTIL.compare(test_groupby_mapping, create_input_series)
    TESTUTIL.compare(test_groupby_mapping_default, create_input_series)
    TESTUTIL.compare(test_groupby_level, create_input_series)
    TESTUTIL.compare(test_groupby_level, create_input_series_with_level)
    TESTUTIL.compare(test_groupby_level, create_input_multiindex_series)
    TESTUTIL.compare(test_groupby_level_label, create_input_multiindex_series)
    TESTUTIL.compare(test_groupby_level, create_input_series_with_na)
    TESTUTIL.compare(test_groupby_level_default, create_input_series)
    TESTUTIL.compare(test_groupby_level_default, create_input_series_with_level)
    TESTUTIL.compare(test_groupby_level_default, create_input_multiindex_series)
    TESTUTIL.compare(test_groupby_level_label_default, create_input_multiindex_series)
    TESTUTIL.compare(test_groupby_level_default, create_input_series_with_na)
    TESTUTIL.compare(test_groupby_drop_false, create_input_series_with_na)
    TESTUTIL.compare(test_groupby_drop_true, create_input_series_with_na)
    TESTUTIL.compare(test_groupby_dict_reduced_mapping, create_input_series)
    TESTUTIL.compare(test_groupby_ser_mapping, create_input_series)
