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
import operator
import pytest

import pandas as pd
import mindpandas as mpd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_series_ops():
    """
    Test series comparison ops
    Description: tests series with comparison such as eq
    Expectation: same output as pandas series comparison ops
    """

    def create_input_series(module):
        ser = module.Series([1, 2, 3, 5])
        return ser

    def create_input_series_with_index(module):
        s = module.Series([100, 250, 300], index=["A", "B", "C"])
        return s

    def create_other_series_with_index(series):
        if isinstance(series, pd.Series):
            return pd.Series([100, 250, 300, 200], index=["A", "B", "C", "D"])
        return mpd.Series([100, 250, 300, 200], index=["A", "B", "C", "D"])

    def create_other_series_with_diff_index(series):
        if isinstance(series, pd.Series):
            return pd.Series([100, 250, 300, 200], index=["A", "F", "C", "D"])
        return mpd.Series([100, 250, 300, 200], index=["A", "F", "C", "D"])

    def create_other_series():
        return [-2, 4, 3, 5]

    def create_other_series_2(ser):
        if isinstance(ser, pd.Series):
            return pd.Series([2, 2, 1, 6])
        return mpd.Series([2, 2, 1, 6])

    def create_series_diff_shape_1(module):
        return module.Series([1, 2, 3, 4, 5])

    def create_series_diff_shape_2(series):
        if isinstance(series, pd.Series):
            return pd.Series([1, 2, 3])
        return mpd.Series([1, 2, 3])

    def create_series_diff_dtype_1(module):
        return module.Series([1.0, 2.0, 3.0])

    def create_series_diff_dtype_2(series):
        if isinstance(series, pd.Series):
            return pd.Series([1, 2, 3])
        return mpd.Series([1, 2, 3])

    def create_series_mixed_1(module):
        return module.Series([1, 2.0, 3, "aa", 4])

    def create_series_mixed_2(series):
        if isinstance(series, pd.Series):
            return pd.Series([1, 2.0, 3, "aa", 4.0])
        return mpd.Series([1, 2.0, 3, "aa", 4.0])

    def create_series_diff_index_1(module):
        return module.Series([1, 2, 3], index=[11, 22, 33])

    def create_series_diff_shape_3(series):
        if isinstance(series, pd.Series):
            return pd.Series([1, 2, 3], index=[11, 22, 333])
        return mpd.Series([1, 2, 3], index=[11, 22, 333])

    def create_series_issue(module):
        return module.Series([2], index=["A"])

    def create_series_issue2(module):
        return module.Series([3], index=["A"])

    def test_series_eq(s):
        return s == 5

    def test_series_ne(ser):
        return ser != 4

    def test_series_gt(ser):
        return ser > 1

    def test_series_ge(ser):
        return ser >= 1

    def test_series_lt(ser):
        return ser < 9

    def test_series_le(ser):
        return ser <= 5

    def test_series_eq_array(ser):
        other = create_other_series()
        return ser == other

    def test_series_ne_array(ser):
        other = create_other_series_2(ser)
        return ser != other

    def test_series_gt_array(ser):
        other = create_other_series_2(ser)
        return ser > other

    def test_series_ge_array(ser):
        other = create_other_series()
        return ser >= other

    def test_series_lt_array(ser):
        other = create_other_series()
        return ser < other

    def test_series_le_array(ser):
        other = create_other_series_2(ser)
        return ser <= other

    def test_series_apply(ser):
        return ser.apply(operator.sub, args=(3,))

    def test_series_filter(ser):
        return ser[[True, False, False, True]]

    def test_series_index(ser):
        return ser[1]

    def test_series_filter_with_series(module):
        ser1 = module.Series([1, 2, 3, 5])
        ser2 = module.Series([True, False, False, True])
        return ser1[ser2]

    def result(df):
        return df

    def test_series_fill_value_eq(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.eq(other_series, fill_value=0)

    def test_multidex_series_eq(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.eq(other_series)

    def test_series_fill_value_ne(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.ne(other_series, fill_value=0)

    def test_multidex_series_ne(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.ne(other_series)

    def test_series_fill_value_le(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.le(other_series, fill_value=0)

    def test_multidex_series_le(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.le(other_series)

    def test_series_fill_value_lt(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.lt(other_series, fill_value=0)

    def test_multidex_series_lt(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.lt(other_series)

    def test_series_fill_value_ge(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.ge(other_series, fill_value=0)

    def test_multidex_series_ge(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.ge(other_series)

    def test_series_fill_value_gt(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.gt(other_series, fill_value=0)

    def test_multidex_series_gt(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.gt(other_series)

    def test_err_msg_series_axis(series):
        other_series = TESTUTIL.create_another_hierarchical_series_with_nan(series)
        return series.le(other_series, axis=1)

    def test_err_msg_series_axis_2(series):
        other_series = create_other_series()
        return series.eq(other_series, axis=5)

    def test_series_diff_shape(series):
        other_series = create_series_diff_shape_2(series)
        return series.equals(other_series)

    def test_series_diff_dtype(series):
        other_series = create_series_diff_dtype_2(series)
        return series.equals(other_series)

    def test_series_mixed(series):
        other_series = create_series_mixed_2(series)
        return series.equals(other_series)

    def test_series_diff_index(series):
        other_series = create_series_diff_shape_3(series)
        return series.equals(other_series)

    def test_series_le_other_tuple(series):
        other = (1, 2, 3, 4,)
        return series.le(other)

    def test_series_le_other_tuple_2(series):
        other = (-2, 4, 3, 5)
        return series.le(other)

    def test_series_lt_other_tuple(series):
        other = (1, 2, 3, 4)
        return series.lt(other)

    def test_series_lt_other_tuple_2(series):
        other = (-2, 4, 3, 5)
        return series.lt(other)

    def test_series_eq_other_tuple(series):
        other = (1, 2, 3, 4)
        return series.eq(other)

    def test_series_eq_other_tuple_2(series):
        other = (-2, 4, 3, 5)
        return series.eq(other)

    def test_series_ne_other_tuple(series):
        other = (1, 2, 3, 4)
        return series.ne(other)

    def test_series_ne_other_tuple_2(series):
        other = (-2, 4, 3, 5)
        return series.ne(other)

    def test_series_eq_other_dict(series):
        other = {-2: -2, 4: 4, 3: 3, 5: 5}
        return series.eq(other)

    def test_series_eq_other_dict2(series):
        other = {-2: -2, 4: 4, 3: 3, 5: 5, 7: 9}
        return series.eq(other)

    def test_series_ne_other_dict(series):
        other = {-2: -2, 4: 4, 3: 3, 5: 5}
        return series.ne(other)

    def test_series_ne_other_dict2(series):
        other = {-2: -2, 4: 4, 3: 3, 5: 5, 7: 9}
        return series.ne(other)

    def test_series_eq_other_set(series):
        other = {-2, 4, 3, 5}
        return series.eq(other)

    def test_series_ne_other_set(series):
        other = {-2, 4, 3, 5}
        return series.ne(other)

    def test_series_eq_compare_series_diff_len(series):
        other_series = create_other_series_with_index(series)
        return series.eq(other_series)

    def test_series_eq_compare_series_diff_len2(series):
        other_series = create_other_series_with_diff_index(series)
        return series.eq(other_series)

    def test_series_ne_compare_series_diff_len(series):
        other_series = create_other_series_with_index(series)
        return series.ne(other_series)

    def test_series_ne_compare_series_diff_len2(series):
        other_series = create_other_series_with_diff_index(series)
        return series.ne(other_series)

    def test_series_gt_compare_series_diff_len(series):
        other_series = create_other_series_with_index(series)
        return series.gt(other_series)

    def test_series_gt_compare_series_diff_len2(series):
        other_series = create_other_series_with_diff_index(series)
        return series.gt(other_series)

    def test_series_lt_compare_series_diff_len(series):
        other_series = create_other_series_with_index(series)
        return series.lt(other_series)

    def test_series_lt_compare_series_diff_len2(series):
        other_series = create_other_series_with_diff_index(series)
        return series.lt(other_series)

    def test_series_with_index_eq_set(series):
        return series.eq({100, 150, 200})

    def test_series_with_index_eq_dict(series):
        return series.eq({100: 1, 150: 2, 200: 3})

    def test_comp_op_err(series, opt):
        other_input = create_other_series_with_index(series)
        result = pd.eval(f"series {opt} {other_input}")
        return result

    def test_series_issue(series):
        return series.eq(other=[3], fill_value=None)

    TESTUTIL.compare(test_series_eq, create_input_series)
    TESTUTIL.compare(test_series_ne, create_input_series)
    TESTUTIL.compare(test_series_gt, create_input_series)
    TESTUTIL.compare(test_series_ge, create_input_series)
    TESTUTIL.compare(test_series_lt, create_input_series)
    TESTUTIL.compare(test_series_le, create_input_series)
    TESTUTIL.compare(test_series_eq_array, create_input_series)
    TESTUTIL.compare(test_series_ne_array, create_input_series)
    TESTUTIL.compare(test_series_gt_array, create_input_series)
    TESTUTIL.compare(test_series_ge_array, create_input_series)
    TESTUTIL.compare(test_series_lt_array, create_input_series)
    TESTUTIL.compare(test_series_le_array, create_input_series)
    TESTUTIL.compare(test_series_apply, create_input_series)
    TESTUTIL.compare(test_series_filter, create_input_series)
    TESTUTIL.compare(test_series_index, create_input_series)
    TESTUTIL.compare(result, test_series_filter_with_series)
    # new tests for multindex Series  <level>, <fill_value>
    TESTUTIL.compare(test_series_fill_value_eq,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_multidex_series_eq,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_series_fill_value_ne,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_multidex_series_ne,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_series_fill_value_le,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_multidex_series_le,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_series_fill_value_lt,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_multidex_series_lt,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_series_fill_value_ge,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_multidex_series_ge,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_series_fill_value_gt,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_multidex_series_gt,
                     create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.compare(test_series_diff_shape, create_fn=create_series_diff_shape_1)
    TESTUTIL.compare(test_series_diff_dtype, create_fn=create_series_diff_dtype_1)
    TESTUTIL.compare(test_series_mixed, create_fn=create_series_mixed_1)
    TESTUTIL.compare(test_series_diff_index, create_fn=create_series_diff_index_1)
    TESTUTIL.compare(test_series_le_other_tuple, create_fn=create_input_series)
    TESTUTIL.compare(test_series_le_other_tuple_2, create_fn=create_input_series)
    TESTUTIL.compare(test_series_lt_other_tuple, create_fn=create_input_series)
    TESTUTIL.compare(test_series_lt_other_tuple_2, create_fn=create_input_series)
    TESTUTIL.compare(test_series_eq_other_tuple, create_fn=create_input_series)
    TESTUTIL.compare(test_series_eq_other_tuple_2, create_fn=create_input_series)
    TESTUTIL.compare(test_series_ne_other_tuple, create_fn=create_input_series)
    TESTUTIL.compare(test_series_ne_other_tuple_2, create_fn=create_input_series)
    TESTUTIL.compare(test_series_eq_other_dict, create_fn=create_input_series)
    TESTUTIL.compare(test_series_ne_other_dict, create_fn=create_input_series)
    TESTUTIL.compare(test_series_eq_other_set, create_fn=create_input_series)
    TESTUTIL.compare(test_series_ne_other_set, create_fn=create_input_series)
    TESTUTIL.compare(test_series_eq_other_dict2, create_fn=create_input_series)
    TESTUTIL.compare(test_series_ne_other_dict2, create_fn=create_input_series)
    TESTUTIL.compare(test_series_eq_compare_series_diff_len,
                     create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_ne_compare_series_diff_len,
                     create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_gt_compare_series_diff_len,
                     create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_lt_compare_series_diff_len,
                     create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_eq_compare_series_diff_len2,
                     create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_ne_compare_series_diff_len2,
                     create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_gt_compare_series_diff_len2,
                     create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_lt_compare_series_diff_len2,
                     create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_with_index_eq_set, create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_with_index_eq_dict, create_fn=create_input_series_with_index)
    TESTUTIL.compare(test_series_issue, create_fn=create_series_issue)
    TESTUTIL.compare(test_series_issue, create_fn=create_series_issue2)
    # new tests for error message
    TESTUTIL.run_compare_error(test_err_msg_series_axis, ValueError,
                               create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.run_compare_error(test_err_msg_series_axis_2, ValueError,
                               create_fn=create_input_series)
    TESTUTIL.run_compare_error(test_err_msg_series_axis, ValueError,
                               create_fn=TESTUTIL.create_input_hierarchical_series_with_nan)
    TESTUTIL.run_compare_error(test_err_msg_series_axis_2, ValueError,
                               create_fn=create_input_series)

    for opt in ["<", "<=", ">", ">=", "==", "!="]:
        TESTUTIL.run_compare_error_special(
            test_comp_op_err, ValueError, create_input_series_with_index, opt)
