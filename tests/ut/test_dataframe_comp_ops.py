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
import mindpandas as mpd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_dataframe_comp_ops():
    """
    Test dataframe_comp_ops
    Description: tests df comparison such as df.eq
    Expectation: same output as pandas df comparison
    """

    def create_input_df(module):
        np.random.seed(100)
        return module.DataFrame(np.random.randint(10, size=(100, 100)))

    def create_other_df():
        np.random.seed(200)
        return np.random.randint(10, size=(100, 100))

    def create_other_df_2(df):
        np.random.seed(200)
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(np.random.randint(10, size=(100, 100)))
        return mpd.DataFrame(np.random.randint(10, size=(100, 100)))

    def create_other_series():
        np.random.seed(200)
        return np.random.randint(10, size=(100,))

    def create_other_series_2(df):
        np.random.seed(200)
        if isinstance(df, pd.DataFrame):
            return pd.Series(np.random.randint(10, size=(100)))
        return mpd.Series(np.random.randint(10, size=(100)))

    def create_df1_diff_col_name_1(module):
        data1 = {'key1': ['a', 'b', 'c', 'd'], 'key2': ['e', 'f', 'g', 'h']}
        return module.DataFrame(data1)

    def create_df1_diff_col_name_2(df):
        data2 = {'key1': ['a', 'b', 'c', 'd'], 'key3': ['e', 'f', 'g', 'h']}
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(data2)
        return mpd.DataFrame(data2)

    def create_df1_diff_index_1(module):
        data1 = {'key1': ['a', 'b', 'c', 'd'], 'key2': ['e', 'f', 'g', 'h']}
        index1 = ['k', 'l', 'm', 'n']
        return module.DataFrame(data1, index=index1)

    def create_df1_diff_index_2(df):
        data1 = {'key1': ['a', 'b', 'c', 'd'], 'key2': ['e', 'f', 'g', 'h']}
        index2 = ['p', 'q', 'u', 'v']
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(data1, index=index2)
        return mpd.DataFrame(data1, index=index2)

    def create_df1_diff_shape_1(module):
        data1 = {'key1': ['a', 'b', 'c', 'd'], 'key2': ['e', 'f', 'g', 'h']}
        return module.DataFrame(data1)

    def create_df1_diff_shape_2(df):
        data2 = {'key1': ['a', 'b', 'c', 'd']}
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(data2)
        return mpd.DataFrame(data2)

    def create_df1_diff_dtypes_1(module):
        data1 = {'key1': [1.0, 2.0, 3.0], 'key2': [4, 5, 6]}
        return module.DataFrame(data1)

    def create_df1_diff_dtypes_2(df):
        data2 = {'key1': [1, 2, 3], 'key2': [4, 5, 6]}
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(data2)
        return mpd.DataFrame(data2)

    def create_df1_diff_dtypes_3(df):
        data2 = {'key1': ["aa", "bb", "cc"], 'key2': [4, 5, 6]}
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(data2)
        return mpd.DataFrame(data2)

    def create_issue_datafrane(module):
        return module.DataFrame({'cost': ["100", 150, 100],
                                 "revenue": [100, 250, 300]},
                                index=['A', 'B', 'C'])

    # big multilevel, multindex dataframe (625*4)
    def create_big_df_multindex_multilevel(module):
        np.random.seed(188)

        def make_lable(prefix, num):
            return [f"{prefix}er{i}er" for i in range(num)]

        miindex = pd.MultiIndex.from_product(
            [make_lable("A", 5), make_lable("B", 5),
             make_lable("C", 5), make_lable("D", 5)])
        micolumns = pd.MultiIndex.from_tuples(
            [("class_1", "c1_type1"), ("class_1", "c1_type2"),
             ("class_2", "c2_type1"), ("class_2", "c2_type2")],
            names=["upper_level", "lower_level"])

        df = (
            module.DataFrame(
                np.array(np.random.randint(10, size=(len(miindex) * len(micolumns)))).reshape(
                    (len(miindex), len(micolumns))
                ),
                index=miindex,
                columns=micolumns,
            )
            .sort_index()
            .sort_index(axis=1))
        return df

    def create_big_compared_df_multindex_multilevel(df):
        np.random.seed(188)

        def make_lable(prefix, num):
            return [f"{prefix}er{i}er" for i in range(num)]

        miindex = pd.MultiIndex.from_product(
            [make_lable("A", 5), make_lable("B", 5),
             make_lable("C", 5), make_lable("D", 5)])
        micolumns = pd.MultiIndex.from_tuples(
            [("class_1", "c1_type1"), ("class_1", "c1_type2"),
             ("class_2", "c2_type1"), ("class_2", "c2_type2")],
            names=["upper_level", "lower_level"])

        if isinstance(df, pd.DataFrame):
            df_return = (
                pd.DataFrame(
                    np.array(np.random.randint(10, size=(len(miindex) * len(micolumns)))).reshape(
                        (len(miindex), len(micolumns))
                    ),
                    index=miindex,
                    columns=micolumns,
                )
                .sort_index()
                .sort_index(axis=1))

        elif isinstance(df, mpd.DataFrame):
            df_return = (
                mpd.DataFrame(
                    np.array(np.random.randint(10, size=(len(miindex) * len(micolumns)))).reshape(
                        (len(miindex), len(micolumns))
                    ),
                    index=miindex,
                    columns=micolumns,
                )
                .sort_index()
                .sort_index(axis=1))
        return df_return

    def create_df_random(module):
        data = np.random.randn(10, 5)
        return module.DataFrame(data)

    def create_df_random_index(df):
        other_data = np.random.randn(10, 5)
        other_index = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(other_data, index=other_index)
        return mpd.DataFrame(other_data, index=other_index)

    def test_df_eq(df):
        return df == 5

    def test_df_ne(df):
        return df != 4

    def test_df_gt(df):
        return df > 1

    def test_df_ge(df):
        return df >= 1

    def test_df_lt(df):
        return df < 9

    def test_df_le(df):
        return df <= 5

    def test_df_eq_array(df):
        other = create_other_df()
        return df == other

    def test_df_ne_array(df):
        other = create_other_df_2(df)
        return df.ne(other, axis="columns")

    def test_df_gt_array(df):
        other = create_other_series()
        return df > other

    def test_df_ge_array(df):
        other = create_other_df()
        return df.ge(other, axis="index")

    def test_df_lt_array(df):
        other = create_other_series_2(df)
        return df.lt(other, axis=0)

    def test_df_le_array(df):
        other = create_other_df_2(df)
        return df.le(other, axis=1)

    def test_df_equal(df):
        other = create_other_df()
        return df.equals(other)

    def test_df_equal_2(df):
        return df.equals(df)

    def test_df_equal_series(df):
        other = create_other_series()
        return df.equals(other)

    def test_df_equals_col_name(df):
        other = create_df1_diff_col_name_2(df)
        return df.equals(other)

    def test_df_equals_index(df):
        other = create_df1_diff_index_2(df)
        return df.equals(other)

    def test_df_equals_shape(df):
        other = create_df1_diff_shape_2(df)
        return df.equals(other)

    def test_df_equals_dtypes(df):
        other = create_df1_diff_dtypes_2(df)
        return df.equals(other)

    def test_df_eq_list(df):
        other = [1, 2]
        return df.eq(other)

    def test_df_eq_level(df):
        other_df = create_big_compared_df_multindex_multilevel(df)
        return df.eq(other_df, level="upper_level")

    # new tests in multindex, multlevel Dataframe
    def test_df_le_2(df):
        return df <= 60

    def test_df_le_3(df):
        return df <= 99

    def test_df_le_df(df):
        other_df = create_big_compared_df_multindex_multilevel(df)
        return df.le(other_df)

    def test_df_ge_2(df):
        return df >= 19

    def test_df_ge_3(df):
        return df >= 23

    def test_df_ge_df(df):
        other_df = create_big_compared_df_multindex_multilevel(df)
        return df.ge(other_df)

    def test_df_lt_2(df):
        return df < 9

    def test_df_lt_3(df):
        return df < 13

    def test_df_lt_df(df):
        other_df = create_big_compared_df_multindex_multilevel(df)
        return df.lt(other_df)

    def test_df_gt_2(df):
        return df > 79

    def test_df_gt_3(df):
        return df > 93

    def test_df_gt_df(df):
        other_df = create_big_compared_df_multindex_multilevel(df)
        return df.gt(other_df)

    def test_df_eq_2(df):
        return df == 78

    def test_df_eq_3(df):
        return df == 98

    def test_df_eq_df(df):
        other_df = create_big_compared_df_multindex_multilevel(df)
        return df.eq(other_df)

    def test_df_ne_2(df):
        return df != 63

    def test_df_ne_3(df):
        return df != 28

    def test_df_ne_df(df):
        other_df = create_big_compared_df_multindex_multilevel(df)
        return df.ne(other_df)

    def test_df_eq_diff_type(df):
        other_df = create_df1_diff_dtypes_3(df)
        return df == other_df

    def test_df_ne_diff_type(df):
        other_df = create_df1_diff_dtypes_3(df)
        return df != other_df

    def test_df_le_issue(df):
        other_df = create_df_random_index(df)
        return df.le(other_df)


    def test_df_diff_index_str(df):
        other_df = TESTUTIL.create_df_index_str_list_2(df, len(df.index)//2)
        return df.ge(other_df)

    def test_df_diff_index_int(df):
        other_df = TESTUTIL.create_df_index_integer_list_2(df, len(df.index)//2)
        return df.ge(other_df)

    def test_df_diff_index_range(df):
        other_df = TESTUTIL.create_df_index_range_2(df, len(df.index)//2)
        return df.ge(other_df)

    # def test_err_msg_shape(df, opt):
    #     other_df = create_df1_diff_shape_2(df)
    #     result = pd.eval("df opt other_df")
    #     return result

    # def test_err_msg_index(df, opt):
    #     other_df = create_df1_diff_index_2(df)
    #     result = pd.eval("df opt other_df")
    #     return result

    # def test_err_msg_type(df, opt):
    #     other_df = create_df1_diff_dtypes_3(df)
    #     result = pd.eval(f"{df} {opt} {other_df}")
    #     return result

    def test_df_dict_issue2(df):
        return df.eq(other=(100, 20), axis=1)

    TESTUTIL.compare(test_df_eq, create_input_df)
    TESTUTIL.compare(test_df_ne, create_input_df)
    TESTUTIL.compare(test_df_gt, create_input_df)
    TESTUTIL.compare(test_df_ge, create_input_df)
    TESTUTIL.compare(test_df_lt, create_input_df)
    TESTUTIL.compare(test_df_le, create_input_df)
    TESTUTIL.compare(test_df_eq_array, create_input_df)
    TESTUTIL.compare(test_df_ne_array, create_input_df)
    TESTUTIL.compare(test_df_gt_array, create_input_df)
    TESTUTIL.compare(test_df_ge_array, create_input_df)
    TESTUTIL.compare(test_df_lt_array, create_input_df)
    TESTUTIL.compare(test_df_le_array, create_input_df)
    TESTUTIL.compare(test_df_equal, create_input_df)
    TESTUTIL.compare(test_df_equal_2, create_input_df)
    TESTUTIL.compare(test_df_equal_series, create_input_df)
    # new tests
    TESTUTIL.compare(test_df_le_2, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_le_3, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_le_df, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_ge_2, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_ge_3, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_ge_df, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_lt_2, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_lt_3, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_lt_df, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_gt_2, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_gt_3, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_gt_df, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_eq_2, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_eq_3, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_eq_df, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_ne_2, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_ne_3, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_ne_df, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_dict_issue2, create_fn=create_issue_datafrane)
    # test shape, column names, index and dtypes
    TESTUTIL.compare(test_df_equals_col_name, create_fn=create_df1_diff_col_name_1)
    TESTUTIL.compare(test_df_equals_index, create_fn=create_df1_diff_index_1)
    TESTUTIL.compare(test_df_equals_shape, create_fn=create_df1_diff_shape_1)
    TESTUTIL.compare(test_df_equals_dtypes, create_fn=create_df1_diff_dtypes_1)
    TESTUTIL.compare(test_df_eq_diff_type, create_fn=create_df1_diff_dtypes_1)
    TESTUTIL.compare(test_df_ne_diff_type, create_fn=create_df1_diff_dtypes_1)
    TESTUTIL.compare(test_df_eq_list, create_fn=create_df1_diff_col_name_1)
    TESTUTIL.compare(test_df_eq_level, create_fn=create_big_df_multindex_multilevel)
    TESTUTIL.compare(test_df_le_issue, create_fn=create_df_random)

    # test copartition for two dataframes with different index
    TESTUTIL.compare(test_df_diff_index_str, create_fn=TESTUTIL.create_df_index_str_list)
    TESTUTIL.compare(test_df_diff_index_int, create_fn=TESTUTIL.create_df_index_integer_list)
    TESTUTIL.compare(test_df_diff_index_range, create_fn=TESTUTIL.create_df_index_range)


    # for opt in ["<", "<=", ">", ">=", "==", "!="]:
    #     TESTUTIL.run_compare_error_special(
    #         test_err_msg_shape, ValueError, create_df1_diff_shape_1, opt)
    #     TESTUTIL.run_compare_error_special(
    #         test_err_msg_index, ValueError, create_df1_diff_index_1, opt)
    #     if opt not in ["==", "!="]:
    #         TESTUTIL.run_compare_error_special(
    #             test_err_msg_type, TypeError, create_df1_diff_dtypes_2, opt)
