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
# ==============================================================================
import pytest
import numpy as np

import pandas as pd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_default_to_pandas_boolean_ops():
    """
    Test groupby default to pandas boolean ops
    Description: tests df.groupby.all and df.groupby.any
    Expectation: same output as pandas for those ops
    """
    def create_groupby_default_to_pandas_test_frame_boolean(module):
        data = {'A': ['a', 'b', 'c', 'a', 'b', 'c'],
                'B': [1, 1, 0, 1, 0, 0]}
        return module.DataFrame(data)

    def groupby_all(dataframe):
        return dataframe.groupby(by='A').all()

    def groupby_any(dataframe):
        return dataframe.groupby(by='B').any()

    TESTUTIL.compare(
        groupby_all, create_fn=create_groupby_default_to_pandas_test_frame_boolean)
    TESTUTIL.compare(
        groupby_any, create_fn=create_groupby_default_to_pandas_test_frame_boolean)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_default_to_pandas_nunique_op():
    """
    Test groupby with nunique
    Description: tests df.groupby.nunique
    Expectation: same output as pandas df.groupby.nunique
    """
    def create_groupby_default_to_pandas_test_frame_nunique(module):
        data = {'id': ['spam', 'egg', 'egg', 'spam', 'ham', 'ham'],
                'value1': [1, 5, 5, 2, 5, 5],
                'value2': list('abbaxy')}
        return module.DataFrame(data)

    def groupby_nunique(dataframe):
        return dataframe.groupby(by='id').nunique()

    TESTUTIL.compare(
        groupby_nunique, create_fn=create_groupby_default_to_pandas_test_frame_nunique)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_default_to_pandas_nan_ops():
    """
    Test groupby with ops that fill nan
    Description: tests df.groupby with ops that fill nan
    Expectation: same output as pandas for those ops
    """
    def create_groupby_default_to_pandas_test_frame_nan(module):
        data = {'A': ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
                'B': [np.nan, np.nan, 1, np.nan, 2, np.nan, 3, np.nan, np.nan, np.nan, 4]}
        return module.DataFrame(data)

    def groupby_fillna(dataframe):
        return dataframe.groupby(by='A').fillna(value=0)

    def groupby_bfill(dataframe):
        return dataframe.groupby(by='A').bfill()

    def groupby_backfill(dataframe):
        return dataframe.groupby(by='A').backfill(limit=2)

    def groupby_ffill(dataframe):
        return dataframe.groupby(by='A').ffill()

    def groupby_pad(dataframe):
        return dataframe.groupby(by='A').pad(limit=1)

    TESTUTIL.compare(
        groupby_fillna, create_fn=create_groupby_default_to_pandas_test_frame_nan)
    TESTUTIL.compare(
        groupby_bfill, create_fn=create_groupby_default_to_pandas_test_frame_nan)
    TESTUTIL.compare(groupby_backfill,
                     create_fn=create_groupby_default_to_pandas_test_frame_nan)
    TESTUTIL.compare(
        groupby_ffill, create_fn=create_groupby_default_to_pandas_test_frame_nan)
    TESTUTIL.compare(
        groupby_pad, create_fn=create_groupby_default_to_pandas_test_frame_nan)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_default_to_pandas_corrwith_filter_ops():
    """
    Test groupby with corrwith and filter
    Description: tests df.groupby.corrwith and df.groupby.filter
    Expectation: same output as pandas for those ops
    """
    def create_groupby_default_to_pandas_test_frame_two_dfs(module):
        data1 = {'A': ['a', 'b', 'a', 'b', 'a', 'b'],
                 'B': [4, 8, 2, 6, 1, 3],
                 'C': [7.0, 5., 2., 2., 0., 9.]}
        data2 = {'A': ['a', 'b', 'a', 'b', 'a', 'b'],
                 'B': [5, 7, 3, 5, 2, 4],
                 'C': [3.6, 2.6, 7.5, 4.9, 8.8, 0.9]}
        return module.DataFrame(data1), module.DataFrame(data2)

    def groupby_corrwith(dfs):
        return dfs[0].groupby(by='A').corrwith(other=dfs[1])

    def groupby_filter(dfs):
        return dfs[0].groupby(by='A').filter(func=lambda x: x['B'].mean() > 3.)

    def groupby_as_index(dfs):
        return dfs[0].groupby(by='A', as_index=False).filter(func=lambda x: x['B'].mean() <= 3.)

    TESTUTIL.compare(
        groupby_corrwith, create_fn=create_groupby_default_to_pandas_test_frame_two_dfs)
    TESTUTIL.compare(
        groupby_filter, create_fn=create_groupby_default_to_pandas_test_frame_two_dfs)
    TESTUTIL.compare(
        groupby_as_index, create_fn=create_groupby_default_to_pandas_test_frame_two_dfs)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_default_to_pandas_periodic_ops():
    """
    Test groupby with periodic ops
    Description: tests df.groupby with periodic ops
    Expectation: same output as pandas for those ops
    """
    def create_groupby_default_to_pandas_test_frame_periodic(module):
        idx = pd.date_range('1/1/2000', periods=4, freq='T')
        data = 4 * [range(2)]
        columns = ['a', 'b']
        dataframe = module.DataFrame(data, index=idx, columns=columns)
        dataframe.iloc[2, 0] = 5
        return dataframe

    def groupby_shift(dataframe):
        return dataframe.groupby(by='a').shift(periods=1)

    def groupby_tshift(dataframe):
        return dataframe.groupby(by='a').tshift(periods=1, freq=pd.infer_freq(dataframe.index))

    def groupby_pct_change(dataframe):
        return dataframe.groupby(by='a').pct_change()

    TESTUTIL.compare(
        groupby_shift, create_fn=create_groupby_default_to_pandas_test_frame_periodic)
    TESTUTIL.compare(
        groupby_tshift, create_fn=create_groupby_default_to_pandas_test_frame_periodic)
    TESTUTIL.compare(groupby_pct_change,
                     create_fn=create_groupby_default_to_pandas_test_frame_periodic)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_default_to_pandas_sample_op():
    """
    Test groupby with sample op
    Description: tests df.groupby.sample
    Expectation: same output as pandas df.groupby.sample
    """
    def create_groupby_default_to_pandas_test_frame_sample(module):
        data = {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
        return module.DataFrame(data)

    def groupby_sample(dataframe):
        return dataframe.groupby(by="a").sample(n=1, random_state=1)

    TESTUTIL.compare(
        groupby_sample, create_fn=create_groupby_default_to_pandas_test_frame_sample)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_default_to_pandas_take_op():
    """
    Test groupby with take
    Description: tests df.groupby.take
    Expectation: same output as pandas df.groupby.take
    """
    def create_groupby_default_to_pandas_test_frame_take(module):
        data = [('falcon', 'bird', 389.0),
                ('parrot', 'bird', 24.0),
                ('lion', 'mammal', 80.5),
                ('monkey', 'mammal', 25.0)]
        columns = ['name', 'class', 'max_speed']
        index = [0, 2, 3, 1]
        return module.DataFrame(data, index, columns)

    def groupby_take(dataframe):
        return dataframe.groupby(by='name').take(indices=[0])

    TESTUTIL.compare(
        groupby_take, create_fn=create_groupby_default_to_pandas_test_frame_take)


@pytest.mark.usefixtures("set_mode", "set_shape")
def groupby_default_to_pandas_ohlc_prod_diff_rank_cumprod_ops(by_key, csv_file):
    """
    Test groupby with ohlc, prod, diff, rank, cumprod ops
    Description: tests df.groupby with ohlc, prod, diff, rank, cumprod ops
    Expectation: same output as pandas for those ops
    """
    def create_groupby_default_to_pandas_test_frame_simple_csv(module):
        return module.read_csv(csv_file)

    def groupby_ohlc(dataframe):
        return dataframe.groupby(by=by_key).ohlc()

    def groupby_prod(dataframe):
        return dataframe.groupby(by=by_key).prod(numeric_only=False)

    def groupby_diff(dataframe):
        return dataframe.groupby(by=by_key).diff()

    def groupby_rank(dataframe):
        return dataframe.groupby(by=by_key).rank()

    def groupby_cumprod(dataframe):
        return dataframe.groupby(by=by_key).cumprod()

    TESTUTIL.compare(
        groupby_ohlc, create_fn=create_groupby_default_to_pandas_test_frame_simple_csv)
    TESTUTIL.compare(
        groupby_prod, create_fn=create_groupby_default_to_pandas_test_frame_simple_csv)
    TESTUTIL.compare(
        groupby_rank, create_fn=create_groupby_default_to_pandas_test_frame_simple_csv)
    TESTUTIL.compare(
        groupby_cumprod, create_fn=create_groupby_default_to_pandas_test_frame_simple_csv)
    TESTUTIL.compare(
        groupby_diff, create_fn=create_groupby_default_to_pandas_test_frame_simple_csv)


@pytest.mark.usefixtures("set_mode", "set_shape")
def groupby_default_to_pandas_major_ops(by_key, csv_file):
    """
    Test groupby with most ops
    Description: tests df.groupby with most ops
    Expectation: same output as pandas for those ops
    """
    def create_groupby_default_to_pandas_test_frame_csv(module):
        return module.read_csv(csv_file)

    def groupby_boxplot(dataframe):
        return dataframe.groupby(by=by_key).boxplot()

    def groupby_cumcount(dataframe):
        return dataframe.groupby(by=by_key).cumcount()

    def groupby_cummax(dataframe):
        return dataframe.groupby(by=by_key).cummax()

    def groupby_cummin(dataframe):
        return dataframe.groupby(by=by_key).cummin()

    def groupby_cumsum(dataframe):
        return dataframe.groupby(by=by_key).cumsum()

    def groupby_first(dataframe):
        return dataframe.groupby(by=by_key).first(numeric_only=False, min_count=50)

    def groupby_head(dataframe):
        return dataframe.groupby(by=by_key).head(n=5)

    def groupby_last(dataframe):
        return dataframe.groupby(by=by_key).last(numeric_only=False, min_count=50)

    def groupby_max(dataframe):
        return dataframe.groupby(by=by_key).max()

    def groupby_mean(dataframe):
        return dataframe.groupby(by=by_key).mean()

    def groupby_median(dataframe):
        return dataframe.groupby(by=by_key).median()

    def groupby_min(dataframe):
        return dataframe.groupby(by=by_key).min()

    def groupby_ngroup(dataframe):
        return dataframe.groupby(by=by_key).ngroup()

    def groupby_nth(dataframe):
        return dataframe.groupby(by=by_key).nth(n=[12, 6, 2, 11, 5, 10, 1, 8])

    def groupby_sem(dataframe):
        return dataframe.groupby(by=by_key).sem()

    def groupby_std(dataframe):
        return dataframe.groupby(by=by_key).std()

    def groupby_var(dataframe):
        return dataframe.groupby(by=by_key).var()

    def groupby_tail(dataframe):
        return dataframe.groupby(by=by_key).tail()

    def groupby_corr(dataframe):
        return dataframe.groupby(by=by_key).corr(method='pearson')

    def groupby_cov(dataframe):
        return dataframe.groupby(by=by_key).cov()

    def groupby_describe(dataframe):
        return dataframe.groupby(by=by_key).describe()

    def groupby_hist(dataframe):
        return dataframe.groupby(by=by_key).hist()

    def groupby_idxmax(dataframe):
        return dataframe.groupby(by=by_key).idxmax()

    def groupby_idxmin(dataframe):
        return dataframe.groupby(by=by_key).idxmin()

    def groupby_mad(dataframe):
        return dataframe.groupby(by=by_key).mad(axis=0)

    def groupby_plot(dataframe):
        return dataframe.groupby(by=by_key).plot()

    def groupby_quantile(dataframe):
        return dataframe.groupby(by=by_key).quantile(q=0.5, interpolation='linear')

    def groupby_skew(dataframe):
        return dataframe.groupby(by=by_key).skew(axis=0)

    # Check visually instead of with assert comparison
    TESTUTIL.compare(
        groupby_boxplot, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_hist, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_plot, create_fn=create_groupby_default_to_pandas_test_frame_csv)

    # Check with assert comparison
    TESTUTIL.compare(groupby_cumcount,
                     create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_cummax, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_cummin, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_cumsum, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_head, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_nth, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_tail, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_corr, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_first, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_last, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_max, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_mean, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_median, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_min, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_ngroup, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_sem, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_std, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_var, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_cov, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(groupby_describe,
                     create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_idxmax, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_idxmin, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_mad, create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(groupby_quantile,
                     create_fn=create_groupby_default_to_pandas_test_frame_csv)
    TESTUTIL.compare(
        groupby_skew, create_fn=create_groupby_default_to_pandas_test_frame_csv)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_basic():
    """
    Test groupby with basic cases
    Description: tests df.groupby with basic cases
    Expectation: same output as pandas for those cases
    """

    def test_gb_count(df):
        return df.groupby(by=by_keys).count()

    def test_gb_size(df):
        return df.groupby(by=by_keys).size()

    def test_gb_min(df):
        return df.groupby(by=by_keys).min()

    def test_gb_max(df):
        return df.groupby(by=by_keys).max()

    def test_gb_sum(df):
        return df.groupby(by=by_keys).sum()

    def test_gb_prod(df):
        return df.groupby(by=by_keys).prod()

    def test_gb_mean(df):
        return df.groupby(by=by_keys).mean()

    def test_gb_cumsum(df):
        return df.groupby(by=by_keys).cumsum()

    def test_gb_median(df):
        return df.groupby(by=by_keys).median()

    def test_gb_size_2(df):
        return df.groupby(by=by_keys, as_index=False).size()

    def test_gb_count_2(df):
        return df.groupby(by=by_keys, as_index=False).count()

    def test_gb_min_2(df):
        return df.groupby(by=by_keys, as_index=False).min()

    def test_gb_sum_2(df):
        return df.groupby(by=by_keys, as_index=False).sum()

    def test_gb_prod_2(df):
        return df.groupby(by=by_keys, as_index=False).prod()

    def test_gb_mean_2(df):
        return df.groupby(by=by_keys, as_index=False).mean()

    def test_gb_cumsum_2(df):
        x = df.groupby(by=by_keys, as_index=False).cumsum()
        return x

    def test_gb_median_2(df):
        x = df.groupby(by=by_keys, as_index=False).median()
        return x

    by_keys = 'A'
    TESTUTIL.compare(test_gb_count)
    TESTUTIL.compare(test_gb_size)
    TESTUTIL.compare(test_gb_min)
    TESTUTIL.compare(test_gb_max)
    TESTUTIL.compare(test_gb_sum)
    TESTUTIL.compare(test_gb_prod)
    TESTUTIL.compare(test_gb_mean)
    TESTUTIL.compare(test_gb_cumsum)
    TESTUTIL.compare(test_gb_median)
    TESTUTIL.compare(test_gb_count_2)
    TESTUTIL.compare(test_gb_min_2)
    TESTUTIL.compare(test_gb_sum_2)
    TESTUTIL.compare(test_gb_prod_2)
    by_keys = ['A']
    TESTUTIL.compare(test_gb_size_2)
    TESTUTIL.compare(test_gb_mean_2)
    TESTUTIL.compare(test_gb_cumsum_2)
    TESTUTIL.compare(test_gb_median_2)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_sum():
    """
    Test groupby with sum
    Description: tests df.groupby.sum
    Expectation: same output as pandas df.groupby.sum
    """

    def create_df(module):
        return module.DataFrame({"A": list("aaabba"), "B": [1, 1, 2, 3, 2, 1]})

    def test_sum_min_count(df):
        return df.groupby("A").sum(min_count=4)

    TESTUTIL.compare(test_sum_min_count, create_fn=create_df)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_grouper():
    """
    Test groupby with grouper
    Description: tests df.groupby with grouper
    Expectation: same output as pandas df.groupby with grouper
    """

    def test_gb_count(df):
        return df.groupby(by=by_keys).count()

    def test_gb_sum(df):
        return df.groupby(by=by_keys).sum()

    by_keys = pd.Grouper(key='A')
    TESTUTIL.compare(test_gb_count)
    TESTUTIL.compare(test_gb_sum)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_multicol():
    """
    Test groupby with multiple columns
    Description: tests df.groupby with multiple columns
    Expectation: same output as pandas df.groupby with multiple columns
    """

    def groupby_multicol_sum(df):
        return df.groupby([0, 1]).sum()

    def groupby_multicol_count(df):
        return df.groupby([1, 0]).count()

    TESTUTIL.compare(groupby_multicol_sum, create_fn=TESTUTIL.create_df_gb_frame)
    TESTUTIL.compare(groupby_multicol_count, create_fn=TESTUTIL.create_df_gb_frame)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_getitem():
    """
    Test groupby with getitem
    Description: tests df.groupby with getitem
    Expectation: same output as pandas df.groupby with getitem
    """

    def groupby_getitem_series(df):
        return df.groupby([0, 1])[2].count()

    def groupby_getitem_multicol(df):
        return df.groupby([0, 1])[[2, 3]].count()

    TESTUTIL.compare(groupby_getitem_series, create_fn=TESTUTIL.create_df_gb_frame)
    TESTUTIL.compare(groupby_getitem_multicol, create_fn=TESTUTIL.create_df_gb_frame)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_key():
    """
    Test type support for input parameter 'by' of groupby
    Description: tests df.groupby with different 'by' types
    Expectation: same output as pandas for those types
    """

    def create_input_dataframe(module):
        df = module.DataFrame({'Animal': ['Falcon', 'Falcon',
                                          'Parrot', 'Parrot'],
                               'Color': ['black', 'white', 'black', 'white'],
                               'Height': [3, 2, 2, 3],
                               'Max Speed': [380., 370., 24., 26.]})
        return module, df

    def create_input_series(module):
        df = module.Series(['Color', 'Max Speed'])
        return df

    def create_single_index_dataframe(module):
        data = [("bird", "Falconiformes", 389.0), ("bird", "Psittaciformes", 24.0),
                ("mammal", "Carnivora", 80.2), ("mammal", "Primates", np.nan),
                ("mammal", "Carnivora", 58)]
        index = ["falcon", "parrot", "lion", "monkey", "leopard"]
        columns = ("class", "order", "max_speed")
        return module.DataFrame(data, index=index, columns=columns)

    def test_list(mo_df):
        _, df = mo_df
        return df.groupby([1, 2, 3, 4]).count()

    def test_list_sum(mo_df):
        _, df = mo_df
        return df.groupby([1, 2, 3, 4]).sum()

    def test_list_of_series(mo_df):
        _, df = mo_df
        return df.groupby([df['Animal'], df['Color']]).sum()

    def test_list_of_labels(mo_df):
        """ This is syntactic sugar of test_list_of_series """
        _, df = mo_df
        return df.groupby(['Animal', 'Color']).sum()

    def test_list_of_dict(mo_df):
        _, df = mo_df
        return df.groupby([{'a': 'Falcon', 'b': 'Falcon', 'c': 'Parrot', 'd': 'Parrot'},
                           {'a': 'black', 'b': 'white', 'c': 'white'}]).sum()

    def test_series_mapping(mo_df):
        _, df = mo_df
        return df.groupby(df['Animal']).count()

    def test_dict_mapping(mo_df):
        _, df = mo_df
        return df.groupby({0: 'Falcon', 1: 'Falcon', 2: 'Parrot', 3: 'Parrot'}).count()

    def test_dict_reduced_mapping(df):
        mapping = {'falcon': 'falcon_sec', 'lion': 'lion_sec'}
        return df.groupby(mapping).count()

    def test_nparray_mapping(mo_df):
        """ nparray only support mapping, not support labels """
        _, df = mo_df
        return df.groupby(np.array(['Falcon', 'Falcon', 'Falcon', 4])).count()

    def test_getitem_by_str_from_label_groupby(mo_df):
        """ test getitem by a str label of a datagrame.groupby created by a label """
        _, df = mo_df
        return df.groupby('Animal')['Max Speed'].sum()

    def test_getitem_by_list_from_label_groupby(mo_df):
        _, df = mo_df
        return df.groupby('Animal')[['Color', 'Max Speed']].all()

    def test_getitem_by_list_from_lstofindex_groupby(mo_df):
        _, df = mo_df
        return df.groupby([1, 2, 3, 4])[['Color', 'Max Speed']].all()

    def test_getitem_by_str_from_lstoflabels_groupby(mo_df):
        _, df = mo_df
        return df.groupby(['Animal', 'Height'])['Max Speed'].sum()

    def test_getitem_by_list_from_lstoflabels_groupby(mo_df):
        _, df = mo_df
        return df.groupby(['Animal', 'Height'])[['Color', 'Max Speed']].all()

    def test_getitem_by_str_from_mapping_groupby(mo_df):
        _, df = mo_df
        groupby_obj = df.groupby([df['Animal'], df['Height']])
        return groupby_obj['Max Speed'].sum()

    def test_getitem_by_list_from_mapping_groupby(mo_df):
        _, df = mo_df
        groupby_obj = df.groupby([df['Animal'], df['Height']])
        return groupby_obj[['Color', 'Max Speed']].count()

    def test_getitem_by_tuple(mo_df):
        _, df = mo_df
        groupby_obj = df.groupby([df['Animal'], df['Height']])
        return groupby_obj[('Color', 'Max Speed')].all()

    def test_getitem_by_pd_series(mo_df):
        _, df = mo_df
        ser = pd.Series(['Color', 'Max Speed'])
        return df.groupby([df['Animal'], df['Height']])[ser].all()

    def test_getitem_by_mpd_series(mo_df):
        module, df = mo_df
        ser = module.Series(['Color', 'Max Speed'])
        return df.groupby([df['Animal'], df['Height']])[ser].all()

    def test_err_groupby_getitem_by_df(mo_df):
        _, df = mo_df
        df_key = pd.DataFrame(['Color', 'Max Speed'])
        df.groupby([df['Animal'], df['Height']])[df_key].all()

    def test_err_groupby_getitem_by_dict(mo_df):
        _, df = mo_df
        dct = {0: 'Color', 1: 'Max Speed'}
        df.groupby([df['Animal'], df['Height']])[dct].all()

    def test_err_groupby_by_class(mo_df):
        _, df = mo_df
        class ClassA:
            def __init__(self):
                self.var = 0

        grouyby_obj = ClassA()
        df.groupby(grouyby_obj)[['Color', 'Max Speed']].all()

    def test_err_groupby_by_lst_of_class(mo_df):
        _, df = mo_df
        class ClassA:
            def __init__(self):
                self.var = 0

        grouyby_obj = ClassA()
        df.groupby([grouyby_obj])[['Color', 'Max Speed']].all()

    def test_err_groupby_by_empty_list(mo_df):
        _, df = mo_df
        df.groupby([])[['Color', 'Max Speed']].all()

    def test_err_input_not_df_by_list(mo_df):
        """ input_dataframe not mpd.DataFrame, by list
            df should put series here
        """
        _, df = mo_df
        df.groupby(['Animal', 'Color'])[['Color', 'Max Speed']].all()

    def test_err_input_not_df_by_ser(mo_df):
        """ input_dataframe not mpd.DataFrame, by ser
            df should put series here
        """
        _, df = mo_df
        df.groupby({0: 'Falcon', 1: 'Falcon', 2: 'Parrot', 3: 'Parrot'})[
            ['Color', 'Max Speed']].all()

    def test_err_mapping_with_unaligned_number_of_elements(mo_df):
        _, df = mo_df
        df.groupby([3, 2])[['Color', 'Max Speed']].all()

    TESTUTIL.compare(test_list, create_input_dataframe)
    TESTUTIL.compare(test_list_sum, create_input_dataframe)
    TESTUTIL.compare(test_list_of_labels, create_input_dataframe)
    TESTUTIL.compare(test_list_of_dict, create_input_dataframe)
    TESTUTIL.compare(test_list_of_series, create_input_dataframe)
    TESTUTIL.compare(test_series_mapping, create_input_dataframe)
    TESTUTIL.compare(test_dict_mapping, create_input_dataframe)
    TESTUTIL.compare(test_dict_reduced_mapping, create_single_index_dataframe)
    TESTUTIL.compare(test_nparray_mapping, create_input_dataframe)
    # !Note: UDF does not support yet
    # TESTUTIL.compare(test_udf, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_str_from_label_groupby, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_list_from_label_groupby, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_list_from_lstofindex_groupby, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_str_from_lstoflabels_groupby, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_list_from_lstoflabels_groupby, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_str_from_mapping_groupby, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_list_from_mapping_groupby, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_tuple, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_pd_series, create_input_dataframe)
    TESTUTIL.compare(test_getitem_by_mpd_series, create_input_dataframe)
    TESTUTIL.run_compare_error(test_err_groupby_getitem_by_df, TypeError, create_input_dataframe)
    TESTUTIL.run_compare_error(test_err_groupby_getitem_by_dict, TypeError, create_input_dataframe)
    TESTUTIL.run_compare_error(test_err_groupby_by_class, TypeError, create_input_dataframe)
    TESTUTIL.run_compare_error(test_err_groupby_by_lst_of_class, TypeError, create_input_dataframe)
    TESTUTIL.run_compare_error(test_err_groupby_by_empty_list, ValueError, create_input_dataframe)
    TESTUTIL.run_compare_error(test_err_input_not_df_by_list, AttributeError, create_input_series)
    TESTUTIL.run_compare_error(test_err_input_not_df_by_ser, AttributeError, create_input_series)
    TESTUTIL.run_compare_error(
        test_err_mapping_with_unaligned_number_of_elements, KeyError, create_input_dataframe)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_arguments():
    """
    Test groupby arguments, with two types of tests
    Description: tests df.groupby with 1) call func_wrapper function: sum(), count()
                                       2) default to pandas function: mean()
    Expectation: same output as pandas for those arguments
    """

    def create_multi_index_dataframe(module):
        arrays = [['Wolverine', 'Falcon', 'Falcon', 'Parrot', 'Parrot'],
                  ['Wild', 'Captive', 'Captive', 'Captive', 'Wild']]
        index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
        df = module.DataFrame({'Max Speed': [1000, 390., 350., 30., 20.],
                               'Max Speed2': [2, 30., 30., 10., 50.],
                               'String': ['a', 'b', 'c', 'd', 'e']},
                              index=index)
        return df

    def create_categorical_dataframe(module):
        df = module.DataFrame({
            "state": pd.Categorical(["AK", "AL", "AK", "AL"]),
            "gender": pd.Categorical(["M", "M", "M", "F"]),
            "name": list("abcd"),
        })
        return df

    def create_na_dataframe(module):
        l = [["a", 12, 12], [None, 12.3, 33.], ["b", 12.3, 123], ["a", 1, 1], [None, 7, 8]]
        df = module.DataFrame(l, columns=["a", "b", "c"])
        return df

    def test_multiindex_groupby_index(df):
        return df.groupby('Animal').sum()

    def test_multiindex_groupby_label(df):
        return df.groupby('Max Speed').sum()

    def test_multiindex_groupby_index_lst(df):
        return df.groupby(['Animal', 'Type']).sum()

    def test_multiindex_groupby_label_lst(df):
        return df.groupby(['Max Speed', 'Max Speed2']).sum()

    def test_multiindex_groupby_index_default(df):
        return df.groupby('Animal').mean()

    def test_multiindex_groupby_label_default(df):
        return df.groupby('Max Speed').mean()

    def test_multiindex_groupby_index_lst_default(df):
        return df.groupby(['Animal', 'Type']).mean()

    def test_multiindex_groupby_label_lst_default(df):
        return df.groupby(['Max Speed', 'Max Speed2']).mean()

    def test_level_int(df):
        return df.groupby(level=0).sum()

    def test_level_label(df):
        return df.groupby(level="Type").sum()

    def test_level_lst_of_int(df):
        return df.groupby(level=[0, 1]).sum()

    def test_level_lst_of_labels(df):
        return df.groupby(level=["Animal", "Type"]).sum()

    def test_level_func_default(df):
        return df.groupby(level=["Animal", "Type"]).mean()

    def test_sort_multi_index_dataframe(df):
        return df.groupby(["Animal", "Type"], sort=False).sum()

    def test_sort_multi_index_dataframe_default(df):
        return df.groupby(["Animal", "Type"], sort=False).mean()

    def test_sort_multi_index_dataframe_true(df):
        return df.groupby(["Animal", "Type"], sort=True).sum()

    def test_sort_multi_index_dataframe_true_default(df):
        return df.groupby(["Animal", "Type"], sort=True).mean()

    def test_observed_default(df):
        return df.groupby(["state", "gender"], observed=True).cumcount()

    def test_dropna1(df):
        return df.groupby(by="a", dropna=False).sum()

    def test_dropna_default(df):
        return df.groupby(by="a", dropna=False).mean()

    def test_dropna_true(df):
        return df.groupby(by="a", dropna=True).sum()

    TESTUTIL.compare(test_multiindex_groupby_index, create_multi_index_dataframe)
    TESTUTIL.compare(test_multiindex_groupby_label, create_multi_index_dataframe)
    TESTUTIL.compare(test_multiindex_groupby_index_lst, create_multi_index_dataframe)
    TESTUTIL.compare(test_multiindex_groupby_label_lst, create_multi_index_dataframe)
    TESTUTIL.compare(test_multiindex_groupby_index_default, create_multi_index_dataframe)
    TESTUTIL.compare(test_multiindex_groupby_label_default, create_multi_index_dataframe)
    TESTUTIL.compare(test_multiindex_groupby_index_lst_default, create_multi_index_dataframe)
    TESTUTIL.compare(test_multiindex_groupby_label_lst_default, create_multi_index_dataframe)
    TESTUTIL.compare(test_level_int, create_multi_index_dataframe)
    TESTUTIL.compare(test_level_label, create_multi_index_dataframe)
    TESTUTIL.compare(test_level_lst_of_int, create_multi_index_dataframe)
    TESTUTIL.compare(test_level_lst_of_labels, create_multi_index_dataframe)
    TESTUTIL.compare(test_level_func_default, create_multi_index_dataframe)
    TESTUTIL.compare(test_sort_multi_index_dataframe, create_multi_index_dataframe)
    TESTUTIL.compare(test_sort_multi_index_dataframe_default, create_multi_index_dataframe)
    TESTUTIL.compare(test_sort_multi_index_dataframe_true, create_multi_index_dataframe)
    TESTUTIL.compare(test_sort_multi_index_dataframe_true_default, create_multi_index_dataframe)
    # NOTE:test_observed for categorical_dataframe can get the exact same result as the original
    # pandas using print. but multi-index cannot be recognized by pandas which makes the
    # pandas.DataFrame.equals()outputs False value.
    # TESTUTIL.compare(test_observed, create_categorical_dataframe)
    TESTUTIL.compare(test_observed_default, create_categorical_dataframe)
    TESTUTIL.compare(test_dropna1, create_na_dataframe)
    TESTUTIL.compare(test_dropna_default, create_na_dataframe)
    TESTUTIL.compare(test_dropna_true, create_na_dataframe)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_axis1():
    """
    Test groupby with axis=1
    Description: tests df.groupby with axis=1
    Expectation: same output as pandas groupby with axis=1
    """

    def create_index_dataframe(module):
        np.random.seed(0)
        data = {
            'key1': ['a', 'a', 'b', 'b', 'a'],
            'key2': ['one', 'two', 'one', 'two', 'one'],
            'data1': np.random.randn(5),
            'data2': np.random.randn(5),
            'data3': np.random.randn(5),
            'data4': np.random.randn(5),
        }
        index = pd.Index(['A', 'B', 'C', 'D', 'E'])
        return module.DataFrame(data, index=index)

    def test_axis1_dict_mapping(df):
        mapping = {'data1': 1, 'data2': 'data20', 'data3': 1, 'data4': 'data20'}
        return df.groupby(mapping, axis=1).sum()

    def test_axis1_ser_mapping(df):
        mapping = {'data1': 1, 'data2': 'data20', 'data3': 1, 'data4': 'data20'}
        ser = pd.Series(mapping)
        return df.groupby(ser, axis=1).sum()

    def test_axis1_lst_injective_mapping(df):
        lst = [1, 2, 3, 4, 5, 6]
        return df.groupby(lst, axis=1).sum()

    def test_axis1_nparray_injective_mapping(df):
        lst = [1, 2, 3, 4, 5, 6]
        arr = np.array(lst)
        return df.groupby(arr, axis=1).sum()

    def test_none_axis(df):
        # ValueError: No axis named None for object type DataFrame
        df.groupby('key1', axis=None).sum()

    TESTUTIL.compare(test_axis1_dict_mapping, create_index_dataframe)
    TESTUTIL.compare(test_axis1_ser_mapping, create_index_dataframe)
    TESTUTIL.compare(test_axis1_lst_injective_mapping, create_index_dataframe)
    TESTUTIL.compare(test_axis1_nparray_injective_mapping, create_index_dataframe)
    TESTUTIL.run_compare_error(test_none_axis, ValueError, create_index_dataframe)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_groupby_groups():
    """
    Test groupby with groups
    Description: tests df.groupby with groups
    Expectation: same output as pandas
    """

    def create_input_dataframe(module):
        data = [("bird", "Falconiformes", 389.0), ("bird", "Psittaciformes", 24.0),
                ("mammal", "Carnivora", 80.2), ("mammal", "Primates", np.nan),
                ("mammal", "Carnivora", 58)]
        index = pd.Index(["falcon", "parrot", "lion", "monkey", "leopard"], name='animal')
        columns = ("class", "order", "max_speed")
        return module.DataFrame(data, index=index, columns=columns)

    def test_groupby_column_label(df):
        return df.groupby('class').groups

    def test_groupby_level_label(df):
        return df.groupby('animal').groups

    def test_groupby_level1(df):
        return df.groupby(level='animal').groups

    def test_groupby_level2(df):
        return df.groupby(level=0).groups

    def test_groupby_list_of_labels(df):
        return df.groupby(["class", "order"]).groups

    def test_groupby_list_of_ser(df):
        return df.groupby([df["class"], df["order"]]).groups

    def test_groupby_ser_mapping(df):
        return df.groupby(pd.Series({"falcon": "falcon0", "parrot": "parrot0"})).groups

    def test_groupby_dict_mapping(df):
        return df.groupby({"falcon": "falcon0", "parrot": "parrot0"}).groups

    def test_groupby_list_of_dict_mapping(df):
        return df.groupby([{"falcon": "falcon0", "parrot": "parrot0"}]).groups

    def test_axis1_lst_injective_mapping(df):
        return df.groupby([1, 2, 3], axis=1).groups

    def test_axis1_dict_mapping(df):
        mapping = {'class': 1, 'order': 'order0', 'max_speed': 1}
        return df.groupby(mapping, axis=1).groups

    def test_groupby_indices_column_label(df):
        return df.groupby('class').indices

    def test_groupby_indices_level_label(df):
        return df.groupby('animal').indices

    def test_groupby_indices_level1(df):
        return df.groupby(level='animal').indices

    TESTUTIL.compare(test_groupby_column_label, create_input_dataframe)
    TESTUTIL.compare(test_groupby_level_label, create_input_dataframe)
    TESTUTIL.compare(test_groupby_level1, create_input_dataframe)
    TESTUTIL.compare(test_groupby_level2, create_input_dataframe)
    TESTUTIL.compare(test_groupby_list_of_labels, create_input_dataframe)
    TESTUTIL.compare(test_groupby_list_of_ser, create_input_dataframe)
    TESTUTIL.compare(test_groupby_ser_mapping, create_input_dataframe)
    TESTUTIL.compare(test_groupby_dict_mapping, create_input_dataframe)
    TESTUTIL.compare(test_groupby_list_of_dict_mapping, create_input_dataframe)
    TESTUTIL.compare(test_axis1_lst_injective_mapping, create_input_dataframe)
    TESTUTIL.compare(test_axis1_dict_mapping, create_input_dataframe)
    TESTUTIL.compare(test_groupby_indices_column_label, create_input_dataframe)
    TESTUTIL.compare(test_groupby_indices_level_label, create_input_dataframe)
    TESTUTIL.compare(test_groupby_indices_level1, create_input_dataframe)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", default=True, type=bool)
    parser.add_argument("--csv_file_list", nargs='+',
                        action='append', default=None)
    parser.add_argument("--by_keys_list", nargs='+',
                        action='append', default=None)
    args = parser.parse_args()
    if args.performance is True:
        TESTUTIL.set_perf_mode()

    test_groupby_default_to_pandas_boolean_ops()
    test_groupby_default_to_pandas_nunique_op()
    test_groupby_default_to_pandas_periodic_ops()
    test_groupby_default_to_pandas_corrwith_filter_ops()
    test_groupby_default_to_pandas_sample_op()
    test_groupby_default_to_pandas_take_op()
    test_groupby_default_to_pandas_nan_ops()
    test_groupby_basic()
    test_groupby_sum()
    test_groupby_grouper()
    test_groupby_multicol()
    test_groupby_getitem()
    test_groupby_key()
    test_groupby_arguments()
    test_groupby_axis1()
    test_groupby_groups()

    if args.csv_file_list is not None:
        print(args.by_keys_list)
        print(args.csv_file_list)
        groupby_default_to_pandas_ohlc_prod_diff_rank_cumprod_ops(args.by_keys_list[0][0],
                                                                  args.csv_file_list[0][0])
        groupby_default_to_pandas_major_ops(args.by_keys_list[1][0],
                                            args.csv_file_list[1][0])
