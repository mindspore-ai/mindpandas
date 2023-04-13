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
""" Test mindpandas lazy mode """
import logging as logger
import pytest

import mindpandas as mpd
import numpy as np
import pandas as pd


# currenltly we set test_mode ="quick"
# testing_mode = "quick"  # Option is 'quick' or 'full'

def compare_df(pd_res, mpd_res):
    """
    Compare the executed results with original pandas
    """
    if isinstance(pd_res, pd.DataFrame) and isinstance(mpd_res, mpd.DataFrame):
        mpd_res = mpd_res.to_pandas()
        pd.testing.assert_frame_equal(mpd_res, pd_res, check_names=False)
    elif isinstance(pd_res, pd.Series) and isinstance(mpd_res, mpd.Series):
        mpd_res = mpd_res.to_pandas()
        pd.testing.assert_series_equal(mpd_res, pd_res, check_names=False)
    else:
        raise AssertionError("Mismatched type")


def helper_op_pd_testing(data, op_name, kwargs, enable_debug):
    """
    Help test using data and op_name as op, based on the arguments in kwargs given.
    """
    enable_debug = True
    if isinstance(data, pd.DataFrame):
        new_data = data.copy(deep=True)
    else:  # numpy array
        new_data = data.copy()

    mpd_df = mpd.DataFrame(new_data)
    mpd_func = getattr(mpd.DataFrame, op_name)
    mpd_res = mpd_func(mpd_df, **kwargs)

    # Lazy mode does not support inplace
    # Prevent extra typecasting in pandas when inplace set to True by setting it to False
    if "inplace" in kwargs and kwargs["inplace"]:
        pd_df = pd.DataFrame(new_data.copy())
        pd_func = getattr(pd.DataFrame, op_name)
        pd_func(pd_df, **kwargs)
        pd_res = pd_df
    else:
        pd_df = pd.DataFrame(new_data)
        pd_func = getattr(pd.DataFrame, op_name)
        pd_res = pd_func(pd_df, **kwargs)

    if enable_debug and "inplace" in kwargs and not kwargs["inplace"]:
        print("xxx mpd_df before mpd.debug\n", mpd_df)
        mpd.explain(mpd_res)
        print("xxx mpd_res before mpd.debug:\n", mpd_res)
        mpd_res = mpd.debug(mpd_res, pr_details=True)
        print("xxx mpd_res after mpd.debug:\n", mpd_res)
        print(
            "mpd_res partitions:\n",
            mpd_res.backend_frame.partitions.shape,
            mpd_res.backend_frame.partitions[0],
        )
    else:
        mpd_res = mpd.run(mpd_res)

    compare_df(pd_res, mpd_res)


# ========= QUICK TEST ==========
def helper_quick_test(op_name, enable_debug=False):
    """Help perform quick regression test"""

    def _create_dataframe():
        return pd.DataFrame(np.random.randn(11, 7),
                            columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
                            index=["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"])

    # Note this quick test will generate 1 dataset for all, doesn't cover levels yet
    mpd.set_lazy_mode(True)
    data = _create_dataframe()

    list_axis = [0, 1]
    for axis in list_axis:
        if op_name == "apply":
            kwargs = {
                "func": (lambda x: pd.Series(["1X"] * int(x.count()))),
                "axis": axis,
                "raw": False,
                "result_type": "expand",
                "args": (),
            }
        elif op_name == "applymap":
            kwargs = {"func": (lambda x: x * 3), "na_action": "ignore"}
        elif op_name == "count":
            kwargs = {"axis": axis, "numeric_only": None}
        elif op_name == "fillna":
            kwargs = {
                "value": None,
                "method": "ffill",
                "axis": 0,
                "inplace": True,
                "limit": 2,
                "downcast": "infer",
            }
        elif op_name == "max":
            # TODO: add numeric_only True later
            kwargs = {"axis": axis, "skipna": True, "numeric_only": None}
        elif op_name == "mean":
            kwargs = {"axis": axis, "skipna": True, "numeric_only": None}
        elif op_name == "median":
            kwargs = {"axis": axis, "skipna": True, "numeric_only": None}
        elif op_name == "min":
            kwargs = {"axis": axis, "skipna": True, "numeric_only": None}
        elif op_name == "sum":
            kwargs = {"axis": axis, "skipna": True, "numeric_only": None}
            # TODO: add min_count
            # kwargs = {'axis': axis, 'skipna': True, 'numeric_only': None, 'min_count': 5}
        else:
            logger.warning(
                "%s as op name is invalid or not yet implemented in quick test with axis=%s", op_name, axis
            )
            continue

        helper_op_pd_testing(data, op_name, kwargs, enable_debug)


# ========= DEFAULT TO PANDAS CASES ==========
def helper_default_to_pandas_test(op_name, enable_debug=False):
    '''Helper to perform default to pandas tests'''
    def _create_nan_dataframe():
        nan_df = pd.DataFrame({'A': [np.nan, 2, np.nan, 0],
                               'B': [3, 4, np.nan, 1],
                               'C': [np.nan, 5, np.nan, np.nan],
                               'D': [np.nan, 3, np.nan, 4]})
        return nan_df

    def _create_multiindex_dataframe():
        multicol1 = pd.MultiIndex.from_tuples([('weight', 'kg'),
                                               ('weight', 'pounds')])
        multiindex_df = pd.DataFrame([[1, 2], [2, 4]],
                                     index=['cat', 'dog'],
                                     columns=multicol1)
        return multiindex_df

    def _create_multilevel_dataframe():
        df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).set_index([0, 1]).rename_axis(['a', 'b'])
        df.columns = pd.MultiIndex.from_tuples([('c', 'e'), ('d', 'f')], names=['level_1', 'level_2'])
        return df

    mpd.set_lazy_mode(True)

    if op_name == "notna":
        data = _create_nan_dataframe()
        kwargs = {}
    elif op_name == "stack":
        data = _create_multiindex_dataframe()
        kwargs = {"level": -1}
    elif op_name == "unstack":
        data = _create_multiindex_dataframe()
        kwargs = {"level": 0}
    elif op_name == "droplevel":
        data = _create_multilevel_dataframe()
        kwargs = {"level": 'a'}

    helper_op_pd_testing(data, op_name, kwargs, enable_debug)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_apply():
    """
    Test apply
    Description: tests df.apply in lazy mode
    Expectation: same output as pandas.DataFrame.apply
    """
    helper_quick_test(op_name="apply")


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_applymap():
    """
    Test applymap
    Description: tests df.applymap in lazy mode
    Expectation: same output as pandas.DataFrame.applymap
    """
    helper_quick_test(op_name="applymap")


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_count():
    """
    Test count
    Description: tests df.count in lazy mode
    Expectation: same output as pandas.DataFrame.count
    """
    helper_quick_test(op_name="count")


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_fillna():
    """
    Test fillna
    Description: tests df.fillna in lazy mode
    Expectation: same output as pandas.DataFrame.fillna
    """
    helper_quick_test(op_name="fillna")


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_max():
    """
    Test max
    Description: tests df.max in lazy mode
    Expectation: same output as pandas.DataFrame.max
    """
    helper_quick_test(op_name="max")


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_mean():
    """
    Test mean
    Description: tests df.mean in lazy mode
    Expectation: same output as pandas.DataFrame.mean
    """
    helper_quick_test(op_name="mean")


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_median():
    """
    Test median
    Description: tests df.median in lazy mode
    Expectation: same output as pandas.DataFrame.median
    """
    helper_quick_test(op_name="median")


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_min():
    """
    Test min
    Description: tests df.min in lazy mode
    Expectation: same output as pandas.DataFrame.min
    """
    helper_quick_test(op_name="min")


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_sum():
    """
    Test sum
    Description: tests df.sum in lazy mode
    Expectation: same output as pandas.DataFrame.sum
    """
    helper_quick_test(op_name="sum")

@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_notna():
    """
    Test sum
    Description: tests df.sum in lazy mode
    Expectation: same output as pandas.DataFrame.sum
    """
    helper_default_to_pandas_test(op_name="notna")

@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_stack():
    """
    Test sum
    Description: tests df.sum in lazy mode
    Expectation: same output as pandas.DataFrame.sum
    """
    helper_default_to_pandas_test(op_name="stack")

@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_unstack():
    """
    Test sum
    Description: tests df.sum in lazy mode
    Expectation: same output as pandas.DataFrame.sum
    """
    helper_default_to_pandas_test(op_name="unstack")

@pytest.mark.usefixtures("set_mode", "set_shape")
def test_batch_notdist_droplevel():
    """
    Test sum
    Description: tests df.sum in lazy mode
    Expectation: same output as pandas.DataFrame.sum
    """
    helper_default_to_pandas_test(op_name="droplevel")
