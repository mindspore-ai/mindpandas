import os.path

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import mindpandas as mpd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("concurrency_mode", ["multithread", "multiprocess"])
def test_end_to_end(concurrency_mode):
    """
    Feature: System Test
    Description: Test multiple APIs
    Expectation: Runs successfully with the same result
    """
    mpd.set_adaptive_concurrency(False)
    mpd.set_concurrency_mode(concurrency_mode, address="127.0.0.1")
    mpd.set_partition_shape((16, 3))

    file_name = os.path.join(DATA_DIR, "raw_data.csv")
    df = pd.read_csv(file_name)
    mdf = mpd.read_csv(file_name)
    assert_frame_equal(df, mdf.to_pandas())

    df_head = df.head()
    mdf_head = mdf.head()
    assert_frame_equal(df_head, mdf_head.to_pandas())
    df_tail = df.tail()
    mdf_tail = mdf.tail()
    assert_frame_equal(df_tail, mdf_tail.to_pandas())

    df = df.dropna(axis=0)
    mdf = mdf.dropna(axis=0)
    assert_frame_equal(df, mdf.to_pandas())

    df = df.drop(columns="index")
    mdf = mdf.drop(columns="index")
    assert_frame_equal(df, mdf.to_pandas())

    df = df.sort_values("col1").reset_index(drop=True)
    mdf = mdf.sort_values("col1").reset_index(drop=True)
    mdf.repartition((16, 3))
    assert_frame_equal(df, mdf.to_pandas())

    df["col3"] = df["col3"].apply(lambda x: x.replace("_", ""))
    mdf["col3"] = mdf["col3"].apply(lambda x: x.replace("_", ""))
    assert_frame_equal(df, mdf.to_pandas())

    assert df.groupby("col5").size()["Y"] == mdf.groupby("col5").size()["Y"]

    df["col4"] = df["col4"] * 100
    mdf["col4"] = mdf["col4"] * 100
    assert_frame_equal(df, mdf.to_pandas())

    assert df["col4"].max() == mdf["col4"].max()
    assert df["col4"].min() == mdf["col4"].min()

    res1 = df[df["col5"] == "Y"].groupby("col2")["col4"].mean()
    res2 = mdf[mdf["col5"] == "Y"].groupby("col2")["col4"].mean()
    assert_series_equal(res1, res2.to_pandas())

    assert_frame_equal(df, mdf.to_pandas())
