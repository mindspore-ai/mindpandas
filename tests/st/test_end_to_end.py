import hashlib
import os

import numpy as np
import pandas as pd
import pytest

import mindpandas as ms_pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("concurrency_mode", ["multithread"])
def test_end_to_end(concurrency_mode):
    """
    Feature: System Test
    Description: Test multiple APIs
    Expectation: Runs successfully with the same result
    """
    ms_pd.set_adaptive_concurrency(False)
    ms_pd.set_concurrency_mode(concurrency_mode)
    ms_pd.set_partition_shape((16, 3))

    def hash_item(val, item_size=10000000, offset=0):
        if isinstance(val, str):
            return abs(int(hashlib.sha256(val.encode('utf-8')).hexdigest(), 16)) % item_size
        return abs(hash(val)) % item_size + offset

    dense_feat_names = ["col6", "col8", "col9", "col10", "col13", "col17", "col18", "col19", "col22", "col23", "col37"]
    sparse_feat_names = ["col1", "col2", "col3", "col4", "col5", "col7", "col11", "col15", "col20", "col30", "col35"]

    file_name = os.path.join(DATA_DIR, "data.csv")
    df = pd.read_csv(file_name)
    mdf = ms_pd.read_csv(file_name)

    df["col40"] = df["col40"].apply(lambda x: x.split(':') if isinstance(x, str) else [])
    mdf["col40"] = mdf["col40"].apply(lambda x: x.split(':') if isinstance(x, str) else [])

    df = df.replace("x", pd.NaT)
    mdf = mdf.replace("x", pd.NaT)

    df["label"] = df["col10"]
    mdf["label"] = mdf["col10"]

    df = df[:-100]
    mdf = mdf[:-100]

    df = df.sort_values("col3")
    mdf = mdf.sort_values("col3")
    mdf.repartition((16, 3))

    df = df.reset_index(drop=True)
    mdf = mdf.reset_index(drop=True)

    train_len = int(len(df) * 0.7)
    m_train_len = int(len(mdf) * 0.7)

    df["is_training"] = [1] * train_len + [0] * (len(df) - train_len)
    mdf["is_training"] = [1] * train_len + [0] * (len(mdf) - m_train_len)

    df = df.drop(columns=dense_feat_names)
    mdf = mdf.drop(columns=dense_feat_names)

    df = df.fillna("-1")
    mdf = mdf.fillna("-1")

    df[sparse_feat_names] = df[sparse_feat_names].applymap(hash_item)
    mdf[sparse_feat_names] = mdf[sparse_feat_names].applymap(hash_item)

    df = df.drop_duplicates(subset=['col30'])
    mdf = mdf.drop_duplicates(subset=['col30'])

    df = df.reset_index(drop=True)
    mdf = mdf.reset_index(drop=True)

    np.random.seed(100)
    data = np.random.rand(len(df))
    df.insert(20, "inserted", pd.Series(data))
    mdf.insert(20, "inserted", ms_pd.Series(data))

    assert np.array_equal(df.values, mdf.values)
