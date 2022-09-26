# Copyright 2020 Huawei Technologies Co., Ltd
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

"""
Module contains ``test_io`` class which is responsible for testing the correct
implementations of functions related to input/output like read_csv().
"""

import io
import os

import numpy as np
import pytest

import mindpandas as mpd
from util import TESTUTIL, DATA_DIR


# Always use csv mode for testcases

@pytest.mark.usefixtures("set_mode", "set_shape")
def test_read_csv_basic():
    """
    Feature: read_csv
    Description: Test read_csv
    Expectation: Runs successfully with the same result
    """

    def create_module(module):
        return module

    TESTUTIL.set_use_csv()
    TESTUTIL.compare(TESTUTIL.create_single_column_df, create_fn=create_module)
    TESTUTIL.compare(TESTUTIL.create_df_range, create_fn=create_module)
    TESTUTIL.compare(TESTUTIL.create_df_range_float, create_fn=create_module)
    TESTUTIL.unset_use_csv()


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_read_csv_edge_cases():
    """
    Feature: read_csv
    Description: Test read_csv edge cases
    Expectation: Runs successfully with the same result
    """

    def create_module(module):
        return module

    def read_csv_string(module):
        data = "a b c\n1 2 3"
        df = module.read_csv(io.StringIO(data))
        return df

    def read_csv_compress(module):
        df = module.read_csv(os.path.join(DATA_DIR, "compress.csv.gz"))
        return df

    def read_csv_empty(module):
        df = module.read_csv(os.path.join(DATA_DIR, "empty.csv"))
        return df

    def read_csv_invalid(module):
        df = module.read_csv(os.path.join(DATA_DIR, "test_invalid.csv"))
        return df

    def read_csv_type(module):
        df = module.read_csv(os.path.join(DATA_DIR, "test_simple.csv"), header=None)
        return df

    def read_csv_symbol(module):
        df = module.read_csv(os.path.join(DATA_DIR, "test_symbol.csv"))
        return df

    TESTUTIL.compare(read_csv_string, create_fn=create_module)
    TESTUTIL.compare(read_csv_compress, create_fn=create_module)
    TESTUTIL.compare(read_csv_empty, create_fn=create_module)
    TESTUTIL.compare(read_csv_invalid, create_fn=create_module)
    # This testcase needs more changes... currently there is a string on line1 so all dtypes will be string
    # When we split the file, we don't have this first row and the system assumes they are ints
    TESTUTIL.compare(read_csv_type, create_fn=create_module)
    TESTUTIL.compare(read_csv_symbol, create_fn=create_module)


def create_module_df(module):
    """
    Create DataFrame for testing purpose.
    """
    df = TESTUTIL.create_df_range_float(module)
    return module, df


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_to_csv_compression():
    """
    Feature: read_csv
    Description: Test read_csv to read compressed files
    Expectation: Runs successfully with the same result
    """

    def to_csv_no_compression(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv.csv"))
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv.csv"))
        return df

    def to_csv_gzip(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv.gz"), compression="gzip")
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv.gz"))
        return df

    def to_csv_bz2(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv.bz2"), compression="bz2")
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv.bz2"))
        return df

    def to_csv_xz(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv.xz"), compression="xz")
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv.xz"))
        return df

    def to_csv_zip(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv.zip"), compression="zip")
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv.zip"))
        return df

    def to_csv_zip_archive_name(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv.zip"), compression=dict(method="zip", archive_name="csv"))
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv.zip"))
        return df

    TESTUTIL.compare(to_csv_no_compression, create_fn=create_module_df)
    TESTUTIL.compare(to_csv_gzip, create_fn=create_module_df)
    TESTUTIL.compare(to_csv_bz2, create_fn=create_module_df)
    TESTUTIL.compare(to_csv_xz, create_fn=create_module_df)
    TESTUTIL.compare(to_csv_zip, create_fn=create_module_df)
    TESTUTIL.compare(to_csv_zip_archive_name, create_fn=create_module_df)
    # Todo: remove files after completes.  Also ensure they are unique in case tests run in parallel


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_to_csv_arguments():
    """
    Feature: read_csv
    Description: Test read_csv with arguments
    Expectation: Runs successfully with the same result
    """

    def create_df(module):
        data = {"price": [10, np.nan, 70, 80], "Date": ["2019/12/01", "2020/12/01", "2021/12/01", "2022/12/01"]}
        index = ["one", "two", "three", "four"]
        return module, module.DataFrame(data, index)

    def to_csv_header(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_0.csv"), header=True)
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_0.csv"))
        return df

    def to_csv_sep(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_1.csv"), sep="*", header=True)
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_1.csv"))
        return df

    def to_csv_index_label(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_1.csv"), sep="*", header=True, index_label=["table"])
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_1.csv"))
        return df

    def to_csv_2(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_2.csv"), na_rep="1000", columns=["price", "Date"], index=False,
                  index_label="table", decimal=",")
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_2.csv"))
        return df

    def to_csv_3(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_3.csv"), columns=["price", "Date"], index_label="table",
                  date_format="%Y%m%d", doublequote=False, escapechar="/")
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_3.csv"))
        return df

    def to_csv_4(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_4.csv"), columns=module.Series(["price", "Date"]),
                  index_label=module.Series(["table"]))
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_4.csv"))
        return df

    def to_csv_5(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_5.csv"), index_label="table")
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_5.csv"))
        return df

    def to_csv_6(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_6.csv"), index_label=False)
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_6.csv"))
        return df

    def to_csv_7(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "to_csv_7.csv"), index_label=10000)
        df = module.read_csv(os.path.join(DATA_DIR, "to_csv_7.csv"))
        return df

    def read_csv_index_col(df_mod):
        module, df = df_mod
        df.to_csv(os.path.join(DATA_DIR, "tmp.csv"))
        df = module.read_csv(os.path.join(DATA_DIR, "tmp.csv"), index_col="A")
        return df

    TESTUTIL.compare(to_csv_header, create_fn=create_df)
    TESTUTIL.compare(to_csv_sep, create_fn=create_df)
    TESTUTIL.compare(to_csv_index_label, create_fn=create_df)
    TESTUTIL.compare(to_csv_2, create_fn=create_df)
    TESTUTIL.compare(to_csv_3, create_fn=create_df)
    TESTUTIL.compare(to_csv_4, create_fn=create_df)
    TESTUTIL.compare(to_csv_5, create_fn=create_df)
    TESTUTIL.compare(to_csv_6, create_fn=create_df)
    TESTUTIL.compare(to_csv_7, create_fn=create_df)
    TESTUTIL.compare(read_csv_index_col, create_fn=create_module_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", default=False, type=bool)
    parser.add_argument("--rows", default=-1, type=int)
    parser.add_argument("--partitions", default=-1, type=int)

    args = parser.parse_args()
    if args.performance:
        print("Running in performance mode with larger table")
        TESTUTIL.set_perf_mode()
        if args.rows == -1:
            TESTUTIL.set_size(rows=10000000, cols=50, unique=5000)

    if args.rows != -1:
        TESTUTIL.set_size(rows=args.rows, cols=50, unique=5000)
    if args.partitions != -1:
        mpd.config.set_partition_shape((args.partitions, args.partitions))

    test_read_csv_edge_cases()
    test_read_csv_basic()
    test_to_csv_compression()
