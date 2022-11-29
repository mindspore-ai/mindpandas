import random
import pandas as pd
import mindpandas as mpd
import numpy as np
import pytest

DENSE_NUM = 13
SPARSE_NUM = 26
ROW_NUM = 10000


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("concurrency_mode", ["multithread", "multiprocess"])
def test_end_to_end(concurrency_mode):
    """
    Feature: System Test
    Description: Test multiple APIs
    Expectation: Runs successfully with the same result
    """
    cat_val, int_val, lab_val = [], [], []
    max_dict, min_dict = {}, {}
    mpd.set_adaptive_concurrency(False)
    mpd.set_concurrency_mode(concurrency_mode, address="127.0.0.1")
    mpd.set_partition_shape((16, 3))

    def get_cat_feature(length):
        result = hex(random.randint(0, 16 ** length)).replace('0x', '').upper()
        if len(result) < length:
            result = '0' * (length - len(result)) + result
        return str(result)

    def get_int_feature():
        return random.randint(-10, 10000)

    def get_lab_feature():
        x = random.randint(0, 1)
        return round(x)

    def get_weight(x):
        ret = []
        for index, val in enumerate(x):
            if index < DENSE_NUM:
                col = f'I{index + 1}'
                ret.append((val - min_dict[col]) / (max_dict[col] - min_dict[col]))
            else:
                ret.append(1)
        return ret

    def get_id(x):
        ret = []
        for index, val in enumerate(x):
            if index < DENSE_NUM:
                ret.append(index + 1)
            else:
                ret.append(val)
        return ret

    def get_label(x):
        return np.array([x])

    for i in range(ROW_NUM * SPARSE_NUM):
        cat_val.append(get_cat_feature(8))
    np_cat = np.array(cat_val).reshape(ROW_NUM, SPARSE_NUM)
    df_cat = pd.DataFrame(np_cat, columns=[f'C{i + 1}' for i in range(SPARSE_NUM)])
    mdf_cat = mpd.DataFrame(np_cat, columns=[f'C{i + 1}' for i in range(SPARSE_NUM)])
    assert df_cat.equals(mdf_cat.to_pandas())

    for i in range(ROW_NUM * DENSE_NUM):
        int_val.append(get_int_feature())
    np_int = np.array(int_val).reshape(ROW_NUM, DENSE_NUM)
    df_int = pd.DataFrame(np_int, columns=[f'I{i + 1}' for i in range(DENSE_NUM)])
    mdf_int = mpd.DataFrame(np_int, columns=[f'I{i + 1}' for i in range(DENSE_NUM)])
    assert df_int.equals(mdf_int.to_pandas())

    for i in range(ROW_NUM):
        lab_val.append(get_lab_feature())
    np_lab = np.array(lab_val).reshape(ROW_NUM, 1)
    df_lab = pd.DataFrame(np_lab, columns=['label'])
    mdf_lab = mpd.DataFrame(np_lab, columns=['label'])
    assert df_lab.equals(mdf_lab.to_pandas())

    df = pd.concat([df_lab, df_int, df_cat], axis=1)
    mdf = mpd.concat([mdf_lab, mdf_int, mdf_cat], axis=1)
    assert df.equals(mdf.to_pandas())

    for i, j in enumerate(df_int.max()):
        max_dict[f'I{i + 1}'] = j

    for i, j in enumerate(df_int.min()):
        min_dict[f'I{i + 1}'] = j

    features = df.iloc[:, 1:40]
    feat_id = features.apply(get_id, axis=1)
    feat_weight = features.apply(get_weight, axis=1)

    mfeatures = mdf.iloc[:, 1:40]
    mfeat_id = mfeatures.apply(get_id, axis=1)
    mfeat_weight = mfeatures.apply(get_weight, axis=1)
    assert features.equals(mfeatures.to_pandas())
    assert feat_id.equals(mfeat_id.to_pandas())
    assert feat_weight.equals(mfeat_weight.to_pandas())

    df['weight'] = feat_weight
    df['id'] = feat_id
    df['label'] = df['label'].apply(get_label)

    mdf['weight'] = mfeat_weight
    mdf['id'] = mfeat_id
    mdf['label'] = mdf['label'].apply(get_label)
    assert df.equals(mdf.to_pandas())

    df = df[['id', 'weight', 'label']]
    mdf = mdf[['id', 'weight', 'label']]
    assert np.array_equal(df.values, mdf.values)
