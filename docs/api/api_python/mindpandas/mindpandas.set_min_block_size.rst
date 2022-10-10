mindpandas.set_min_block_size
=============================

.. py:function:: mindpandas.set_min_block_size(min_block_size)

    用户可以使用此API设置每个分片的最小块大小， `min_block_size` 表示分片的每个轴的最小尺寸。每个分片的大小将大于或等于 `(min_block_size，min_block_size)` ，除非原始数据就小于 `(min_block_size，min_block_size)` 。例如，对于一个只有16列、分片维度为(2, 2)的"DataFrame"，如果 `min_block_size` 设置为32，在分片时不会进一步拆分列。

    参数：
        - **min_block_size** (int) - 分片最小块的最小行数和列数。

    异常：
        - **ValueError** - `min_block_size` 不是int类型。