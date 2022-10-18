.. py:function:: mindpandas.set_partition_shape(shape)

    用户可以设置数据分片的shape，其中shape[0]是行方向上的分片数量，shape[1]是列方向上的分片数量。例如，设置shape为(16, 16)时，MindPandas会尝试将数据切分为16*16个分片。

    参数：
        - **shape** (tuple) - 在每个轴上期望的分片数。shape是一个包含两个正整数的元组，第一个元素是行方向上的分片数量，第二个元素是列方向上的分片数量。

    异常：
        - **ValueError** - shape不是tuple类型或者shape的值不是正整数。