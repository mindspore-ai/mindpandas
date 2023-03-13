.. py:class:: mindpandas.channel.DataSender(address, namespace='default', num_shards=1, dataset_name='dataset', full_batch=False, max_queue_size=10)

    channel的发送方（输入端），通过channel发送新对象。

    参数：
        - **address** (str) - 当前sender运行的节点的ip地址。
        - **namespace** (str, 可选) - channel所属的命名空间。默认值：`default` ，sender将在命名空间 `default` 中运行。不同命名空间的DataSender和DataReceiver不能相互连接。
        - **num_shards** (int, 可选) - 指定将数据划分为多少个分片。默认值：1。
        - **dataset_name** (str, 可选) - 数据集的名称。默认值：`dateset` 。
        - **full_batch** (bool, 可选) - 如果为True，则每个分片将获得sender发送的完整数据。否则，每个分片只能获取部分数据。默认值：False。
        - **max_queue_size** (int, 可选) - 队列中能够缓存的最大元素数量。默认值：10。

    异常：
        - **ValueError** - 当 `num_shards` 为无效值时。

    .. note::
        分布式执行引擎必须提前启动。

    .. py:method:: send(obj)

        通过channel发送对象。

        参数：
            - **obj** (Union[numpy.ndarray, list, mindpandas.DataFrame]) - 要发送的对象。

        异常：
            - **AttributeError** - 当对象没有 `len()` 时。
            - **TypeError** - 当 `obj` 不能使用[]进行索引操作时。

    .. py:method:: num_shards
        :property:

        返回当前channel的 `num_shards` 。

    .. py:method:: full_batch
        :property:

        返回 `full_batch` 的值。

    .. py:method:: get_queue(shard_id=None)

        返回与指定的 `shard_id` 对应的数据集中尚未消费的对象引用。

        参数：
            - **shard_id** (int, 可选) - 请求分片的id。默认值：None，将返回所有分片。

        返回：
            List，存储分片中数据的引用的列表。
