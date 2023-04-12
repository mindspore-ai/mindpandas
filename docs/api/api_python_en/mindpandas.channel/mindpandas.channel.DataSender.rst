.. py:class:: mindpandas.channel.DataSender(address, namespace='default', num_shards=1, dataset_name='dataset', full_batch=False, max_queue_size=10)

    The sender (input side) of the channel. It can be used for sending new object through the channel.

    Parameters
        - **address** (str) - The ip address of the node current sender runs on.
        - **namespace** (str, optional) - The namespace that the channel belongs to. By default the value is `default` and the sender will be running in namespace `default`. DataSender and DataReceiver in different namespaces cannot connect to each other.
        - **num_shards** (int, optional) - Specifies how many shards the data will be divided into. By default the value is 1.
        - **dataset_name** (str, optional) - The name of the dataset. By default the value is `dataset`.
        - **full_batch** (bool, optional) - If true, each shard will get complete data sent by the sender. Otherwise each shard only gets part of the data. By default the value is False.
        - **max_queue_size** (int, optional) - The maximum number of data that can be cached in the queue. By default the value is 10.

    Raises
        - **ValueError** - When `num_shards` is an invalid value.

    .. note::
        Distributed executor has to be started in advance.

    .. py:method:: send(obj)

        Send object through the channel.

        Parameters
            - **obj** (Union[numpy.ndarray, list, mindpandas.DataFrame]) - The object to send.

        Raises
            - **TypeError** - If the type of the `obj` is invalid.
            - **ValueError** - If the length of the `obj` is not a positive integer or cannot be evenly divided by the number of shards.

    .. py:method:: num_shards
        :property:

        Returns the `num_shards` of current channel.

    .. py:method:: full_batch
        :property:

        Returns the value of `full_batch`.

    .. py:method:: get_queue(shard_id=None)

        Returns the object references that haven't been consumed in the shard specified by `shard_id`.

        Parameters
            - **shard_id** (int, optional) - The id of the requested shard. By default the value is None and it will return all shards.

        Returns
            List，stores references of the data in the shard.
