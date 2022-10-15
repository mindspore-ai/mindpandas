# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
This module defines DataSender and DataLoader class which is used to transfer data between processes.
"""
import time
import logging
from collections import deque

import pandas
import mindpandas
import yr


__all__ = ['DataSender', 'DataReceiver']


@yr.instance
class Actor:
    """The actor that stores the object references. The whole dataset is divided into several shards, and data within
    the same shard is stored in a deque and the deques are stored in a dictionary where the key is shard_id.

    Args:
        num_shards(int): The number of shard (receiver). Each shard will retrieve different data from the actor.
    """

    def __init__(self, num_shards=1):
        self.object_pool = {}
        self.shard_id = 0
        self.num_shards = num_shards
        for i in range(num_shards):
            self.object_pool[i] = deque()

    def put(self, ref_list, shard_id=-1, full_batch=False):
        """Put an object into object store and append the reference to the queue specified by shard_id.

        Args:
            ref_list (list[ObjectReference]): The object reference in object store.
            shard_id (int, optional): The shard that obj belongs to. If -1, use internal shard_id. By default the value
                is -1.
            full_batch (bool, optional): If true, all shards will receive the object. By default the value is False.
        """
        ref = ref_list[0]
        if full_batch:
            for q in self.object_pool.values():
                q.append(ref)
        else:
            if shard_id == -1:
                self.object_pool[self.shard_id].append(ref)
                self.shard_id = (self.shard_id + 1) % self.num_shards
            else:
                self.object_pool[shard_id].append(ref)

    def get(self, shard_id):
        """Get the last object reference from the queue specified by shard_id.

        Args:
            shard_id: The shard to get data from.

        Returns:
            A object reference if queue is not empty, otherwise return None.
        """
        if len(self.object_pool[shard_id]):
            ref = self.object_pool[shard_id].popleft()
            return ref
        return None

    def get_num_shards(self):
        return self.num_shards

    def get_queue(self, shard_id):
        if shard_id is None:
            return self.object_pool
        return self.object_pool.get(shard_id, None)


class BaseChannel:
    """The base class for DataSender and DataReceiver. Initializes distributed executor if necessary.

    Args:
        address (str): The ip address of the master node of distributed executor.
    """

    def __init__(self, address):
        self.initialized = False

        if not yr.is_initialized():
            logging.info('No yr cluster detected, starting a new one.')

        conf = yr.Config(function_id="sn:cn:yrk:12345678901234561234567890123456:function:0-default-func:$latest",
                         in_cluster=True,
                         recycle_time=300,
                         server_address=address,
                         ds_address=address)
        yr.init(conf)


class DataSender(BaseChannel):
    """The sender(input side) of the channel. It can be used for sending new object through the channel.

    Args:
        address (str): The ip address of the node current sender runs on.
        namespace (str, optional): The namespace that the channel belongs to. By default the value is "default" and the
            sender will be running in namespace "default". DataSender and DataReceiver in different namespaces cannot
            connect to each other.
        num_shards (int, optional): Specifies how many shards the data will be divided into. By default the value is 1.
        dataset_name (str, optional): The name of the dataset. By default the value is "dataset".
        full_batch (bool, optional): If true, each shard will get complete data sent by the sender. Otherwise each shard
            only gets part of the data. By default the value is False.

    Raises:
        ValueError: If `num_shards` is an invalid value.

    Note:
        Distributed executor has to be started in advance.

    Examples:
        >>> from mindpandas.channel import DataSender
        >>> sender = DataSender(address="127.0.0.1")
    """

    def __init__(self,
                 address,
                 namespace='default',
                 num_shards=1,
                 dataset_name='dataset',
                 full_batch=False
                 ):
        if not isinstance(num_shards, int) or num_shards <= 0:
            raise ValueError(f"num_shards has to be a positive integer, got {num_shards} of type {type(num_shards)}")
        self._num_shards = num_shards
        self._full_batch = full_batch

        super(DataSender, self).__init__(address=address)

        try:
            old_actor = yr.get_instance(name=dataset_name, namespace=namespace)
            old_actor.terminate()
        except RuntimeError:
            pass

        option = yr.InvokeOptions(name=dataset_name, namespace=namespace)
        actor = Actor.options(option).invoke(num_shards=self.num_shards)
        self.actor = actor

    def send(self, obj):
        """Send object through the channel.

        Args:
            obj (array-like): The object to send. It should be an array-like object(e.g. numpy.ndarray,
                python list) that has "len" property, or a DataFrame object.

        Raises:
            AttributeError: When the object has no len().
            TypeError: When obj is not subscriptable.

        Examples:
            >>> # sender is an instance object of DataSender
            >>> data = [1, 2, 3, 4]
            >>> sender.send(data)
        """
        if self.full_batch:
            ref = yr.put(obj)
            self.actor.put.invoke([ref], full_batch=True)
        else:
            if isinstance(obj, mindpandas.DataFrame):
                obj.repartition((self.num_shards, 1))
                array_of_parts = obj.remote_to_numpy().flatten()
                for i, part in enumerate(array_of_parts):
                    if hasattr(part, 'data_id'):
                        ref = part.data_id
                    else:
                        data = part.get()
                        ref = yr.put(data)
                    self.actor.put.invoke([ref], shard_id=i)
            else:
                try:
                    quo = len(obj) // self.num_shards
                    for i in range(self.num_shards):
                        ref = yr.put(obj[i * quo:(i + 1) * quo])
                        self.actor.put.invoke([ref], shard_id=i)
                except (AttributeError, TypeError) as e:
                    raise e

    @property
    def num_shards(self):
        """Returns the `num_shards` of current channel."""
        return self._num_shards

    @property
    def full_batch(self):
        """Returns the value of `full_batch`."""
        return self._full_batch

    def get_queue(self, shard_id=None):
        """Returns the object references that haven't been consumed in the shard specified by `shard_id`.

        Args:
            shard_id (int, optional): The id of the requested shard. By default the value is None and it will return all
                shards.

        Returns:
            List, the queue that stores references of the data that haven't been consumed in the shard.

        Examples:
            >>> # sender is an instance object of DataSender
            >>> sender.get_queue()
        """
        return yr.get(self.actor.get_queue.invoke(shard_id))


class DataReceiver(BaseChannel):
    """The receiver (output side) of the channel. It can be used for receiving new object from the channel.

    Args:
        address (str): The ip address of the node current receiver runs on.
        namespace (str, optional): The namespace that the channel belongs to. By default the value is "default" and the
            receiver will be running in namespace "default". DataSender and DataReceiver in different namespaces cannot
            connect to each other.
        shard_id (int, optional): Specifies the shard of data that is received by current receiver. By default the value
            is 0 and the receiver will get data from the shard with id 0.
        dataset_name (str, optional): The name of the dataset. By default the value is "dataset".

    Note:
        Distributed executor has to be started and a DataSender has to be initialized in advance. To pair with the
            correct DataSender, the `namespace` and `dataset_name` have to be identical to the DataSender.

    Examples:
        >>> from mindpandas.channel import DataReceiver
        >>> sender = DataReceiver(address="127.0.0.1")
    """

    def __init__(self,
                 address,
                 namespace='default',
                 shard_id=0,
                 dataset_name='dataset'
                 ):
        self.cool_down = 0.1
        self._shard_id = shard_id
        self.actor = None
        super(DataReceiver, self).__init__(address=address)
        self.actor = yr.get_instance(name=dataset_name, namespace=namespace)
        logging.info(f'Retrieved dataset "{dataset_name}"')
        self._num_shards = yr.get(self.actor.get_num_shards.invoke())

    def recv(self):
        """Get data from the channel.

        Returns:
            object, the least recent object in the shard that haven't been consumed.

        Raises:
            ValueError: When the `shard_id` of current receiver is invalid.

        Examples:
            >>> # receiver is an instance object of DataReceiver
            >>> data = receiver.recv()
        """
        if self.shard_id < 0 or self.shard_id >= self.num_shards:
            raise ValueError(f"Shard id '{self.shard_id}'out of range, should be in [0, {self.num_shards})")
        dref = None
        while dref is None:
            rref = self.actor.get.invoke(self.shard_id)
            dref = yr.get(rref)
            self._wait()
        result = yr.get(dref)
        if isinstance(result, pandas.DataFrame) and result.shape == (1, 1):
            result = result.squeeze()
        return result

    def _wait(self):
        time.sleep(self.cool_down)

    @property
    def shard_id(self):
        """Returns the `shard_id` of current receiver."""
        return self._shard_id

    @property
    def num_shards(self):
        """Returns the `num_shards` of current channel."""
        return self._num_shards
