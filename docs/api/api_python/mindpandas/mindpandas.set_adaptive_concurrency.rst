mindpandas.set_adaptive_concurrency
======================

.. py:function:: mindpandas.set_adaptive_concurrency(**kwargs)

    设置后端运行模式。

    .. note::
        - 可以设置mode为multithread或yr两种后端模式，默认值为multithread。multithread模式为多线程后端，yr模式为多进程后端。

    参数：
        - **mode** (str) - 设置后端运行模式。

    异常：
        - **ValueError** - 该模式不支持。