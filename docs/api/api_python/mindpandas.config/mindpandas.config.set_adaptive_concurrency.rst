.. py:function:: mindpandas.config.set_adaptive_concurrency(adaptive, **kwargs)

    用户可以选择是否开启自适应并发模式。

    参数：
        - **adaptive** (bool) - 是否开启自适应并发模式。设置为True时开启，从 `read_csv` 读取的文件大小超过18MB时、使用"pandas.DataFrame"初始化的"mindpandas.DataFrame"时或者内存占用大于1GB时，将使用多进程模式，否则使用多线程模式。设置为False时关闭自适应并发模式，使用当前环境设置的并发模式。
        - **\*\*kwargs** - 当 `adaptive` 为False时不需要额外的参数，为True时 `kwargs` 包括：

          - address (str) : master节点的IP地址。可选，默认使用"127.0.0.1"。
          - cpu (int) : 用户设定使用的CPU核数。可选，默认使用当前节点的所有核。
          - datamem (int) : 共享内存的大小，单位是MB。可选，默认使用当前空闲内存的30%。
          - mem (int) : MindPandas使用的总内存（包含共享内存），单位是MB。可选，默认使用当前空闲内存的90%。
          - tmp_dir (str) : 临时文件的存放路径。可选，默认使用"/tmp/mindpandas/"作为临时文件的存放路径。
          - tmp_file_size_limit (int) : 临时文件的大小限制，单位是MB。可选，默认上限为当前空闲磁盘空间的95%。

    异常：
        - **ValueError** - `adaptive` 不是True或者False。