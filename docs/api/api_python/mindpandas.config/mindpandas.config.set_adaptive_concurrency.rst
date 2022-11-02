.. py:function:: mindpandas.config.set_adaptive_concurrency(adaptive)

    用户可以设置自适应并发，让 `read_csv` 基于文件大小自动选择并发模式。可选项为"True"或"False"。设置为True时，从 `read_csv` 读取的文件大小超过18MB，或者使用"pandas.DataFrame"初始化的"mindpandas.DataFrame"，内存占用大于1GB时将使用多进程模式，否则使用多线程模式。设置为False时，会使用当前的并发模式。

    参数：
        - **adaptive** (bool) - 为True时开启自适应模式，为False时关闭自适应模式。

    异常：
        - **ValueError** - `adaptive` 不是True或者False。