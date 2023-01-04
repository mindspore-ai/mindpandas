# MindPandas Release Notes

## MindPandas 0.2.0 Release Notes

### 主要特性和增强

- [STABLE] 提供了基于共享内存的数据管道，数据无需落盘即可从MindPandas数据处理进程传输至MindSpore训练进程，解决了数据分析框架与AI训练框架割裂的问题。
- [STABLE] 简化单机多进程模式使用方式，无需手动部署分布式计算引擎。
- [STABLE] 支持Python3.9。
- [STABLE] 增加了落盘功能，当datamem使用率超过预设的阈值时使用磁盘空间。
- [STABLE] 优化分布式计算引擎冷启动时间。
- [STABLE] 优化多进程模式内存占用。
- [STABLE] 重构统计类和比较类API，并提升部分性能。

### API 变更

- [STABLE] 新增API `mindpandas.DataFrame.memory_usage`。
- [STABLE] 新增API `mindpandas.DataFrame.count`。
- [STABLE] 新增API `mindpandas.DataFrame.product`。
- [STABLE] 新增API `mindpandas.DataFrame.var`。
- [STABLE] 新增API `mindpandas.DataFrame.prod`。
- [STABLE] 新增API `mindpandas.Series.prod`。
- [STABLE] 新增API `mindpandas.Series.isin`。
- [STABLE] 新增API `mindpandas.Series.item`。
- [STABLE] 新增API `mindpandas.Series.cummin`。
- [STABLE] 新增API `mindpandas.Series.count`。
- [STABLE] 新增API `mindpandas.Series.cummax`。

### Bug Fixes

- [BUGFIX] 修复了 `read_csv` 无法处理URL的问题。
- [BUGFIX] 修复了 `drop` API在某些情况下计算结果错误的问题。
- [BUGFIX] 修复了当使用 `yrctl start` 启动分布式计算引擎过程时出错，需要手动退出的问题。
- [BUGFIX] 修复了设置代理后分布式计算引擎无法启动的问题。

### 贡献者

感谢以下人员做出的贡献:

caiyimeng, chenyue li, dessyang, liyuxia, lichen_101010, Martin Yang, panfengfeng, RobinGrosman, shenghong96, Tom Chen, wangyue, weisun092, xiaohanzhang, xutianyu, yanghaitao, youtianming

欢迎以任何形式对项目提供贡献！

## MindPandas 0.1.0 Release Notes

MindPandas是一款兼容Pandas接口，同时提供分布式处理能力的数据分析框架，致力于提供支持大数据量、高性能的表格类型数据处理能力，同时又能与训练流程无缝结合，使得昇思MindSpore支持完整AI模型训练全流程的能力。

### 主要特性

#### MindPandas

- [STABLE] MindPandas提供100+分布式化Pandas API，修改少量代码即可从原生Pandas切换到MindPandas。
- [STABLE] 提供多进程和多线程两种执行方式，以单机或集群的方式提供数据的并行处理能力，提高数据处理的性能。
- [STABLE] 高效利用集群资源以处理大规模数据，解决原生Pandas因内存限制无法处理大数据量的问题。

### 贡献者

感谢以下人员做出的贡献:

caiyimeng, chenyue li, dessyang, liyuxia, lichen_101010, Martin Yang, panfengfeng, RobinGrosman, shenghong96, Tom Chen, wangyue, weisun092, xiaohanzhang, xutianyu, yanghaitao, youtianming

欢迎以任何形式对项目提供贡献！