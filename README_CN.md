# 欢迎来到MindPandas

[View English](./README.md)

<!-- TOC -->

- [MindPandas介绍](#mindpandas介绍)
    - [概述](#概述)
    - [总体架构](#总体架构)
- [安装方式](#安装方式)
    - [pip安装](#pip安装)
    - [源码编译安装](#源码编译安装)
- [快速入门](#快速入门)
- [文档](#文档)
- [贡献](#贡献)
- [版本说明](#版本说明)
- [许可证](#许可证)

<!-- /TOC -->

## MindPandas介绍

### 概述
MindPandas使用分布式计算引擎来加速pandas运算，与现有pandas代码无缝集成和兼容，使用MindPandas进行计算，可以使用计算机上所有的CPU核心，这使得MindPandas在处理较大的数据集上效果特别好。

### 总体架构
MindPandas采用分布式实现，而原生pandas是基于单线程实现的。这意味着每次只能使用一个CPU核。

然而，MindPandas能使用机器上更多的线程和内核，或者整个集群的所有内核。

## 安装方式

### pip安装

安装PyPI上的版本

```
pip install mindpandas
```
### 源码编译安装

1.从代码仓下载源码

```
git clone https://gitee.com/mindspore/mindpandas.git
```
2.编译安装mindpandas

构建whl包进行安装，首先进入源码的根目录，先执行build目录下的MindPandas编译脚本，再执行命令安装output目录下生成的whl包。

```
cd mindpandas
bash build/build.sh
pip install output/mindpandas-0.1.0-cp37-cp37m-linux_x86_64.whl
```
### 验证是否安装成功

在Python交互式命令行执行如下命令成功，即为安装成功。

```
import mindpandas as mpd
```
## 快速入门

首先安装依赖库和包：pandas、numpy，
然后通过如下命令导入MindPandas。

```
import mindspore.pandas as ms_pd
```
通过如下命令设置MindPandas的运行模式，这样可以加快您的pandas工作流程。

```
ms_pd.config.set_concurrency_mode('multithread') 

```
完整示例如下：

```python
>>> import mindspore.pandas as mpd
>>> mpd.config.set_concurrency_mode('multithread')
>>> mpd.set_partition_shape((16, 2))
>>> mpd_df = mpd.DataFrame([[1, 2, 3], [4, 5, 6]])
>>> sum = mpd_df.sum()
>>> print(sum)
0 5
1 7
2 9
Name: sum, dtype: int64
```

## 文档

有关安装指南、教程和API的更多详细信息，请参阅[用户文档]()。

## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 版本说明

版本说明请参阅[RELEASE](https://gitee.com/mindspore/mindpandas/blob/master/RELEASE.md)。

## 许可证

[Apache License 2.0](https://gitee.com/mindspore/mindpandas/blob/master/LICENSE)
