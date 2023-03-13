# 欢迎来到MindPandas

[View English](./README.md)

<!-- TOC -->

- [MindPandas介绍](#MindPandas介绍)
    - [概述](#概述)
    - [总体架构](#总体架构)
- [安装方式](#安装方式)
    - [确认系统环境信息](#确认系统环境信息)
    - [pip安装](#pip安装)
    - [源码安装](#源码安装)
- [验证安装是否成功](#验证安装是否成功)
- [快速入门](#快速入门)
- [文档](#文档)
- [贡献](#贡献)
- [版本说明](#版本说明)
- [许可证](#许可证)

<!-- /TOC -->

## MindPandas介绍

### 概述

MindPandas使用分布式计算引擎来加速Pandas运算，与现有Pandas代码无缝集成和兼容，使用MindPandas进行计算，可以使用计算机上所有的CPU核心，这使得MindPandas在处理较大的数据集上效果特别好。

### 总体架构

MindPandas采用分布式实现，而原生Pandas是基于单线程实现的。这意味着每次只能使用一个CPU核。

然而，MindPandas能使用机器上更多的线程和内核，或者整个集群的所有内核。

详细架构设计，请参阅[官网文档](https://www.mindspore.cn/mindpandas/docs/zh-CN/master/index.html)。

## 安装方式

### 确认系统环境信息

下表列出了安装、编译和运行MindPandas所需的系统环境：

| 软件名称 |                版本                |
| :------: |:--------------------------------:|
|  Linux-x86_64操作系统 | Ubuntu \>=18.04<br/>Euler \>=2.9 |
|  Python  |             3.8-3.9              |
|  glibc  |             \>=2.25              |

- 请确保环境中安装了libxml2-utils。
- 其他的第三方依赖请参考[requirements文件](https://gitee.com/mindspore/mindpandas/blob/master/requirements.txt)。

### pip安装

请从[MindPandas下载页面](https://www.mindspore.cn/versions)下载whl包，使用`pip`指令安装。

> 在联网状态下，安装whl包时会自动下载MindPandas安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。

### 源码安装

下载[源码](https://gitee.com/mindspore/mindpandas.git)，下载后进入mindpandas目录，运行build.sh脚本。

```shell
git clone https://gitee.com/mindspore/mindpandas.git
cd mindpandas
bash build.sh
```

编译完成后，whl包在output目录下，使用pip安装：

```shell
pip install output/mindpandas-0.1.0-cp38-cp38-linux_x86_64.whl
```

## 验证安装是否成功

在shell中执行以下命令，如果没有报错`No module named 'mindpandas'`，则说明安装成功。

```shell
python -c "import mindpandas"
```

## 快速入门

首先通过如下命令导入MindPandas。

```python
import mindpandas as pd
```

通过如下命令设置MindPandas的运行模式，这样可以加快您的MindPandas工作流程。

```python
pd.set_concurrency_mode('multithread')
```

完整示例如下：

```python
>>> import mindpandas as pd
>>> pd.set_concurrency_mode('multithread')
>>> pd.set_partition_shape((16, 2))
>>> pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
>>> sum = pd_df.sum()
>>> print(sum)
0 5
1 7
2 9
Name: sum, dtype: int64
```

## 文档

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://www.mindspore.cn/mindpandas/docs/zh-CN/master/index.html)。

## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING_CN.md)。

## 版本说明

版本说明请参阅[RELEASE](https://gitee.com/mindspore/mindpandas/blob/master/RELEASE.md)。

## 许可证

[Apache License 2.0](https://gitee.com/mindspore/mindpandas/blob/master/LICENSE)
