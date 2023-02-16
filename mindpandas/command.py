# Copyright 2023 Huawei Technologies Co., Ltd
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
"""MindPandas lazy mode command file"""
import mindpandas as mpd
from .compiler.lazy.workspace import WorkSpace as ws


def explain(node):
    """print the IR info"""
    print(ws.explain(node))


def explain_detail(node):
    """print the IR info in details"""
    print(ws.explain(node, pr_stats=False, pr_details=True))


def explain_stats(node):
    """print the IR info with the statistics"""
    print(ws.explain(node, pr_stats=True, pr_details=False))


def explain_full(node):
    """print the IR info in details with the statistics"""
    print(ws.explain(node, pr_stats=True, pr_details=True))


def run(node, debug_mode=False):
    """API to execute the mindpandas IR"""
    if mpd.is_lazy_mode():
        result = ws.run(node, debug=debug_mode)
        return result
    return node


def debug(node, pr_stats=False, pr_details=False, pr_runtime=False):
    """execute the mindpandas IR in a debug mode"""
    result = ws.run(node, True, pr_stats, pr_details)
    if pr_runtime:
        ws.show_snapshot()
    return result


def disable(phase, op, rule):
    """disable the certain phase of rules to be applied to operators"""
    ws.disable(phase, op, rule)


def enable(phase, op, rule):
    """enable the certain phase of rules to be applied to operators"""
    ws.enable(phase, op, rule)
