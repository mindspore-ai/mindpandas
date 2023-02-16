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
"""MindSpore Pandas Lazy Mode WorkSpace Class"""
import mindpandas.dist.client as dist_client

from .checker import Checker
from .graph import DirectedGraph
from .optimizer import Optimizer
from .rewrite import Rewrite
from .snapshot import SnapShot


class WorkSpace:
    """MindSpore Pandas Lazy Mode WorkSpace Class"""
    dag = DirectedGraph()
    # A dict of 3-ary tuple: (phase, op, rule)
    blacklist = dict()
    snapshot = SnapShot()

    @classmethod
    def disable(cls, phase, op, rule):
        cls.blacklist[(phase, op, rule)] = 1

    @classmethod
    def enable(cls, phase, op, rule):
        try:
            cls.blacklist.pop((phase, op, rule))
        except KeyError:
            pass

    @classmethod
    def is_disable(cls, phase, op, rule):
        try:
            _ = cls.blacklist[(phase, op, rule)]
        except KeyError:
            return False
        else:
            return True

    @classmethod
    def _cache(cls, node_id, result):
        result.node_id = node_id
        cls.dag.cache(node_id, result)

    @classmethod
    def explain(cls, plan, pr_stats=False, pr_details=False):
        """print the IR info"""
        node_id = plan.node_id
        out: str = ""
        out += "Plan:" + '\n'
        out += "=====" + '\n'
        out += cls.dag.print(node_id, pr_stats, pr_details)
        out += "== End ==" + '\n'
        return out

    @classmethod
    def show_snapshot(cls):
        cls.snapshot.dump_acc_time()

    @classmethod
    def _check(cls, workspace, root, debug, pr_stats, pr_details):
        checker = Checker(workspace, root, debug, pr_stats, pr_details)
        validated_plan = checker.run()
        return validated_plan

    @classmethod
    def _rewrite(cls, workspace, plan, debug, pr_stats, pr_details):
        rewrite = Rewrite(workspace, plan, debug, pr_stats, pr_details)
        logical_plan = rewrite.run()
        return logical_plan

    @classmethod
    def _optimize(cls, workspace, plan, debug, pr_stats, pr_details):
        # TODO: Generate a physical plan from the logical plan
        optimizer = Optimizer(workspace, plan, debug, pr_stats, pr_details)
        physical_plan = optimizer.run()
        return physical_plan

    @classmethod
    def _execute(cls, plan, debug, pr_stats, pr_details):
        client = dist_client.Client(plan, debug, pr_stats, pr_details)
        result = client.run()
        return result

    @classmethod
    def run(cls, node, debug=False, pr_stats=False, pr_details=False):
        """
        pandas script
            |
         +-------+
         | Parse | -> Pre-validated plan
         +-------+       |
                   +----------+
                   | Semantic |
                   |  check   | -> Validated logical plan
                   +----------+       |
                                  +---------+
                                  | Rewrite | -> Optimal logical plan
                                  +---------+       |
                                              +----------+
                                              | Optimize | -> Physical plan
                                              +----------+       |
                                                           +-------+
                                                           | Build | -> Byte code
                                                           +-------+     |
                                                                     +-----+
                                                                     | Run |
                                                                     +-----+
                                                                         |
        Result <---------------------------------------------------------+

        """
        import mindpandas.internal_config as i_config

        cls.snapshot.reset()
        try:
            cls.snapshot.entry("compile")
            # This a hack to return eager Dataframe. Will figure out a better way later
            i_config.set_lazy_mode(False)
            node_id = node.backend_frame.node_id
            # Perform semantic checking
            # This phase does not modify the input plan. It only flags the first error, if any.
            validated_plan = cls._check(cls, node_id, debug, pr_stats, pr_details)
            # Run the rule-based optimizer to rewrite the validated plan into an optimal logical plan
            logical_plan = cls._rewrite(cls, validated_plan, debug, pr_stats, pr_details)
            # Run the cost-based optimizer to pick the lowest cost physical plan
            physical_plan = cls._optimize(cls, logical_plan, debug, pr_stats, pr_details)
            cls.snapshot.exit("compile")
            # No "Build" phase yet.
            # We run the plan by processing each operator in the plan in post-order
            cls.snapshot.entry("execution")
            result = cls._execute(physical_plan, debug, pr_stats, pr_details)
            cls.snapshot.exit("execution")
        finally:
            i_config.set_lazy_mode(True)

        # cache the result?
        cls._cache(node_id, result)
        return result
